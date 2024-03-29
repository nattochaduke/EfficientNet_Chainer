from __future__ import print_function
import argparse
import chainer
import chainer.cuda
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions

import chainermn
from chainerui.utils import save_args
from chainercv.links.model.senet import SEResNeXt50

from model.efficient_net import EfficientNet
from datasets.datasets import ImageNetDataset
from datasets.augmentations import get_transforms
from functions.soft import soft_softmax_cross_entropy, soft_accuracy
from functions import lr_schedules

chainer.global_config.autotune = True
chainer.global_config.type_check = False
chainer.global_config.cudnn_fast_batch_normalization = True

# chainermn.create_multi_node_evaluator can be also used with user customized
# evaluator classes that inherit chainer.training.extensions.Evaluator.
class TestModeEvaluator(extensions.Evaluator):

    def evaluate(self):
        model = self.get_target('main')
        model.train = False
        ret = super(TestModeEvaluator, self).evaluate()
        model.train = True
        return ret


def main():
    # Check if GPU is available
    # (ImageNet example does not support CPU execution)
    if not chainer.cuda.available:
        raise RuntimeError('ImageNet requires GPU support.')

    archs = [f'b{i}' for i in range(8)] + ['se']
    patchsizes = {'b0': 224, 'b1': 240, 'b2': 260, 'b3': 300, 'b4': 380, 'b5': 456, 'b6': 528, 'b7': 600, 'se': 224}

    parser = argparse.ArgumentParser(
        description='Learning convnet from ILSVRC2012 dataset')
    parser.add_argument('--arch', '-a', choices=archs, default='b0')
    parser.add_argument('--patchsize', default=None, type=int, help='The input size of images. If not specifed,\
                                                                     architecture-wise default values wil be used.')
    parser.add_argument('--batchsize', '-B', type=int, default=32,
                        help='Learning minibatch size')
    parser.add_argument('--optimizer', default='RMSProp')
    parser.add_argument('--lr', default=0.256, type=float)
    parser.add_argument('--cosine_annealing', action='store_true')
    parser.add_argument('--exponent', type=float, default=0.97)
    parser.add_argument('--exponent_trigger', type=float, default=2.6)
    parser.add_argument('--soft_label', action='store_true')
    parser.add_argument('--epoch', '-E', type=int, default=350,
                        help='Number of epochs to train')
    parser.add_argument('--initmodel',
                        help='Initialize the model from given file')
    parser.add_argument('--loaderjob', '-j', type=int, default=3,
                        help='Number of parallel data loading processes')
    parser.add_argument('--resume', '-r', default='',
                        help='Initialize the trainer from given file')
    parser.add_argument('--out', '-o', default='result',
                        help='Output directory')
    parser.add_argument('--root', '-R', default='../ssd/imagenet',
                        help='Root directory path of image files')
    parser.add_argument('--val_batchsize', '-b', type=int, default=32,
                        help='Validation minibatch size')
    parser.add_argument('--workerwisebn', action='store_true')
    parser.add_argument('--no_dropconnect', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--communicator', default='pure_nccl')
    parser.add_argument('--no_autoaugment', action='store_true')
    parser.add_argument('--dtype', default='float32', choices=['mixed16', 'float32'], help='For now do not use mixed16')
    parser.set_defaults(test=False)
    args = parser.parse_args()

    chainer.global_config.dtype = args.dtype
    comm = chainermn.create_communicator(args.communicator)
    device = comm.intra_rank

    if comm.rank == 0:
        print('==========================================')
        print('Num process (COMM_WORLD): {}'.format(comm.size))
        print('Using {} communicator'.format(args.communicator))
        print('Using {} arch'.format(args.arch))
        print('Num Minibatch-size: {}'.format(args.batchsize))
        print('Num epoch: {}'.format(args.epoch))
        mode = 'workerwise' if args.workerwisebn else 'synchronized'
        print(f'BatchNorm is {mode}')
        print('==========================================')


    if args.soft_label:
        accfun = soft_accuracy
        lossfun = soft_softmax_cross_entropy
    else:
        accfun = F.accuracy
        lossfun = F.softmax_cross_entropy

    if args.arch != 'se':
        model = EfficientNet(args.arch, workerwisebn=args.workerwisebn, no_dropconnect=args.no_dropconnect)
    else:
        model = SEResNeXt50()
    model = L.Classifier(model, lossfun=lossfun, accfun=accfun)
    if args.initmodel:
        print('Load model from', args.initmodel)
        chainer.serializers.load_npz(args.initmodel, model)

    chainer.cuda.get_device_from_id(device).use()  # Make the GPU current
    model.to_gpu()

    # Split and distribute the dataset. Only worker 0 loads the whole dataset.
    # Datasets of worker 0 are evenly split and distributed to all workers.

    patchsize = patchsizes[args.arch] if args.patchsize is None else args.patchsize
    patchsize = (patchsize, patchsize)
    train_transform, val_transform, _ = get_transforms(patchsize, no_autoaugment=args.no_autoaugment,
                                                       soft=args.soft_label)
    if comm.rank == 0:
        train = ImageNetDataset(args.root, 'train')
        val = ImageNetDataset(args.root, 'val')
    else:
        train = None
        val = None
    train = chainermn.scatter_dataset(train, comm, shuffle=True)
    val = chainermn.scatter_dataset(val, comm)
    train = chainer.datasets.TransformDataset(train, train_transform)
    val = chainer.datasets.TransformDataset(val, val_transform)

    # A workaround for processes crash should be done before making
    # communicator above, when using fork (e.g. MultiProcessIterator)
    # along with Infiniband.
    train_iter = chainer.iterators.MultiprocessIterator(
        train, args.batchsize, n_processes=args.loaderjob)
    val_iter = chainer.iterators.MultiprocessIterator(
        val, args.val_batchsize, repeat=False, n_processes=args.loaderjob)

    # Create a multi node optimizer from a standard Chainer optimizer.
    symbol = 'lr'
    if args.optimizer.lower() == 'rmsprop':
        optimizer = chainer.optimizers.RMSprop(lr=args.lr, alpha=0.9)
    elif args.optimizer.lower() == 'momentumsgd':
        optimizer = chainer.optimizers.MomentumSGD(lr=args.lr)
    elif args.optimizer.lower() == 'corrected':
        optimizer = chainer.optimizers.CorrectedMomentumSGD(lr=args.lr)
    elif args.optimizer.lower() == 'adabound':
        optimizer = chainer.optimizers.AdaBound(alpha=args.lr, final_lr=0.5)
        symbol = 'alpha'
    optimizer = chainermn.create_multi_node_optimizer(optimizer, comm)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(1e-5))

    args.out = f'experiments/{args.arch}' + args.out
    save_args(args, args.out)
    # Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=device)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), args.out)

    checkpoint_interval = (10, 'iteration') if args.test else (1, 'epoch')
    val_interval = (10, 'iteration') if args.test else (2, 'epoch')
    log_interval = (10, 'iteration') if args.test else (2, 'epoch')

    checkpointer = chainermn.create_multi_node_checkpointer(
        name='imagenet-example', comm=comm)
    checkpointer.maybe_load(trainer, optimizer)
    trainer.extend(checkpointer, trigger=checkpoint_interval)

    if args.cosine_annealing:
        schedule = lr_schedules.CosineLRSchedule(args.lr)
        if args.optimizer in ['MomentumSGD', 'Corrected']:
            trainer.extend(lr_schedules.LearningRateScheduler(schedule))
    else:
        trainer.extend(extensions.ExponentialShift(symbol, args.exponent), trigger=(args.exponent_trigger, 'epoch'))

    # Create a multi node evaluator from an evaluator.
    evaluator = TestModeEvaluator(val_iter, model, device=device)
    evaluator = chainermn.create_multi_node_evaluator(evaluator, comm)
    trainer.extend(evaluator, trigger=val_interval)

    # Some display and output extensions are necessary only for one worker.
    # (Otherwise, there would just be repeated outputs.)
    if comm.rank == 0:
        trainer.extend(extensions.DumpGraph('main/loss'))
        trainer.extend(extensions.snapshot_object(
            model, 'model_iter_{.updater.iteration}.npz'), trigger=val_interval)
        trainer.extend(extensions.LogReport(trigger=log_interval))
        trainer.extend(extensions.observe_lr(), trigger=log_interval)
        trainer.extend(extensions.PrintReport([
            'epoch', 'iteration', 'main/loss', 'validation/main/loss',
            'main/accuracy', 'validation/main/accuracy', 'lr'
        ]), trigger=log_interval)
        trainer.extend(extensions.ProgressBar(update_interval=100))

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()


if __name__ == '__main__':
    main()