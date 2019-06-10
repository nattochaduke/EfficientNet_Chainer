from __future__ import division
import math
from collections import namedtuple

import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import initializers
from chainercv.links.connection import Conv2DBNActiv
from chainercv.links import PickableSequentialChain

from .mb_conv_block import RepeatedMBConvBlock


# RGB order
# This is channel wise mean of mean image distributed at
# https://github.com/KaimingHe/deep-residual-networks
_imagenet_mean = np.array(
    [123.15163084, 115.90288257, 103.0626238],
    dtype=np.float32)[:, np.newaxis, np.newaxis]


def round_channels(channels, global_params):
    """Round number of channels based on depth multiplier"""
    multiplier = global_params["width_coefficient"]
    divisor = global_params["depth_divisor"]
    min_depth = global_params["min_depth"]
    if not multiplier:
        return channels

    channels *= multiplier
    min_depth = min_depth or divisor
    new_channels = max(min_depth, int(channels + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_channels < 0.9 * channels:
        new_channels += divisor
    return int(new_channels)


def round_repeats(repeats, global_params):
    """Round number of filters based on depth multiplier."""
    multiplier = global_params["depth_coefficient"]
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


class EfficientNetTrainingLink(chainer.Chain):
    def __init__(self, name='b0', act='swish', comm=None, workerwisebn=False, no_dropconnect=False):
        super(EfficientNetTrainingLink, self).__init__()
        with self.init_scope():
            self.net = EfficientNet(name, act, comm, workerwisebn, no_dropconnect)


class EfficientNet(PickableSequentialChain):

    def get_global_params(self, model_name):

        params_dict = {
            # (width_coefficient, depth_coefficient, resolution, dropout_ratio)
            'b0': (1.0, 1.0, 224, 0.2),
            'b1': (1.0, 1.1, 240, 0.2),
            'b2': (1.1, 1.2, 260, 0.3),
            'b3': (1.2, 1.4, 300, 0.3),
            'b4': (1.4, 1.8, 380, 0.4),
            'b5': (1.6, 2.2, 456, 0.4),
            'b6': (1.8, 2.6, 528, 0.5),
            'b7': (2.0, 3.1, 600, 0.5),
        }

        params = params_dict[model_name]
        global_params = {'width_coefficient': params[0],
                         'depth_coefficient': params[1],
                         'resolution': params[2],
                         'dropout_ratio': params[3],
                         'batch_norm_momentum': 0.99,
                         'batch_norm_epsilon': 1e-3,
                         'drop_connect_ratio': 0.2, # In the repo this is set 0.2 but in the paper
                                                    # page 7, left column it is set 0.3.
                                                    # And seems this value is never used.
                         'classes': 1000,
                         'depth_divisor': 8,
                         'min_depth': None,
                         }
        return global_params

    _block_args = \
        [
            {'num_repeat': 1, 'ksize': 3, 'stride': 1, 'expand_ratio': 1, 'in_channels': 32, 'out_channels': 16,
             'se_ratio': 0.25},
            {'num_repeat': 2, 'ksize': 3, 'stride': 2, 'expand_ratio': 6, 'in_channels': 16, 'out_channels': 24,
             'se_ratio': 0.25},
            {'num_repeat': 2, 'ksize': 5, 'stride': 2, 'expand_ratio': 6, 'in_channels': 24, 'out_channels': 40,
             'se_ratio': 0.25},
            {'num_repeat': 3, 'ksize': 3, 'stride': 2, 'expand_ratio': 6, 'in_channels': 40, 'out_channels': 80,
             'se_ratio': 0.25},
            {'num_repeat': 3, 'ksize': 5, 'stride': 1, 'expand_ratio': 6, 'in_channels': 80, 'out_channels': 112,
             'se_ratio': 0.25},
            {'num_repeat': 4, 'ksize': 5, 'stride': 2, 'expand_ratio': 6, 'in_channels': 112, 'out_channels': 192,
             'se_ratio': 0.25},
            {'num_repeat': 1, 'ksize': 3, 'stride': 1, 'expand_ratio': 6, 'in_channels': 192, 'out_channels': 320,
             'se_ratio': 0.25},
        ]
    # The resolutions in stages contradicts between paper and official repo.
    # stage 1 (stem) -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 8-> 9 (head and fc)
    #      224         112  112  56   28   28   14   7    7  (paper)
    #      224         112  112  56   28   14   14   7    7  (official and this repo)

    def __init__(self, name='b0', act='swish', comm=None, workerwisebn=False, no_dropconnect=False):

        global_params = self.get_global_params(name)
        if act == 'swish':
            act = lambda x: x * F.sigmoid(x)
        elif act == 'relu':
            act = F.relu
        initialW = initializers.HeNormal(fan_option='fan_out')

        super(EfficientNet, self).__init__()
        self._global_params = global_params
        bn_kwargs = {'eps': global_params['batch_norm_epsilon'],
                     'decay': global_params['batch_norm_momentum']}
        if (not workerwisebn) and (comm is not None):
            bn_kwargs['comm'] = comm
        if no_dropconnect:
            self._global_params['drop_connect_ratio'] = 0

        # stem part
        with self.init_scope():
            out_channels = round_channels(32, self._global_params)
            self.conv_stem = Conv2DBNActiv(
                in_channels=3,
                out_channels=out_channels,
                ksize=3,
                stride=2,
                pad=1,
                nobias=True,
                initialW=initialW,
                activ=act,
                bn_kwargs=bn_kwargs
            )

            total_length = sum([ba['num_repeat'] for ba in self._block_args])
            drop_ratios = np.linspace(0, global_params['drop_connect_ratio'], total_length)
            blocks_cnt = 0
            for i, block_args in enumerate(self._block_args):
                in_channels = round_channels(block_args['in_channels'], self._global_params)
                out_channels = round_channels(block_args['out_channels'], self._global_params)
                num_repeat = round_repeats(block_args['num_repeat'], self._global_params)
                ksize = block_args['ksize']
                stride = block_args['stride']
                expand_ratio = block_args['expand_ratio']
                se_ratio = block_args['se_ratio']
                block = RepeatedMBConvBlock(in_channels, out_channels, ksize, stride, num_repeat, expand_ratio, se_ratio,
                                            drop_ratios=drop_ratios[blocks_cnt: blocks_cnt+num_repeat],
                                            global_params=self._global_params, act=act, initialW=initialW, bn_kwargs=bn_kwargs)
                setattr(self, f'block{1+i}', block)
            # Head part
            in_channels = out_channels
            out_channels = round_channels(1280, self._global_params)

            self.conv_head = Conv2DBNActiv(
                in_channels=in_channels,
                out_channels=out_channels,
                ksize=1,
                nobias=True,
                initialW=initialW,
                activ=act,
                bn_kwargs=bn_kwargs
            )

            self.avg_pooling = lambda x: F.average(x, axis=(2, 3))
            self.fc = L.Linear(out_channels, self._global_params['classes'])
            if self._global_params['dropout_ratio'] > 0:
                self.dropout = lambda x: F.dropout(x, self._global_params['dropout_ratio'])
            #self.prob = F.softmax
