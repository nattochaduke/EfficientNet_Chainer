import chainer
import chainer.links as L
from chainer.functions import identity
from chainer.functions import relu
from chainer.links import BatchNormalization
from chainer.links import Convolution2D

try:
    from chainermn.links import MultiNodeBatchNormalization
except ImportError:
    pass


class DepthwiseConv2DBNActiv(chainer.Chain):

    def __init__(self, in_channels, channel_multiplier, ksize=None,
                 stride=1, pad=0, nobias=True,
                 initialW=None, initial_bias=None, activ=relu, bn_kwargs={}):
        self.activ = activ
        #print(bn_kwargs)
        super(DepthwiseConv2DBNActiv, self).__init__()

        with self.init_scope():
            self.conv = L.DepthwiseConvolution2D(
                in_channels, channel_multiplier, ksize, stride, pad,
                nobias, initialW, initial_bias)
            if 'comm' in bn_kwargs:
                self.bn = MultiNodeBatchNormalization(
                    int(in_channels*channel_multiplier), **bn_kwargs)
            else:
                self.bn = BatchNormalization(
                    int(in_channels * channel_multiplier), **bn_kwargs)

    def forward(self, x):
        h = self.conv(x)
        h = self.bn(h)
        if self.activ is None:
            return h
        else:
            return self.activ(h)
