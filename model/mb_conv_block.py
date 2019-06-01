
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import six
from six.moves import xrange  # pylint: disable=redefined-builtin
from functools import partial

import chainer
import chainer.functions as F
import chainer.links as L
from chainercv.links.connection import Conv2DBNActiv

from .depthwise_conv_2d_bn_activ import DepthwiseConv2DBNActiv

DTYPE = np.float32


class SEBlock(chainer.Chain):
    def __init__(self, n_channel, ratio, act):
        super(SEBlock, self).__init__()
        self.act = act
        reduction_size = n_channel // ratio

        with self.init_scope():
            self.down = L.Linear(n_channel, reduction_size)
            self.up = L.Linear(reduction_size, n_channel)

    def forward(self, u):
        B, C, H, W = u.shape

        z = F.average(u, axis=(2, 3))
        x = self.act(self.down(z))
        x = F.sigmoid(self.up(x))

        x = F.broadcast_to(x, (H, W, B, C))
        x = x.transpose((2, 3, 0, 1))

        return u * x


class MBConvBlock(chainer.Chain):
    """A class of MBConv: Mobile Inveretd Residual Bottleneck.

    """

    def __init__(self, in_channels, out_channels, ksize, stride, expand_ratio, se_ratio, global_params=None,
                 act=lambda x: x*F.sigmoid(x), initialW=None, bn_kwargs=None):
        """Initializes a MBConv block.
        Args:

        """

        super(MBConvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ksize = ksize
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.stride = stride
        self.has_se = (se_ratio is not None) and (se_ratio > 0) and (se_ratio <= 1)
        self.relu_fn = act

        mid_channels = in_channels * expand_ratio
        with self.init_scope():
            if expand_ratio != 1:
                self.expand_conv = Conv2DBNActiv(
                    in_channels=in_channels,
                    out_channels=mid_channels,
                    ksize=1,
                    stride=1,
                    pad=0,
                    nobias=True,
                    initialW=initialW,
                    activ=act,
                    bn_kwargs=bn_kwargs)


            pad = (ksize - 1) // 2
            # Depth-wise convolution phase
            self.depthwise_conv = DepthwiseConv2DBNActiv(
                in_channels=mid_channels,
                channel_multiplier=1,
                ksize=ksize,
                stride=stride,
                pad=pad,
                nobias=True,
                initialW=initialW,
                activ=act,
                bn_kwargs=bn_kwargs)

            if self.has_se: # In official implementation they use 2d convolution to squeeze-and-expad(excite).
                self.seblock = SEBlock(mid_channels, int(1/se_ratio), act)

            self.project_conv = Conv2DBNActiv(
                    in_channels=mid_channels,
                    out_channels=out_channels,
                    ksize=1,
                    stride=1,
                    nobias=True,
                    initialW=initialW,
                    activ=act,
                    bn_kwargs=bn_kwargs)


    def forward(self, x):
        if self.expand_ratio != 1:
            h = self.expand_conv(x)
            h = self.depthwise_conv(h)
        else:
            h = self.depthwise_conv(x)
        if self.has_se:
            h = self.seblock(h)
        h = self.project_conv(h)
        return h


class RepeatedMBConvBlock(chainer.Chain):
    def __init__(self, in_channels, out_channels, ksize, stride, num_repeat, expand_ratio, se_ratio, global_params=None,
                 act=lambda x: x * F.sigmoid(x), initialW=None, bn_kwargs=None):
        self.num_repeat = num_repeat
        super(RepeatedMBConvBlock, self).__init__()
        with self.init_scope():
            self.link0 = MBConvBlock(in_channels, out_channels, ksize, stride, expand_ratio, se_ratio, global_params, act, initialW, bn_kwargs)
            in_channels = out_channels
            stride = 1
            if num_repeat > 1:
                for i in range(1, num_repeat):
                    link = MBConvBlock(in_channels, out_channels, ksize, stride, expand_ratio, se_ratio, global_params, act, initialW, bn_kwargs)
                    setattr(self, f'link{i}', link)

    def forward(self, x):
        for i in range(self.num_repeat):
            link = getattr(self, f'link{i}')
            x = link(x)
        return x
