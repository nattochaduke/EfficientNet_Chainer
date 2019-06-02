import chainer
import chainer.functions as F
from chainer import configuration
from chainer import backend
import numpy


def drop_connect(inputs, p):
    """ Drop connect. """

    if not configuration.config.train:
        return inputs

    xp = backend.get_array_module(inputs)
    keep_prob = 1 - p
    batch_size = inputs.shape[0]
    random_tensor = keep_prob
    random_tensor += xp.random.uniform(size=[batch_size, 1, 1, 1])
    binary_tensor = xp.floor(random_tensor)
    output = (inputs / keep_prob) * binary_tensor
    return output
