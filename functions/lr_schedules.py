import math
import numpy as np

from chainer.training import extension
from chainer.training.triggers.interval_trigger import IntervalTrigger


def get_total_progress(trainer):
    """Returns a number in [0, 1] which describes how much of the total training is finished."""

    stop_trigger = trainer.stop_trigger
    assert isinstance(stop_trigger, IntervalTrigger)

    if stop_trigger.unit == 'epoch':
        return trainer.updater.epoch_detail / stop_trigger.period
    elif stop_trigger.unit == 'iteration':
        return trainer.updater.iteration / stop_trigger.period
    else:
        assert False


class LearningRateScheduler(extension.Extension):

    # Based on ExponentialShift
    def __init__(self, func, attr='lr', target=None, optimizer=None):
        self._attr = attr
        self._func = func
        self._target = target
        self._optimizer = optimizer
        self._t = 0
        self._last_value = None

    def __call__(self, trainer):
        optimizer = self._get_optimizer(trainer)
        updater = trainer.updater
        val = self._func(updater.epoch, updater.epoch_detail, updater.iteration,
                         get_total_progress(trainer))
        self._update_value(optimizer, val)

    def _get_optimizer(self, trainer):
        return self._optimizer or trainer.updater.get_optimizer('main')

    def _update_value(self, optimizer, value):
        setattr(optimizer, self._attr, value)
        self._last_value = value


class _Warmup(object):

    def __init__(self, warmup_iteration, warmup_factor):
        self.warmup_iteration = warmup_iteration
        self.warmup_factor = warmup_factor

    def __call__(self, epoch, epoch_detail, iteration):
        # Regardless of length_def, 'iteration' is used during warmup
        # scheduling.
        if iteration < self.warmup_iteration:
            alpha = float(iteration) / self.warmup_iteration
            warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            return warmup_factor
        else:
            return 1.0


class StepLRSchedule(object):

    def __init__(self, base_lr, step_sizes=(),
                 length_def='epoch',
                 interval=2.6,
                 warmup_iteration=500, warmup_factor=1 / 3.):
        self.base_lr = base_lr
        self.step_sizes = np.array(step_sizes)
        self.interval = interval
        self.length_def = length_def
        self.warmup = _Warmup(warmup_iteration, warmup_factor)

    def __call__(self, epoch, epoch_detail, iteration, total=None):
        warmup_factor = self.warmup(epoch, epoch_detail, iteration)

        if self.length_def == 'iteration':
            length = iteration
        elif self.length_def == 'epoch':
            length = epoch_detail

        if np.all(length < self.step_sizes):
            return warmup_factor * self.base_lr

        if np.all(length >= self.step_sizes):
            idx = len(self.step_sizes)
        else:
            # Example input:
            # length: 500
            # step_sizes: 100, 400, 600
            # Output:
            # (length - step_sizes) >= 0: [T, T, F]
            # idx == 2
            idx = np.argmin((length - np.array(self.step_sizes)) >= 0)

        lr = self.base_lr
        for i in range(idx):
            lr *= 0.1
        return warmup_factor * lr


class CosineLRSchedule(object):

    def __init__(self, base_lr,
                 warmup_iteration=500, warmup_factor=1 / 3.,
                 initial_total_progress=0.0):
        self.initial_total_progress = initial_total_progress
        self.base_lr = base_lr
        self.warmup = _Warmup(warmup_iteration, warmup_factor)

    def __call__(self, epoch, epoch_detail, iteration, actual_total_progress):
        total_progress = self.initial_total_progress + (1.0 - self.initial_total_progress) * actual_total_progress
        warmup_factor = self.warmup(epoch, epoch_detail, iteration)
        ratio = 0.5 * (math.cos(math.pi * total_progress) + 1)
        return warmup_factor * self.base_lr * ratio


class CyclicCosineLRSchedule(object):

    def __init__(self, base_lr, n_cycles,
                 warmup_iteration=500, warmup_factor=1 / 3.):
        self.cosine_lr_schedule = CosineLRSchedule(base_lr, warmup_iteration, warmup_factor)
        self.n_cycles = n_cycles
        self.warmup = _Warmup(warmup_iteration, warmup_factor)

    def __call__(self, epoch, epoch_detail, iteration, total_progress):
        p = total_progress * self.n_cycles
        partial_total_progress = p - int(p)
        return self.cosine_lr_schedule(epoch, epoch_detail, iteration, partial_total_progress)