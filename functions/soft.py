import chainer.functions as F
from chainer import backend
from chainer.functions.evaluation.accuracy import Accuracy

def soft_softmax_cross_entropy(preds, soft_labels):
    """
    Calculates softmax cross entropy with soft labels.
    """
    if len(soft_labels.shape) == 1: # If soft_labels is actually hard, then return softmax_cross_entropy
        return F.softmax_cross_entropy(preds, soft_labels)
    return - F.sum(F.log_softmax(preds) * soft_labels) / len(soft_labels)


class SoftAccuracy(Accuracy):

    def check_type_forward(self, in_types):
        pass

    def forward(self, inputs):
        xp = backend.get_array_module(*inputs)
        y, t = inputs
        if len(t.shape) == 2: # If t is soft then transform to be hard.
            t = F.argmax(t, -1).data

        if self.ignore_label is not None:
            mask = (t == self.ignore_label)
            ignore_cnt = mask.sum()

            pred = xp.where(mask, self.ignore_label,
                            y.argmax(axis=1).reshape(t.shape))
            count = (pred == t).sum() - ignore_cnt
            total = t.size - ignore_cnt

            if total == 0:
                return xp.asarray(0.0, dtype=y.dtype),
            else:
                return xp.asarray(float(count) / total, dtype=y.dtype),
        else:
            pred = y.argmax(axis=1).reshape(t.shape)
            return xp.asarray((pred == t).mean(dtype=y.dtype)),


def soft_accuracy(y, t, ignore_label=None):
    """
    accuracy function for soft labels
    """
    return SoftAccuracy(ignore_label=ignore_label)(y, t)