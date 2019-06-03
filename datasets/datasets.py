import os
from PIL import Image

from chainer.dataset import DatasetMixin
import numpy as np

mean = (123, 117, 104)


class ImageNetDataset(DatasetMixin):
    def __init__(self, root, file):
        with open(file, 'r') as f:
            data = f.read().splitlines()
        data = [line.split() for line in data]
        data = [[os.path.join(root, name), int(label)] for [name, label] in data]
        self._pairs = data
        self._length = len(data)

    def __len__(self):
        return self._length

    def get_example(self, i):
        file, label = self._pairs[i]
        image = np.asarray(Image.open(file))
        return image, label
