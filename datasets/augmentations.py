from PIL import Image

import chainer
from chainercv import transforms
import numpy as np

from .autoaugment import ImageNetPolicy
from .datasets import mean
from functions.soft import hard_to_soft


def get_transforms(patchsize=(224, 224), no_autoaugment=False, mean=mean, soft=False, dtype=None):
    if dtype is None:
        dtype = chainer.get_dtype()
    policy = ImageNetPolicy(fillcolor=mean)
    mean = np.array(mean).reshape(3, 1, 1)
    resize_size = (int(1.1423*patchsize[0]), int(1.1423*patchsize[1])) # (224, 224) -> (256, 256)

    def train_transform(sample):
        img, label = sample
        img = np.transpose(img, (2, 0, 1))
        img = transforms.random_sized_crop(img)
        img = transforms.resize(img, patchsize)
        img = transforms.random_flip(img, x_random=True)
        if not no_autoaugment:
            img = np.transpose(img, (1, 2, 0))
            img = Image.fromarray(img)
            img = policy(img)
            img = np.asarray(img)
            img = np.transpose(img, (2, 0, 1))
        img = img - mean
        img = img.astype(dtype)
        if soft:
            label = hard_to_soft(label, dtype=dtype)
        return img, label

    def val_transform(sample):
        img, label = sample
        img = np.transpose(img, (2, 0, 1))
        img = transforms.resize(img, resize_size)
        img = transforms.center_crop(img, patchsize)
        img = img - mean
        img = img.astype(dtype)
        if soft:
            label = hard_to_soft(label, dtype=dtype)
        return img, label

    def test_transform(sample):
        img = sample
        if len(img.shape) == 2: # Grayscale
            img = np.stack([img, img, img], 2)
        img = np.transpose(img, (2, 0, 1))
        img = transforms.resize(img, resize_size)
        img = transforms.center_crop(img, patchsize)
        img = img - mean
        img = img.astype(dtype)
        return img

    return train_transform, val_transform, test_transform
