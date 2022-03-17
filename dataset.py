#!/usr/bin/env python3

"""
@author: xi
"""

import imgaug.augmenters as iaa
import numpy as np
from docset import DocSet
from torch.utils.data import Dataset


class UnsupervisedDataset(Dataset):

    def __init__(self, path, image_size):
        super(UnsupervisedDataset, self).__init__()
        self.ds = DocSet(path, 'r')
        self.transform = iaa.Sequential([
            iaa.Sometimes(0.5, iaa.Rotate((-10, 10), mode='edge')),
            iaa.Crop(percent=(0, 0.3), sample_independently=True, keep_size=False),
            iaa.Resize({'height': image_size, 'width': image_size}, interpolation='linear'),
            iaa.Fliplr(0.5),
            iaa.Sometimes(0.8, [
                iaa.AddToHue((-50, 50)),
                iaa.MultiplySaturation((0.2, 1.8)),
                iaa.MultiplyBrightness((0.2, 1.8)),
                iaa.LinearContrast((0.2, 1.8)),
            ]),
            iaa.Sometimes(0.2, iaa.Grayscale()),
            iaa.Cutout(nb_iterations=(0, 2), size=0.2, squared=False),
        ])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        doc = self.ds[index]
        feature = doc['feature']
        feature1 = self.transform(image=feature)
        feature2 = self.transform(image=feature)
        # from matplotlib import pyplot as plt
        # plt.subplot(211)
        # plt.imshow(feature1)
        # plt.subplot(212)
        # plt.imshow(feature2)
        # plt.show()
        # exit()
        feature1 = np.transpose((np.array(feature1, np.float32) - 127.5) / 127.5, (2, 0, 1))
        feature2 = np.transpose((np.array(feature2, np.float32) - 127.5) / 127.5, (2, 0, 1))
        feature1 = np.ascontiguousarray(feature1)
        feature2 = np.ascontiguousarray(feature2)
        return {
            'feature': (feature1, feature2),
            'label': doc['label']
        }


class SupervisedDataset(Dataset):

    def __init__(self, path, image_size):
        super(SupervisedDataset, self).__init__()
        self.ds = DocSet(path, 'r')
        self.transform = iaa.Resize({'height': image_size, 'width': image_size})

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        doc = self.ds[index]
        feature = doc['feature']
        feature = self.transform(image=feature)
        feature = np.transpose((np.array(feature, np.float32) - 127.5) / 127.5, (2, 0, 1))
        feature = np.ascontiguousarray(feature)
        return {
            'feature': feature,
            'label': doc['label']
        }
