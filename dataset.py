#!/usr/bin/env python3

"""
@author: xi
"""

import imgaug.augmenters as iaa
import numpy as np
from PIL import Image
from docset import DocSet
from torch.utils.data import ConcatDataset, Dataset
from torchvision import transforms


class UnsupervisedDataset(Dataset):

    def __init__(self, path, image_size):
        super(UnsupervisedDataset, self).__init__()
        self.ds = DocSet(path, 'r')
        random_size = (image_size, int(image_size * 1.2))
        self.augmenter = iaa.Sequential([
            iaa.Resize({'height': random_size, 'width': random_size}),
            iaa.CropToFixedSize(height=image_size, width=image_size),
            iaa.Fliplr(0.5),
            iaa.Sometimes(0.8, iaa.Sequential([
                iaa.MultiplyBrightness((0.2, 1.8)),
                iaa.MultiplySaturation((0.2, 1.8)),
                iaa.MultiplyHue((0.8, 1.2))
            ])),
            iaa.Sometimes(0.2, iaa.Grayscale())
        ])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        doc = self.ds[index]
        feature = doc['feature']
        feature1 = self.augmenter(image=feature)
        feature2 = self.augmenter(image=feature)
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
        self.augmenter = iaa.Resize({'height': image_size, 'width': image_size})

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        doc = self.ds[index]
        feature = doc['feature']
        feature = self.augmenter(image=feature)
        feature = np.transpose((np.array(feature, np.float32) - 127.5) / 127.5, (2, 0, 1))
        feature = np.ascontiguousarray(feature)
        return {
            'feature': feature,
            'label': doc['label']
        }


class DSDataset(Dataset):

    def __init__(self, ds_file, fn):
        self._ds = DocSet(ds_file, 'r')
        self._fn = fn

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, index):
        doc = self._ds[index]
        if self._fn is not None:
            doc = self._fn(doc)
        return doc


def create_unsupervised_dataset(ds_file_list):
    data_transforms = transforms.Compose([
        transforms.RandomResizedCrop(size=96),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor()
    ])

    def fn(doc):
        image = Image.fromarray(doc['feature'])
        feature1 = data_transforms(image)
        feature2 = data_transforms(image)
        doc['feature'] = (feature1, feature2)
        return doc

    return ConcatDataset([
        DSDataset(tar_path, fn=fn)
        for tar_path in ds_file_list
    ])


def create_supervised_dataset(ds_file_list):
    data_transforms = transforms.ToTensor()

    def fn(doc):
        image = Image.fromarray(doc['feature'])
        doc['feature'] = data_transforms(image)
        return doc

    return ConcatDataset([
        DSDataset(tar_Path, fn=fn)
        for tar_Path in ds_file_list
    ])
