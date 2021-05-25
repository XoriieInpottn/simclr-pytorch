#!/usr/bin/env python3


"""
@author: xi
"""

from PIL import Image
from docset import DocSet
from torch.utils.data import ConcatDataset, Dataset
from torchvision import transforms


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
