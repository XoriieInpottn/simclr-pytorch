#!/usr/bin/env python3

"""
@author: xi
"""

from typing import Union, Tuple

import imgaug.augmenters as iaa
import numpy as np
from docset import DocSet
from torch.utils.data import Dataset


class ColorJitter(iaa.Sequential):

    def __init__(
            self,
            hue_shift: Union[float, Tuple[float], None] = 0.05,
            saturation_factor: Union[float, Tuple[float], None] = 0.2,
            brightness_factor: Union[float, Tuple[float], None] = 0.2,
            contrast_factor: Union[float, Tuple[float], None] = 0.2
    ) -> None:
        """Randomly change the hue, saturation, brightness and contrast of an image.

        Args:
            hue_shift: How much to jitter hue.
                hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
                Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
            saturation_factor: How much to jitter saturation.
                saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
                or the given [min, max]. Should be non negative numbers.
            brightness_factor: How much to jitter brightness.
                brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
                or the given [min, max]. Should be non negative numbers.
            contrast_factor: How much to jitter contrast.
                contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
                or the given [min, max]. Should be non negative numbers.
        """
        if isinstance(hue_shift, float):
            h = (-int(hue_shift * 255), int(hue_shift * 255))
        elif isinstance(hue_shift, (tuple, list)) and len(hue_shift) == 2:
            h = (int(hue_shift[0] * 255), int(hue_shift[1] * 255))
        elif hue_shift is None:
            h = None
        else:
            raise RuntimeError(f'Invalid hue_shift {hue_shift}.')

        if isinstance(saturation_factor, float):
            s = (max(1.0 - saturation_factor, 0), 1.0 + saturation_factor)
        elif isinstance(saturation_factor, (tuple, list)) and len(saturation_factor) == 2:
            s = saturation_factor
        elif saturation_factor is None:
            s = None
        else:
            raise RuntimeError(f'Invalid saturation_factor {saturation_factor}.')

        if isinstance(brightness_factor, float):
            v = (max(1.0 - brightness_factor, 0), 1.0 + brightness_factor)
        elif isinstance(brightness_factor, (tuple, list)) and len(brightness_factor) == 2:
            v = brightness_factor
        elif brightness_factor is None:
            v = None
        else:
            raise RuntimeError(f'Invalid brightness_factor {brightness_factor}.')

        if isinstance(contrast_factor, float):
            c = (max(1.0 - contrast_factor, 0), 1.0 + contrast_factor)
        elif isinstance(contrast_factor, (tuple, list)) and len(contrast_factor) == 2:
            c = contrast_factor
        elif contrast_factor is None:
            c = None
        else:
            raise RuntimeError(f'Invalid contrast_factor {contrast_factor}.')

        super(ColorJitter, self).__init__([
            iaa.WithColorspace(
                from_colorspace=iaa.CSPACE_RGB,
                to_colorspace=iaa.CSPACE_HSV,
                children=iaa.Sequential([
                    iaa.WithChannels(0, iaa.Add(h)) if h else iaa.Identity(),
                    iaa.WithChannels(1, iaa.Multiply(s)) if s else iaa.Identity(),
                    iaa.WithChannels(2, iaa.Multiply(v)) if v else iaa.Identity()
                ])
            ),
            iaa.LinearContrast(c) if c else iaa.Identity()
        ])


class ResizedCrop(iaa.Sequential):

    def __init__(
            self,
            width: int,
            height: int,
            scale: float = 1.0,
            ratio: float = 1.33,
            interpolation='linear'
    ) -> None:
        assert scale >= 1.0, f'Invalid scale {scale}. It should >= 1.'
        assert ratio > 0, f'Invalid ratio {ratio}. It should > 0.'
        if ratio < 1.0:
            ratio = 1.0 / ratio
        min_width = int(width * scale)
        max_width = int(min_width * ratio)
        min_height = int(height * scale)
        max_height = int(min_height * ratio)
        super(ResizedCrop, self).__init__([
            iaa.Resize(
                {'width': (min_width, max_width), 'height': (min_height, max_height)},
                interpolation=interpolation
            ),
            iaa.CropToFixedSize(width=width, height=height),
        ])


class UnsupervisedDataset(Dataset):

    def __init__(self, path, image_size):
        super(UnsupervisedDataset, self).__init__()
        self.ds = DocSet(path, 'r')
        self.transform = iaa.Sequential([
            ResizedCrop(image_size, image_size, 1.1, 1.4),
            iaa.Fliplr(0.5),
            iaa.Sometimes(0.8, ColorJitter(0.1, 0.8, 0.8, 0.8)),
            iaa.Sometimes(0.2, iaa.Grayscale()),
        ])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        doc = self.ds[index]
        feature = doc['feature']
        feature1 = self.transform(image=feature)
        feature2 = self.transform(image=feature)

        # from matplotlib import pyplot as plt
        # plt.figure(figsize=(8, 8))
        # n = 5
        # for i in range(n * n):
        #     f = self.transform(image=feature)
        #     plt.subplot(n, n, i + 1)
        #     plt.imshow(f)
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
