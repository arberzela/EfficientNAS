from __future__ import absolute_import

from torchvision.transforms import *

import random
import math
import torch
import numpy as np
from torch import randperm

from .operations import *


def mixup_data(x, y, alpha=1.0, is_cuda=True):
    lam = np.random.beta(alpha, alpha) if alpha > 0. else 1.
    batch_size = x.size()[0]
    index = randperm(batch_size).cuda() if is_cuda else randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class RandomErasing(object):
    """
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al. 
    
    Args:
        probability: The probability that the operation will be performed.
        sl: min erasing area
        sh: max erasing area
        r1: min aspect ratio
        mean: erasing value
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.mean = mean

    def __call__(self, img):
        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1/self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                    img[1, x1:x1+h, y1:y1+w] = self.mean[1]
                    img[2, x1:x1+h, y1:y1+w] = self.mean[2]
                else:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[1]
                return img

        return img


class Cutout(object):
    """
    Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = int(np.clip(y - self.length / 2, 0, h))
            y2 = int(np.clip(y + self.length / 2, 0, h))
            x1 = int(np.clip(x - self.length / 2, 0, w))
            x2 = int(np.clip(x + self.length / 2, 0, w))

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

class AutoAugment(object):
    
    def __init__(self):
        pass

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        """
        policies = {'Policy_0':  [Invert(0.1, 7),       Contrast(0.2, 6)],
                    'Policy_1':  [Rotate(0.7, 2),       TranslateX(0.3, 9)],
                    'Policy_2':  [Sharpness(0.8, 1),    Sharpness(0.9, 3)], 
                    'Policy_3':  [ShearY(0.5, 8),       TranslateY(0.7, 9)],
                    'Policy_4':  [AutoContrast(0.5, 8), Equalize(0.9, 2)],
                    'Policy_5':  [ShearY(0.2, 7),       Posterize(0.3, 7)],
                    'Policy_6':  [Color(0.4, 3),        Brightness(0.6, 7)],
                    'Policy_7':  [Sharpness(0.3, 9),    Brightness(0.7, 9)],
                    'Policy_8':  [Equalize(0.6, 5),     Equalize(0.5, 1)],
                    'Policy_9':  [Contrast(0.6, 7),     Sharpness(0.6, 5)],
                    'Policy_10': [Color(0.7, 7),        TranslateX(0.5, 8)],
                    'Policy_11': [Equalize(0.3, 7),     AutoContrast(0.4, 8)],
                    'Policy_12': [TranslateY(0.4, 3),   Sharpness(0.2, 6)],
                    'Policy_13': [Brightness(0.9, 6),   Color(0.2, 8)],
                    'Policy_14': [Solarize(0.5, 2),     Invert(0.0, 3)],
                    'Policy_15': [Equalize(0.2, 0),     AutoContrast(0.6, 0)],
                    'Policy_16': [Equalize(0.2, 8),     Equalize(0.6, 4)],
                    'Policy_17': [Color(0.9, 9),        Equalize(0.6, 6)],
                    'Policy_18': [AutoContrast(0.8, 4), Solarize(0.2, 8)],
                    'Policy_19': [Brightness(0.1, 3),   Color(0.7, 0)],
                    'Policy_20': [Solarize(0.4, 5),     AutoContrast(0.9, 3)],
                    'Policy_21': [TranslateY(0.9, 9),   TranslateY(0.7, 9)],
                    'Policy_22': [AutoContrast(0.9, 2), Solarize(0.8, 3)],
                    'Policy_23': [Equalize(0.8, 8),     Invert(0.1, 3)],
                    'Policy_24': [TranslateY(0.7, 9),   AutoContrast(0.9, 1)],
                    }

        policy = random.choice([policies['Policy_%d'%i] for i in range(25)])

        for op in policy:
            img = op(img)

        return img
