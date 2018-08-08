import numpy as np

import torch

from torchvision import datasets, models, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from util.transforms import Cutout

def Loader(config):
    if config.dataset == 'cifar10':
        mean=[x / 255.0 for x in[125.3, 123.0, 113.9]]
        std=[x / 255.0 for x in [63.0, 62.1, 66.7]]
    elif config.dataset == 'cifar100':
        mean=[x / 255.0 for x in[129.3, 124.1, 112.4]]
        std=[x / 255.0 for x in [68.2, 65.4, 70.4]]
    elif config.dataset == 'svhn':
        mean=[x / 255.0 for x in[109.9, 109.7, 113.8]]
        std=[x / 255.0 for x in [50.1, 50.6, 50.8]]
    
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    if config.auto_aug:
        train_transform.transforms.append(transforms.ToPILImage())
        train_transform.transforms.append(AutoAugment())
        train_transform.transforms.append(transforms.ToTensor())

    if config.cutout:
        train_transform.transforms.append(Cutout(n_holes=config.n_holes, length=config.length))

    if config.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(
            config.gen, train=True, transform=train_transform, download=True)
        valid_dataset = datasets.CIFAR10(
            config.gen, train=True, transform=train_transform, download=True)
        test_dataset = datasets.CIFAR10(
            config.gen, train=False, transform=test_transform, download=True)
    elif config.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(
            config.gen, train=True, transform=train_transform, download=True)
        test_dataset = datasets.CIFAR100(
            config.gen, train=False, transform=test_transform, download=True)
    elif config.dataset == 'svhn':
        train_dataset = datasets.SVHN(
            config.gen, split='train', transform=train_transform, download=True)
        extra_dataset = datasets.SVHN(
            config.gen, split='extra', transform=train_transform, download=True)
        
        # Combine both training splits (https://arxiv.org/pdf/1605.07146.pdf)
        data = np.concatenate([train_dataset.data, extra_dataset.data], axis=0)
        labels = np.concatenate([train_dataset.labels, extra_dataset.labels], axis=0)
        train_dataset.data = data
        train_dataset.labels = labels

        test_dataset = datasets.SVHN(
            config.gen, split='test', transform=test_transform, download=True)

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(config.valid_frac * num_train))

    np.random.seed(config.manual_seed)
    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        sampler=train_sampler,
        pin_memory=True,
        num_workers=config.num_threads,
        drop_last=True,
    )

    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=config.batch_size,
        sampler=valid_sampler,
        pin_memory=True,
        num_workers=config.num_threads,
        drop_last=False,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=config.num_threads,
        drop_last=False,
    )

    return train_loader, valid_loader, test_loader

