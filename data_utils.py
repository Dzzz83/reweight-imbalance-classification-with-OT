import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision
import numpy as np
import copy
import random
from torch.utils.data import Dataset

def build_dataset(dataset, num_meta, data_root=None, download=False):
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                          (4, 4, 4, 4), mode='reflect').squeeze()),
        transforms.ToPILImage(),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    if dataset == 'cifar10':
        data_root = data_root or '../cifar-10'
        train_dataset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=download, transform=transform_train)
        test_dataset = torchvision.datasets.CIFAR10(data_root, train=False, transform=transform_test)
        img_num_list = [num_meta] * 10
        num_classes = 10

    if dataset == 'cifar100':
        data_root = data_root or '../cifar-100'
        train_dataset = torchvision.datasets.CIFAR100(root=data_root, train=True, download=download, transform=transform_train)
        test_dataset = torchvision.datasets.CIFAR100(data_root, train=False, transform=transform_test)
        img_num_list = [num_meta] * 100
        num_classes = 100

    data_list_val = {}
    for j in range(num_classes):
        data_list_val[j] = [i for i, label in enumerate(train_dataset.targets) if label == j]

    idx_to_meta = []
    idx_to_train = []
    print(img_num_list)

    meta_random = np.random.RandomState(6)
    for cls_idx, img_id_list in data_list_val.items():
        meta_random.shuffle(img_id_list)
        img_num = img_num_list[int(cls_idx)]
        idx_to_meta.extend(img_id_list[:img_num])
        idx_to_train.extend(img_id_list[img_num:])
    train_data = copy.deepcopy(train_dataset)
    train_data_meta = copy.deepcopy(train_dataset)

    train_data_meta.data = np.delete(train_dataset.data, idx_to_train,axis=0)
    train_data_meta.targets = np.delete(train_dataset.targets, idx_to_train, axis=0)
    train_data.data = np.delete(train_dataset.data, idx_to_meta, axis=0)
    train_data.targets = np.delete(train_dataset.targets, idx_to_meta, axis=0)
    train_data.original_indices = np.delete(np.arange(len(train_dataset)), idx_to_meta)
    train_data_meta.original_indices = np.delete(np.arange(len(train_dataset)), idx_to_train)

    return train_data_meta, train_data, test_dataset

def get_img_num_per_cls(dataset, imb_factor=None, num_meta=None):

    if dataset == 'cifar10':
        img_max = (50000-num_meta)/10
        cls_num = 10

    if dataset == 'cifar100':
        img_max = (50000-num_meta)/100
        cls_num = 100

    if imb_factor is None:
        return [img_max] * cls_num
    img_num_per_cls = []
    for cls_idx in range(cls_num):
        num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
        img_num_per_cls.append(int(num))
    return img_num_per_cls


def build_experiment_data(dataset, num_meta, imb_factor, data_root, download=False):
    """Reproduce the release's meta split (seed 6) and retention (seed 42)."""
    meta, train, test = build_dataset(dataset, num_meta, data_root, download)
    num_classes = 10 if dataset == 'cifar10' else 100
    counts = get_img_num_per_cls(dataset, imb_factor, num_meta * num_classes)
    retention_random = random.Random(42)
    deleted = []
    for class_id in range(num_classes):
        indices = [i for i, label in enumerate(train.targets) if label == class_id]
        retention_random.shuffle(indices)
        if len(indices) < counts[class_id] or counts[class_id] < 1:
            raise ValueError('Invalid retained count for class {}'.format(class_id))
        deleted.extend(indices[counts[class_id]:])
    train.data = np.delete(train.data, deleted, axis=0)
    train.targets = np.delete(train.targets, deleted, axis=0)
    train.original_indices = np.delete(train.original_indices, deleted)
    return meta, train, test, counts

class new_dataset(Dataset):
    def __init__(self, dataset, train=None):
        self.data = dataset.data
        self.targets = dataset.targets
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        if train:
            self.transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),(4, 4, 4, 4), mode='reflect').squeeze()),
                                transforms.ToPILImage(),
                                transforms.RandomCrop(32),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                normalize,
                            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                normalize
            ])


    def __getitem__(self, index):
        img, label = self.data[index, ::], self.targets[index]
        img = self.transform(img)
        label = torch.LongTensor([np.int64(label)])

        return img, label, index

    def __len__(self):
        return len(self.data)

