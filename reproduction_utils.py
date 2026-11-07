"""Small execution helpers; model and transport mathematics stay upstream."""

import json
import inspect
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import resnet
from data_utils import new_dataset


def select_device(name):
    device = torch.device(name)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Full reproduction needs a working CUDA environment; --device cpu is available for checks.')
    return device


def seed_training(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def create_model(num_classes, device):
    # Preserve upstream device-side initialization, including its RNG behavior.
    resnet.device = device
    model = resnet.ResNet32(num_classes).to(device)
    for name, tensor in list(model.named_params(model)):
        if not tensor.is_leaf:
            model.set_param(model, name, tensor.detach().requires_grad_(True))
    torch.backends.cudnn.benchmark = device.type == 'cuda'
    return model


def make_loader(dataset, batch_size, train, shuffle=None):
    return DataLoader(new_dataset(dataset, train=train), batch_size=batch_size,
                      shuffle=train if shuffle is None else shuffle, num_workers=0)


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def split_state(meta, train, counts):
    return {'meta_indices': meta.original_indices.tolist(),
            'train_indices': train.original_indices.tolist(),
            'class_counts': counts}


def load_checkpoint(path, device):
    # These are local experiment checkpoints, not arbitrary downloaded objects.
    options = {'map_location': device}
    if 'weights_only' in inspect.signature(torch.load).parameters:
        options['weights_only'] = False
    return torch.load(path, **options)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for images, labels, _ in loader:
        labels = labels.to(device).reshape(-1)
        _, logits = model(images.to(device))
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.numel()
    return correct / total


def require_finite(loss, name):
    if not torch.isfinite(loss):
        raise RuntimeError('Nonfinite {} loss'.format(name))
