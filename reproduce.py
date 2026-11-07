"""Run one complete CIFAR-100 experiment: 200 CE epochs, then 40 OT epochs."""

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torchvision

import OT_train
import pretrain
from reproduction_utils import save_json, select_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', default='../cifar-100')
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--recipe', choices=['paper', 'released-defaults'], default='paper')
    args = parser.parse_args()
    select_device(args.device)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / 'experiment.json', {
        **vars(args), 'upstream_commit': '01f35a8c08610c4f09e19e346150af3a3428174e',
        'torch': torch.__version__, 'torchvision': torchvision.__version__,
        'numpy': np.__version__, 'cuda_runtime': torch.version.cuda,
        'gpu': torch.cuda.get_device_name() if args.device.startswith('cuda') else None,
        'momentum': 0.9, 'nesterov': True, 'weight_decay': 0.0005,
        'meta_split_seed': 6, 'retention_seed': 42,
        'sinkhorn_epsilon': 0.1, 'sinkhorn_max_iterations': 200,
        'sinkhorn_stop_threshold': 0.1, 'meta_loss_coefficient': 10,
        'stage1_milestones_one_based': [160, 180], 'stage1_decay_factor': 0.01,
    })
    shared = {'dataset': 'cifar100', 'data_root': args.data_root,
              'download': args.download, 'num_meta': 10, 'imb_factor': 0.01,
              'batch_size': 16, 'seed': args.seed, 'device': args.device}
    checkpoint = pretrain.run(SimpleNamespace(
        **shared, epochs=200, lr=0.1, output_dir=str(output / 'pretrain')))
    metrics = OT_train.run(SimpleNamespace(
        **shared, epochs=40 if args.recipe == 'paper' else 90, lr=2e-5,
        weight_lr=0.001 if args.recipe == 'paper' else 0.01,
        meta_set='prototype', checkpoint=str(checkpoint),
        output_dir=str(output / 'ot')))
    save_json(output / 'result.json', metrics)
    print('Final accuracy: {:.2%}; best test accuracy: {:.2%}; paper: {:.2%}'.format(
        metrics['final_accuracy'], metrics['best_accuracy'], metrics['paper_accuracy']), flush=True)


if __name__ == '__main__':
    main()
