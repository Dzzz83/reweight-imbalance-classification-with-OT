"""Run the released OT training algorithm from a Stage-1 checkpoint."""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data_utils import build_experiment_data
from Sinkhorn_distance_fl import SinkhornDistance
from reproduction_utils import (create_model, evaluate, load_checkpoint, make_loader,
                                require_finite, save_json, seed_training,
                                select_device, split_state)


def initialize_weights(labels, class_counts, device):
    beta = 0.9999
    effective_num = 1.0 - np.power(beta, class_counts)
    class_weights = (1.0 - beta) / effective_num
    class_weights = class_weights / np.sum(class_weights) * len(class_counts)
    return torch.tensor(class_weights, dtype=torch.float32, device=device)[
        torch.as_tensor(labels, dtype=torch.long, device=device)]


def train_epoch(model, train_loader, meta_loader, weightsbuffer, optimizer,
                criterion, device, num_classes, weight_lr, meta_set='prototype'):
    model.train()
    meta_images, meta_labels, _ = next(iter(meta_loader))
    meta_images = meta_images.to(device)
    meta_labels = meta_labels.to(device).reshape(-1)
    if meta_set == 'prototype':
        # Upstream averages augmented images, then extracts their features.
        reference_images = torch.stack([
            meta_images[meta_labels == c].mean(0) for c in range(num_classes)])
        reference_labels = torch.arange(num_classes, device=device)
    else:
        reference_images, reference_labels = meta_images, meta_labels
    reference_onehot = F.one_hot(reference_labels, num_classes).double()
    reference_features, _ = model(reference_images)
    reference_features = reference_features.detach()

    loss_sum = 0.0
    ot_sum = 0.0
    count = 0
    max_marginal_error = 0.0
    for images, labels, positions in train_loader:
        images = images.to(device)
        labels = labels.to(device).reshape(-1)
        positions = positions.to(device)
        labels_onehot = F.one_hot(labels, num_classes).double()
        weights = weightsbuffer[positions].detach().requires_grad_(True)
        model.eval()
        # Intentionally recreate this optimizer each batch, as in the release.
        weight_optimizer = torch.optim.SGD([weights], lr=weight_lr,
                                           momentum=0.9, weight_decay=5e-4)
        features, _ = model(images)
        probabilities = weights.softmax(0)
        ot_loss = criterion(reference_features, features.detach(), reference_onehot,
                            labels_onehot, probabilities)
        require_finite(ot_loss, 'OT')
        max_marginal_error = max(max_marginal_error, criterion.last_marginal_error)
        weight_optimizer.zero_grad()
        ot_loss.backward()
        weight_optimizer.step()
        weightsbuffer[positions] = weights.detach()

        model.train()
        optimizer.zero_grad()
        _, logits = model(images)
        _, meta_logits = model(meta_images)
        # Preserve raw weights, summed training CE and the supervised meta loss.
        loss = (F.cross_entropy(logits, labels, reduction='none') * weights.detach()).sum()
        loss = loss + 10 * F.cross_entropy(meta_logits, meta_labels)
        require_finite(loss, 'classification')
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * len(labels)
        ot_sum += ot_loss.item() * len(labels)
        count += len(labels)
    return {'loss': loss_sum / count, 'ot_cost': ot_sum / count,
            'max_marginal_error': max_marginal_error,
            'min_weight': weightsbuffer.min().item(),
            'max_weight': weightsbuffer.max().item(),
            'negative_weights': (weightsbuffer < 0).sum().item()}


def run(args):
    device = select_device(args.device)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError('Output directory exists: {}'.format(output_dir))
    num_classes = 10 if args.dataset == 'cifar10' else 100
    seed_training(args.seed)
    meta, train, test, counts = build_experiment_data(
        args.dataset, args.num_meta, args.imb_factor, args.data_root, args.download)
    split = split_state(meta, train, counts)
    checkpoint = load_checkpoint(args.checkpoint, device)
    if checkpoint['split'] != split:
        raise ValueError('Stage-1 and Stage-2 data splits do not match')
    if checkpoint['config']['dataset'] != args.dataset:
        raise ValueError('Stage-1 dataset does not match')
    seed_training(args.seed)
    model = create_model(num_classes, device)
    model.load_state_dict(checkpoint['model_state_dict'])
    # Upstream optimizes only the classifier. Suppress unused backbone gradients
    # to fit the full-meta forward on small GPUs; keep its train-mode BN updates.
    for name, tensor in model.named_params(model):
        if not name.startswith('linear.'):
            tensor.requires_grad_(False)
    weights = initialize_weights(train.targets, counts, device)
    train_loader = make_loader(train, args.batch_size, True)
    meta_loader = make_loader(meta, len(meta), True, shuffle=False)
    test_loader = make_loader(test, args.batch_size, False)
    optimizer = torch.optim.SGD([model.linear.weight, model.linear.bias],
                                lr=args.lr, momentum=0.9, nesterov=True,
                                weight_decay=5e-4)
    criterion = SinkhornDistance(eps=0.1, max_iter=200, reduction=None)
    output_dir.mkdir(parents=True)
    save_json(output_dir / 'config.json', vars(args))
    save_json(output_dir / 'split.json', split)
    # DataLoader iteration consumes CPU RNG even without shuffling. Reporting
    # the starting accuracy must not change the subsequent training sequence.
    random_state = torch.get_rng_state()
    initial_accuracy = evaluate(model, test_loader, device)
    torch.set_rng_state(random_state)
    history = []
    best_accuracy = -1.0
    best_epoch = None
    for epoch in range(1, args.epochs + 1):
        start = time.monotonic()
        record = train_epoch(model, train_loader, meta_loader, weights, optimizer,
                             criterion, device, num_classes, args.weight_lr, args.meta_set)
        accuracy = evaluate(model, test_loader, device)
        record.update(epoch=epoch, test_accuracy=accuracy, learning_rate=args.lr,
                      epoch_seconds=time.monotonic() - start)
        history.append(record)
        state = {'model_state_dict': model.state_dict(), 'epoch': epoch,
                 'optimizer': optimizer.state_dict(), 'weights': weights.detach().cpu(),
                 'config': vars(args), 'split': split, 'test_accuracy': accuracy}
        if accuracy > best_accuracy:
            best_accuracy, best_epoch = accuracy, epoch
            torch.save(state, output_dir / 'best.pt')
        torch.save(state, output_dir / 'final.pt')
        save_json(output_dir / 'history.json', history)
        print('OT epoch {}/{} loss={:.4f} test={:.2%} best={:.2%}'.format(
            epoch, args.epochs, record['loss'], accuracy, best_accuracy), flush=True)
    metrics = {'pretrained_accuracy': initial_accuracy, 'final_accuracy': accuracy,
               'best_accuracy': best_accuracy, 'best_epoch': best_epoch,
               'paper_accuracy': 0.4854 if args.dataset == 'cifar100' and args.imb_factor == 0.01 else None}
    save_json(output_dir / 'metrics.json', metrics)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['cifar10', 'cifar100'], default='cifar100')
    parser.add_argument('--data-root', default='../cifar-100')
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--num-meta', type=int, default=10)
    parser.add_argument('--imb-factor', type=float, default=0.01)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=40, help='Number of Stage-2 epochs')
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--weight-lr', type=float, default=1e-3)
    parser.add_argument('--meta-set', choices=['prototype', 'whole'], default='prototype')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.num_meta < 1:
        parser.error('epochs, batch-size and num-meta must be positive')
    run(args)


if __name__ == '__main__':
    main()
