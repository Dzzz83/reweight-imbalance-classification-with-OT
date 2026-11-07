"""Reconstructed cross-entropy pretraining for the authors' ResNet-32."""

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from data_utils import build_experiment_data
from reproduction_utils import (create_model, evaluate, make_loader, require_finite,
                                save_json, seed_training, select_device, split_state)


def learning_rate_for_epoch(epoch, initial_lr=0.1):
    # Literal paper schedule; epoch is one-based.
    return initial_lr * 0.01 ** sum(epoch >= milestone for milestone in (160, 180))


def train_epoch(model, loader, optimizer, device):
    model.train()
    loss_sum = 0.0
    sample_count = 0
    for images, labels, _ in loader:
        images = images.to(device)
        labels = labels.to(device).reshape(-1)
        optimizer.zero_grad()
        _, logits = model(images)
        loss = F.cross_entropy(logits, labels)
        require_finite(loss, 'pretraining')
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * len(labels)
        sample_count += len(labels)
    return loss_sum / sample_count


def run(args):
    device = select_device(args.device)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError('Output directory exists: {}'.format(output_dir))
    seed_training(args.seed)
    meta, train, test, counts = build_experiment_data(
        args.dataset, args.num_meta, args.imb_factor, args.data_root, args.download)
    split = split_state(meta, train, counts)
    seed_training(args.seed)
    model = create_model(len(counts), device)
    optimizer = torch.optim.SGD(model.params(), lr=args.lr, momentum=0.9,
                                nesterov=True, weight_decay=5e-4)
    loader = make_loader(train, args.batch_size, True)
    output_dir.mkdir(parents=True)
    save_json(output_dir / 'config.json', vars(args))
    save_json(output_dir / 'split.json', split)
    history = []
    for epoch in range(1, args.epochs + 1):
        start = time.monotonic()
        lr = learning_rate_for_epoch(epoch, args.lr)
        for group in optimizer.param_groups:
            group['lr'] = lr
        loss = train_epoch(model, loader, optimizer, device)
        history.append({'epoch': epoch, 'loss': loss, 'learning_rate': lr,
                        'epoch_seconds': time.monotonic() - start})
        torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch,
                    'optimizer': optimizer.state_dict(), 'config': vars(args),
                    'split': split}, output_dir / 'final.pt')
        save_json(output_dir / 'history.json', history)
        print('Pretrain epoch {}/{} loss={:.4f} lr={:.6g}'.format(
            epoch, args.epochs, loss, lr), flush=True)
    accuracy = evaluate(model, make_loader(test, args.batch_size, False), device)
    save_json(output_dir / 'metrics.json', {'final_accuracy': accuracy})
    print('Pretrained accuracy: {:.2%}'.format(accuracy), flush=True)
    return output_dir / 'final.pt'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['cifar10', 'cifar100'], default='cifar100')
    parser.add_argument('--data-root', default='../cifar-100')
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--num-meta', type=int, default=10)
    parser.add_argument('--imb-factor', type=float, default=0.01)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.num_meta < 1:
        parser.error('epochs, batch-size and num-meta must be positive')
    run(args)


if __name__ == '__main__':
    main()
