"""Focused checks for the reconstructed stage and preserved upstream behavior."""

import ast
import copy
import random
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

import OT_train
import data_utils
import pretrain
from Sinkhorn_distance_fl import SinkhornDistance
from reproduction_utils import create_model, load_checkpoint, split_state

torch.set_num_threads(1)
UPSTREAM = '01f35a8c08610c4f09e19e346150af3a3428174e'

def assert_close(actual, expected):
    assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-6)


def upstream_source(filename):
    return subprocess.check_output(
        ['git', 'show', UPSTREAM + ':' + filename],
        cwd=str(Path(__file__).parent), universal_newlines=True)


def test_split_matches_released_selection(monkeypatch):
    class FakeCifar:
        def __init__(self, root, train, **kwargs):
            size = 50000 if train else 10000
            self.data = np.arange(size).reshape(-1, 1)
            self.targets = np.tile(np.arange(100), size // 100).tolist()

        def __len__(self):
            return len(self.targets)

    monkeypatch.setattr(data_utils.torchvision.datasets, 'CIFAR100', FakeCifar)
    namespace = {}
    exec(upstream_source('data_utils.py'), namespace)
    old_meta, old_train, _ = namespace['build_dataset']('cifar100', 10)
    counts = namespace['get_img_num_per_cls']('cifar100', 0.01, 1000)
    retention_random = random.Random(42)
    deleted = []
    for class_id in range(100):
        indices = [i for i, label in enumerate(old_train.targets) if label == class_id]
        retention_random.shuffle(indices)
        deleted.extend(indices[counts[class_id]:])
    expected_train = np.delete(old_train.data, deleted, axis=0).reshape(-1)
    meta, train, _, actual_counts = data_utils.build_experiment_data(
        'cifar100', 10, 0.01, 'unused')
    assert actual_counts == counts
    assert len(train) == 10626
    assert len(meta) == 1000
    np.testing.assert_array_equal(train.original_indices, expected_train)
    np.testing.assert_array_equal(meta.original_indices, old_meta.data.reshape(-1))
    assert not np.intersect1d(meta.original_indices, train.original_indices).size
    # Split selection must be independent of ambient RNG/training seed.
    np.random.seed(123)
    random.seed(99)
    again_meta, again_train, _, _ = data_utils.build_experiment_data(
        'cifar100', 10, 0.01, 'unused')
    np.testing.assert_array_equal(again_train.original_indices, train.original_indices)
    np.testing.assert_array_equal(again_meta.original_indices, meta.original_indices)


def tensor_loaders():
    generator = torch.Generator().manual_seed(7)
    images = torch.randn(4, 3, 32, 32, generator=generator)
    labels = torch.tensor([[0], [0], [1], [1]])
    dataset = TensorDataset(images, labels, torch.arange(4))
    return DataLoader(dataset, batch_size=2), DataLoader(dataset, batch_size=4)


def test_stage2_matches_upstream_epoch():
    # Execute only upstream functions, replacing hardcoded CUDA transfers with
    # CPU transfers. This compares complete update trajectories, not a rewrite.
    source = upstream_source('OT_train.py')
    names = {'train_OT', 'to_var', 'softmax_normalize', 'AverageMeter',
             'accuracy', 'to_categorical'}
    nodes = [node for node in ast.parse(source).body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    module = ast.Module(body=nodes, type_ignores=[])
    class DeviceCalls(ast.NodeTransformer):
        def visit_Call(self, node):
            self.generic_visit(node)
            if isinstance(node.func, ast.Attribute):
                if node.func.attr == 'cuda':
                    node.func.attr = 'to'
                    node.args = [ast.Name(id='device', ctx=ast.Load())]
                elif node.func.attr == 'to' and node.args and getattr(node.args[0], 'value', getattr(node.args[0], 's', None)) == 'cuda':
                    node.args[0] = ast.Name(id='device', ctx=ast.Load())
            return node
    adapted = compile(ast.fix_missing_locations(DeviceCalls().visit(module)), 'upstream_OT_train.py', 'exec')
    device = torch.device('cpu')
    namespace = {'torch': torch, 'nn': torch.nn, 'F': torch.nn.functional,
                 'Variable': torch.autograd.Variable, 'device': device,
                 'args': SimpleNamespace(num_classes=2, num_meta=2,
                                         meta_set='prototype', cost='combined', print_freq=100),
                 'criterion_fl': SinkhornDistance(0.1, 200, reduction=None)}
    exec(adapted, namespace)
    torch.manual_seed(5)
    model = create_model(2, device)
    old_model = copy.deepcopy(model)
    for name, tensor in model.named_params(model):
        if not name.startswith('linear.'):
            tensor.requires_grad_(False)
    old_weights = torch.tensor([0.7, 0.7, 1.3, 1.3])
    weights = old_weights.clone()
    before = {name: tensor.detach().clone() for name, tensor in model.named_params(model)}
    before_bn = model.bn1.running_mean.clone()
    def optimizer_for(network):
        return torch.optim.SGD([network.linear.weight, network.linear.bias],
                               lr=2e-5, momentum=0.9, nesterov=True, weight_decay=5e-4)
    train, meta = tensor_loaders()
    namespace['train_OT'](train, meta, old_weights, old_model, optimizer_for(old_model),
                          0, torch.nn.CrossEntropyLoss(reduction='none'))
    record = OT_train.train_epoch(model, train, meta, weights, optimizer_for(model),
                                  SinkhornDistance(0.1, 200, reduction=None), device, 2, 0.01)
    assert_close(weights, old_weights)
    for name, value in model.state_dict().items():
        assert_close(value, old_model.state_dict()[name])
    assert not torch.equal(weights, torch.tensor([0.7, 0.7, 1.3, 1.3]))
    for name, value in model.named_params(model):
        if not name.startswith('linear.'):
            assert torch.equal(value, before[name])
    assert not torch.equal(model.linear.weight, before['linear.weight'])
    assert not torch.equal(model.bn1.running_mean, before_bn)
    assert np.isfinite(record['max_marginal_error'])


@pytest.mark.parametrize('epoch, expected', [(1, 0.1), (159, 0.1), (160, 0.001),
                                            (179, 0.001), (180, 0.00001), (200, 0.00001)])
def test_literal_paper_schedule(epoch, expected):
    assert pretrain.learning_rate_for_epoch(epoch) == pytest.approx(expected)


def test_single_example_transport_gradient():
    scores = torch.zeros(1, requires_grad=True)
    solver = SinkhornDistance(0.1, 200, reduction=None)
    loss = solver(torch.randn(2, 64), torch.randn(1, 64),
                  torch.eye(2).double(), torch.tensor([[1., 0.]]).double(), scores.softmax(0))
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(scores.grad).all()
    assert np.isfinite(solver.last_marginal_error)


def test_pretrain_checkpoint_to_ot_with_partial_batches(tmp_path, monkeypatch):
    rng = np.random.RandomState(17)
    class FakeDataset(SimpleNamespace):
        def __len__(self):
            return len(self.targets)
    def dataset(labels, offset):
        return FakeDataset(data=rng.randint(0, 256, (len(labels), 32, 32, 3), dtype=np.uint8),
                               targets=np.array(labels), original_indices=np.arange(len(labels)) + offset)
    meta = dataset(list(range(100)), 1000)
    train = dataset(list(range(100)) + [0], 0)
    test = dataset([0, 1, 99], 2000)
    counts = np.bincount(train.targets, minlength=100).tolist()
    def fake_data(*args):
        return meta, train, test, counts
    monkeypatch.setattr(pretrain, 'build_experiment_data', fake_data)
    monkeypatch.setattr(OT_train, 'build_experiment_data', fake_data)
    shared = dict(dataset='cifar100', num_meta=1, imb_factor=0.01, data_root='unused',
                  download=False, seed=42, device='cpu', batch_size=50, epochs=1)
    torch.manual_seed(42)
    initial = create_model(100, torch.device('cpu'))
    before = initial.conv1.weight.detach().clone()
    checkpoint_path = pretrain.run(SimpleNamespace(
        **shared, lr=0.1, output_dir=str(tmp_path / 'pretrain')))
    checkpoint = load_checkpoint(checkpoint_path, torch.device('cpu'))
    assert not torch.equal(before, checkpoint['model_state_dict']['conv1.weight'])
    restored = create_model(100, torch.device('cpu'))
    restored.load_state_dict(checkpoint['model_state_dict'])
    restored.eval()
    probe = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        expected = restored(probe)[1]
    restored.load_state_dict(load_checkpoint(checkpoint_path, torch.device('cpu'))['model_state_dict'])
    with torch.no_grad():
        assert_close(restored(probe)[1], expected)
    args = SimpleNamespace(**shared, lr=2e-5, weight_lr=0.001, meta_set='prototype',
                           checkpoint=str(checkpoint_path), output_dir=str(tmp_path / 'ot'))
    metrics = OT_train.run(args)
    assert 0 <= metrics['final_accuracy'] <= 1
    final = load_checkpoint(tmp_path / 'ot' / 'final.pt', torch.device('cpu'))
    assert final['epoch'] == 1
    assert final['split'] == split_state(meta, train, counts)
    assert (tmp_path / 'ot' / 'best.pt').exists()
    checkpoint['split']['train_indices'][0] = -1
    torch.save(checkpoint, checkpoint_path)
    args.output_dir = str(tmp_path / 'mismatch')
    with pytest.raises(ValueError, match='splits do not match'):
        OT_train.run(args)
    assert not (tmp_path / 'mismatch').exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA device unavailable')
def test_cuda_model_optimizer_leaves():
    model = create_model(100, torch.device('cuda'))
    assert all(tensor.is_leaf for tensor in model.params())
    optimizer = torch.optim.SGD(model.params(), lr=0.1)
    _, logits = model(torch.randn(2, 3, 32, 32, device='cuda'))
    logits.square().mean().backward()
    optimizer.step()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA device unavailable')
def test_cuda_full_meta_batch_and_partial_training_batch():
    device = torch.device('cuda')
    model = create_model(100, device)
    for name, tensor in model.named_params(model):
        if not name.startswith('linear.'):
            tensor.requires_grad_(False)
    generator = torch.Generator().manual_seed(3)
    labels = torch.arange(100).repeat_interleave(10).reshape(-1, 1)
    images = torch.randn(1000, 3, 32, 32, generator=generator)
    meta = DataLoader(TensorDataset(images, labels, torch.arange(1000)), batch_size=1000)
    train = DataLoader(TensorDataset(images[:17], labels[:17], torch.arange(17)), batch_size=16)
    weights = torch.ones(17, device=device)
    optimizer = torch.optim.SGD([model.linear.weight, model.linear.bias], lr=2e-5,
                                momentum=0.9, nesterov=True, weight_decay=5e-4)
    record = OT_train.train_epoch(model, train, meta, weights, optimizer,
                                  SinkhornDistance(0.1, 200, reduction=None), device, 100, 0.001)
    assert np.isfinite(record['loss'])
    assert torch.isfinite(weights).all()
