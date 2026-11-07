# One-run CIFAR-100 reproduction attempt

Upstream: DandanGuo1993/reweight-imbalance-classification-with-OT, commit
`01f35a8c08610c4f09e19e346150af3a3428174e`.
This directory is independent of the shared expert_method pipeline.

## Run once

Run the full experiment on the user's CUDA server, not the local development
machine. Transfer this modified repository, then from its root, with the
environment active:

```bash
python reproduce.py --download --device cuda --seed 42 --recipe paper --output-dir outputs/cifar100_if100_seed42
```

This executes exactly one 200-epoch pretraining followed by one 40-epoch OT
stage. It downloads CIFAR-100 to `../cifar-100` unless `--data-root` is given.
Output directories must be new. No five-seed sweep runs automatically.
`--recipe released-defaults` instead uses 90 OT epochs and weight LR 0.01;
this is a separate, optional experiment, not part of the one-run command.

The authors list Python 3.6, PyTorch 1.7.1, torchvision 0.8.2, NumPy 1.19.2,
and tqdm 4.19.9. `requirements-reproduction.txt` preserves those versions and
adds the imports their release omitted. Install it only in a dedicated legacy
environment; compatible CUDA drivers/wheels must be supplied by that machine.
The code also supports modern PyTorch for smoke checks, but the recorded
environment must accompany any research result. An environment change can affect
accuracy; a modern-stack run is not an exact software-environment reproduction.

For local correctness checks (no download):

```bash
python -m pytest -q
```

## Recipe and assumptions

The source is Sections 5–6.1 and Table 1 of the
[paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/a39a9aceda771cded859ae7560530e09-Paper-Conference.pdf).

| Setting | Value |
|---|---|
| Target | CIFAR-100, reported IF 100, CLI retention factor 0.01 |
| Model | Authors' custom ResNet-32, Option-A shortcuts |
| Data | Meta selection seed 6; retention seed 42; original order preserved |
| Counts | 1,000 balanced meta images, 10,626 training images, 10,000 test images |
| Pretraining | 200 epochs, ordinary mean CE, SGD LR 0.1 |
| Pretraining schedule | LR ×0.01 at the start of one-based epochs 160 and 180 |
| OT stage | 40 epochs, classifier SGD LR 0.00002, weight SGD LR 0.001 |
| SGD | Momentum 0.9, weight decay 0.0005, classifier/pretrain Nesterov enabled |
| Batch | 16 training/test images; all meta images in one batch |
| Transport | Combined cost, epsilon 0.1, maximum 200 iterations, upstream early stop |

The pretraining script was missing. Applying batch size 16 and the same SGD
settings to pretraining is a reconstruction. Nesterov comes from the released
classifier optimizer. The LR decay factor is the literal paper value, not a
conventional factor 0.1; milestone placement is explicitly one-based here.
The exact authors' five seeds and pretrained checkpoints are unavailable.

The paper-settings recipe changes the released weight LR (0.01 → 0.001) and OT
duration (90 → 40). It preserves the released algorithm, including:

- Class-balanced effective-number initialization (beta 0.9999).
- One batch-softmax OT weight update; a fresh weight optimizer per batch.
- Raw weights for the summed classification loss, plus 10 × mean meta CE.
- Augmented image means as prototypes, computed once per OT epoch in train mode.
- Classifier-only SGD updates; batch-normalization running statistics still update.
- Unused backbone gradients are disabled to reduce memory, without changing
  forward values, classifier gradients, or batch-normalization behavior.
- Combined cost of 0.5 × cosine distance plus 0.5 × mean squared one-hot distance.
  The label mismatch contribution is 0.01 for 100 classes.
- Upstream Sinkhorn early stop on potential changes, threshold 0.1. Marginal
  errors are measured, not used to silently change the solve.

These choices differ from a clean reimplementation of the paper's mathematical
description. They are retained to check the released code rather than redesign
the algorithm. No feature cache, new loss, global weight normalization, or extra
training split is introduced. Raw weights are not clamped if they turn negative;
their ranges and negative count are logged. Nonfinite losses stop execution.

## Files and outputs

- `pretrain.py`: reconstructed Stage 1 using `model.params()`.
- `OT_train.py`: released Stage-2 computation with explicit inputs and duration.
- `reproduce.py`: one complete experiment, no seed sweep.
- `reproduction_utils.py`: checkpoint, device, loader, and reporting helpers.
- `data_utils.py`: shared original split and retained positions for both stages.
- `resnet.py`: explicit construction device and current spelling of Kaiming init.
- Sinkhorn files: input-device allocation, scalar-batch robustness, missing
  staticmethod repair, and observational marginal diagnostics; same solver math.

`experiment.json` records software versions and fixed settings. Each stage saves
`config.json`, `split.json`, `history.json`, and `final.pt`. Stage 2 also saves
`best.pt` using the release's best-test selection, and `metrics.json` reports
pretrained, final, and best test accuracy as fractions. `result.json` repeats the
final summary. Both stages verify identical split indices before OT training.
Checkpoints are trusted local experiment files and use PyTorch pickle loading.

The paper's target is 51.46% error = 48.54% accuracy averaged over five runs.
Our single-run result can establish approximate agreement, not reproduce that
five-run average. Report the actual difference without inventing a pass threshold.
Stage 1 passes its final scheduled checkpoint to Stage 2. Both final and best-test
Stage-2 scores are reported; the paper does not clearly establish which selection
rule produced its table, while the release selects best test accuracy.

## Execution status

All 12 focused tests passed locally with CPU and CUDA access. Checks include
exact data-selection comparison with upstream, a Stage-2 update comparison,
partial batches, a synthetic checkpoint handoff, and a 1,000-image synthetic meta
batch on GPU. They establish execution and preserved computation, not research
accuracy. The checked environment was PyTorch 2.14.0+cu126, torchvision
0.29.0+cu126, NumPy 2.5.2, and Python 3.14.8; the legacy pinned stack is untested.

The local full-run command was stopped during downloading, before any full-data
training began. Its incomplete download and empty run artifacts were removed.
Full training is reserved for the user's server. No accuracy result is available.
