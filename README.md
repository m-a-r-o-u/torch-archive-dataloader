# torch-archive-dataloader

This repository is organized around runnable data-loading experiments.

## Installation

Set up a local virtual environment with `uv`, then install the runtime dependencies:

```bash
uv venv
source .venv/bin/activate
uv pip install torch torchvision
```

## Experiments

- [Original Minimal ImageNet-C TTA Example](experiments/original-imagenetc-tta/README.md)
- [Archive-backed ImageNet-C TTA Example](experiments/archive-imagenetc-tta/README.md)
- [WebDataset-streaming ImageNet-C TTA Example](experiments/webdataset-imagenetc-tta/README.md)
