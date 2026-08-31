# Agentic Discovery of Non-Canonical Antimicrobial Peptides with AMPGAN v3

This repository contains the official code implementation for the paper **Agentic Discovery of Non-Canonical Antimicrobial Peptides with AMPGAN v3**. 

You can read the full paper here: [Link to Paper](https://arxiv.org/html/2606.17127v1)

Please contact jay-hwasung.jung@uvm.edu 

## Overview

The repository has two halves:

- **`training/`** — the AMPGAN v3 conditional GAN (transformer generator + two discriminators), trained on DBAASP. Conditioned on target species, MIC, and sequence length.
- **`pepcraft/`** — the agentic discovery loop (LangGraph planner/executor) that calls the trained generator, then filters and verifies candidates.

## Requirements

Python 3.11–3.12 and an NVIDIA GPU (CPU works, just slower). Dependencies are declared in `pyproject.toml`.

```bash
# with uv (recommended)
uv sync
source .venv/bin/activate
```

The default `torch` wheel targets **CUDA 12.6**. On a CPU-only machine, change the
index URL in `pyproject.toml` to `https://download.pytorch.org/whl/cpu` before syncing.

Two features need external tooling that is *not* installed by `uv sync`:

| Feature | Requires |
| --- | --- |
| `pepcraft/tools/Generating/Predict_Structure.py` | the [`simplefold`](https://github.com/apple/ml-simplefold) CLI on `$PATH` |
| `pepcraft/tools/Verifying/Verify_{DBAASP,SwissProt}.py` | NCBI BLAST+ (`blastp`) and local BLAST databases |

`pepcraft` calls Gemini, so set `GOOGLE_API_KEY` in your environment before running it.

## Pretrained weights

Released checkpoints live in [`weights/`](weights/) — run 7, epoch 200:

| File | Model | Size |
| --- | --- | --- |
| `Generator_7_200.pth` | Generator | 9.4 MB |
| `Discriminator1_7_200.pth` | GAN discriminator (`D_GAN`) | 1.0 MB |
| `Discriminator2_7_200.pth` | MIC discriminator (`D_MIC`) | 1.0 MB |

These are plain `state_dict` files, loadable with `weights_only=True`:

```python
import torch
from models.Generator import Generator
from data_utils import AMPDatasets

dataset = AMPDatasets(max_length=68, data_path="training/data")
model = Generator(output_shape=(68, len(dataset.tokens)), species_shape=(6,), embed_dim=128)
model.load_state_dict(torch.load("weights/Generator_7_200.pth", weights_only=True))
model.eval()
```

## Usage

### Training

```bash
cd training
uv run train.py run=1          # checkpoints land in training/logs/{Generator,Discriminator1,Discriminator2}
```

Config is Hydra-based (`training/conf/`); override anything on the command line,
e.g. `uv run train.py run=7 trainer.max_epochs=201 batch_size=256`.
Training logs to Weights & Biases (project `AMPGANv3`) — run `wandb offline` to disable.

### Sampling from a checkpoint

```bash
cd training
uv run generate_samples.py 7   # uses weights/Generator_7_200.pth by default
```

Point it at your own runs with `AMPGAN_CKPT_DIR=logs/Generator AMPGAN_CKPT_EPOCH=100`.

### Agentic discovery

```bash
cd pepcraft
export GOOGLE_API_KEY=...
uv run Initialization.py <run_tag>
```

## Citation
If you use this code or find our work helpful, please consider citing our paper:

```bibtex
@misc{jung2026agenticdiscoverynoncanonicalantimicrobial,
      title={Agentic Discovery of Non-Canonical Antimicrobial Peptides with AMPGAN v3}, 
      author={Jay Jung and Xiaohan Zhang and Shenghan Song and Mahmoud Sayedahmed and Chijian Xiang and Yunong Xu and Ahmed AbdelKhalek and Severin T. Schneebeli and Matthew J. Wargo and Jianing Li and Safwan Wshah},
      year={2026},
      eprint={2606.17127},
      archivePrefix={arXiv},
      primaryClass={q-bio.QM},
      url={https://arxiv.org/abs/2606.17127}, 
}
