# Installation

Use Python 3.10 or later. Run commands from the source root; editable
installation keeps the configuration and script paths available.

## CPU source checks

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
python -m pytest -q tests
```

These tests exercise code contracts with synthetic inputs. They do not
validate a trained checkpoint or execute a full benchmark.

## GPU inference

Install a CUDA-enabled PyTorch build appropriate for the machine first,
then install the inference extras:

```bash
python -m pip install -e ".[inference]"
```

The inference extras pin PyTorch 2.7.0 and Transformers 4.57.3.
The PyTorch version follows the repository runtime requirements; the
Transformers version follows the retained ViSS model configuration.
The extras declare Accelerate, qwen-vl-utils, and Decord as well.
They specify an installation recipe, not a claim that every GPU/driver
combination has been validated.

Install FFmpeg and ensure `ffprobe` is on `PATH`. Multi-GPU shell launchers
also use Bash 4 or newer, `jq`, `sha256sum`, and `nvidia-smi`.

## Training and attention

```bash
python -m pip install -e ".[inference,training]"
python -m pip install flash-attn==2.7.4.post1 --no-build-isolation
```

FlashAttention installation needs a compatible CUDA toolkit and compiler.
The trainer defaults to `flash_attention_2`. Use `--attn-implementation sdpa`
for a development run when FlashAttention is unavailable; this changes the
runtime configuration.

The package includes no model weights. Before a GPU run, confirm that the
selected checkpoint loads with `AutoModelForImageTextToText` and that the
processor matches the intended evaluation contract.

## Environment records

Save the installed package versions and launch command alongside your own
runs. Generated outputs belong under `outputs/` or `checkpoints/` and are
excluded from the source archive. Do not commit credentials, benchmark media,
or local machine paths.
