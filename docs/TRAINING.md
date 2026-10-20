# Training ViSS

Install the training environment described in [Installation](INSTALL.md) and
prepare a sealed training release as described in [Data preparation](DATA.md).

## Paper recipe

The manuscript specifies one epoch, learning rate `2e-6`, cosine scheduling,
warmup ratio `0.03`, seed `59`, maximum length `8192`, maximum image pixels
`200704`, and a global batch of `12`. The vision tower is frozen; the language
model and multimodal projector are trainable.

For two GPUs, a microbatch of one and six accumulation steps give the stated
global batch:

```bash
torchrun --nproc_per_node=2 -m experiments.train_qwen3vl_sft \
  --model checkpoints/Qwen3-VL-8B-Instruct \
  --train data/viss_release/train.jsonl \
  --output checkpoints/viss \
  --epochs 1 \
  --lr 2e-6 \
  --lr-scheduler-type cosine \
  --warmup-ratio 0.03 \
  --per-device-batch-size 1 \
  --gradient-accumulation-steps 6 \
  --max-length 8192 \
  --max-pixels 200704 \
  --seed 59 \
  --train-order sequential \
  --freeze-vision-tower \
  --train-vision-projector \
  --attn-implementation flash_attention_2 \
  --deepspeed configs/training/deepspeed_zero3_bf16.json
```

Replace `train.jsonl` with the actual row file inside the sealed release.
The sequential setting preserves prepared row order. The paper preparation
uses a parent stage followed by capability rows, with stage shuffle seed 59;
that ordering must already be reflected in the input release.

Changing the GPU count requires adjusting accumulation if the global batch
is to remain fixed. Check the trainer's complete options with:

```bash
python -m experiments.train_qwen3vl_sft --help
```

The training entry point requires Transformers even for `--help`. Training
outputs and logs are generated at runtime and are not part of this archive.
