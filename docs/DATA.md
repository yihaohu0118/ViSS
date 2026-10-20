# Data preparation

The code package distributes construction and validation code. Training
records, source media, benchmark exports, and calibration artifacts are
external inputs.

## Training records

The trainer consumes a sealed semantic release. Its validation checks the
manifest, source inventory, user/assistant message schema, media identity,
and permitted targets. See:

- `viss_core/data_v0/schema.py`
- `viss_core/data_v0/semantic_release.py`
- `viss_core/data_v0/trainer_adapter.py`

The source-aligned builder combines already prepared source releases:

```bash
python -m data_pipeline.temporal_scope.build_source_aligned_sft_release \
  --source data/prepared_source_a \
  --source data/prepared_source_b \
  --output data/viss_release \
  --release-id my_viss_release \
  --copy-mode copy
```

`prepared_source_a` and `prepared_source_b` are placeholders for valid
source releases, not raw-video directories. This generic example does not
reconstruct the paper's accepted training mixture by itself.

The paper describes the training mixture. Company training records and their
manifests are not distributed. The training guide assumes users have prepared
their own valid source releases, media, example weights, and ordering.

Set `VISS_UPSTREAM_MANIFEST_SHA256` to the trusted upstream identity for your
training inputs before starting the builder or trainer. The validator retains
the identity check, but the code package embeds no company-specific digest.
Private legacy source adapters are excluded; supply the standard semantic
release format instead.

## Evaluation assets

For each benchmark, prepare:

1. Official annotations and authorized video files.
2. A query export using the fields read by `viss_core/benchmarks.py`.
3. The benchmark's official scoring installation.
4. Model and processor files compatible with the locked evaluation protocol.

`configs/evaluation/recent4.example.json` describes the public sampling,
query-count, prompt, and processor conventions. It contains no run dates,
company file identities, or query records. Follow the evaluation guide to
create a local configuration bound to your own query exports and processor.
The generated configuration is excluded from publication.

## TVS calibration

The TVS policy consumes acquisition/commit calibration artifacts and aligned
view predictions from one model. Calibrate on source-disjoint data, then
freeze the thresholds for benchmark evaluation. The archive includes the
selection and analysis implementation, but not the paper's calibration
records or precomputed predictions.
