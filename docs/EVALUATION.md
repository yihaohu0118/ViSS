# Inference, TVS, and scoring

Run from the repository root after [installation](INSTALL.md) and
[data preparation](DATA.md). The code separates model inference, view-policy
composition, and official scoring.

## 1. Generate recent-view proposals

Create a local configuration from the public template and your own assets:

```bash
python -m scripts.configure_evaluation_protocol \
  --streaming-samples data/streamingbench/queries.jsonl \
  --ovo-samples data/ovo/queries.jsonl \
  --processor-config checkpoints/viss/locked_processor/preprocessor_config.json \
  --output configs/evaluation/recent4_full_v1.json
```

The generated file contains local paths and file identities. It remains
private and is excluded from the source archive. Then validate coverage:

```bash
python -m scripts.validate_evaluation_protocol \
  --protocol configs/evaluation/recent4_full_v1.json
```

This check requires the external benchmark exports. It is not a data-free
installation test.

Use the command in the main README with the prepared checkpoint and locked
processor. The runner accepts `--benchmark streamingbench` or
`--benchmark ovo`; annotations, query exports, and media must match.

Use `--record-choice-confidence` when the proposals will be passed to TVS.
For a reproducibility run, supply `--model-manifest` with the matching model
identity artifact. The same identity must be retained across all views.
The runner also supports `--start-index` and `--end-index` for sharding.

The optional multi-GPU launcher is:

```text
bash scripts/run_recent4_single_benchmark.sh \
  REPO MODEL MODEL_MANIFEST OUTPUT PYTHON BENCHMARK GPU_CSV MODEL_ID
```

It additionally requires the benchmark paths documented by its environment
variable checks. Start with the Python entry point when preparing new assets.

## 2. Apply the TVS acquisition and commitment policy

The event policy is `causalbudget20`; the compact-history policy is
`causalhybrid32`. Use the same checkpoint and the view encodings prescribed
by your protocol. Selecting a policy name alone does not fix image resolution
or the processor configuration.

Given recent-view predictions and a frozen calibration artifact:

```bash
python -m experiments.select_uncertainty_escalation_uids \
  --recent outputs/recent.json \
  --threshold data/calibration/thresholds.json \
  --samples data/benchmark/queries.jsonl \
  --output-uids outputs/event_uids.json \
  --output-samples outputs/event_queries.jsonl \
  --output-audit outputs/selection.json
```

Run event-view inference on the selected queries. Then compose the first
two stages:

```bash
python -m experiments.materialize_uncertainty_temporal_agent \
  --recent outputs/recent.json \
  --temporal outputs/event.json \
  --selection-audit outputs/selection.json \
  --threshold-audit data/calibration/thresholds.json \
  --output outputs/two_view.json
```

Once the required compact-history proposals are available:

```bash
python -m experiments.materialize_confidence_gated_third_view_agent \
  --recent outputs/recent.json \
  --event outputs/event.json \
  --base-agent outputs/two_view.json \
  --third-view outputs/history.json \
  --threshold-audit data/calibration/thresholds.json \
  --output outputs/tvs.json
```

These commands name user-generated files; no prediction or score artifacts
are shipped. Materialization combines existing proposals and does not call
the VLM. `materialize_tvs_view_oracle.py` uses reference correctness for
analysis and is not the deployed selection policy.

## 3. Score saved predictions

```bash
python -m scripts.score_official_benchmark_predictions \
  --benchmark streamingbench \
  --annotation data/streamingbench/annotations.json \
  --predictions outputs/tvs.json \
  --official-root external/StreamingBench \
  --output-json outputs/streamingbench_score.json
```

Use `--benchmark ovo` and the corresponding official assets for OVO-Bench.
The adapter checks query coverage and invokes the original scorer. For OVO,
the primary metric is the task-then-group macro average. The official scorer
may have additional dependencies; install those from its own repository.

Generated predictions and scores are your run outputs. They are excluded
from version control and from the source-release archive.
