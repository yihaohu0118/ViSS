#!/usr/bin/env bash
set -euo pipefail

# Run one half of the locked recent4_full_v1 evaluation.  Two independent
# invocations may therefore evaluate StreamingBench and OVO concurrently on
# disjoint GPU sets while retaining the exact common runner and official
# scorers used by the serial baseline launcher.

if [[ $# -ne 8 ]]; then
  echo "usage: $0 REPO MODEL MODEL_MANIFEST OUTPUT PYTHON BENCHMARK GPU_CSV MODEL_ID" >&2
  exit 2
fi

REPO=$1
MODEL=$2
MODEL_MANIFEST=$3
OUTPUT=$4
PYTHON=$5
BENCHMARK=$6
GPU_CSV=$7
MODEL_ID=$8
PROCESSOR=${STREAMEVA_PROCESSOR:-$MODEL}
EVAL_BATCH_SIZE=${STREAMEVA_EVAL_BATCH_SIZE:-16}
DECODE_PREFETCH_BATCHES=${STREAMEVA_DECODE_PREFETCH_BATCHES:-2}
FLUSH_EVERY_BATCHES=${STREAMEVA_EVAL_FLUSH_EVERY_BATCHES:-16}
RECORD_CHOICE_CONFIDENCE=${STREAMEVA_RECORD_CHOICE_CONFIDENCE:-1}
FRAME_POLICY=${STREAMEVA_FRAME_POLICY:-recent4}
VISUAL_INPUT_MODE=${STREAMEVA_VISUAL_INPUT_MODE:-image-list}
ALLOW_SHARED_GPU=${STREAMEVA_ALLOW_SHARED_GPU:-0}
MIN_FREE_MIB=${STREAMEVA_MIN_FREE_MIB:-35840}
PROTOCOL=$REPO/configs/evaluation/recent4_full_v1.json
RUNNER=${STREAMEVA_EVALUATOR_ENTRYPOINT:-$REPO/experiments/evaluate_qwen3vl_recent4_full.py}
MODEL_IDENTITY_SHA256=${STREAMEVA_MODEL_IDENTITY_SHA256:-}

# Keep non-login tmux jobs aligned with the CUDA wheel runtime used by
# training.  In particular, libcusparse from the wheel requires the matching
# bundled nvJitLink library to precede system CUDA libraries.
PYTHON_SITE=$($PYTHON -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
NVIDIA_LIB_DIRS=$(find "$PYTHON_SITE/nvidia" -mindepth 2 -maxdepth 2 \
  -type d -name lib -print 2>/dev/null | LC_ALL=C sort | paste -sd: -)
if [[ -n $NVIDIA_LIB_DIRS ]]; then
  export LD_LIBRARY_PATH="$NVIDIA_LIB_DIRS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
[[ $EVAL_BATCH_SIZE =~ ^[1-9][0-9]*$ ]]
[[ $DECODE_PREFETCH_BATCHES =~ ^[0-9]+$ ]]
[[ $FLUSH_EVERY_BATCHES =~ ^[1-9][0-9]*$ ]]
[[ $RECORD_CHOICE_CONFIDENCE =~ ^[01]$ ]]
[[ $FRAME_POLICY =~ ^(recent4|wide8|wide12|wide16|prefix8|prefix16|prefix32|prefix64|causalbudget20)$ ]]
[[ $VISUAL_INPUT_MODE =~ ^(image-list|native-video|hybrid-history-video-recent-images)$ ]]
[[ $ALLOW_SHARED_GPU =~ ^[01]$ ]]
[[ $MIN_FREE_MIB =~ ^[1-9][0-9]*$ ]]

case "$BENCHMARK" in
  streamingbench)
    ANNOTATION=${STREAM_ANNOTATION:?Set STREAM_ANNOTATION to the StreamingBench annotation file}
    SAMPLES=${STREAM_SAMPLES:?Set STREAM_SAMPLES to the frozen StreamingBench query export}
    VIDEO_ROOT=${STREAM_VIDEO_ROOT:?Set STREAM_VIDEO_ROOT to authorized benchmark media}
    OFFICIAL_ROOT=${STREAM_OFFICIAL_ROOT:?Set STREAM_OFFICIAL_ROOT to the official scorer}
    EXPECTED=2495
    SMOKE_UIDS=(streaming_0_0 streaming_8_0 streaming_61_4)
    ;;
  ovo)
    OVO_ROOT=${OVO_ROOT:?Set OVO_ROOT to authorized OVO runtime assets}
    ANNOTATION=${OVO_ANNOTATION:-$OVO_ROOT/data/ovo_bench_new.json}
    OVO_RUNTIME_REPORT=${OVO_RUNTIME_REPORT:-$OVO_ROOT/data/runtime_assets.json}
    SAMPLES=${OVO_SAMPLES:?Set OVO_SAMPLES to the frozen OVO query export}
    VIDEO_ROOT=${OVO_VIDEO_ROOT:-$OVO_ROOT}
    OFFICIAL_ROOT=${OVO_OFFICIAL_ROOT:-$OVO_ROOT}
    EXPECTED=3035
    SMOKE_UIDS=(ovo_0 ovo_1117 ovo_1468_0 ovo_1516_0 ovo_1558_0)
    ;;
  *) echo "BENCHMARK must be streamingbench or ovo" >&2; exit 2 ;;
esac

IFS=, read -r -a GPUS <<<"$GPU_CSV"
[[ ${#GPUS[@]} -gt 0 ]] || exit 2
declare -A SEEN=()
for gpu in "${GPUS[@]}"; do
  [[ $gpu =~ ^[0-7]$ || $gpu =~ ^GPU-[0-9A-Fa-f-]{36}$ ]] || {
    echo "GPU_CSV must contain local indices or GPU UUIDs" >&2
    exit 2
  }
  if [[ -n ${SEEN[$gpu]:-} && $ALLOW_SHARED_GPU -eq 0 ]]; then
    echo "GPU_CSV may repeat an index only with STREAMEVA_ALLOW_SHARED_GPU=1" >&2
    exit 2
  fi
  SEEN[$gpu]=1
done
[[ $MODEL_ID =~ ^[A-Za-z0-9._-]+$ ]] || exit 2
for path in "$MODEL/config.json" \
  "$PROCESSOR/preprocessor_config.json" "$PROTOCOL" "$RUNNER" \
  "$ANNOTATION" "$SAMPLES" "$VIDEO_ROOT" "$OFFICIAL_ROOT"; do
  test -e "$path"
done
if [[ $MODEL_MANIFEST == none ]]; then
  [[ $MODEL_IDENTITY_SHA256 =~ ^[0-9a-f]{64}$ ]] || {
    echo "manifest-free evaluation requires STREAMEVA_MODEL_IDENTITY_SHA256" >&2
    exit 2
  }
  MODEL_IDENTITY_ARGS=(--model-identity-sha256 "$MODEL_IDENTITY_SHA256")
else
  test -s "$MODEL_MANIFEST"
  MODEL_IDENTITY_ARGS=(--model-manifest "$MODEL_MANIFEST")
fi
if [[ $BENCHMARK == ovo ]]; then
  test -s "$OVO_RUNTIME_REPORT"
  ANNOTATION_SHA256=$(sha256sum "$ANNOTATION" | cut -d' ' -f1)
  FROZEN_EXPORT_SHA256=$(sha256sum "$SAMPLES" | cut -d' ' -f1)
  jq -e --arg annotation_sha256 "$ANNOTATION_SHA256" \
    --arg frozen_export_sha256 "$FROZEN_EXPORT_SHA256" '
      .status == "pass"
      and .expanded_queries == 3035
      and .covered_queries == 3035
      and .coverage == 1
      and .missing_videos == 0
      and .annotation_sha256 == $annotation_sha256
      and .frozen_export_sha256 == $frozen_export_sha256
    ' "$OVO_RUNTIME_REPORT" >/dev/null
fi
test -x "$PYTHON"
export PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}

if [[ ${PREFLIGHT_ONLY:-0} == 1 ]]; then
  "$PYTHON" -c 'import PIL, qwen_vl_utils, torch, transformers'
  printf '{"status":"cpu_preflight_pass","benchmark":"%s","protocol":"recent4_full_v1","batch_size":%s,"decode_prefetch_batches":%s}\n' \
    "$BENCHMARK" "$EVAL_BATCH_SIZE" "$DECODE_PREFETCH_BATCHES"
  exit 0
fi

mkdir -p "$OUTPUT/logs" "$OUTPUT/$BENCHMARK"
if [[ -e $OUTPUT/$BENCHMARK/complete ]]; then
  echo "$BENCHMARK is already complete"
  exit 0
fi
if [[ $MODEL_MANIFEST == none ]]; then
  :
elif [[ -n ${PREVERIFIED_MODEL_MANIFEST_SHA256:-} ]]; then
  [[ $PREVERIFIED_MODEL_MANIFEST_SHA256 =~ ^[0-9a-f]{64}$ ]] || exit 2
  [[ $(sha256sum "$MODEL_MANIFEST" | cut -d' ' -f1) == \
    "$PREVERIFIED_MODEL_MANIFEST_SHA256" ]] || {
    echo "preverified model manifest hash mismatch" >&2
    exit 3
  }
else
  (cd "$MODEL" && sha256sum --quiet -c SHA256SUMS)
fi
if [[ $MODEL_MANIFEST != none ]]; then
  MODEL_IDENTITY_SHA256=$(sha256sum "$MODEL_MANIFEST" | cut -d' ' -f1)
fi
"$PYTHON" "$REPO/scripts/validate_evaluation_protocol.py" --protocol "$PROTOCOL" \
  >"$OUTPUT/$BENCHMARK/protocol_validation.json"

wait_gpu_free() {
  local gpu=$1 pids used free
  while true; do
    pids=$(nvidia-smi -i "$gpu" --query-compute-apps=pid \
      --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')
    used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used \
      --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')
    free=$(nvidia-smi -i "$gpu" --query-gpu=memory.free \
      --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')
    if [[ $ALLOW_SHARED_GPU -eq 0 ]]; then
      if [[ -z $pids && ${used:-999999} -le 16 ]]; then return 0; fi
    elif [[ ${free:-0} -ge $MIN_FREE_MIB ]]; then
      return 0
    fi
    echo "$(date -u +%FT%TZ) waiting for GPU $gpu (used=${used:-unknown} MiB free=${free:-unknown} MiB shared=$ALLOW_SHARED_GPU minimum_free=$MIN_FREE_MIB MiB)"
    sleep 20
  done
}

validate_result() {
  "$PYTHON" - "$1" "$2" "$FRAME_POLICY" "$VISUAL_INPUT_MODE" "$RECORD_CHOICE_CONFIDENCE" <<'PY'
import json, sys
x = json.load(open(sys.argv[1], encoding="utf-8"))
assert x["evaluation_protocol"] == "recent4_full_v1"
assert x["num_samples"] == int(sys.argv[2])
assert x["num_errors"] == 0
assert all(str(row.get("prediction") or "").strip() for row in x["rows"])
requested_policy = sys.argv[3]
visual_input_mode = sys.argv[4]
record_choice_confidence = bool(int(sys.argv[5]))
assert x["processor_geometry"]["visual_input_mode"] == visual_input_mode
assert x["comparable_under_locked_recent4_protocol"] == (
    requested_policy == "recent4" and visual_input_mode == "image-list"
)
for row in x["rows"]:
    sampling = row["sampling"]
    if requested_policy == "prefix8":
        assert sampling["num_sampled_frames"] == 8
    elif requested_policy == "prefix16":
        assert sampling["num_sampled_frames"] == 16
    elif requested_policy == "prefix32":
        assert sampling["num_sampled_frames"] == 32
    elif requested_policy == "prefix64":
        assert sampling["num_sampled_frames"] == 64
    elif requested_policy == "causalbudget20":
        assert 1 <= sampling["num_sampled_frames"] <= 20
        assert sampling["controller"]["benchmark_task_or_group_router"] is False
        assert sampling["controller"]["future_frame_violations"] == 0
    else:
        assert sampling["num_sampled_frames"] == 4
    assert max(sampling["frame_timestamps"]) <= row["query_time"] + 1e-3
    assert row["visual_input_mode"] == visual_input_mode
    assert bool(row.get("generation_confidence")) == record_choice_confidence
    assert bool(row["video_grid_thw"]) == (
        visual_input_mode in {"native-video", "hybrid-history-video-recent-images"}
        and bool(row.get("native_video_metadata"))
    )
PY
}

CONFIDENCE_ARGS=()
if [[ $RECORD_CHOICE_CONFIDENCE -eq 1 ]]; then
  CONFIDENCE_ARGS+=(--record-choice-confidence)
fi

SMOKE_OUTPUT=$OUTPUT/$BENCHMARK/smoke.json
if ! test -s "$SMOKE_OUTPUT" \
    || ! validate_result "$SMOKE_OUTPUT" "${#SMOKE_UIDS[@]}"; then
  wait_gpu_free "${GPUS[0]}"
  smoke_batch=$EVAL_BATCH_SIZE
  smoke_rc=1
  for smoke_attempt in 1 2 3; do
    smoke_rc=0
    CUDA_VISIBLE_DEVICES=${GPUS[0]} PYTHONPATH=$REPO PYTHONUNBUFFERED=1 \
      TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      "$PYTHON" "$RUNNER" --benchmark "$BENCHMARK" \
        --annotation "$ANNOTATION" --samples "$SAMPLES" \
        --video-root "$VIDEO_ROOT" --model "$MODEL" --processor "$PROCESSOR" \
        "${MODEL_IDENTITY_ARGS[@]}" --protocol "$PROTOCOL" \
        --output "$SMOKE_OUTPUT" --uids "${SMOKE_UIDS[@]}" \
        --frame-policy "$FRAME_POLICY" \
        --visual-input-mode "$VISUAL_INPUT_MODE" \
        --batch-size "$smoke_batch" \
        --decode-prefetch-batches "$DECODE_PREFETCH_BATCHES" \
        --flush-every-batches "$FLUSH_EVERY_BATCHES" \
        "${CONFIDENCE_ARGS[@]}" \
        --resume --retry-errors >>"$OUTPUT/logs/${BENCHMARK}_smoke.log" 2>&1 || smoke_rc=$?
    if validate_result "$SMOKE_OUTPUT" "${#SMOKE_UIDS[@]}"; then break; fi
    echo "warning: $BENCHMARK smoke attempt=$smoke_attempt batch=$smoke_batch failed rc=$smoke_rc" >&2
    if (( smoke_batch > 1 )); then smoke_batch=$(( (smoke_batch + 1) / 2 )); fi
  done
  if validate_result "$SMOKE_OUTPUT" "${#SMOKE_UIDS[@]}"; then
    if [[ $smoke_rc -ne 0 ]]; then
      echo "warning: $BENCHMARK smoke exited rc=$smoke_rc after writing a complete validated result" >&2
    fi
  else
    echo "$BENCHMARK smoke failed validation after runner rc=$smoke_rc" >&2
    if [[ $smoke_rc -eq 0 ]]; then smoke_rc=1; fi
    exit "$smoke_rc"
  fi
fi

SHARD_UID_ROOT=$OUTPUT/$BENCHMARK/shard_uids
mkdir -p "$SHARD_UID_ROOT"
"$PYTHON" - "$SAMPLES" "$SHARD_UID_ROOT" "${#GPUS[@]}" <<'PY'
import json, sys
from pathlib import Path
source, output, count = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
uids = [str(json.loads(line)["uid"]) for line in source.read_text().splitlines() if line]
assert len(uids) == len(set(uids))
for index in range(count):
    shard = uids[index::count]
    (output / f"shard{index}.uids").write_text("\n".join(shard) + "\n")
assert sum(1 for path in output.glob("shard*.uids") for line in path.read_text().splitlines() if line) == len(uids)
PY

run_shard() {
  local index=$1 gpu=$2 uid_file=$3 expected=$4 output log attempt runner_rc attempt_batch
  output=$OUTPUT/$BENCHMARK/shard${index}.json
  log=$OUTPUT/logs/${BENCHMARK}_shard${index}.log
  wait_gpu_free "$gpu"
  attempt_batch=$EVAL_BATCH_SIZE
  for attempt in 1 2 3; do
    runner_rc=0
    CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$REPO PYTHONUNBUFFERED=1 \
      TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      "$PYTHON" "$RUNNER" --benchmark "$BENCHMARK" \
        --annotation "$ANNOTATION" --samples "$SAMPLES" \
        --video-root "$VIDEO_ROOT" --model "$MODEL" --processor "$PROCESSOR" \
        "${MODEL_IDENTITY_ARGS[@]}" --protocol "$PROTOCOL" \
        --output "$output" --uids-file "$uid_file" \
        --frame-policy "$FRAME_POLICY" \
        --visual-input-mode "$VISUAL_INPUT_MODE" \
        --batch-size "$attempt_batch" \
        --decode-prefetch-batches "$DECODE_PREFETCH_BATCHES" \
        --flush-every-batches "$FLUSH_EVERY_BATCHES" \
        "${CONFIDENCE_ARGS[@]}" \
        --resume --retry-errors >>"$log" 2>&1 || runner_rc=$?
    if validate_result "$output" "$expected"; then
      if [[ $runner_rc -ne 0 ]]; then
        echo "warning: $BENCHMARK shard$index exited rc=$runner_rc after writing a complete validated result" >&2
      fi
      return 0
    fi
    echo "warning: $BENCHMARK shard$index attempt=$attempt batch=$attempt_batch failed validation after runner rc=$runner_rc" >&2
    if (( attempt_batch > 1 )); then attempt_batch=$(( (attempt_batch + 1) / 2 )); fi
  done
  return 1
}

pids=()
shards=()
for index in "${!GPUS[@]}"; do
  uid_file=$SHARD_UID_ROOT/shard${index}.uids
  expected=$(awk 'NF {count++} END {print count+0}' "$uid_file")
  shards+=("$OUTPUT/$BENCHMARK/shard${index}.json")
  run_shard "$index" "${GPUS[$index]}" "$uid_file" "$expected" &
  pids+=("$!")
done
failed=0
for index in "${!pids[@]}"; do
  pid=${pids[$index]}
  if wait "$pid"; then
    echo "$(date -u +%FT%TZ) $BENCHMARK shard$index exited rc=0"
  else
    rc=$?
    echo "$(date -u +%FT%TZ) $BENCHMARK shard$index exited rc=$rc" >&2
    failed=1
  fi
done
[[ $failed -eq 0 ]]

"$PYTHON" "$REPO/scripts/merge_eval_shards.py" --inputs "${shards[@]}" \
  --output "$OUTPUT/$BENCHMARK/full.json" --expected-count "$EXPECTED" \
  --expected-uids "$SAMPLES"
validate_result "$OUTPUT/$BENCHMARK/full.json" "$EXPECTED"
"$PYTHON" "$REPO/scripts/score_official_benchmark_predictions.py" \
  --benchmark "$BENCHMARK" --official-root "$OFFICIAL_ROOT" \
  --annotation "$ANNOTATION" --predictions "$OUTPUT/$BENCHMARK/full.json" \
  --output-json "$OUTPUT/$BENCHMARK/official_score.json" \
  --official-input-json "$OUTPUT/$BENCHMARK/official_input.json" \
  >"$OUTPUT/logs/${BENCHMARK}_official_score.log" 2>&1

RUNTIME_REPORT_ARG=${OVO_RUNTIME_REPORT:-}
"$PYTHON" - "$OUTPUT/$BENCHMARK" "$MODEL_IDENTITY_SHA256" "$PROTOCOL" \
  "$MODEL_ID" "$BENCHMARK" "$PROCESSOR" "$RUNTIME_REPORT_ARG" <<'PY'
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
out, model_identity, protocol = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
model_id, benchmark, processor = sys.argv[4:7]
processor = Path(processor)
runtime_report = Path(sys.argv[7]) if sys.argv[7] else None
score = json.loads((out / "official_score.json").read_text())
full = json.loads((out / "full.json").read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
summary = {
    "format": "iclr_video.single_benchmark.v2",
    "benchmark": benchmark,
    "model_id": model_id,
    "protocol_id": "recent4_full_v1",
    "inference_policy": full["inference_policy"],
    "inference_policy_hash": full["inference_policy_hash"],
    "policy_config": full["policy_config"],
    "comparable_under_locked_recent4_protocol": full.get(
        "comparable_under_locked_recent4_protocol"
    ),
    "model_identity_sha256": model_identity,
    "processor_path": str(processor),
    "processor_config_sha256": sha(processor / "preprocessor_config.json"),
    "processor_manifest_sha256": (
        sha(processor / "SHA256SUMS")
        if (processor / "SHA256SUMS").is_file()
        else None
    ),
    "protocol_sha256": sha(protocol),
    "benchmark_runtime_assets_sha256": (
        sha(runtime_report) if runtime_report is not None else None
    ),
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "official_score": score,
}
(out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
PY
touch "$OUTPUT/$BENCHMARK/complete"
echo "$(date -u +%FT%TZ) completed $BENCHMARK for $MODEL_ID"
