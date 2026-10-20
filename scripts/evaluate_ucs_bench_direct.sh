#!/usr/bin/env bash
# dest1-direct UCS-Bench eval. Consumes CUDA_VISIBLE_DEVICES ordinals.
# Never uses scheduler. Never rewrites placement. Caller must pick idle GPUs.
# Paper protocol is timestamp_causal_model_only. Structured-memory is not scheduled.
set -Eeuo pipefail

ROOT="${ROOT:?set repository root}"
PY="${PY:?set Python executable}"
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || {
  echo "set CUDA_VISIBLE_DEVICES to idle GPU ordinals" >&2; exit 2; }
IFS=',' read -r -a _visible <<< "$CUDA_VISIBLE_DEVICES"
LOCAL_WORLD_SIZE="${#_visible[@]}"
export LOCAL_WORLD_SIZE
MODEL="${MODEL:?set complete materialized model}"
PROCESSOR="${PROCESSOR:-$MODEL}"
QUESTIONS="${QUESTIONS:?set materialized UCS questions JSONL}"
VIDEOS="${VIDEOS:?set UCS video directory}"
OUTPUT="${OUTPUT:?set a fresh output directory}"
EXPECTED_ROWS="${EXPECTED_ROWS:-8114}"
EXPECTED_VIDEOS="${EXPECTED_VIDEOS:-532}"
EXPECTED_PREFIXES="${EXPECTED_PREFIXES:-7287}"
PROTOCOL="${PROTOCOL:?set timestamp_causal_model_only or timestamp_causal_structured_memory}"
ID_FILE="${ID_FILE:-}"
EVAL_ALLOW_RESUME="${EVAL_ALLOW_RESUME:-0}"
UCS_SKIP_STATE_WRITER="${UCS_SKIP_STATE_WRITER:-0}"
UCS_FRAME_CACHE_DIR="${UCS_FRAME_CACHE_DIR:-$(dirname "$OUTPUT")/.ucs_uniform128_frame_cache}"
UCS_CACHE_READY="${UCS_CACHE_READY:-$UCS_FRAME_CACHE_DIR/DATA_READY}"
UCS_CACHE_EXPECTED_ROWS="${UCS_CACHE_EXPECTED_ROWS:-$EXPECTED_ROWS}"
UCS_SUPPORT_READY_ROOT="${UCS_SUPPORT_READY_ROOT:-}"
UCS_SUPPORT_EXPECTED_ROWS="${UCS_SUPPORT_EXPECTED_ROWS:-$EXPECTED_ROWS}"
UCS_FRAME_POLICY="${UCS_FRAME_POLICY:-uniform}"
UCS_NFRAMES="${UCS_NFRAMES:-128}"
UCS_SEMANTIC_RECENT_NFRAMES="${UCS_SEMANTIC_RECENT_NFRAMES:-8}"
UCS_SEMANTIC_REFERENCE_NFRAMES="${UCS_SEMANTIC_REFERENCE_NFRAMES:-8}"
UCS_SHARD_MODE="${UCS_SHARD_MODE:-interleaved}"
STATE_PACKET_MODE="${STATE_PACKET_MODE:-state_and_evidence}"
STATE_EVIDENCE_FRAMES="${STATE_EVIDENCE_FRAMES:-32}"
MODEL_IDENTITY_FILE="${MODEL_IDENTITY_FILE:-$MODEL/MATERIALIZATION_COMPLETE.json}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE_OVERRIDE:-1}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS_OVERRIDE:-8}"
EVAL_ANSWER_PARSER="${EVAL_ANSWER_PARSER:-native_letter}"
EVAL_SAVE_RAW_REPLY="${EVAL_SAVE_RAW_REPLY:-0}"
EVAL_CPU_THREADS="${OVOS_EVAL_CPU_THREADS:-4}"
EVAL_PREFETCH_WORKERS="${OVOS_EVAL_FRAME_PREFETCH_WORKERS:-1}"
EVAL_PREFETCH_DEPTH="${OVOS_EVAL_FRAME_PREFETCH_DEPTH:-$EVAL_PREFETCH_WORKERS}"

case "$LOCAL_WORLD_SIZE" in 1|2|4|8) ;; *) echo "UCS evaluator requires 1/2/4/8 GPUs" >&2; exit 2 ;; esac
[[ "$EXPECTED_ROWS" =~ ^[1-9][0-9]*$ ]] || { echo "invalid EXPECTED_ROWS" >&2; exit 2; }
[[ "$STATE_EVIDENCE_FRAMES" =~ ^[1-9][0-9]*$ ]] || { echo "invalid STATE_EVIDENCE_FRAMES" >&2; exit 2; }
case "$EVAL_BATCH_SIZE" in 1|2|4) ;; *) echo "EVAL_BATCH_SIZE_OVERRIDE must be 1, 2, or 4" >&2; exit 2 ;; esac
[[ "$EVAL_MAX_NEW_TOKENS" =~ ^[1-9][0-9]*$ ]] || { echo "invalid EVAL_MAX_NEW_TOKENS_OVERRIDE" >&2; exit 2; }
case "$EVAL_ANSWER_PARSER" in native_letter|canonical_json_value) ;; *) echo "unsupported EVAL_ANSWER_PARSER=$EVAL_ANSWER_PARSER" >&2; exit 2 ;; esac
case "$EVAL_SAVE_RAW_REPLY" in 0|1) ;; *) echo "EVAL_SAVE_RAW_REPLY must be 0 or 1" >&2; exit 2 ;; esac
[[ "$EVAL_CPU_THREADS" =~ ^[1-9][0-9]*$ ]] || { echo "invalid OVOS_EVAL_CPU_THREADS" >&2; exit 2; }
[[ "$EVAL_PREFETCH_WORKERS" =~ ^[1-9][0-9]*$ ]] || { echo "invalid OVOS_EVAL_FRAME_PREFETCH_WORKERS" >&2; exit 2; }
[[ "$EVAL_PREFETCH_DEPTH" =~ ^[1-9][0-9]*$ ]] || { echo "invalid OVOS_EVAL_FRAME_PREFETCH_DEPTH" >&2; exit 2; }
[[ "$UCS_NFRAMES" =~ ^[1-9][0-9]*$ ]] || { echo "invalid UCS_NFRAMES" >&2; exit 2; }
[[ "$UCS_SEMANTIC_RECENT_NFRAMES" =~ ^[1-9][0-9]*$ ]] || { echo "invalid UCS_SEMANTIC_RECENT_NFRAMES" >&2; exit 2; }
[[ "$UCS_SEMANTIC_REFERENCE_NFRAMES" =~ ^[1-9][0-9]*$ ]] || { echo "invalid UCS_SEMANTIC_REFERENCE_NFRAMES" >&2; exit 2; }
[[ "$UCS_CACHE_EXPECTED_ROWS" =~ ^[1-9][0-9]*$ ]] || { echo "invalid UCS_CACHE_EXPECTED_ROWS" >&2; exit 2; }
[[ "$UCS_SUPPORT_EXPECTED_ROWS" =~ ^[1-9][0-9]*$ ]] || { echo "invalid UCS_SUPPORT_EXPECTED_ROWS" >&2; exit 2; }
case "$UCS_FRAME_POLICY" in uniform|recent|log_decay|cyclic|semantic_adaptive) ;; *) echo "unsupported UCS_FRAME_POLICY=$UCS_FRAME_POLICY" >&2; exit 2 ;; esac
for path in "$PY" "$MODEL/config.json" "$MODEL_IDENTITY_FILE" \
  "$PROCESSOR/preprocessor_config.json" "$QUESTIONS" \
  "$UCS_CACHE_READY" \
  "$ROOT/synthesis/eval_ovos_uniform.py" "$ROOT/scripts/merge_ucs_causal_shards.py" \
  "$ROOT/scripts/build_ucs_causal_state_snapshots.py" \
  "$ROOT/scripts/merge_ucs_causal_state_snapshots.py"; do
  [[ -e "$path" ]] || { echo "missing UCS input: $path" >&2; exit 2; }
done
"$PY" - "$UCS_CACHE_READY" "$QUESTIONS" "$UCS_FRAME_CACHE_DIR" "$UCS_CACHE_EXPECTED_ROWS" <<'PY'
import json,sys
from pathlib import Path
ready=json.load(open(sys.argv[1],encoding="utf-8"))
assert ready.get("format")=="iclr_spatial.ucs_uniform128_cache_ready.v1"
assert ready.get("ok") is True and ready.get("cache_only") is True
assert ready.get("resize_fallback") is False
assert ready.get("all_cache_entries_present") is True
assert int(ready.get("rows") or 0)==int(sys.argv[4])
assert Path(ready["questions"]).resolve()==Path(sys.argv[2]).resolve()
assert Path(ready["cache_dir"]).resolve()==Path(sys.argv[3]).resolve()
PY
if [[ "$UCS_FRAME_POLICY" == semantic_adaptive ]]; then
  [[ -n "$UCS_SUPPORT_READY_ROOT" ]] || {
    echo "semantic_adaptive UCS requires UCS_SUPPORT_READY_ROOT" >&2; exit 2; }
  "$PY" - "$UCS_SUPPORT_READY_ROOT/semantic_DATA_READY" \
    "$UCS_SUPPORT_EXPECTED_ROWS" "$UCS_FRAME_CACHE_DIR" <<'PY'
import json,sys
from pathlib import Path
payload=json.load(open(sys.argv[1],encoding="utf-8"))
expected=int(sys.argv[2]); cache=Path(sys.argv[3]).resolve()
assert payload.get("format")=="iclr_spatial.ovos_frame_cache_data_ready.v1"
assert payload.get("ok") is True and int(payload.get("rows") or 0)==expected
assert payload.get("frame_policy")=="semantic_adaptive"
assert int(payload.get("nframes") or 0)==128
assert Path(payload["cache_dir"]).resolve()==cache
routes=payload.get("resolved_route_counts") or {}
assert sum(int(value) for value in routes.values())==expected
assert set(routes)<= {"uniform","recent","log_decay"}
PY
fi
"$PY" - "$MODEL_IDENTITY_FILE" <<'PY'
import json,sys
payload=json.load(open(sys.argv[1],encoding="utf-8"))
fmt = payload.get("format")
assert fmt in {
    "iclr_spatial.hf_materialization_complete.v2",
    "iclr_spatial.model_identity.v1",
    "iclr_video.hf_model_snapshot.v1",
}
if fmt == "iclr_video.hf_model_snapshot.v1":
    assert payload.get("repo_id") == "Qwen/Qwen3-VL-8B-Instruct"
    requested = str(payload.get("requested_revision") or "")
    resolved = str(payload.get("resolved_revision") or "")
    assert len(resolved) == 40 and requested == resolved
PY
[[ -d "$VIDEOS" ]] || { echo "missing UCS video root: $VIDEOS" >&2; exit 2; }
if [[ -n "$ID_FILE" ]]; then [[ -s "$ID_FILE" ]] || { echo "missing UCS ID file" >&2; exit 2; }; fi
case "$PROTOCOL" in
  timestamp_causal_model_only) state_compiler=none ;;
  timestamp_causal_structured_memory) state_compiler=persistent_snapshots ;;
  *) echo "unsupported UCS protocol: $PROTOCOL" >&2; exit 2 ;;
esac
case "$EVAL_ALLOW_RESUME" in
  0) [[ ! -e "$OUTPUT" ]] || { echo "refusing existing UCS OUTPUT=$OUTPUT" >&2; exit 3; } ;;
  1) mkdir -p "$OUTPUT" ;;
  *) echo "EVAL_ALLOW_RESUME must be 0 or 1" >&2; exit 2 ;;
esac
case "$UCS_SKIP_STATE_WRITER" in
  0) ;;
  1)
    [[ "$state_compiler" == persistent_snapshots ]] || {
      echo "UCS_SKIP_STATE_WRITER requires timestamp_causal_structured_memory" >&2
      exit 2
    }
    ;;
  *) echo "UCS_SKIP_STATE_WRITER must be 0 or 1" >&2; exit 2 ;;
esac
case "$STATE_PACKET_MODE" in
  evidence_only|state_and_evidence) ;;
  *) echo "unsupported STATE_PACKET_MODE=$STATE_PACKET_MODE" >&2; exit 2 ;;
esac
case "$UCS_SHARD_MODE" in contiguous|interleaved) ;; *) echo "unsupported UCS_SHARD_MODE=$UCS_SHARD_MODE" >&2; exit 2 ;; esac

mkdir -p "$OUTPUT/shards" "$OUTPUT/logs" "$UCS_FRAME_CACHE_DIR"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OVOS_FRAME_CACHE_DIR="$UCS_FRAME_CACHE_DIR"
export OVOS_FRAME_CACHE_ONLY=1
export OVOS_FRAME_CACHE_RESIZE_FALLBACK=0
unset OVOS_FRAME_LONG_QUERY_CAP_SEC OVOS_FRAME_LONG_QUERY_CAP_PATH_SUBSTR

state_snapshot_args=()
if [[ "$state_compiler" == persistent_snapshots ]]; then
  state_root="$OUTPUT/state_snapshots"
  merged_state="$state_root/merged.json"
  if [[ "$UCS_SKIP_STATE_WRITER" == 1 ]]; then
    "$PY" - "$merged_state" "$EXPECTED_VIDEOS" "$EXPECTED_PREFIXES" <<'PY'
import json, sys

path, expected_videos, expected_prefixes = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
payload = json.load(open(path, encoding="utf-8"))
assert payload.get("format") == "iclr_spatial.ucs_query_independent_state_release.v1"
assert payload.get("complete") is True
assert payload.get("question_or_options_visible_to_writer") is False
assert int(payload.get("videos") or 0) == expected_videos
assert int(payload.get("unique_prefixes") or 0) == expected_prefixes
store = payload.get("snapshot_store") or {}
snapshots = store.get("snapshots") or []
assert len(snapshots) == expected_prefixes
PY
  else
    mkdir -p "$state_root/shards"
    state_pids=()
    state_shards=()
    state_id_args=()
    if [[ -n "$ID_FILE" ]]; then state_id_args=(--id-file "$ID_FILE"); fi
    for ((rank=0; rank<LOCAL_WORLD_SIZE; rank++)); do
      state_shard="$state_root/shards/rank$(printf '%02d' "$rank").json"
      state_shards+=("$state_shard")
      state_resume=()
      if [[ "$EVAL_ALLOW_RESUME" == 1 ]]; then state_resume=(--resume); fi
      LOCAL_RANK="$rank" OVOS_EVAL_CPU_THREADS=4 "$PY" \
        "$ROOT/scripts/build_ucs_causal_state_snapshots.py" \
        --model "$MODEL" --processor "$PROCESSOR" --annotation "$QUESTIONS" \
        --video-base "$VIDEOS" --output "$state_shard" \
        --shard-rank "$rank" --shard-world-size "$LOCAL_WORLD_SIZE" \
        --frames-per-update 16 --frame-size 448 --max-new-tokens 384 \
        --checkpoint-every 10 "${state_id_args[@]}" "${state_resume[@]}" \
        >"$OUTPUT/logs/state_rank$(printf '%02d' "$rank").log" 2>&1 &
      state_pids+=("$!")
    done
    state_status=0
    for pid in "${state_pids[@]}"; do wait "$pid" || state_status=$?; done
    (( state_status == 0 )) || exit "$state_status"
    if [[ ! -s "$merged_state" ]]; then
      merge_state_args=()
      for shard in "${state_shards[@]}"; do
        merge_state_args+=(--shard "$shard")
      done
      "$PY" "$ROOT/scripts/merge_ucs_causal_state_snapshots.py" \
        "${merge_state_args[@]}" --expected-videos "$EXPECTED_VIDEOS" \
        --expected-prefixes "$EXPECTED_PREFIXES" --output "$merged_state"
    fi
  fi
  state_snapshot_args=(
    --state-snapshot-manifest "$merged_state"
    --state-packet-mode "$STATE_PACKET_MODE"
    --state-evidence-frames "$STATE_EVIDENCE_FRAMES"
  )
  state_compiler=none
fi

pids=()
shards=()
for ((rank=0; rank<LOCAL_WORLD_SIZE; rank++)); do
  shard_args=()
  if [[ "$UCS_SHARD_MODE" == interleaved ]]; then
    # Round-robin assignment balances long-prefix decode cost substantially
    # better than contiguous row ranges without changing the evaluated set.
    count=$(( (EXPECTED_ROWS + LOCAL_WORLD_SIZE - 1 - rank) / LOCAL_WORLD_SIZE ))
    shard_args=(--shard-rank "$rank" --shard-world-size "$LOCAL_WORLD_SIZE")
  else
    # Compatibility lane for checkpoints created before interleaved sharding.
    start=$(( EXPECTED_ROWS * rank / LOCAL_WORLD_SIZE ))
    end=$(( EXPECTED_ROWS * (rank + 1) / LOCAL_WORLD_SIZE ))
    count=$(( end - start ))
    shard_args=(--offset "$start")
  fi
  shard="$OUTPUT/shards/rank$(printf '%02d' "$rank").json"
  shards+=("$shard")
  if [[ "$EVAL_ALLOW_RESUME" == 1 && -s "$shard" ]] && \
     "$PY" - "$shard" "$count" <<'PY' >/dev/null 2>&1
import json,sys
x=json.load(open(sys.argv[1],encoding="utf-8"))
raise SystemExit(0 if len(x.get("results") or []) == int(sys.argv[2]) else 1)
PY
  then
    continue
  fi
  extra=(--state-compiler "$state_compiler" "${state_snapshot_args[@]}")
  if [[ "$state_compiler" != none ]]; then
    extra+=(--state-compiler-max-new-tokens 384 --state-compiler-chunk-frames 16 --state-compiler-overlap-frames 4)
  fi
  ids=()
  if [[ -n "$ID_FILE" ]]; then ids=(--id-file "$ID_FILE"); fi
  resume=()
  if [[ "$EVAL_ALLOW_RESUME" == 1 ]]; then resume=(--resume); fi
  raw_reply=()
  if [[ "$EVAL_SAVE_RAW_REPLY" == 1 ]]; then raw_reply=(--save-raw-reply); fi
  LOCAL_RANK="$rank" OVOS_EVAL_CPU_THREADS="$EVAL_CPU_THREADS" \
    OVOS_EVAL_FRAME_PREFETCH_WORKERS="$EVAL_PREFETCH_WORKERS" \
    OVOS_EVAL_FRAME_PREFETCH_DEPTH="$EVAL_PREFETCH_DEPTH" \
    "$PY" "$ROOT/synthesis/eval_ovos_uniform.py" \
    --model "$MODEL" --processor "$PROCESSOR" --annotation "$QUESTIONS" \
    --video-base "$VIDEOS" --frame-policy "$UCS_FRAME_POLICY" --nframes "$UCS_NFRAMES" --frame-size 448 \
    --semantic-recent-nframes "$UCS_SEMANTIC_RECENT_NFRAMES" \
    --semantic-reference-nframes "$UCS_SEMANTIC_REFERENCE_NFRAMES" \
    --prompt-style default --media-serialization ordered_image_blocks \
    --decode-mode generate --max-new-tokens "$EVAL_MAX_NEW_TOKENS" \
    --answer-parser "$EVAL_ANSWER_PARSER" --batch-size "$EVAL_BATCH_SIZE" \
    "${shard_args[@]}" \
    --limit "$count" --checkpoint-every 5 \
    --out "$shard" "${ids[@]}" "${extra[@]}" "${raw_reply[@]}" "${resume[@]}" \
    >"$OUTPUT/logs/rank$(printf '%02d' "$rank").log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do wait "$pid" || status=$?; done
(( status == 0 )) || exit "$status"

merge_args=()
for shard in "${shards[@]}"; do merge_args+=(--shard "$shard"); done
"$PY" "$ROOT/scripts/merge_ucs_causal_shards.py" \
  "${merge_args[@]}" --expected-rows "$EXPECTED_ROWS" \
  --output "$OUTPUT/predictions.json"
score_ids=()
if [[ -n "$ID_FILE" ]]; then score_ids=(--id-file "$ID_FILE"); fi
"$PY" "$ROOT/scripts/evaluate_ucs_bench.py" \
  --predictions "$OUTPUT/predictions.json" --questions "$QUESTIONS" \
  --protocol "$PROTOCOL" --model "$MODEL" --expected-total "$EXPECTED_ROWS" \
  --output "$OUTPUT/official_score.json" "${score_ids[@]}"
printf 'ok\n' >"$OUTPUT/complete"
printf '%s\n' "DIRECT_UCS_EVAL_COMPLETE=$OUTPUT"
