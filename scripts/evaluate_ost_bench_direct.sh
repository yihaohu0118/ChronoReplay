#!/usr/bin/env bash
# dest1-direct OST-Bench eval. Consumes CUDA_VISIBLE_DEVICES ordinals.
# Never uses scheduler. Never rewrites placement. Caller must pick idle GPUs.
set -Eeuo pipefail

fail_closed() {
  local rc=$?
  if (( rc != 0 )) && [[ -n "${OUTPUT_ROOT:-}" && ! -e "${OUTPUT_ROOT}/complete" ]]; then
    mkdir -p "$OUTPUT_ROOT"
    printf 'fail\n' >"$OUTPUT_ROOT/failed"
  fi
  exit "$rc"
}
trap fail_closed EXIT

ROOT="${ROOT:?set deployed code root}"
PY="${PY:?set evaluation Python}"
MODEL="${MODEL:?set a model or official-base path}"
MODEL_READY_FILE="${MODEL_READY_FILE:-$MODEL/MATERIALIZATION_COMPLETE.json}"
ANNOTATION="${ANNOTATION:?set OST_bench.json}"
IMAGE_ROOT="${IMAGE_ROOT:?set the OST-Bench image root}"
MANIFEST="${MANIFEST:?set the scene-disjoint manifest}"
OUTPUT_ROOT="${OUTPUT_ROOT:?set a fresh evaluation output root}"
CHUNK_SIZE="${CHUNK_SIZE:-48}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
CLASSICAL_TRAJECTORY="${CLASSICAL_TRAJECTORY:-0}"
SELF_REFINE="${SELF_REFINE:-0}"
SPATIAL_LEDGER_PROMPT="${SPATIAL_LEDGER_PROMPT:-0}"
PERSISTENT_STATE_WRITER="${PERSISTENT_STATE_WRITER:-0}"
COMPACT_STATE_WRITER="${COMPACT_STATE_WRITER:-0}"
APPEND_ONLY_STATE_WRITER="${APPEND_ONLY_STATE_WRITER:-0}"
WRITER_MAX_TOKENS="${WRITER_MAX_TOKENS:-0}"
WRITER_ARBITRATION="${WRITER_ARBITRATION:-0}"
SELF_CONSISTENCY_SAMPLES="${SELF_CONSISTENCY_SAMPLES:-1}"
SELF_CONSISTENCY_TEMPERATURE="${SELF_CONSISTENCY_TEMPERATURE:-0.3}"
CANDIDATE_A_DIR="${CANDIDATE_A_DIR:-}"
CANDIDATE_B_DIR="${CANDIDATE_B_DIR:-}"
CANDIDATE_CHOICE_ONLY="${CANDIDATE_CHOICE_ONLY:-0}"
OPERATOR_GROUNDED_CANDIDATE_CHOICE="${OPERATOR_GROUNDED_CANDIDATE_CHOICE:-0}"
CANDIDATE_ANSWER_ONLY="${CANDIDATE_ANSWER_ONLY:-0}"
CANDIDATE_ORDER_FLIP="${CANDIDATE_ORDER_FLIP:-0}"
CANDIDATE_INDEPENDENT_VERIFICATION="${CANDIDATE_INDEPENDENT_VERIFICATION:-0}"
CAUSAL_SUPPORT_VIEW="${CAUSAL_SUPPORT_VIEW:-full}"
PROPOSAL_LOGPROBS="${PROPOSAL_LOGPROBS:-0}"
RESTRICT_TO_MANIFEST="${RESTRICT_TO_MANIFEST:-1}"
SHARD_INDEX="${SHARD_INDEX:-0}"
SHARD_COUNT="${SHARD_COUNT:-1}"
SKIP_SCORE="${SKIP_SCORE:-0}"
SCORE_ONLY="${SCORE_ONLY:-0}"

score_ost() {
  "$PY" -m benchmark.ost.score \
    --predictions "$OUTPUT_ROOT/predictions" \
    --annotation "$ANNOTATION" \
    --manifest "$MANIFEST" \
    --model-path "$MODEL" \
    --output "$OUTPUT_ROOT/official_score.json"
  printf 'ok\n' >"$OUTPUT_ROOT/complete"
  printf '%s\n' "DIRECT_OST_EVAL_COMPLETE=$OUTPUT_ROOT"
}

if [[ "$SCORE_ONLY" == "1" ]]; then
  for path in "$PY" "$ANNOTATION" "$MANIFEST" "$MODEL/config.json" \
    "$ROOT/benchmark/ost/score.py"; do
    [[ -s "$path" ]] || { echo "missing required artifact: $path" >&2; exit 2; }
  done
  export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
  score_ost
  exit 0
fi

[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || {
  echo "set CUDA_VISIBLE_DEVICES to idle GPU ordinals" >&2; exit 2; }
IFS=',' read -r -a _visible <<< "$CUDA_VISIBLE_DEVICES"
LOCAL_WORLD_SIZE="${#_visible[@]}"
if (( SHARD_COUNT > 1 )); then
  (( LOCAL_WORLD_SIZE == 1 )) || {
    echo "sharded OST uses one GPU per shard" >&2; exit 2; }
else
  (( LOCAL_WORLD_SIZE == 1 || LOCAL_WORLD_SIZE == 2 || LOCAL_WORLD_SIZE == 4 )) || {
    echo "direct OST supports 1, 2, or 4 idle GPUs" >&2; exit 2; }
fi
export LOCAL_WORLD_SIZE

for path in "$PY" "$ANNOTATION" "$MANIFEST" "$MODEL/config.json" \
  "$MODEL_READY_FILE" \
  "$ROOT/benchmark/ost/inference.py" "$ROOT/benchmark/ost/score.py"; do
  [[ -s "$path" ]] || { echo "missing required artifact: $path" >&2; exit 2; }
done
[[ -d "$IMAGE_ROOT" ]] || { echo "missing image root: $IMAGE_ROOT" >&2; exit 2; }

ost_root="$(dirname "$ANNOTATION")"
for forbidden in OST_bench_train.json img_train.zip; do
  [[ -e "$ost_root/$forbidden" ]] && {
    echo "OST-Bench training split present: $ost_root/$forbidden" >&2; exit 2; }
done

predictions="$OUTPUT_ROOT/predictions"
mkdir -p "$predictions" "$OUTPUT_ROOT/logs"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn

manifest_arg=()
[[ "$RESTRICT_TO_MANIFEST" == "1" ]] && manifest_arg=(--manifest "$MANIFEST")
traj_arg=()
case "$CLASSICAL_TRAJECTORY" in
  0) ;;
  1) traj_arg=(--classical-trajectory) ;;
  *) echo "CLASSICAL_TRAJECTORY must be 0 or 1" >&2; exit 2 ;;
esac
refine_arg=()
case "$SELF_REFINE" in
  0) ;;
  1) refine_arg=(--self-refine) ;;
  *) echo "SELF_REFINE must be 0 or 1" >&2; exit 2 ;;
esac
ledger_arg=()
case "$SPATIAL_LEDGER_PROMPT" in
  0) ;;
  1) ledger_arg=(--spatial-ledger-prompt) ;;
  *) echo "SPATIAL_LEDGER_PROMPT must be 0 or 1" >&2; exit 2 ;;
esac
writer_arg=()
case "$PERSISTENT_STATE_WRITER" in
  0) ;;
  1) writer_arg=(--persistent-state-writer) ;;
  *) echo "PERSISTENT_STATE_WRITER must be 0 or 1" >&2; exit 2 ;;
esac
compact_writer_arg=()
case "$COMPACT_STATE_WRITER" in
  0) ;;
  1) compact_writer_arg=(--compact-state-writer) ;;
  *) echo "COMPACT_STATE_WRITER must be 0 or 1" >&2; exit 2 ;;
esac
append_writer_arg=()
case "$APPEND_ONLY_STATE_WRITER" in
  0) ;;
  1) append_writer_arg=(--append-only-state-writer) ;;
  *) echo "APPEND_ONLY_STATE_WRITER must be 0 or 1" >&2; exit 2 ;;
esac
(( PERSISTENT_STATE_WRITER + COMPACT_STATE_WRITER + APPEND_ONLY_STATE_WRITER <= 1 )) || {
  echo "writer modes are exclusive" >&2; exit 2; }
[[ $WRITER_MAX_TOKENS =~ ^[0-9]+$ ]] || {
  echo "WRITER_MAX_TOKENS must be non-negative" >&2; exit 2; }
writer_budget_arg=()
(( WRITER_MAX_TOKENS > 0 )) && writer_budget_arg=(--writer-max-tokens "$WRITER_MAX_TOKENS")
arbitration_arg=()
case "$WRITER_ARBITRATION" in
  0) ;;
  1) arbitration_arg=(--writer-arbitration) ;;
  *) echo "WRITER_ARBITRATION must be 0 or 1" >&2; exit 2 ;;
esac
(( WRITER_ARBITRATION == 0 || PERSISTENT_STATE_WRITER + COMPACT_STATE_WRITER + APPEND_ONLY_STATE_WRITER == 1 )) || {
  echo "WRITER_ARBITRATION requires exactly one writer mode" >&2; exit 2; }
[[ $SELF_CONSISTENCY_SAMPLES =~ ^[1-9][0-9]*$ ]] || {
  echo "invalid SELF_CONSISTENCY_SAMPLES" >&2; exit 2; }
consistency_args=(
  --self-consistency-samples "$SELF_CONSISTENCY_SAMPLES"
  --self-consistency-temperature "$SELF_CONSISTENCY_TEMPERATURE"
)
candidate_args=()
if [[ -n "$CANDIDATE_A_DIR" || -n "$CANDIDATE_B_DIR" ]]; then
  [[ -d "$CANDIDATE_A_DIR" && -d "$CANDIDATE_B_DIR" ]] || {
    echo "candidate verifier requires two prediction directories" >&2; exit 2; }
  candidate_args=(--candidate-a-dir "$CANDIDATE_A_DIR" --candidate-b-dir "$CANDIDATE_B_DIR")
fi
case "$CANDIDATE_CHOICE_ONLY" in
  0) ;;
  1)
    [[ ${#candidate_args[@]} -gt 0 ]] || { echo "candidate choice requires candidates" >&2; exit 2; }
    candidate_args+=(--candidate-choice-only)
    ;;
  *) echo "CANDIDATE_CHOICE_ONLY must be 0 or 1" >&2; exit 2 ;;
esac
operator_candidate_args=()
case "$OPERATOR_GROUNDED_CANDIDATE_CHOICE" in
  0) ;;
  1)
    [[ "$CANDIDATE_CHOICE_ONLY" == 1 ]] || {
      echo "operator-grounded candidate choice requires CANDIDATE_CHOICE_ONLY=1" >&2
      exit 2
    }
    operator_candidate_args+=(--operator-grounded-candidate-choice)
    ;;
  *) echo "OPERATOR_GROUNDED_CANDIDATE_CHOICE must be 0 or 1" >&2; exit 2 ;;
esac
candidate_control_args=()
for value in "$CANDIDATE_ANSWER_ONLY" "$CANDIDATE_ORDER_FLIP" "$CANDIDATE_INDEPENDENT_VERIFICATION"; do
  case "$value" in 0|1) ;; *) echo "candidate controls must be 0 or 1" >&2; exit 2 ;; esac
done
if [[ "$CANDIDATE_ANSWER_ONLY" == 1 || "$CANDIDATE_ORDER_FLIP" == 1 || "$CANDIDATE_INDEPENDENT_VERIFICATION" == 1 ]]; then
  [[ "$CANDIDATE_CHOICE_ONLY" == 1 ]] || {
    echo "candidate controls require CANDIDATE_CHOICE_ONLY=1" >&2; exit 2; }
fi
[[ "$CANDIDATE_ANSWER_ONLY" == 1 ]] && candidate_control_args+=(--candidate-answer-only)
[[ "$CANDIDATE_ORDER_FLIP" == 1 ]] && candidate_control_args+=(--candidate-order-flip)
[[ "$CANDIDATE_INDEPENDENT_VERIFICATION" == 1 ]] && candidate_control_args+=(--candidate-independent-verification)
case "$CAUSAL_SUPPORT_VIEW" in full|recent|event|cyclic) ;; *) echo "invalid CAUSAL_SUPPORT_VIEW" >&2; exit 2 ;; esac
case "$PROPOSAL_LOGPROBS" in 0|1) ;; *) echo "invalid PROPOSAL_LOGPROBS" >&2; exit 2 ;; esac
proposal_logprob_arg=()
[[ "$PROPOSAL_LOGPROBS" == 1 ]] && proposal_logprob_arg=(--proposal-logprobs)

ATTEMPTS="${ATTEMPTS:-4}"
log="$OUTPUT_ROOT/logs/shard${SHARD_INDEX}of${SHARD_COUNT}.log"
attempt=1
while (( attempt <= ATTEMPTS )); do
  if "$PY" -m benchmark.ost.inference \
    --model "$MODEL" \
    --annotation "$ANNOTATION" \
    --image-root "$IMAGE_ROOT" \
    --output "$predictions" \
    --shard-index "$SHARD_INDEX" \
    --shard-count "$SHARD_COUNT" \
    --chunk-size "$CHUNK_SIZE" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --gpu-mem-util "$GPU_MEM_UTIL" \
    --tensor-parallel-size "$LOCAL_WORLD_SIZE" \
    --causal-support-view "$CAUSAL_SUPPORT_VIEW" \
    "${manifest_arg[@]}" \
    "${traj_arg[@]}" \
    "${refine_arg[@]}" \
    "${ledger_arg[@]}" \
    "${writer_arg[@]}" \
    "${compact_writer_arg[@]}" \
    "${append_writer_arg[@]}" \
    "${writer_budget_arg[@]}" \
    "${arbitration_arg[@]}" \
    "${consistency_args[@]}" \
    "${candidate_args[@]}" \
    "${operator_candidate_args[@]}" \
    "${candidate_control_args[@]}" \
    "${proposal_logprob_arg[@]}" >>"$log" 2>&1; then
    break
  fi
  printf '%s tensor_parallel_attempt=%s failed; resuming completed scans\n' \
    "$(date -Is)" "$attempt" >>"$log"
  attempt=$((attempt + 1))
  sleep 60
done
(( attempt <= ATTEMPTS )) || {
  echo "OST-Bench inference failed; see $log" >&2; exit 3; }

if [[ "$SKIP_SCORE" == "1" ]]; then
  printf '%s\n' "DIRECT_OST_SHARD_COMPLETE=$OUTPUT_ROOT shard=${SHARD_INDEX}/${SHARD_COUNT}"
  exit 0
fi
score_ost
