#!/bin/bash
# Pre- vs post-adaptation for training-free in-context learning (gr00t/rag/icl.py).
#
#   ./run_icl_eval.sh [k] [num_clients_per_arm]
#
# Both arms serve the same plain checkpoint through scripts/rag/7_serve_icl.py: "pre" with
# k=0 (stock policy), "post" with k retrieved in-context frames. Each arm runs the same fixed
# variant list (VARIANTS, default eval_results/icl/variants10.json), episode 0, seed 0.
# One server per arm; its clients share it (ZMQ REQ/REP serialises the calls).
set -uo pipefail
cd "$(dirname "$0")"

K="${1:-1}"
NUM_CLIENTS="${2:-5}"
source env.sh
source env_eval.sh

SUITE=libero_10
CHECKPOINT="${CHECKPOINT:-$REPO_ROOT/models/public_finetunes/libero_long_posttrain}"
MEMORY="${MEMORY:-$REPO_ROOT/models/icl/libero_10/memory.pt}"
VARIANTS="${VARIANTS:-$REPO_ROOT/eval_results/icl/variants10.json}"
EXEC_HORIZON="${EXEC_HORIZON:-8}"
TAG="${TAG:-k${K}}"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/eval_results/icl/$TAG}"
LOGS="$REPO_ROOT/logs/icl/$TAG"
mkdir -p "$LOGS" "$OUT_ROOT"

PIDS=()
cleanup() { for pid in "${PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done; }
trap cleanup EXIT

declare -A ARM_K=([pre]=0 [post]=$K)
declare -A ARM_PORT=([pre]=8871 [post]=8872)
for arm in pre post; do
  CUDA_VISIBLE_DEVICES="$GPU" $CONDA_PY scripts/rag/7_serve_icl.py \
    --checkpoint-path "$CHECKPOINT" --memory-path "$MEMORY" --k "${ARM_K[$arm]}" \
    --retrieval-log "$OUT_ROOT/$arm/retrieval.jsonl" --port "${ARM_PORT[$arm]}" --cuda 0 \
    > "$LOGS/${arm}_server.log" 2>&1 &
  PIDS+=($!)
  mkdir -p "$OUT_ROOT/$arm"
done

for arm in pre post; do
  for _ in $(seq 1 120); do
    grep -q "Server is ready" "$LOGS/${arm}_server.log" 2>/dev/null && break
    sleep 10
  done
  grep -q "Server is ready" "$LOGS/${arm}_server.log" || { echo "$arm server failed"; tail -20 "$LOGS/${arm}_server.log"; exit 1; }
done
echo ">>> servers ready (k=$K)"

CLIENT_PIDS=()
for arm in pre post; do
  for i in $(seq 0 $((NUM_CLIENTS - 1))); do
    PYTHONPATH="$EVAL_PYTHONPATH" LD_PRELOAD="$EVAL_LD_PRELOAD" \
      $EVAL_PY examples/Libero/eval/run_libero_plus_eval.py \
      --task-suite-name "$SUITE" --port "${ARM_PORT[$arm]}" \
      --shard-index "$i" --num-shards "$NUM_CLIENTS" \
      --variants-file "$VARIANTS" --num-trials-per-task 1 \
      --exec-horizon "$EXEC_HORIZON" --timeout-ms 600000 \
      --out-dir "$OUT_ROOT/$arm" \
      > "$LOGS/${arm}_client${i}.log" 2>&1 &
    CLIENT_PIDS+=($!); PIDS+=($!)
  done
done
echo ">>> ${#CLIENT_PIDS[@]} clients running; logs in $LOGS"

FAILED=0
for pid in "${CLIENT_PIDS[@]}"; do wait "$pid" || FAILED=1; done

for arm in pre post; do
  echo "=== $arm (k=${ARM_K[$arm]}) ==="
  $EVAL_PY examples/Libero/eval/score_libero_plus.py --result-dir "$OUT_ROOT/$arm" --suite "$SUITE" \
    | tee "$LOGS/${arm}_score.log"
done
exit $FAILED
