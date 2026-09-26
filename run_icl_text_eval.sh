#!/bin/bash
# Instruction-augmentation arms of the in-context study (gr00t/rag/icl.py, text_mode).
#
#   ./run_icl_text_eval.sh <arm> [<arm> ...]      arms: pre | retrieved | generated
#
# Every arm serves the same plain checkpoint through scripts/rag/7_serve_icl.py with no
# feature/action concatenation (k=0); they differ only in the text appended to the
# instruction. "generated" needs the hint VLM (scripts/rag/8_serve_hint_vlm.py) already
# listening on HINT_URL. All arms run VARIANTS (default: the original 10 + 10 hard-axis
# variants; VARIANTS=all runs every LIBERO-Plus variant of the suite), episode 0, seed 0.
# Each arm gets SERVERS_PER_ARM policy servers; its NUM_CLIENTS clients are spread over them.
set -uo pipefail
cd "$(dirname "$0")"
[ $# -ge 1 ] || { echo "usage: $0 <pre|retrieved|generated>..."; exit 1; }
ARMS=("$@")

source env.sh
source env_eval.sh

SUITE=libero_10
CHECKPOINT="${CHECKPOINT:-$REPO_ROOT/models/public_finetunes/libero_long_posttrain}"
MEMORY="${MEMORY:-$REPO_ROOT/models/icl/libero_10/memory.pt}"
VARIANTS="${VARIANTS:-$REPO_ROOT/eval_results/icl/variants20.json}"
HINT_URL="${HINT_URL:-http://127.0.0.1:8890/hint}"
NUM_CLIENTS="${NUM_CLIENTS:-10}"
SERVERS_PER_ARM="${SERVERS_PER_ARM:-1}"
if [ "$VARIANTS" = all ]; then
  SELECTION=(--num-tasks-per-axis 0)
else
  SELECTION=(--variants-file "$VARIANTS")
fi
EXEC_HORIZON="${EXEC_HORIZON:-8}"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/eval_results/icl/text}"
LOGS="${LOGS_DIR:-$REPO_ROOT/logs/icl/text}"
mkdir -p "$LOGS"

# Server j of an arm listens on PORT[arm] + 10*j.
declare -A PORT=([pre]=8881 [retrieved]=8882 [generated]=8883)
PIDS=()
cleanup() { for pid in "${PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done; }
trap cleanup EXIT

for arm in "${ARMS[@]}"; do
  mode=$arm; [ "$arm" = pre ] && mode=none
  mkdir -p "$OUT_ROOT/$arm"
  for j in $(seq 0 $((SERVERS_PER_ARM - 1))); do
    CUDA_VISIBLE_DEVICES="$GPU" $CONDA_PY scripts/rag/7_serve_icl.py \
      --checkpoint-path "$CHECKPOINT" --memory-path "$MEMORY" --k 0 \
      --text-mode "$mode" --hint-url "$HINT_URL" \
      --retrieval-log "$OUT_ROOT/$arm/retrieval.jsonl" \
      --port "$((PORT[$arm] + 10 * j))" --cuda 0 \
      > "$LOGS/${arm}_server${j}.log" 2>&1 &
    PIDS+=($!)
  done
done

for arm in "${ARMS[@]}"; do
  for j in $(seq 0 $((SERVERS_PER_ARM - 1))); do
    log="$LOGS/${arm}_server${j}.log"
    for _ in $(seq 1 120); do
      grep -q "Server is ready" "$log" 2>/dev/null && break
      sleep 10
    done
    grep -q "Server is ready" "$log" || { echo "$arm server $j failed"; tail -20 "$log"; exit 1; }
  done
done
echo ">>> servers ready: ${ARMS[*]}"

CLIENT_PIDS=()
for arm in "${ARMS[@]}"; do
  for i in $(seq 0 $((NUM_CLIENTS - 1))); do
    PYTHONPATH="$EVAL_PYTHONPATH" LD_PRELOAD="$EVAL_LD_PRELOAD" \
      $EVAL_PY examples/Libero/eval/run_libero_plus_eval.py \
      --task-suite-name "$SUITE" --port "$((PORT[$arm] + 10 * (i % SERVERS_PER_ARM)))" \
      --shard-index "$i" --num-shards "$NUM_CLIENTS" \
      "${SELECTION[@]}" --num-trials-per-task 1 \
      --exec-horizon "$EXEC_HORIZON" --timeout-ms 900000 \
      --out-dir "$OUT_ROOT/$arm" \
      > "$LOGS/${arm}_client${i}.log" 2>&1 &
    CLIENT_PIDS+=($!); PIDS+=($!)
  done
done
echo ">>> ${#CLIENT_PIDS[@]} clients running; logs in $LOGS"

FAILED=0
for pid in "${CLIENT_PIDS[@]}"; do wait "$pid" || FAILED=1; done

for arm in "${ARMS[@]}"; do
  echo "=== $arm ==="
  $EVAL_PY examples/Libero/eval/score_libero_plus.py --result-dir "$OUT_ROOT/$arm" --suite "$SUITE" \
    | tee "$LOGS/${arm}_score.log"
done
exit $FAILED
