#!/bin/bash
# pi0.5 (pi05_libero) version of the in-context study; same client, variant lists, scorer.
#
#   ./run_pi05_icl_eval.sh <out_name> <variants|all> <arm> [<arm> ...]
#   arms: pre | concat | retrieved | generated
#     pre       stock openpi policy
#     concat    k=1 retrieved prefix KV + generated action chunk concatenated
#     retrieved retrieved clean instruction appended to the prompt
#     generated hint-VLM sentence appended (needs scripts/rag/8_serve_hint_vlm.py on HINT_URL)
#
# Actions per policy call default to 5 of pi0.5's 10 (openpi's LIBERO protocol).
set -uo pipefail
cd "$(dirname "$0")"
[ $# -ge 3 ] || { echo "usage: $0 <out_name> <variants.json|all> <arm>..."; exit 1; }
NAME=$1; VARIANTS=$2; shift 2; ARMS=("$@")

source env_eval.sh
export REPO_ROOT=/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/cross_emb_2/NewTest/RA-VLA-Reproduce
OPENPI=/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/csil-for-vlas/openpi
PI_PY=$OPENPI/.venv/bin/python
GPU="${GPU:-0}"
NUM_CLIENTS="${NUM_CLIENTS:-10}"
SERVERS_PER_ARM="${SERVERS_PER_ARM:-1}"
EXEC_HORIZON="${EXEC_HORIZON:-5}"
PORT_OFFSET="${PORT_OFFSET:-0}"
HINT_URL="${HINT_URL:-http://127.0.0.1:8890/hint}"
OUT_ROOT="$REPO_ROOT/eval_results/icl_pi05/$NAME"
LOGS="$REPO_ROOT/logs/icl_pi05/$NAME"
mkdir -p "$LOGS"
if [ "$VARIANTS" = all ]; then SELECTION=(--num-tasks-per-axis 0); else SELECTION=(--variants-file "$VARIANTS"); fi

declare -A ARGS=([pre]="--k 0" [concat]="--k 1" [retrieved]="--k 0 --text-mode retrieved"
                 [generated]="--k 0 --text-mode generated --hint-url $HINT_URL")
declare -A PORT=([pre]=$((8971 + PORT_OFFSET)) [concat]=$((8972 + PORT_OFFSET)) [retrieved]=$((8973 + PORT_OFFSET)) [generated]=$((8974 + PORT_OFFSET)))
PIDS=()
cleanup() { for pid in "${PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done; }
trap cleanup EXIT

for arm in "${ARMS[@]}"; do
  mkdir -p "$OUT_ROOT/$arm"
  for j in $(seq 0 $((SERVERS_PER_ARM - 1))); do
    CUDA_VISIBLE_DEVICES="$GPU" XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONNOUSERSITE=1 \
      OPENPI_DATA_HOME=/dev/shm/openpi_cache PYTHONPATH="$REPO_ROOT" \
      $PI_PY scripts/pi05/serve_pi05_icl.py ${ARGS[$arm]} \
      --retrieval-log "$OUT_ROOT/$arm/retrieval.jsonl" --port "$((PORT[$arm] + 10 * j))" \
      > "$LOGS/${arm}_server${j}.log" 2>&1 &
    PIDS+=($!)
  done
done
for arm in "${ARMS[@]}"; do
  for j in $(seq 0 $((SERVERS_PER_ARM - 1))); do
    log="$LOGS/${arm}_server${j}.log"
    for _ in $(seq 1 120); do grep -q "Server is ready" "$log" 2>/dev/null && break; sleep 10; done
    grep -q "Server is ready" "$log" || { echo "$arm server $j failed"; tail -20 "$log"; exit 1; }
  done
done
echo ">>> servers ready: ${ARMS[*]}"

CLIENT_PIDS=()
for arm in "${ARMS[@]}"; do
  for i in $(seq 0 $((NUM_CLIENTS - 1))); do
    PYTHONPATH="$EVAL_PYTHONPATH" LD_PRELOAD="$EVAL_LD_PRELOAD" \
      $EVAL_PY examples/Libero/eval/run_libero_plus_eval.py \
      --task-suite-name libero_10 --port "$((PORT[$arm] + 10 * (i % SERVERS_PER_ARM)))" \
      --shard-index "$i" --num-shards "$NUM_CLIENTS" "${SELECTION[@]}" --num-trials-per-task 1 \
      --exec-horizon "$EXEC_HORIZON" --timeout-ms 900000 --out-dir "$OUT_ROOT/$arm" \
      > "$LOGS/${arm}_client${i}.log" 2>&1 &
    CLIENT_PIDS+=($!); PIDS+=($!)
  done
done
echo ">>> ${#CLIENT_PIDS[@]} clients running; logs in $LOGS"
FAILED=0
for pid in "${CLIENT_PIDS[@]}"; do wait "$pid" || FAILED=1; done
for arm in "${ARMS[@]}"; do
  echo "=== $arm ==="
  $EVAL_PY examples/Libero/eval/score_libero_plus.py --result-dir "$OUT_ROOT/$arm" --suite libero_10 \
    | tee "$LOGS/${arm}_score.log"
done
exit $FAILED
