# RA-VLA reproduction environment, one LIBERO suite at a time.
#
# Pick the suite and the GPU before sourcing, e.g.
#   SUITE=libero_goal GPU=1 source env.sh
# Everything downstream (dataset, retriever, checkpoints, results) is keyed off SUITE, so
# the four arms never write into each other's directories.
export REPO_ROOT=/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/cross_emb_2/NewTest/RA-VLA-Reproduce
export CONDA_PY=/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/miniconda3/envs/motif/bin/python

export SUITE="${SUITE:-libero_10}"

# The box used to expose GPU3; it now has two A100-80GB, index 0 and 1. Scripts always
# pass --cuda 0 because CUDA_VISIBLE_DEVICES remaps the chosen device to logical 0.
export GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"

# ~/.local site-packages carries huggingface_hub 1.30.0 which breaks transformers 4.51.3
export PYTHONNOUSERSITE=1
# dtaidistance installed side-by-side so the shared `motif` env stays untouched
export PYTHONPATH=$REPO_ROOT/.deps:$REPO_ROOT

# local GR00T-N1.5-3B mirror (no HF download)
# Overridable so a variant can start from a LIBERO-finetuned backbone instead of the
# stock release; unset, it stays the vanilla GR00T-N1.5-3B mirror every run has used.
export PRETRAINED_VLA_PATH="${PRETRAINED_VLA_PATH:-/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/nghiem/groot/GR00T-N1.5-3B}"

export DATASET=$REPO_ROOT/datasets/$SUITE
export RUN_SUFFIX="${RUN_ID:+/$RUN_ID}"
export RETRIEVER=$REPO_ROOT/models${RUN_SUFFIX}/retriever/$SUITE
export RAVLA=$REPO_ROOT/models${RUN_SUFFIX}/ravla/$SUITE
export LOG_DIR=$REPO_ROOT/logs${RUN_SUFFIX}
export GR00T_FRAME_CACHE_DIR="${GR00T_FRAME_CACHE_DIR:-memory}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
