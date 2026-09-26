# LIBERO-Plus evaluation environment (client side).
#
# The sim runs in miniforge3/envs/vlash-libero (robosuite 1.4.0, mujoco, bddl) against
# triquang's LIBERO-plus checkout, which is used read-only -- disk quota here has no room
# for a second 6.4 GB copy. Two things that checkout needs and no env on this box has,
# `wand` and `scikit-image`, live in .deps_eval, and ImageMagick (which `wand` binds to,
# and which is not installed system-wide) in .imagemagick.
export REPO_ROOT=/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/cross_emb_2/NewTest/RA-VLA-Reproduce
export EVAL_PY=/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/miniforge3/envs/vlash-libero/bin/python
export LIBERO_PLUS=/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/triquang/robot-ood/external/LIBERO-plus

export LIBERO_CONFIG_PATH=$LIBERO_PLUS/libero
export LIBERO_PLUS_TASK_CLASSIFICATION=$LIBERO_PLUS/libero/libero/benchmark/task_classification.json

export MAGICK_HOME=$REPO_ROOT/.imagemagick
export LD_LIBRARY_PATH=$REPO_ROOT/.imagemagick/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

export PYTHONNOUSERSITE=1
# Kept out of PYTHONPATH so it cannot clobber the server's: the runner applies it to the
# client process only. .deps_eval first, then the LIBERO-plus fork ahead of any `libero`
# already installed in the env.
export EVAL_PYTHONPATH=$REPO_ROOT/.deps_eval:$LIBERO_PLUS:$REPO_ROOT

# Results are keyed by suite so the four arms stay separate.
export SUITE="${SUITE:-libero_10}"
export EVAL_OUT="${EVAL_OUT:-$REPO_ROOT/eval_results${RUN_ID:+/$RUN_ID}/$SUITE/libero_plus}"
export EVAL_OUT_CLEAN="${EVAL_OUT_CLEAN:-$REPO_ROOT/eval_results${RUN_ID:+/$RUN_ID}/$SUITE/clean}"

# Clean (unperturbed) LIBERO for the in-distribution baseline: the stock package that the
# vlash-libero env has installed editable, with its bddl/init files from ~/.libero. The
# LIBERO-plus checkout is kept off this path entirely -- there its suites hold the ~2500
# perturbed variants under the same suite names, not the 10 originals.
export CLEAN_PYTHONPATH=$REPO_ROOT/.deps_eval:$REPO_ROOT
export CLEAN_LIBERO_CONFIG_PATH=$HOME/.libero

# Software rendering. EGL exposes exactly one device in this container no matter how
# CUDA_VISIBLE_DEVICES is set, so the renderer cannot be pinned to GPU3 -- and measured
# EGL throughput here matches osmesa (~6 steps/s, no GPU memory bound), so nothing is
# lost by rendering on CPU and leaving GPU3 to the policy server alone.
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export LD_LIBRARY_PATH=$(dirname $EVAL_PY)/../lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

# The vlash-libero env's own libOSMesa.so.8 (osmesa 12.2.2) links against libgcrypt.so.11,
# which this host no longer has, so every rollout died at the first render. The env's
# python has a DT_RPATH to its own lib, which outranks LD_LIBRARY_PATH, so a newer OSMesa
# cannot simply be put in front of it -- it is preloaded instead, and PyOpenGL's dlopen of
# "libOSMesa.so.8" then matches the already-loaded SONAME. EGL is not an alternative here
# any more either: this container exposes no EGL device (no /dev/dri, no egl_vendor.d).
# Applied to the client process only, like EVAL_PYTHONPATH.
# Only the preload: that library carries an $ORIGIN rpath, so its own dependencies
# resolve from its env without putting a foreign lib directory on LD_LIBRARY_PATH -- which
# would shadow what ImageMagick links against and break `wand`.
export MESA_LIB=/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/miniconda3/envs/lpb/lib
export EVAL_LD_PRELOAD=$MESA_LIB/libOSMesa.so.8.0.0
