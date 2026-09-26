"""Offline sanity check for the pi0.5 in-context port, mirroring the GR00T one.

For a sample of clean demo frames, retrieval is restricted to other demonstrations (the
query's own trajectory is masked out), then:
  - top-1 retrieved-task accuracy,
  - action error vs the demonstrator for the stock policy (k=0) and with k=1 context
    concatenated (prefix KV + generated action chunk).
"""
import argparse

import jax
import jax.numpy as jnp
import numpy as np

from pi05_icl.icl import Pi05ICL

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default="/dev/shm/openpi_cache/openpi-assets/checkpoints/pi05_libero")
parser.add_argument("--memory", default="/dev/shm/pi05_icl/memory.npz")
parser.add_argument("--frames", default="/dev/shm/pi05_icl/demo_frames.npz")
parser.add_argument("--num", type=int, default=140)
args = parser.parse_args()

f = np.load(args.frames)
icl = Pi05ICL(args.checkpoint, memory_path=args.memory, k=1)
mem = icl.memory
traj = np.asarray(mem.traj_ids)
picks = np.random.default_rng(0).choice(len(traj), args.num, replace=False)
hits, err = 0, {0: [], 1: []}
for n, i in enumerate(picks):
    obs = {"observation/image": f["image"][i], "observation/wrist_image": f["wrist_image"][i],
           "observation/state": f["state"][i], "prompt": str(f["task"][i])}
    gt = f["actions"][i][:10]
    inputs, observation = icl.prepare(obs)
    _, _, key = icl._encode(icl.model, observation)
    scores = jnp.where(jnp.asarray(traj == traj[i]), -jnp.inf, key[0] @ mem.keys.T)
    j = int(jnp.argmax(scores))
    hits += mem.tasks[j] == str(f["task"][i])
    for k in (0, 1):
        rng = jax.random.key(n)
        if k == 0:
            actions = icl.policy._sample_actions(rng, observation, **icl.policy._sample_kwargs)
        else:
            actions = icl._sample_icl(icl.model, rng, observation, mem.tokens[j][None],
                                      mem.mask[j][None], mem.actions[j][None])
        raw = np.array(icl.unnormalize(inputs, actions), dtype=np.float32)
        raw[:, 6] = (1.0 - raw[:, 6]) / 2.0
        err[k].append(((raw - gt) ** 2).mean())
print(f"frames {len(picks)}: top-1 retrieved-task accuracy {hits / len(picks):.2%}")
print(f"action MSE vs GT: k=0 {np.mean(err[0]):.5f}  k=1 {np.mean(err[1]):.5f}")
