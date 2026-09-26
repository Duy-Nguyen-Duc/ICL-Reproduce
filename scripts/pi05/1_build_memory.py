"""Build the pi0.5 in-context memory from the exported demo frames (openpi venv).

Same 30 demonstrations / 2,039 frames as the GR00T memory. Per frame: visual key, prefix
embedding sequence, and the action chunk pi0.5 generates. Prints the generated-vs-GT
action error as a check that images, state and the action convention line up.
"""
import argparse
import os

import numpy as np
from tqdm import tqdm

from pi05_icl.icl import Pi05ICL, drop_masked_slot

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default="/dev/shm/openpi_cache/openpi-assets/checkpoints/pi05_libero")
parser.add_argument("--frames", default="/dev/shm/pi05_icl/demo_frames.npz")
parser.add_argument("--out", default="/dev/shm/pi05_icl/memory.npz")
parser.add_argument("--limit", type=int, default=0)
args = parser.parse_args()

frames = np.load(args.frames)
n = args.limit or len(frames["task"])
icl = Pi05ICL(args.checkpoint)
keys, tokens, masks, actions, errors = [], [], [], [], []
for i in tqdm(range(n)):
    obs = {"observation/image": frames["image"][i], "observation/wrist_image": frames["wrist_image"][i],
           "observation/state": frames["state"][i], "prompt": str(frames["task"][i])}
    inputs, observation = icl.prepare(obs)
    tok, mask, key = icl._encode(icl.model, observation)
    tok, mask = drop_masked_slot(np.asarray(tok), np.asarray(mask))
    generated = icl.generate(obs)
    raw = np.array(icl.unnormalize(inputs, generated), dtype=np.float32)
    raw[:, 6] = (1.0 - raw[:, 6]) / 2.0          # pi0.5 -1 open/+1 close -> dataset 1 open/0 close
    H = raw.shape[0]
    errors.append(((raw - frames["actions"][i][:H]) ** 2).mean(0))
    keys.append(np.asarray(key[0]))
    tokens.append(tok[0].view(np.uint16))      # bfloat16 bits; numpy has no bf16 dtype
    masks.append(mask[0])
    actions.append(np.asarray(generated[0], dtype=np.float32))

err = np.stack(errors)
print(f"{n} frames; generated-vs-GT raw-action MSE {err.mean():.5f} "
      f"(per dim {np.round(err.mean(0), 5).tolist()})")
os.makedirs(os.path.dirname(args.out), exist_ok=True)
np.savez(args.out, keys=np.stack(keys), tokens=np.stack(tokens), mask=np.stack(masks),
         actions=np.stack(actions), task=frames["task"][:n], traj_id=frames["traj_id"][:n],
         frame_id=frames["frame_id"][:n])
print(f"saved {args.out}")
