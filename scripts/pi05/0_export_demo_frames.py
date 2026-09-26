"""Export the demo frames behind the GR00T ICL memory for the pi0.5 port.

Reads the (traj, frame) list stored in the GR00T memory so both backbones build their
memory from exactly the same 30 demonstrations / 2,039 frames, and writes the raw
observation (both camera views, 8-dim state, instruction) plus the demonstrator's action
chunk to one .npz that the openpi environment can read without the gr00t package.
"""
import argparse

import numpy as np
import torch
from tqdm import tqdm

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.schema import EmbodimentTag
from gr00t.experiment.data_config import load_data_config

STATE_KEYS = [f"state.{k}" for k in ("x", "y", "z", "roll", "pitch", "yaw", "gripper")]
ACTION_KEYS = [f"action.{k}" for k in ("x", "y", "z", "roll", "pitch", "yaw", "gripper")]
LANGUAGE_KEY = "annotation.human.action.task_description"

parser = argparse.ArgumentParser()
parser.add_argument("--gr00t-memory", default="/tmp/icl_memory/memory.pt")
parser.add_argument("--dataset-path", default="datasets/libero_10")
parser.add_argument("--out", default="/dev/shm/pi05_icl/demo_frames.npz")
args = parser.parse_args()

memory = torch.load(args.gr00t_memory, map_location="cpu", weights_only=False)
data_config = load_data_config("examples.Libero.custom_data_config:LiberoDataConfig")
dataset = LeRobotSingleDataset(args.dataset_path, data_config.modality_config(),
                               EmbodimentTag("new_embodiment"), video_backend="torchvision_av")
out = {k: [] for k in ("image", "wrist_image", "state", "actions")}
for traj, frame in tqdm(list(zip(memory["traj_ids"], memory["frame_ids"]))):
    step = dataset.get_step_data(traj, frame)
    out["image"].append(step["video.image"][0])
    out["wrist_image"].append(step["video.wrist_image"][0])
    out["state"].append(np.concatenate([step[k][0] for k in STATE_KEYS]).astype(np.float32))
    out["actions"].append(np.concatenate([step[k] for k in ACTION_KEYS], -1).astype(np.float32))
    assert step[LANGUAGE_KEY][0] == memory["tasks"][len(out["image"]) - 1]

import os
os.makedirs(os.path.dirname(args.out), exist_ok=True)
np.savez(args.out, **{k: np.stack(v) for k, v in out.items()},
         task=np.array(memory["tasks"]), traj_id=np.array(memory["traj_ids"]),
         frame_id=np.array(memory["frame_ids"]))
print({k: np.stack(v).shape for k, v in out.items()}, "->", args.out)
