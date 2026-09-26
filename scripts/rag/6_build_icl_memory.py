"""Build the in-context memory for gr00t/rag/icl.py from a few clean demonstrations.

For each task, `num_demos_per_task` demonstrations are drawn (seeded) and every
`frame_stride`-th frame is passed through the plain policy. Stored per frame: visual key,
raw VLM token sequence, the action chunk the policy's head generates, and, for diagnostics
only, the ground-truth chunk (raw action units; the eval-mode transform drops actions).
"""
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro
from tqdm import tqdm

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.schema import EmbodimentTag
from gr00t.experiment.data_config import load_data_config
from gr00t.model.policy import COMPUTE_DTYPE, unsqueeze_dict_values
from gr00t.rag.icl import ICLPolicy
from gr00t.rag.utils import seed_everything

DATA_CONFIG = "examples.Libero.custom_data_config:LiberoDataConfig"
EMBODIMENT_TAG = "new_embodiment"
ACTION_KEYS = [f"action.{k}" for k in ("x", "y", "z", "roll", "pitch", "yaw", "gripper")]


@dataclass
class ArgsConfig:
    checkpoint_path: str = "models/public_finetunes/libero_long_posttrain"
    dataset_path: str = "datasets/libero_10"
    out_path: str = "models/icl/libero_10/memory.pt"
    num_demos_per_task: int = 3
    frame_stride: int = 4
    denoising_steps: int = 4
    seed: int = 0
    cuda: int = 0


def main(config: ArgsConfig):
    seed_everything(config.seed)
    data_config = load_data_config(DATA_CONFIG)
    policy = ICLPolicy(
        model_path=config.checkpoint_path,
        embodiment_tag=EmbodimentTag(EMBODIMENT_TAG),
        modality_config=data_config.modality_config(),
        modality_transform=data_config.transform(),
        denoising_steps=config.denoising_steps,
        device=f"cuda:{config.cuda}",
    )
    dataset = LeRobotSingleDataset(
        dataset_path=config.dataset_path,
        modality_configs=data_config.modality_config(),
        embodiment_tag=EmbodimentTag(EMBODIMENT_TAG),
        video_backend="torchvision_av",
    )
    head = policy.model.action_head

    rng = random.Random(config.seed)
    keys, actions, generated_raw, gt_actions, features = [], [], [], [], []
    tasks, traj_ids, frame_ids = [], [], []
    for task, task_trajs in sorted(dataset.task_groups.items()):
        for traj_id in sorted(rng.sample(task_trajs, config.num_demos_per_task)):
            assert dataset.trajectory_ids[traj_id] == traj_id, "Trajectory ID != Trajectory Index"
            length = int(dataset.trajectory_lengths[traj_id])
            for frame_id in tqdm(range(0, length, config.frame_stride), desc=f"traj {traj_id}"):
                step = dataset.get_step_data(traj_id, frame_id)
                obs = {k: v for k, v in step.items() if not k.startswith("action.")}
                normalized = policy.apply_transforms(unsqueeze_dict_values(obs))
                with torch.inference_mode(), torch.autocast("cuda", dtype=COMPUTE_DTYPE):
                    _, backbone_out, action_in, key = policy.encode(normalized)
                    feats = backbone_out.backbone_features[0]
                    mask = backbone_out.backbone_attention_mask[0].bool()
                    raw = feats[mask].to(torch.bfloat16).clone()
                    generated = head.get_action(backbone_out, action_in)["action_pred"][0]

                unnormalized = policy.unapply_transforms({"action": generated[None].float().cpu()})
                keys.append(key[0].float().cpu())
                features.append(raw.cpu())
                actions.append(generated.float().cpu())
                generated_raw.append(np.concatenate(
                    [np.asarray(unnormalized[k])[0] for k in ACTION_KEYS], axis=-1))
                gt_actions.append(np.concatenate([step[k] for k in ACTION_KEYS], axis=-1))
                tasks.append(task)
                traj_ids.append(int(traj_id))
                frame_ids.append(frame_id)

    actions = torch.stack(actions)
    generated_raw = torch.as_tensor(np.stack(generated_raw), dtype=torch.float32)
    gt_actions = torch.as_tensor(np.stack(gt_actions), dtype=torch.float32)
    mse = ((generated_raw - gt_actions) ** 2).mean().item()
    print(f"{len(tasks)} frames from {len(set(traj_ids))} demos over {len(set(tasks))} tasks; "
          f"mean VLM tokens {np.mean([f.shape[0] for f in features]):.0f}; "
          f"generated-vs-GT raw-action MSE {mse:.5f}")

    out = Path(config.out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "keys": torch.stack(keys), "actions": actions,
        "generated_raw": generated_raw, "gt_actions": gt_actions,
        "features": features, "tasks": tasks, "traj_ids": traj_ids, "frame_ids": frame_ids,
        "config": vars(config),
    }, out)
    print(f"saved {out}")


if __name__ == "__main__":
    main(tyro.cli(ArgsConfig))
