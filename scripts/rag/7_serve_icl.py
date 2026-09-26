"""Serve a plain GR00T-N1.5 checkpoint with training-free in-context adaptation.

--k 0 is the pre-adaptation arm: the identical code path with no context, i.e. the stock
policy. --k > 0 retrieves that many frames from the memory built by 6_build_icl_memory.py
and concatenates them onto the action head's input (gr00t/rag/icl.py).
"""
import hashlib
from dataclasses import dataclass
from pathlib import Path

import torch
import tyro

from gr00t.data.schema import EmbodimentTag
from gr00t.eval.robot import RobotInferenceServer
from gr00t.experiment.data_config import load_data_config
from gr00t.rag.icl import ICLPolicy
from gr00t.rag.run_metadata import artifact_identity
from gr00t.rag.utils import seed_everything

DATA_CONFIG = "examples.Libero.custom_data_config:LiberoDataConfig"
EMBODIMENT_TAG = "new_embodiment"


@dataclass
class ArgsConfig:
    seed: int = 0
    cuda: int = 0
    checkpoint_path: str = "models/public_finetunes/libero_long_posttrain"
    memory_path: str = "models/icl/libero_10/memory.pt"
    k: int = 1
    """Retrieved in-context frames; 0 serves the unadapted policy."""
    use_features: bool = True
    """Concatenate retrieved VLM tokens onto the cross-attention sequence."""
    use_actions: bool = True
    """Prepend retrieved generated action chunks to the DiT token sequence."""
    text_mode: str = "none"
    """"none" | "retrieved" | "generated": append context-derived text to the instruction."""
    text_k: int = 5
    """Frames voted over to pick the retrieved task for text_mode."""
    hint_url: str = "http://127.0.0.1:8890/hint"
    dataset_path: str = "datasets/libero_10"
    """Demonstrations the memory was built from; text_mode=generated shows their frames."""
    denoising_steps: int = 4
    retrieval_log: str = ""
    host: str = "0.0.0.0"
    port: int = 8871


def run_identity(config):
    root = Path(__file__).resolve().parents[2]
    source_files = ["gr00t/rag/icl.py", "gr00t/model/policy.py",
                    "gr00t/model/action_head/flow_matching_action_head.py",
                    "gr00t/model/action_head/cross_attention_dit.py",
                    "examples/Libero/custom_data_config.py", "scripts/rag/7_serve_icl.py"]
    source_hash = hashlib.sha256(
        b"".join((root / name).read_bytes() for name in source_files)).hexdigest()
    return {
        "checkpoint": artifact_identity(config.checkpoint_path),
        "memory": (artifact_identity(config.memory_path)
                   if config.k > 0 or config.text_mode != "none" else None),
        "text_mode": config.text_mode, "text_k": config.text_k,
        "retrieval_scope": "icl_visual_key", "k": config.k,
        "use_features": config.use_features, "use_actions": config.use_actions,
        "denoising_steps": config.denoising_steps, "seed": config.seed,
        "source_sha256": source_hash,
    }


def main(config: ArgsConfig):
    seed_everything(config.seed)
    data_config = load_data_config(DATA_CONFIG)
    demo_dataset = None
    if config.text_mode == "generated":
        from gr00t.data.dataset import LeRobotSingleDataset
        demo_dataset = LeRobotSingleDataset(
            dataset_path=config.dataset_path, modality_configs=data_config.modality_config(),
            embodiment_tag=EmbodimentTag(EMBODIMENT_TAG), video_backend="torchvision_av")
    policy = ICLPolicy(
        model_path=config.checkpoint_path,
        embodiment_tag=EmbodimentTag(EMBODIMENT_TAG),
        modality_config=data_config.modality_config(),
        modality_transform=data_config.transform(),
        denoising_steps=config.denoising_steps,
        device=f"cuda:{config.cuda}" if torch.cuda.is_available() else "cpu",
        memory_path=config.memory_path, k=config.k,
        use_features=config.use_features, use_actions=config.use_actions,
        retrieval_log=config.retrieval_log or None,
        text_mode=config.text_mode, text_k=config.text_k, hint_url=config.hint_url,
        demo_dataset=demo_dataset,
    )
    if policy.memory is not None:
        print(f"ICL memory: {len(policy.memory)} frames, k={config.k}, "
              f"text_mode={config.text_mode}", flush=True)

    server = RobotInferenceServer(policy, config.host, config.port)
    run_info = run_identity(config)
    server.register_endpoint("get_run_info", lambda: run_info, requires_input=False)
    print("Server is ready", flush=True)
    server.run()


if __name__ == "__main__":
    main(tyro.cli(ArgsConfig))
