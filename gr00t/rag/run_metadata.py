"""Small, deterministic identities for immutable evaluation runs."""
import hashlib
import json
from pathlib import Path


def artifact_identity(path):
    path = Path(path).resolve()
    files = sorted(path.glob("*")) if path.is_dir() else [path]
    entries = []
    for file in files:
        if file.is_file():
            stat = file.stat()
            entries.append([file.name, stat.st_size, stat.st_mtime_ns])
    return {"path": str(path), "files": entries}


def policy_identity(config):
    root = Path(__file__).resolve().parents[2]
    source_files = ["gr00t/rag/ravla.py", "gr00t/rag/retriever.py",
                    "gr00t/model/action_head/cross_attention_dit.py",
                    "gr00t/model/action_head/flow_matching_action_head.py",
                    "examples/Libero/custom_data_config.py", "scripts/rag/4_serve_ravla.py"]
    source_hash = hashlib.sha256(b"".join((root / name).read_bytes() for name in source_files)).hexdigest()
    return {
        "checkpoint": artifact_identity(config.checkpoint_path),
        "retriever": artifact_identity(config.retriever_path),
        "memory": artifact_identity(Path(config.retriever_path) / f"{config.memory_name}.pt"),
        "retrieval_scope": config.retrieval_scope, "k": config.k,
        "denoising_steps": config.denoising_steps, "seed": config.seed,
        "source_sha256": source_hash,
    }
