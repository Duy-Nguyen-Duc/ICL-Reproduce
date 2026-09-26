"""Serve pi0.5 (pi05_libero) behind the GR00T ZMQ/msgpack protocol (openpi venv).

The LIBERO-Plus client (examples/Libero/eval/run_libero_plus_eval.py) sends GR00T-style
observation keys and expects `action.<dim>` chunks back; this server converts both ways so
the same client, variant lists and scoring are used for both backbones.

  images: the client already rotates LIBERO frames 180 degrees, as openpi's LIBERO example
          does; openpi's transforms resize them to 224.
  state:  eef xyz, axis-angle (sent as roll/pitch/yaw), 2 gripper joints -> 8-dim.
  gripper: pi0.5 predicts LIBERO's native command (-1 open, +1 close); the client applies
          y = sign(1 - 2x), so the server sends x = (1 - g) / 2.
"""
import argparse
import hashlib
import io
import json
import os
import pathlib

import msgpack
import numpy as np
import zmq

from pi05_icl.icl import Pi05ICL

ROOT = pathlib.Path(__file__).resolve().parents[2]
DIMS = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]


def decode(obj):
    if "__ndarray_class__" in obj:
        return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
    return obj


def encode(obj):
    if isinstance(obj, np.ndarray):
        buf = io.BytesIO()
        np.save(buf, obj, allow_pickle=False)
        return {"__ndarray_class__": True, "as_npy": buf.getvalue()}
    return obj


def to_openpi(obs):
    state = np.concatenate([np.asarray(obs[f"state.{d}"])[0] for d in DIMS]).astype(np.float32)
    return {
        "observation/image": np.asarray(obs["video.image"])[0],
        "observation/wrist_image": np.asarray(obs["video.wrist_image"])[0],
        "observation/state": state,
        "prompt": str(np.asarray(obs["annotation.human.action.task_description"]).reshape(-1)[0]),
    }


def to_gr00t(actions):
    actions = np.asarray(actions, dtype=np.float32).copy()
    actions[:, 6] = (1.0 - actions[:, 6]) / 2.0
    return {f"action.{d}": actions[:, i:i + 1] for i, d in enumerate(DIMS)}


def file_identity(path):
    if path is None:
        return None
    path = pathlib.Path(path).resolve()
    files = sorted(path.rglob("*")) if path.is_dir() else [path]
    return {"path": str(path), "files": [[str(f.relative_to(path) if path.is_dir() else f.name),
                                          f.stat().st_size] for f in files if f.is_file()]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="/dev/shm/openpi_cache/openpi-assets/checkpoints/pi05_libero")
    p.add_argument("--memory-path", default="/dev/shm/pi05_icl/memory.npz")
    p.add_argument("--frames", default="/dev/shm/pi05_icl/demo_frames.npz")
    p.add_argument("--k", type=int, default=0)
    p.add_argument("--no-features", action="store_true")
    p.add_argument("--no-actions", action="store_true")
    p.add_argument("--text-mode", default="none", choices=["none", "retrieved", "generated"])
    p.add_argument("--text-k", type=int, default=5)
    p.add_argument("--hint-url", default="http://127.0.0.1:8890/hint")
    p.add_argument("--retrieval-log", default="")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--port", type=int, default=8971)
    args = p.parse_args()

    uses_memory = args.k > 0 or args.text_mode != "none"
    frames = dict(np.load(args.frames)) if args.text_mode == "generated" else None
    icl = Pi05ICL(args.checkpoint, memory_path=args.memory_path if uses_memory else None,
                  k=args.k, use_features=not args.no_features, use_actions=not args.no_actions,
                  text_mode=args.text_mode, text_k=args.text_k, hint_url=args.hint_url,
                  demo_frames=frames, retrieval_log=args.retrieval_log or None, seed=args.seed)
    sources = ["pi05_icl/icl.py", "scripts/pi05/serve_pi05_icl.py"]
    run_info = {
        "backbone": "pi05_libero", "checkpoint": file_identity(args.checkpoint),
        "memory": file_identity(args.memory_path) if uses_memory else None,
        "k": args.k, "use_features": not args.no_features, "use_actions": not args.no_actions,
        "text_mode": args.text_mode, "text_k": args.text_k, "seed": args.seed,
        "denoising_steps": 10, "action_horizon": icl.model.action_horizon,
        "source_sha256": hashlib.sha256(b"".join((ROOT / s).read_bytes() for s in sources)).hexdigest(),
    }

    socket = zmq.Context().socket(zmq.REP)
    socket.bind(f"tcp://*:{args.port}")
    print("Server is ready", flush=True)
    while True:
        request = msgpack.unpackb(socket.recv(), object_hook=decode)
        try:
            endpoint = request.get("endpoint", "get_action")
            if endpoint == "ping":
                result = {"status": "ok"}
            elif endpoint == "get_run_info":
                result = run_info
            elif endpoint == "get_action":
                result = to_gr00t(icl.infer(to_openpi(request["data"])))
            else:
                raise ValueError(f"Unknown endpoint: {endpoint}")
        except Exception as exc:  # the client records it and retries the episode
            import traceback
            traceback.print_exc()
            result = {"error": repr(exc)}
        socket.send(msgpack.packb(result, default=encode))


if __name__ == "__main__":
    main()
