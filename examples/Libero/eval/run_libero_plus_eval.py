"""LIBERO-Plus evaluation client for a served RA-VLA policy.

LIBERO-Plus holds perturbed variants of the four LIBERO suites, grouped into seven
perturbation axes; the axis of each variant lives in the checkout's
benchmark/task_classification.json. This script builds the upstream suite, cuts its task
list down to one axis, and rolls the variants out against a policy server started by
`scripts/rag/4_serve_ravla.py`.

The observation mapping and action conversion are the same as `run_libero_eval.py`; the
server protocol is re-implemented here (msgpack over ZeroMQ) so the simulator environment
does not need torch or the gr00t package installed.
"""

import io
import fcntl
import hashlib
import json
import os
import pathlib
import random
import time
from dataclasses import dataclass, field

import msgpack
import numpy as np
import tqdm
import tyro
import zmq

from examples.Libero.eval.utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    normalize_gripper_action,
    quat2axisangle,
)

def _allow_full_unpickling():
    """LIBERO's init-state files predate torch 2.6's weights_only=True default.

    They hold pickled numpy arrays, so loading them raises UnpicklingError on torch >= 2.6.
    The checkout is used read-only and shared with other users, so the default is relaxed
    here, in this process only, rather than edited over there.
    """
    import torch

    original_load = torch.load

    def load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = load


_allow_full_unpickling()

AXES = [
    "Background Textures",
    "Camera Viewpoints",
    "Language Instructions",
    "Light Conditions",
    "Objects Layout",
    "Robot Initial States",
    "Sensor Noise",
]

# Step budgets follow run_libero_eval.py.
MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 600,
    "libero_10": 1000,
    "libero_90": 400,
}


@dataclass
class Args:
    """Which variants to run, and where the policy server is."""

    task_suite_name: str = "libero_10"
    """Suite to evaluate; it should match the suite the policy was trained on."""
    clean: bool = False
    """Evaluate the original, unperturbed suite instead of the LIBERO-Plus variants.

    Requires a stock LIBERO on the path rather than the LIBERO-plus checkout, whose suites
    carry the ~2500 variants under the same names -- `run_eval_libero_clean.sh` sets that
    up. Episodes are recorded with axis "clean" so the same scorer reads them."""
    axes: list[str] = field(default_factory=lambda: list(AXES))
    """Perturbation axes to evaluate. Names carry spaces, as in task_classification.json."""
    num_tasks_per_axis: int = 20
    variants_file: str = ""
    """Path to a JSON list of {"axis", "task"} pairs to run instead of sampling per axis.

    Overrides --axes and --num-tasks-per-axis; used for small fixed-sample comparisons
    where every arm must see exactly the same variants."""
    """Variants sampled per axis. 0 runs every variant of the axis (hours per axis)."""
    num_trials_per_task: int = 1
    num_steps_wait: int = 10
    """Steps of no-op while the simulator settles the objects."""
    seed: int = 0
    """Seeds both the variant sample and the environment."""
    resolution: int = 256
    max_steps: int | None = None
    """Overrides the suite's step budget."""
    exec_horizon: int = 1
    """Actions executed per policy call, out of the 16 the action head predicts.

    1 reproduces run_libero_eval.py, which re-queries every step and throws the rest of
    the chunk away. RA-VLA costs ~1s per call here (the retriever runs the backbone on top
    of the policy's own pass), so 1 puts a 1000-step episode at ~20 minutes; 8 cuts that
    to ~3 while staying inside the predicted chunk. Changing it changes the protocol, so
    only compare runs that used the same value."""

    host: str = "localhost"
    port: int = 8071
    timeout_ms: int = 60000

    shard_index: int = 0
    """This worker's index. Variants are dealt round-robin so every shard spans all axes."""
    num_shards: int = 1
    resume: bool = True
    also_done_dirs: list[str] = field(default_factory=list)
    """Further result directories whose recorded episodes also count as done.

    For finishing a run in a fresh --out-dir (e.g. after an artifact moved, which changes
    the manifest) without re-rolling what the original directory already recorded."""
    """Skip episodes already recorded in this shard's jsonl, so a killed run can continue."""

    base_instructions: str = ""
    """Path to a JSON list of the suite's ORIGINAL training instructions.

    LIBERO-plus derives task.language from the variant filename, so a perturbed task is
    prompted with its tag appended ("...place it on the plate view 0 0 100 2 352 in").
    Upstream evaluates that way (libero/lifelong/main.py:87), so it is the benchmark's
    protocol, not a bug -- set this only to measure what the tag costs, and never compare
    the result against a published LIBERO-plus number. Each variant is matched to the
    longest original instruction whose underscored form prefixes the variant's name;
    unmatched variants (the Language Instructions axis rewrites the whole sentence) keep
    task.language."""

    out_dir: str = "eval_results/libero_plus"
    save_videos: bool = False
    episode_retries: int = 2
    """Off by default; rollout videos are large and disk quota here is tight."""


# --- policy client (mirrors gr00t.eval.service, without the gr00t import) -------------


def _encode(obj):
    if isinstance(obj, np.ndarray):
        buf = io.BytesIO()
        np.save(buf, obj, allow_pickle=False)
        return {"__ndarray_class__": True, "as_npy": buf.getvalue()}
    return obj


def _decode(obj):
    if "__ndarray_class__" in obj:
        obj = np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
    return obj


class PolicyClient:
    def __init__(self, host: str, port: int, timeout_ms: int):
        self.context = zmq.Context()
        self.address = f"tcp://{host}:{port}"
        self.timeout_ms = timeout_ms
        self.socket = None
        self.reconnect()

    def reconnect(self):
        if self.socket is not None:
            self.socket.close(linger=0)
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.socket.connect(self.address)

    def call(self, endpoint: str, data: dict | None = None) -> dict:
        request: dict = {"endpoint": endpoint}
        if data is not None:
            request["data"] = data
        try:
            self.socket.send(msgpack.packb(request, default=_encode))
            response = msgpack.unpackb(self.socket.recv(), object_hook=_decode)
        except zmq.error.ZMQError:
            self.reconnect()
            raise
        if "error" in response:
            raise RuntimeError(f"Server error: {response['error']}")
        return response

    def ping(self) -> bool:
        try:
            self.call("ping")
            return True
        except zmq.error.ZMQError:
            return False


ACTION_KEYS = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]


def build_observation(obs, lang: str) -> dict:
    """LIBERO observation -> GR00T modality dict (identical to run_libero_eval.py)."""
    xyz = obs["robot0_eef_pos"]
    rpy = quat2axisangle(obs["robot0_eef_quat"])
    gripper = obs["robot0_gripper_qpos"]
    img, wrist_img = get_libero_image(obs)
    return {
        "video.image": np.expand_dims(img, axis=0),
        "video.wrist_image": np.expand_dims(wrist_img, axis=0),
        "state.x": np.array([[xyz[0]]]),
        "state.y": np.array([[xyz[1]]]),
        "state.z": np.array([[xyz[2]]]),
        "state.roll": np.array([[rpy[0]]]),
        "state.pitch": np.array([[rpy[1]]]),
        "state.yaw": np.array([[rpy[2]]]),
        "state.gripper": np.expand_dims(gripper, axis=0),
        "annotation.human.action.task_description": [lang],
    }


def to_libero_action(action_chunk: dict, idx: int = 0) -> np.ndarray:
    components = [np.atleast_1d(action_chunk[f"action.{key}"][idx])[0] for key in ACTION_KEYS]
    action = np.array(components, dtype=np.float32)
    action = normalize_gripper_action(action, binarize=True)
    assert len(action) == 7, f"Expected 7-dim action, got {len(action)}"
    return action


# --- axis filtering over an unmodified LIBERO-plus checkout ---------------------------


def load_classification() -> dict:
    path = os.environ.get("LIBERO_PLUS_TASK_CLASSIFICATION")
    if not path:
        from libero.libero import benchmark as _b

        path = os.path.join(
            os.path.dirname(os.path.abspath(_b.__file__)), "task_classification.json"
        )
    with open(path) as fh:
        return json.load(fh)


def suite_for_axis(task_suite_name: str, axis: str, num_tasks: int, seed: int,
                   shard_index: int = 0, num_shards: int = 1, names=None):
    """The upstream suite with its task list cut to one axis, then optionally sampled."""
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()[task_suite_name]()
    if axis == "clean":
        # Stock LIBERO: the suite already is the 10 original tasks, and sharding happens
        # per episode in main() instead, so 10 tasks spread over more than 10 shards.
        if num_tasks and num_tasks < len(suite.tasks):
            suite.tasks = sorted(suite.tasks, key=lambda t: t.name)[:num_tasks]
            suite.n_tasks = len(suite.tasks)
        return suite
    entries = load_classification().get(task_suite_name, [])
    allowed = {e["name"] for e in entries if e.get("category") == axis}
    tasks = [t for t in suite.tasks if t.name in allowed]
    if names is not None:
        # A fixed list is dealt across shards in main(), over the whole list, not per axis.
        suite.tasks = sorted((t for t in tasks if t.name in names), key=lambda t: t.name)
        suite.n_tasks = len(suite.tasks)
        return suite
    if num_tasks and num_tasks < len(tasks):
        # Sampled per axis off a fixed seed so reruns and axes stay comparable.
        tasks = random.Random(f"{seed}:{task_suite_name}:{axis}").sample(tasks, num_tasks)

    tasks.sort(key=lambda t: t.name)  # deal from a stable order so shards never overlap
    if num_shards > 1:
        tasks = tasks[shard_index::num_shards]

    suite.tasks = tasks
    suite.n_tasks = len(tasks)
    return suite


# --- resume bookkeeping --------------------------------------------------------------


def load_done_episodes(path: pathlib.Path) -> set:
    """(axis, task, episode) triples already recorded, so a restart can skip them.

    Every shard file in the directory is read, not just this shard's: a run restarted with
    a different shard count deals the variants differently, and reading only one's own file
    would re-roll episodes another shard had already finished.
    """
    done = set()
    for path in sorted(path.parent.glob("episodes*.jsonl")):
        if not path.exists():
            continue
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a partial line from a killed run
                done.add((rec["axis"], rec["task"], rec["episode"]))
    return done


def run_episode(env, client, task_description, initial_state, args, max_steps, frames):
    env.reset()
    obs = env.set_init_state(initial_state)
    done = False
    t = 0
    while t < max_steps + args.num_steps_wait:
        if t < args.num_steps_wait:
            obs, _, done, _ = env.step(get_libero_dummy_action())
            t += 1
            if done:
                break
            continue

        action_chunk = client.call("get_action", build_observation(obs, task_description))
        for idx in range(args.exec_horizon):
            action = to_libero_action(action_chunk, idx)
            if not np.isfinite(action).all():
                raise ValueError("Policy returned a non-finite action")
            if frames is not None:
                frames.append(get_libero_image(obs)[0])
            obs, _, done, _ = env.step(action.tolist())
            t += 1
            if done or t >= max_steps + args.num_steps_wait:
                break
        if done:
            break
    return done, t



def base_instruction_map(path):
    """(underscored original name, original instruction) pairs, longest name first."""
    with open(path) as fh:
        originals = json.load(fh)
    pairs = [(i.replace(" ", "_"), i) for i in originals]
    return sorted(pairs, key=lambda pair: -len(pair[0]))


def to_base_instruction(pairs, task, fallback):
    for name, instruction in pairs:
        if task.name.startswith(name):
            return instruction
    return fallback


def ensure_run_manifest(out_dir, args, policy_info, max_steps):
    config = {key: getattr(args, key) for key in (
        "task_suite_name", "clean", "num_tasks_per_axis", "num_trials_per_task",
        "num_steps_wait", "seed", "resolution", "exec_horizon", "base_instructions",
        "variants_file")}
    if args.variants_file:
        config["variants"] = json.loads(pathlib.Path(args.variants_file).read_text())
    config.update(axes=sorted(["clean"] if args.clean else args.axes), max_steps=max_steps,
                  policy=policy_info,
                  client_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
                  utils_sha256=hashlib.sha256(pathlib.Path(__file__).with_name("utils.py").read_bytes()).hexdigest(),
                  libero_config_path=os.environ.get("LIBERO_CONFIG_PATH"))
    run_id = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    manifest = {"run_id": run_id, "config": config}
    with open(out_dir / ".manifest.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = out_dir / "run_manifest.json"
        if path.exists():
            if json.loads(path.read_text()) != manifest:
                raise ValueError("Incompatible evaluation configuration; use a fresh --out-dir")
        else:
            if any(out_dir.glob("episodes*.jsonl")):
                raise ValueError("Legacy episode records lack provenance; use a fresh --out-dir")
            path.write_text(json.dumps(manifest, indent=2) + "\n")
    return run_id


def main(args: Args):
    if not 1 <= args.exec_horizon <= 16:
        raise ValueError("exec_horizon must be between 1 and 16")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Invalid shard index/count")
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # One file per shard: parallel workers never write to the same file.
    episodes_path = out_dir / (
        "episodes.jsonl" if args.num_shards == 1
        else f"episodes_shard{args.shard_index:02d}.jsonl"
    )
    client = PolicyClient(args.host, args.port, args.timeout_ms)
    if not client.ping():
        raise RuntimeError(f"No policy server answering on {args.host}:{args.port}")
    print(f"Connected to policy server at {args.host}:{args.port}")

    max_steps = args.max_steps or MAX_STEPS[args.task_suite_name]
    run_id = ensure_run_manifest(out_dir, args, client.call("get_run_info"), max_steps)
    if not args.resume and any(out_dir.glob("episodes*.jsonl")):
        raise ValueError("--no-resume requires a fresh output directory")
    already_done = load_done_episodes(episodes_path) if args.resume else set()
    for other in args.also_done_dirs:
        already_done |= load_done_episodes(pathlib.Path(other) / "episodes.jsonl")
    print(f"Run {run_id[:12]}: {len(already_done)} completed episodes")
    per_axis = {}

    base_pairs = base_instruction_map(args.base_instructions) if args.base_instructions else None
    variants = None
    if args.variants_file:
        variants = {}
        for entry in json.loads(pathlib.Path(args.variants_file).read_text()):
            variants.setdefault(entry["axis"], set()).add(entry["task"])
        args.axes = sorted(variants)
        flat = sorted((a, t) for a, names in variants.items() for t in names)
    axes = ["clean"] if args.clean else args.axes
    for axis in axes:
        suite = suite_for_axis(args.task_suite_name, axis, args.num_tasks_per_axis, args.seed,
                               args.shard_index, args.num_shards,
                               names=variants[axis] if variants else None)
        print(f"\n=== {axis}: {suite.n_tasks} variants x {args.num_trials_per_task} trials ===")

        episodes, successes = 0, 0
        for task_id in tqdm.tqdm(range(suite.n_tasks), desc=axis):
            task = suite.get_task(task_id)
            if variants and flat.index((axis, task.name)) % args.num_shards != args.shard_index:
                continue
            initial_states = suite.get_task_init_states(task_id)

            trials = min(args.num_trials_per_task, len(initial_states))
            pending = [i for i in range(trials)
                       if (axis, task.name, i) not in already_done]
            if args.clean and args.num_shards > 1:
                # Only 10 tasks here, so shards are cut from the flat (task, episode) list.
                pending = [i for i in pending
                           if (task_id * trials + i) % args.num_shards == args.shard_index]
            if not pending:
                continue  # nothing left for this variant; skip the ~2s env build too

            env, task_description = get_libero_env(
                task, resolution=args.resolution,
                horizon=max_steps + args.num_steps_wait + 1,
            )
            if base_pairs is not None:
                task_description = to_base_instruction(base_pairs, task, task_description)
            env.seed(args.seed)

            try:
                for episode_idx in pending:
                    frames = [] if args.save_videos else None
                    started = time.monotonic()
                    for attempt in range(args.episode_retries + 1):
                        try:
                            if frames is not None:
                                frames.clear()
                            done, steps = run_episode(
                                env, client, task_description,
                                initial_states[episode_idx], args, max_steps, frames,
                            )
                            break
                        except Exception as exc:
                            with open(out_dir / f"errors_shard{args.shard_index:02d}.jsonl", "a") as fh:
                                fh.write(json.dumps({"run_id": run_id, "axis": axis,
                                    "task": task.name, "episode": episode_idx,
                                    "attempt": attempt, "error": repr(exc)}) + "\n")
                            if attempt == args.episode_retries:
                                raise
                            client.reconnect()
                    episodes += 1
                    successes += int(done)

                    if args.save_videos:
                        import imageio

                        tag = "success" if done else "failure"
                        name = f"{axis.replace(' ', '_')}_{task.name}_{episode_idx}_{tag}.mp4"
                        imageio.mimwrite(out_dir / name, frames, fps=30)

                    with open(episodes_path, "a") as fh:
                        fh.write(json.dumps({
                            "run_id": run_id,
                            "suite": args.task_suite_name,
                            "axis": axis,
                            "task": task.name,
                            "desc": task_description,
                            "episode": episode_idx,
                            "success": bool(done),
                            "steps": steps,
                            "wall_s": round(time.monotonic() - started, 1),
                        }) + "\n")
            finally:
                env.close()

        rate = successes / episodes if episodes else float("nan")
        per_axis[axis] = (successes, episodes, rate)
        print(f"{axis}: {successes}/{episodes} = {rate:.1%} (this run)")

    print("\n=== LIBERO-Plus summary ===")
    total_s = sum(v[0] for v in per_axis.values())
    total_e = sum(v[1] for v in per_axis.values())
    for axis, (s, e, rate) in per_axis.items():
        print(f"  {axis:24s} {s:4d}/{e:<4d} {rate:6.1%}")
    print(f"  {'OVERALL':24s} {total_s:4d}/{total_e:<4d} "
          f"{(total_s / total_e if total_e else float('nan')):6.1%}")

    summary_name = ("summary.json" if args.num_shards == 1
                    else f"summary_shard{args.shard_index:02d}.json")
    with open(out_dir / summary_name, "w") as fh:
        json.dump({
            "suite": args.task_suite_name,
            "clean": args.clean,
            "num_tasks_per_axis": args.num_tasks_per_axis,
            "num_trials_per_task": args.num_trials_per_task,
            "seed": args.seed,
            "max_steps": max_steps,
            "per_axis": {a: {"successes": s, "episodes": e, "success_rate": r}
                         for a, (s, e, r) in per_axis.items()},
            "overall": {"successes": total_s, "episodes": total_e,
                        "success_rate": (total_s / total_e if total_e else None)},
        }, fh, indent=2)


if __name__ == "__main__":
    main(tyro.cli(Args))
