"""Aggregate LIBERO-Plus episode records into per-axis success rates.

Reads every episodes*.jsonl under a result directory, so it works the same for a single
run and for a sharded parallel one, and is safe to run while a run is still going.
"""

import argparse
import collections
import glob
import json
import os


def load(result_dir: str) -> list[dict]:
    records, seen = [], set()
    manifest_path = os.path.join(result_dir, "run_manifest.json")
    run_id = None
    if os.path.exists(manifest_path):
        with open(manifest_path) as fh:
            run_id = json.load(fh)["run_id"]
    for path in sorted(glob.glob(os.path.join(result_dir, "episodes*.jsonl"))):
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # partial trailing line of a run still in flight
                if run_id is not None and rec.get("run_id") != run_id:
                    raise ValueError(f"Mixed run identities in {path}")
                key = (rec["axis"], rec["task"], rec["episode"])
                if key in seen:
                    continue  # a resumed shard can re-record; count each episode once
                seen.add(key)
                records.append(rec)
    return records


# Variant counts per suite in LIBERO-plus, used only to print progress.
EXPECTED = {
    "libero_spatial": 2402,
    "libero_object": 2518,
    "libero_goal": 2591,
    "libero_10": 2519,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", default="eval_results/libero_plus")
    ap.add_argument("--suite", default=None,
                    help="Suite name; sets --expected from its LIBERO-plus variant count.")
    ap.add_argument("--expected", type=int, default=None,
                    help="Episode count to show progress against.")
    args = ap.parse_args()

    if args.expected is None:
        args.expected = EXPECTED.get(args.suite or "", 0)

    records = load(args.result_dir)
    if not records:
        print(f"No episode records under {args.result_dir}")
        return

    by_axis = collections.defaultdict(lambda: [0, 0])
    steps_total, wall_total = 0, 0.0
    for rec in records:
        stat = by_axis[rec["axis"]]
        stat[0] += int(rec["success"])
        stat[1] += 1
        steps_total += rec.get("steps", 0)
        wall_total += rec.get("wall_s", 0.0)

    total_s = sum(v[0] for v in by_axis.values())
    total_e = sum(v[1] for v in by_axis.values())

    clean = all(rec["axis"] == "clean" for rec in records)
    label = "LIBERO (clean)" if clean else "LIBERO-Plus"
    print(f"{label} results from {args.result_dir}")
    if args.expected and not clean:
        print(f"{total_e} episodes done of {args.expected} expected "
              f"({total_e / args.expected:.1%})\n")
    else:
        print(f"{total_e} episodes recorded\n")
    print(f"  {'axis':24s} {'succ':>6s} {'eps':>6s} {'rate':>8s}")
    for axis in sorted(by_axis):
        s, e = by_axis[axis]
        print(f"  {axis:24s} {s:6d} {e:6d} {s / e:7.1%}")
    print(f"  {'-' * 46}")
    print(f"  {'OVERALL':24s} {total_s:6d} {total_e:6d} {total_s / total_e:7.1%}")
    if total_e:
        print(f"\n  mean {steps_total / total_e:.0f} steps, "
              f"{wall_total / total_e:.0f}s per episode")


if __name__ == "__main__":
    main()
