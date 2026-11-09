#!/usr/bin/env python3
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import run_repair_all

DEFAULT_SEEDS = "0,1,2,3,4,5,6,7,8,9"


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dp-dir", default="experiments/sepsis/decision_points/train")
    parser.add_argument("--normative-model", default="normative_model.pkl")
    parser.add_argument("--out-base", default="experiments/sepsis/repair")
    parser.add_argument("--seeds", default=DEFAULT_SEEDS)
    return parser.parse_known_args(argv)


def build_tasks(seed_list, dp_dir, out_base, tree_cache_dir, normative_model, extra_args):
    tasks = []
    for seed_idx, seed in enumerate(seed_list):
        seed_out = out_base / f"seed_{seed}"
        seed_args = run_repair_all.parse_args([
            "--dp-dir", str(dp_dir),
            "--normative-model", normative_model,
            "--out-base", str(seed_out),
            "--seed", seed,
            "--baseline-tree-cache-dir", str(tree_cache_dir),
            *extra_args,
        ])
        seed_out, fixed_args, log_suffix, extra_flags, dp_files = run_repair_all.prepare(seed_args)
        for dp_csv in sorted(dp_files, key=lambda f: (-f.stat().st_size, f.name)):
            tasks.append({
                "seed_idx": seed_idx,
                "seed": seed,
                "dp_csv": dp_csv,
                "args": seed_args,
                "out_base": seed_out,
                "log_suffix": log_suffix,
                "fixed_args": fixed_args,
                "extra_flags": extra_flags,
            })
    return tasks, seed_args.jobs if seed_list else 1


def run_task(task):
    run_repair_all.run_one(
        task["dp_csv"], task["args"], task["out_base"], task["log_suffix"],
        task["fixed_args"], task["extra_flags"], tag=f"seed={task['seed']} ",
    )


def main(argv=None):
    args, extra_args = parse_args(argv if argv is not None else sys.argv[1:])

    dp_dir = Path(args.dp_dir)
    out_base = Path(args.out_base)
    tree_cache_dir = out_base / "_tree_cache"

    seed_list = [s.strip() for s in args.seeds.split(",") if s.strip()]

    print(f"Running {len(seed_list)} seed(s): {' '.join(seed_list)}", flush=True)
    print(f"Output in {out_base}/seed_<N>/<dp>/mutated/", flush=True)
    print(f"CART/J48/etc. fit cache (Scenario 3): {tree_cache_dir}", flush=True)

    tasks, jobs = build_tasks(seed_list, dp_dir, out_base, tree_cache_dir, args.normative_model, extra_args)
    if not tasks:
        print(f"No dp_*.csv files found in {dp_dir}.", flush=True)
        return 1

    print(f"Parallelism: up to {jobs} job(s) at a time over {len(tasks)} (seed, decision point) job(s).", flush=True)

    first_seed_done = set()
    pending = list(tasks)
    running = {}
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        while pending or running:
            i = 0
            while len(running) < max(1, jobs) and i < len(pending):
                task = pending[i]
                dp_name = task["dp_csv"].name
                if task["seed_idx"] == 0 or dp_name in first_seed_done:
                    pending.pop(i)
                    running[pool.submit(run_task, task)] = task
                else:
                    i += 1
            if not running:
                raise RuntimeError("Scheduler has pending jobs but none can start.")
            done, _ = wait(list(running), return_when=FIRST_COMPLETED)
            for future in done:
                task = running.pop(future)
                future.result()
                if task["seed_idx"] == 0:
                    first_seed_done.add(task["dp_csv"].name)

    print(flush=True)
    print(f"All seeds completed. Output in {out_base}/seed_<N>/<dp>/mutated/", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
