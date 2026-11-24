#!/usr/bin/env python3
import argparse
import hashlib
import sys
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import run_repair_all
from keep_remine_prune.param_grid import GROW_CONFIGS

GRID_HASH = hashlib.md5("|".join(c["name"] for c in GROW_CONFIGS).encode()).hexdigest()[:6]

DEFAULT_SEEDS = "0,1,2,3,4,5,6,7,8,9"
POLL_SECONDS = 30


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dp-dir", default="experiments/sepsis/decision_points/train")
    parser.add_argument("--normative-model", default="normative_model.pkl")
    parser.add_argument("--out-base", default="experiments/sepsis/repair")
    parser.add_argument("--seeds", default=DEFAULT_SEEDS)
    parser.add_argument(
        "--extra-jobs", default="",
        help="Comma-separated seed:dp pairs (e.g. 1:p_21,3:p_22) to run in addition to --seeds, placed at the "
             "front of the queue. A pair that is also covered by --seeds runs once.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip the (seed, decision point) jobs whose output is already complete (baseline_sweep.csv and a "
             "non-empty results.csv for every mining configuration).",
    )
    return parser.parse_known_args(argv)


def dp_name_of(dp_csv):
    name = Path(dp_csv).stem
    return name[len("dp_"):] if name.startswith("dp_") else name


def cache_ready(tree_cache_dir, dp_name):
    return any(Path(tree_cache_dir).glob(f"{dp_name}_af*_sweep_g{GRID_HASH}.pkl"))


def task_complete(task):
    mutated = Path(task["out_base"]) / dp_name_of(task["dp_csv"]) / "mutated"
    sweep = mutated / "baseline_sweep.csv"
    if not sweep.exists() or not (mutated / "manifest.jsonl").exists():
        return False
    for config in GROW_CONFIGS:
        results = mutated / "configs" / config["name"] / "results.csv"
        if not results.exists() or results.stat().st_size == 0:
            return False
    return True


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
    extra_pairs = [tuple(pair.strip().split(":", 1)) for pair in args.extra_jobs.split(",") if pair.strip()]
    if extra_pairs:
        wanted_by_seed = {}
        for seed, dp in extra_pairs:
            wanted_by_seed.setdefault(seed, set()).add(dp)
        extra_tasks = []
        for seed, dps in wanted_by_seed.items():
            seed_tasks, extra_jobs_count = build_tasks([seed], dp_dir, out_base, tree_cache_dir, args.normative_model, extra_args)
            found = {dp_name_of(t["dp_csv"]) for t in seed_tasks} & dps
            missing = dps - found
            if missing:
                raise SystemExit(f"--extra-jobs: no decision point {sorted(missing)} for seed {seed} in {dp_dir}")
            for t in seed_tasks:
                if dp_name_of(t["dp_csv"]) in dps:
                    t["seed_idx"] = 1
                    extra_tasks.append(t)
            jobs = max(jobs, extra_jobs_count) if not tasks else jobs
        extra_keys = {(t["seed"], dp_name_of(t["dp_csv"])) for t in extra_tasks}
        tasks = [t for t in tasks if (t["seed"], dp_name_of(t["dp_csv"])) not in extra_keys]
        extra_tasks.sort(key=lambda t: (-t["dp_csv"].stat().st_size, t["dp_csv"].name, t["seed"]))
        tasks = extra_tasks + tasks
        print(f"Extra jobs: {len(extra_tasks)} (seed, decision point) job(s) placed at the front of the queue.", flush=True)
    if not tasks:
        print(f"No dp_*.csv files found in {dp_dir}.", flush=True)
        return 1

    print(f"Parallelism: up to {jobs} job(s) at a time over {len(tasks)} (seed, decision point) job(s).", flush=True)

    first_seed_done = set()
    pending = list(tasks)
    if args.resume:
        remaining = []
        n_skipped = 0
        for task in pending:
            if task_complete(task):
                n_skipped += 1
                if task["seed_idx"] == 0:
                    first_seed_done.add(task["dp_csv"].name)
            else:
                remaining.append(task)
        pending = remaining
        print(f"Resume: {n_skipped} (seed, decision point) job(s) already complete, {len(pending)} left to run.", flush=True)
        if not pending:
            print("Nothing to run.", flush=True)
            return 0
    running = {}
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        while pending or running:
            i = 0
            while len(running) < max(1, jobs) and i < len(pending):
                task = pending[i]
                dp_name = task["dp_csv"].name
                if task["seed_idx"] == 0 or dp_name in first_seed_done or cache_ready(tree_cache_dir, dp_name_of(task["dp_csv"])):
                    pending.pop(i)
                    running[pool.submit(run_task, task)] = task
                else:
                    i += 1
            if not running:
                raise RuntimeError("Scheduler has pending jobs but none can start.")
            done, _ = wait(list(running), timeout=POLL_SECONDS, return_when=FIRST_COMPLETED)
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
