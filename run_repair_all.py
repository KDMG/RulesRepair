#!/usr/bin/env python3
import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

DEFAULT_W_SIMPS = "0.0,0.001,0.005,0.01,0.02,0.05,0.1,0.2,0.3,0.5,0.75,1.0,1.5,2.0,3.0,5.0"
DEFAULT_W_SIMIS = "0.0,0.001,0.005,0.01,0.02,0.05,0.1,0.2,0.3,0.5,0.75,1.0,1.5"


def default_jobs():
    return os.cpu_count() or 4


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dp-dir", default="decision_points/sepsis/train")
    parser.add_argument("--normative-model", default="normative_model.pkl")
    parser.add_argument("--out-base", default="experiments/sepsis/repair")
    parser.add_argument("--w-simps", default=DEFAULT_W_SIMPS)
    parser.add_argument("--w-simis", default=DEFAULT_W_SIMIS)
    parser.add_argument("--seed", default="0")
    parser.add_argument("--jobs", type=int, default=default_jobs())
    parser.add_argument("--fixed", action="store_true")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--baseline-tree-cache-dir", default=None)
    return parser.parse_args(argv)


def run_one(dp_csv, args, out_base, log_suffix, fixed_args, extra_flags):
    import subprocess

    dp = dp_csv.stem
    if dp.startswith("dp_"):
        dp = dp[len("dp_"):]
    log_path = out_base / f"{dp}{log_suffix}.log"
    print(f"{dp}: start")

    cmd = [
        sys.executable, "-m", "repair.run_repair",
        "--normative-model", args.normative_model,
        "--dp", dp,
        "--data-csv", str(dp_csv),
        "--w-simps", args.w_simps,
        "--w-simis", args.w_simis,
        "--seed", str(args.seed),
        "--out-dir", str(out_base / dp),
        *fixed_args, *extra_flags,
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)

    with open(log_path, "w") as log_file:
        result = subprocess.run(cmd, cwd=REPO_ROOT, env=env, stdout=log_file, stderr=subprocess.STDOUT)

    if result.returncode == 0:
        skipped = False
        with open(log_path) as log_file:
            for line in log_file:
                if line.startswith(f"Skipping '{dp}':"):
                    skipped = True
                    break
        if skipped:
            print(f"{dp}: skipped (too little data, see {log_path})")
        else:
            print(f"{dp}: done")
    else:
        print(f"FAILED: {dp} (see {log_path})")


def main(argv=None):
    args = parse_args(argv if argv is not None else sys.argv[1:])

    dp_dir = Path(args.dp_dir)
    out_base = Path(args.out_base)
    out_base.mkdir(parents=True, exist_ok=True)

    fixed_args = ["--fixed"] if args.fixed else []
    log_suffix = "_fixed" if args.fixed else ""

    extra_flags = []
    if args.skip_baseline:
        extra_flags.append("--skip-baseline")
    if args.baseline_tree_cache_dir:
        extra_flags += ["--baseline-tree-cache-dir", args.baseline_tree_cache_dir]

    dp_files = sorted(dp_dir.glob("dp_*.csv"))
    if not dp_files:
        print(f"No dp_*.csv files found in {dp_dir}.")
        return 1

    print(f"Parallelism: up to {args.jobs} decision point(s) at a time.")

    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = [
            pool.submit(run_one, f, args, out_base, log_suffix, fixed_args, extra_flags)
            for f in dp_files
        ]
        for future in as_completed(futures):
            future.result()  # re-raise any unexpected exception

    print(f"All decision points completed. Output in {out_base}/<dp>/{{baseline,mutated}}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
