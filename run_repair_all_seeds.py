#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

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


def main(argv=None):
    args, extra_args = parse_args(argv if argv is not None else sys.argv[1:])

    dp_dir = Path(args.dp_dir)
    out_base = Path(args.out_base)
    tree_cache_dir = out_base / "_tree_cache"

    seed_list = [s.strip() for s in args.seeds.split(",") if s.strip()]

    print(f"Running {len(seed_list)} seed(s) in sequence: {' '.join(seed_list)}", flush=True)
    print(f"Output in {out_base}/seed_<N>/<dp>/mutated/", flush=True)
    print(f"CART/J48/etc. fit cache (Scenario 3): {tree_cache_dir}", flush=True)

    for seed in seed_list:
        seed_out = out_base / f"seed_{seed}"
        print(flush=True)
        print(f"seed={seed} -> {seed_out}", flush=True)

        cmd = [
            sys.executable, str(REPO_ROOT / "run_repair_all.py"),
            "--dp-dir", str(dp_dir),
            "--normative-model", args.normative_model,
            "--out-base", str(seed_out),
            "--seed", seed,
            "--baseline-tree-cache-dir", str(tree_cache_dir),
        ]
        cmd += extra_args

        subprocess.run(cmd, cwd=REPO_ROOT, check=True)

    print(flush=True)
    print(f"All seeds completed. Output in {out_base}/seed_<N>/<dp>/mutated/", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
