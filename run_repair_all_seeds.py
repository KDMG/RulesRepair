#!/usr/bin/env python3
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

DEFAULT_SEEDS = "0,1,2,3,4,5,6,7,8,9"


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dp-dir", default="decision_points/sepsis/train")
    parser.add_argument("--normative-model", default="normative_model.pkl")
    parser.add_argument("--out-base", default="experiments/sepsis/repair")
    parser.add_argument("--seeds", default=DEFAULT_SEEDS)
    return parser.parse_known_args(argv)


def main(argv=None):
    args, extra_args = parse_args(argv if argv is not None else sys.argv[1:])

    dp_dir = Path(args.dp_dir)
    out_base = Path(args.out_base)
    tree_cache_dir = out_base / "_tree_cache"

    baseline_subdir = "baseline_fixed" if "--fixed" in extra_args else "baseline"

    seed_list = [s.strip() for s in args.seeds.split(",") if s.strip()]

    print(f"Running {len(seed_list)} seed(s) in sequence: {' '.join(seed_list)}")
    print(f"Output in {out_base}/seed_<N>/<dp>/{{baseline,mutated}}/ "
          f"(baseline computed only for the first seed, then copied)")
    print(f"CART/J48/etc. fit cache (Scenario 3): {tree_cache_dir}")

    first_seed = seed_list[0]

    for seed in seed_list:
        seed_out = out_base / f"seed_{seed}"
        print()
        print(f"seed={seed} -> {seed_out}")

        cmd = [
            sys.executable, str(REPO_ROOT / "run_repair_all.py"),
            "--dp-dir", str(dp_dir),
            "--normative-model", args.normative_model,
            "--out-base", str(seed_out),
            "--seed", seed,
            "--baseline-tree-cache-dir", str(tree_cache_dir),
        ]
        if seed != first_seed:
            cmd.append("--skip-baseline")
        cmd += extra_args

        subprocess.run(cmd, cwd=REPO_ROOT, check=True)

        if seed != first_seed:
            first_seed_out = out_base / f"seed_{first_seed}"
            for dp_csv in sorted(dp_dir.glob("dp_*.csv")):
                dp = dp_csv.stem
                if dp.startswith("dp_"):
                    dp = dp[len("dp_"):]
                src = first_seed_out / dp / baseline_subdir
                dst = seed_out / dp / baseline_subdir
                if src.is_dir():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    if dst.exists():
                        shutil.rmtree(dst)
                    shutil.copytree(src, dst)
            print(f"baseline copied from seed_{first_seed} to seed_{seed} for every dp")

    print()
    print(f"All seeds completed. Output in {out_base}/seed_<N>/<dp>/{{baseline,mutated}}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
