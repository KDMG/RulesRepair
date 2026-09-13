#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path

RAW_XES_BY_DATASET = {
    "sepsis": "datasets/sepsis/sepsis.xes",
    "production": "datasets/production/Production.xes",
    "hospital_billing": "datasets/hospital_billing/hospital_billing.xes",
    "road_traffic": "datasets/road_traffic/road_traffic.xes",
    "prepaid_travel_costs": "datasets/prepaid_travel_costs/PrepaidTravelCost.xes",
    "international_declarations": "datasets/international_declarations/InternationalDeclarations.xes_",
}

REPO_ROOT = Path(__file__).resolve().parent


def run_step(args, cwd, env):
    print(f"  $ {' '.join(str(a) for a in args)}")
    subprocess.run(args, cwd=cwd, env=env, check=True)


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset", required=True,
        help=f"Known datasets: {', '.join(RAW_XES_BY_DATASET)}",
    )
    parser.add_argument("--skip-split", action="store_true")
    parser.add_argument("--skip-mine-pn", action="store_true")
    parser.add_argument("--skip-dp", action="store_true")
    parser.add_argument("--skip-normative-model", action="store_true")
    parser.add_argument("--skip-repair", action="store_true")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--noise", default="0.2")
    return parser.parse_known_args(argv)


def main(argv=None):
    args, repair_args = parse_args(argv if argv is not None else sys.argv[1:])

    raw_xes = RAW_XES_BY_DATASET.get(args.dataset)
    if raw_xes is None:
        print(f"ERROR: unknown dataset '{args.dataset}'. "
              f"Known datasets: {', '.join(RAW_XES_BY_DATASET)}")
        print("(to add a new one, add an entry to RAW_XES_BY_DATASET "
              "at the top of this script)")
        return 1

    dataset = args.dataset
    cut_dir = REPO_ROOT / f"datasets/{dataset}_cut"
    pnml = cut_dir / "pn_normative.pnml"
    normative_dp_dir = REPO_ROOT / f"decision_points/{dataset}/normative"
    train_dp_dir = REPO_ROOT / f"decision_points/{dataset}/train"
    normative_model = REPO_ROOT / f"normative_model_{dataset}.pkl"
    out_base = REPO_ROOT / f"experiments/{dataset}/repair"

    import os
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)

    print(f"step 1/5 {dataset}: split_log")
    if args.skip_split:
        print("Skipped (--skip-split).")
    else:
        run_step(
            [sys.executable, "-m", "mining.split_log",
             "--xes", raw_xes,
             "--normative", "0.5", "--train", "0.5", "--test", "0.0",
             "--out-dir", str(cut_dir),
             "--name", dataset],
            REPO_ROOT, env,
        )

    print()
    print(f"step 2/5 {dataset}: mining the normative Petri net (pm4py, Inductive Miner infrequent)")
    normative_xes = cut_dir / f"{dataset}_normative.xes"
    if args.skip_mine_pn:
        print("Skipped (--skip-mine-pn).")
    elif pnml.exists():
        print(f"{pnml} already exists, not remined (delete the file to force "
              f"a new mining, or pass --skip-mine-pn to silence this check).")
    else:
        if not normative_xes.exists():
            print(f"ERROR: {normative_xes} not found, step 1 is required first "
                  f"(don't pass --skip-split on the first run).")
            return 1
        run_step(
            [sys.executable, "-m", "mining.build_pn_normative_pm4py",
             "--xes", str(normative_xes),
             "--out", str(pnml),
             "--noise", args.noise],
            REPO_ROOT, env,
        )

    if not pnml.exists():
        print(f"ERROR: {pnml} still not found after step 2, check "
              f"build_pn_normative_pm4py.py's output above.")
        return 1

    print()
    print(f"step 3/5 {dataset}: rebuilding decision points (normative + train)")
    if args.skip_dp:
        print("Skipped (--skip-dp).")
    else:
        run_step(
            [sys.executable, "-m", "mining.regenerate_decision_points",
             "--pnml", str(pnml),
             "--xes", str(normative_xes),
             "--min-fitness", "0",
             "--out-dir", str(normative_dp_dir)],
            REPO_ROOT, env,
        )
        run_step(
            [sys.executable, "-m", "mining.regenerate_decision_points",
             "--pnml", str(pnml),
             "--xes", str(cut_dir / f"{dataset}_train.xes"),
             "--min-fitness", "0",
             "--out-dir", str(train_dp_dir)],
            REPO_ROOT, env,
        )

    print()
    print(f"step 4/5 {dataset}: rebuilding the normative model")
    if args.skip_normative_model:
        print("Skipped (--skip-normative-model).")
    else:
        run_step(
            [sys.executable, "-m", "mining.build_normative_model",
             "--dp-dir", str(normative_dp_dir),
             "--max-depth", "4",
             "--min-samples-leaf", "0.02",
             "--save", str(normative_model),
             "--quiet"],
            REPO_ROOT, env,
        )

    print()
    print(f"step 5/5 {dataset}: running repair (baseline + mutated) on every dp, seed(s): {args.seeds}")
    if args.skip_repair:
        print("Skipped (--skip-repair).")
    else:
        run_step(
            [sys.executable, str(REPO_ROOT / "run_repair_all_seeds.py"),
             "--dp-dir", str(train_dp_dir),
             "--normative-model", str(normative_model),
             "--out-base", str(out_base),
             "--seeds", args.seeds,
             *repair_args],
            REPO_ROOT, env,
        )

    print()
    print(f"Pipeline for {dataset} completed. Output in {out_base}/seed_<N>/<dp>/{{baseline,mutated}}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
