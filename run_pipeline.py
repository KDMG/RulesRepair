#!/usr/bin/env python3
import os
import sys

if os.environ.get("PYTHONHASHSEED") != "0":
    os.execvpe(sys.executable, [sys.executable] + sys.argv, dict(os.environ, PYTHONHASHSEED="0"))

import argparse
import subprocess
from pathlib import Path

DATASET_INFO = {
    "sepsis": {
        "path": "datasets/sepsis/sepsis.xes",
        "url": "https://doi.org/10.4121/uuid:915d2bfb-7e84-49ad-a286-dc35f063a460",
    },
    "production": {
        "path": "datasets/production/Production.xes",
        "url": "https://doi.org/10.4121/uuid:68726926-5ac5-4fab-b873-ee76ea412399",
    },
    "hospital_billing": {
        "path": "datasets/hospital_billing/Hospital Billing - Event Log.xes.gz",
        "url": "https://doi.org/10.4121/uuid:76c46b83-c930-4798-a1c9-4be94dfeb741",
    },
    "road_traffic": {
        "path": "datasets/road_traffic/Road_Traffic_Fine_Management_Process.xes.gz",
        "url": "https://doi.org/10.4121/uuid:270fd440-1057-4fb9-89a9-b699b47990f5",
    },
    "prepaid_travel_costs": {
        "path": "datasets/prepaid_travel_costs/PrepaidTravelCost.xes",
        "url": "https://doi.org/10.4121/uuid:5d2fe5e1-f91f-4a3b-ad9b-9e4126870165",
    },
    "international_declarations": {
        "path": "datasets/international_declarations/InternationalDeclarations.xes",
        "url": "https://doi.org/10.4121/uuid:2bbf8f6a-fc50-48eb-aa9e-c4ea5ef7e8c5",
    },
}
TUE_SEARCH_URL = "https://data.4tu.nl/search?q=event+log"

REPO_ROOT = Path(__file__).resolve().parent


def run_step(args, cwd, env):
    print(f"  $ {' '.join(str(a) for a in args)}", flush=True)
    subprocess.run(args, cwd=cwd, env=env, check=True)


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset", required=True,
        help=f"Known datasets: {', '.join(DATASET_INFO)}",
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

    info = DATASET_INFO.get(args.dataset)
    if info is None:
        print(f"ERROR: unknown dataset '{args.dataset}'. Known datasets: {', '.join(DATASET_INFO)}", flush=True)
        print(f"For any other event log, search 4TU.ResearchData: {TUE_SEARCH_URL}", flush=True)
        return 1

    raw_xes = info["path"]
    if not (REPO_ROOT / raw_xes).exists():
        print(f"ERROR: '{args.dataset}' event log not found at {raw_xes}", flush=True)
        print(f"Download it from {info['url']} and place it there.", flush=True)
        return 1

    dataset = args.dataset
    legacy_cut_dir = REPO_ROOT / f"datasets/{dataset}_cut"
    legacy_dp_dir = REPO_ROOT / f"decision_points/{dataset}"
    if legacy_cut_dir.exists() or legacy_dp_dir.exists():
        cut_dir = legacy_cut_dir
        normative_dp_dir = legacy_dp_dir / "normative"
        train_dp_dir = legacy_dp_dir / "train"
        normative_model = legacy_dp_dir / "normative_model.pkl"
    else:
        dataset_dir = REPO_ROOT / f"experiments/{dataset}"
        cut_dir = dataset_dir / f"{dataset}_cut"
        normative_dp_dir = dataset_dir / "decision_points/normative"
        train_dp_dir = dataset_dir / "decision_points/train"
        normative_model = dataset_dir / "decision_points/normative_model.pkl"
    pnml = cut_dir / "pn_normative.pnml"
    out_base = REPO_ROOT / f"experiments/{dataset}/repair"

    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)

    print(f"step 1/5 {dataset}: split_log", flush=True)
    if args.skip_split:
        print("Skipped (--skip-split).", flush=True)
    else:
        run_step(
            [sys.executable, "-m", "mining.split_log",
             "--xes", raw_xes,
             "--normative", "0.5", "--train", "0.5", "--test", "0.0",
             "--out-dir", str(cut_dir),
             "--name", dataset],
            REPO_ROOT, env,
        )

    print(flush=True)
    print(f"step 2/5 {dataset}: mining the normative Petri net (pm4py, Inductive Miner infrequent)", flush=True)
    normative_xes = cut_dir / f"{dataset}_normative.xes"
    if args.skip_mine_pn:
        print("Skipped (--skip-mine-pn).", flush=True)
    elif pnml.exists():
        print(f"{pnml} already exists, not remined (delete the file to force "
              f"a new mining, or pass --skip-mine-pn to silence this check).", flush=True)
    else:
        if not normative_xes.exists():
            print(f"ERROR: {normative_xes} not found, step 1 is required first "
                  f"(don't pass --skip-split on the first run).", flush=True)
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
              f"build_pn_normative_pm4py.py's output above.", flush=True)
        return 1

    print(flush=True)
    print(f"step 3/5 {dataset}: rebuilding decision points (normative + train)", flush=True)
    if args.skip_dp:
        print("Skipped (--skip-dp).", flush=True)
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
             "--out-dir", str(train_dp_dir),
             "--show-row-stats"],
            REPO_ROOT, env,
        )

    print(flush=True)
    print(f"step 4/5 {dataset}: rebuilding the normative model", flush=True)
    if args.skip_normative_model:
        print("Skipped (--skip-normative-model).", flush=True)
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

    print(flush=True)
    print(f"step 5/5 {dataset}: running repair (baseline + mutated) on every dp, seed(s): {args.seeds}", flush=True)
    if args.skip_repair:
        print("Skipped (--skip-repair).", flush=True)
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

    print(flush=True)
    print(f"Pipeline for {dataset} completed. Output in {out_base}/seed_<N>/<dp>/{{baseline,mutated}}/", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
