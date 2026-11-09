import argparse
import glob as glob_module
import re
from pathlib import Path

import pandas as pd

from keep_remine_prune.param_grid import GROW_CONFIGS

_DATASET_RE = re.compile(r"experiments[/\\]([^/\\]+)[/\\]repair[/\\]")
_SEED_RE = re.compile(r"seed_([^/\\]+)[/\\]")

ALGORITHM_DISPLAY = {"cart": "CART", "j48": "C4.5", "reptree": "REPTree"}


def infer_dataset(csv_path):
    m = _DATASET_RE.search(str(csv_path))
    return m.group(1) if m else "unknown"


def infer_seed(csv_path):
    m = _SEED_RE.search(str(csv_path))
    return m.group(1) if m else "unknown"


def config_name(algorithm, criterion, pct):
    if isinstance(criterion, str):
        return f"{algorithm}_{criterion}_msl{int(pct)}"
    return f"{algorithm}_msl{int(pct)}"


def load_trial_config_times(csv_paths):
    frames = []
    for raw in csv_paths:
        for path in sorted(glob_module.glob(str(raw))) or [str(raw)]:
            path = Path(path)
            if not path.exists():
                print(f"  {path}: file not found")
                continue
            sweep = pd.read_csv(path)
            required = ["trial_id", "algorithm", "criterion", "min_samples_leaf_pct", "train_time_sec"]
            missing = [c for c in required if c not in sweep.columns]
            if missing:
                raise SystemExit(f"{path}: missing column(s) {missing} -- expected a baseline_sweep.csv file.")
            sweep = sweep.copy()
            sweep["config"] = [
                config_name(a, c, p) for a, c, p in zip(sweep["algorithm"], sweep["criterion"], sweep["min_samples_leaf_pct"])
            ]
            mine = sweep[["trial_id", "config", "algorithm", "train_time_sec"]].rename(
                columns={"train_time_sec": "mine_sec"}
            )

            configs_dir = path.parent / "configs"
            rr_frames = []
            absent = []
            for config in GROW_CONFIGS:
                results_path = configs_dir / config["name"] / "results.csv"
                if not results_path.exists():
                    absent.append(config["name"])
                    continue
                results = pd.read_csv(results_path, usecols=["trial_id", "rulesrepair_grow_time_sec"])
                grouped = results.groupby("trial_id", as_index=False)["rulesrepair_grow_time_sec"].sum()
                grouped["config"] = config["name"]
                rr_frames.append(grouped.rename(columns={"rulesrepair_grow_time_sec": "rulesrepair_sec"}))
            if absent:
                print(f"  Warning: {path.parent}: missing RulesRepair results for {absent}, skipping this file.")
                continue

            merged = pd.concat(rr_frames, ignore_index=True).merge(mine, on=["trial_id", "config"], how="inner")
            merged["dataset"] = infer_dataset(path)
            merged["seed"] = infer_seed(path)
            merged["decision_point"] = path.parents[1].name
            frames.append(merged)
    if not frames:
        raise SystemExit("No usable baseline_sweep.csv files with matching configs/ results found.")
    times = pd.concat(frames, ignore_index=True)
    times["delta_sec"] = times["rulesrepair_sec"] - times["mine_sec"]
    times["algorithm"] = times["algorithm"].map(ALGORITHM_DISPLAY)
    return times


def summarize(times, group_cols):
    out = times.groupby(group_cols).agg(
        n=("delta_sec", "count"),
        rulesrepair_mean_sec=("rulesrepair_sec", "mean"), rulesrepair_std_sec=("rulesrepair_sec", "std"),
        mine_mean_sec=("mine_sec", "mean"), mine_std_sec=("mine_sec", "std"),
        delta_mean_sec=("delta_sec", "mean"), delta_std_sec=("delta_sec", "std"),
    ).reset_index()
    numeric = [c for c in out.columns if c.endswith("_sec")]
    out[numeric] = out[numeric].round(4)
    return out.sort_values(group_cols).reset_index(drop=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv_paths", nargs="+", help="baseline_sweep.csv file(s)")
    parser.add_argument(
        "--out-dir", default=None,
        help="Where to write the timing tables. Default: evaluation/quantitative/<dataset>/timing/ "
             "(evaluation/quantitative/timing/ if the inputs span more than one dataset).",
    )
    args = parser.parse_args(argv)

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        datasets = {infer_dataset(p) for p in args.csv_paths}
        if len(datasets) == 1:
            out_dir = Path("evaluation") / "quantitative" / next(iter(datasets)) / "timing"
        else:
            out_dir = Path("evaluation") / "quantitative" / "timing"
    out_dir.mkdir(parents=True, exist_ok=True)

    times = load_trial_config_times(args.csv_paths)
    print(f"Loaded {len(times)} (trial, configuration) pair(s). RulesRepair time = one run of one mining "
          f"configuration over the whole w_simp x w_simi grid; Mine time = one fit of the same configuration.")

    by_algorithm = summarize(times, ["dataset", "algorithm"])
    by_config = summarize(times, ["dataset", "algorithm", "config"])
    raw_csv = out_dir / "time_per_trial_config.csv"
    algo_csv = out_dir / "time_by_algorithm.csv"
    config_csv = out_dir / "time_by_config.csv"
    times.to_csv(raw_csv, index=False)
    by_algorithm.to_csv(algo_csv, index=False)
    by_config.to_csv(config_csv, index=False)

    print("\nTime (sec) by dataset x algorithm:")
    print(by_algorithm.to_string(index=False))
    print(f"\nWrote {algo_csv}, {config_csv} and {raw_csv}")


if __name__ == "__main__":
    main()
