import argparse
import re
from pathlib import Path

import pandas as pd

_DATASET_RE = re.compile(r"experiments[/\\]([^/\\]+)[/\\]repair[/\\]")
_SEED_RE = re.compile(r"seed_([^/\\]+)[/\\]")

BASELINE_TIME_COLUMNS = ["cart_train_time_sec", "j48_train_time_sec", "reptree_train_time_sec"]


def infer_dataset(csv_path):
    m = _DATASET_RE.search(str(csv_path))
    return m.group(1) if m else "unknown"


def infer_seed(csv_path):
    m = _SEED_RE.search(str(csv_path))
    return m.group(1) if m else "unknown"


def infer_decision_point(csv_path):
    path = Path(csv_path)
    if path.parent.name in ("mutated", "baseline"):
        return path.parent.parent.name
    return path.parent.name


def load_trials(csv_paths):
    frames = []
    for p in csv_paths:
        df = pd.read_csv(p)
        present_baselines = [c for c in BASELINE_TIME_COLUMNS if c in df.columns]
        if "trial_id" not in df.columns or "rulesrepair_grow_time_sec" not in df.columns or not present_baselines:
            continue
        keep_cols = ["trial_id", "rulesrepair_grow_time_sec"] + present_baselines
        df = df[keep_cols].copy()
        df["dataset"] = infer_dataset(p)
        df["seed"] = infer_seed(p)
        df["decision_point"] = infer_decision_point(p)

        agg = {"rulesrepair_grow_time_sec": "sum"}
        for c in present_baselines:
            agg[c] = "first"
        trials = df.groupby(["dataset", "seed", "decision_point", "trial_id"], as_index=False).agg(agg)
        trials = trials.rename(columns={"rulesrepair_grow_time_sec": "rulesrepair_total_sec"})
        trials["mine_min_sec"] = trials[present_baselines].min(axis=1)
        trials["delta_sec"] = trials["rulesrepair_total_sec"] - trials["mine_min_sec"]
        frames.append(trials)
    if not frames:
        raise SystemExit(
            "None of the given file(s) contain trial_id + rulesrepair_grow_time_sec + at least "
            "one baseline *_train_time_sec column."
        )
    trials = pd.concat(frames, ignore_index=True)
    return trials, len(trials)


def summarize_delta(trials):
    grouped = trials.groupby("dataset")["delta_sec"]
    out = grouped.agg(n="count", mean_sec="mean", std_sec="std").reset_index()
    for col in ("mean_sec", "std_sec"):
        out[col] = out[col].round(4)
    return out.sort_values("dataset").reset_index(drop=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv_paths", nargs="+", help="results.csv file(s), glob-expanded by your shell")
    parser.add_argument(
        "--out-dir", default=None,
        help="Write time_delta_overall.csv here. Default: evaluation/quantitative/<dataset>/timing/ "
             "(dataset inferred from the input paths; falls back to evaluation/quantitative/timing/ "
             "if the inputs span more than one dataset).",
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

    trials, n_trials = load_trials(args.csv_paths)
    print(f"Loaded {n_trials} trial(s) from {len(args.csv_paths)} file(s) "
          f"(RulesRepair time summed over its whole w_simp x w_simi grid per trial).")

    table = summarize_delta(trials)
    out_csv = out_dir / "time_delta_overall.csv"
    table.to_csv(out_csv, index=False)
    print("\nRulesRepair time minus fastest Mine algorithm, by dataset:")
    print(table.to_string(index=False))
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
