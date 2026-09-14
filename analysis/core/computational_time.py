"""Aggregate per-trial computation time out of results.csv files (baseline
and/or mutated repair pipeline) into compact summary tables: overall, by
mutation type, and by decision point.

    python -m analysis.core.computational_time experiments/<dataset>/repair/seed_*/*/mutated/results.csv
"""
import argparse
import re
from pathlib import Path

import pandas as pd

_DATASET_RE = re.compile(r"experiments[/\\]([^/\\]+)[/\\]repair[/\\]")
_SEED_RE = re.compile(r"seed_([^/\\]+)[/\\]")

# column -> human-readable algorithm name. rulesrepair_grow_time_sec is the
# repair algorithm itself; the rest are the baselines it's compared against.
TIME_COLUMNS = {
    "rulesrepair_grow_time_sec": "RulesRepair",
    "cart_train_time_sec": "CART",
    "cart_entropy_train_time_sec": "CART-entropy",
    "c45_train_time_sec": "C4.5",
    "j48_train_time_sec": "J48",
    "reptree_train_time_sec": "REPTree",
}

NO_OPERATOR_LABEL = "baseline (no mutation operator)"


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


def load_all(csv_paths):
    id_cols = ["dataset", "seed", "decision_point", "operator"]
    frames = []
    n_trials = 0
    for p in csv_paths:
        df = pd.read_csv(p)
        present = [c for c in TIME_COLUMNS if c in df.columns]
        if not present:
            continue
        n_trials += len(df)
        df = df[present].copy()
        df["operator"] = df["operator"].fillna(NO_OPERATOR_LABEL) if "operator" in df.columns else NO_OPERATOR_LABEL
        df["dataset"] = infer_dataset(p)
        df["seed"] = infer_seed(p)
        df["decision_point"] = infer_decision_point(p)

        long = df.melt(
            id_vars=id_cols, value_vars=present,
            var_name="time_column", value_name="seconds",
        ).dropna(subset=["seconds"])
        if len(long):
            frames.append(long)
    if not frames:
        raise SystemExit(
            "None of the given file(s) contain a *_train_time_sec or rulesrepair_grow_time_sec column."
        )
    long = pd.concat(frames, ignore_index=True)
    long["algorithm"] = long["time_column"].map(TIME_COLUMNS)
    return long, n_trials


def summarize(long, group_cols):
    grouped = long.groupby(group_cols + ["algorithm"])["seconds"]
    out = grouped.agg(
        n="count", mean_sec="mean", median_sec="median", std_sec="std", total_sec="sum",
    ).reset_index()
    for col in ("mean_sec", "median_sec", "std_sec", "total_sec"):
        out[col] = out[col].round(4)
    return out.sort_values(group_cols + ["algorithm"]).reset_index(drop=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv_paths", nargs="+", help="results.csv file(s), glob-expanded by your shell")
    parser.add_argument(
        "--out-dir", default=None,
        help="Write the three summary CSVs here. Default: evaluation/quantitative/timing/",
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir) if args.out_dir else Path("evaluation") / "quantitative" / "timing"
    out_dir.mkdir(parents=True, exist_ok=True)

    long, n_trials = load_all(args.csv_paths)
    print(f"Loaded {n_trials} trial(s) from {len(args.csv_paths)} file(s).")

    tables = [
        ("Overall, by dataset", ["dataset"], "time_overall.csv"),
        ("By mutation type", ["dataset", "operator"], "time_by_mutation_type.csv"),
        ("By decision point", ["dataset", "decision_point"], "time_by_decision_point.csv"),
    ]
    for title, group_cols, filename in tables:
        table = summarize(long, group_cols)
        out_csv = out_dir / filename
        table.to_csv(out_csv, index=False)
        print(f"\n{title}:")
        print(table.to_string(index=False))
        print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
