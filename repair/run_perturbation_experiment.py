import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from mutations.scenario0 import generate_scenario0


def split_adapt_test(df, target, adapt_fraction, split_method, split_seed):
    if split_method == "positional":
        n_adapt = round(len(df) * adapt_fraction)
        return df.iloc[:n_adapt].reset_index(drop=True), df.iloc[n_adapt:].reset_index(drop=True)

    if split_method == "case_chronological":
        if "case_id" not in df.columns or "timestamp" not in df.columns:
            raise SystemExit(
                "--split-method case_chronological requires both 'case_id' and 'timestamp' columns "
                "in --data-csv (to group rows by case and order cases by their first event)."
            )
        case_order = df.groupby("case_id")["timestamp"].min().sort_values().index
        n_adapt_cases = round(len(case_order) * adapt_fraction)
        adapt_cases = set(case_order[:n_adapt_cases])
        df_adapt = df[df["case_id"].isin(adapt_cases)].reset_index(drop=True)
        df_test = df[~df["case_id"].isin(adapt_cases)].reset_index(drop=True)
        return df_adapt, df_test

    try:
        return train_test_split(df, train_size=adapt_fraction, random_state=split_seed, stratify=df[target])
    except ValueError:
        return train_test_split(df, train_size=adapt_fraction, random_state=split_seed)


def run_experiment(data_csv, target, adapt_fraction, split_method, split_seed, out_dir):
    df_data = pd.read_csv(data_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trials = generate_scenario0(df_data, target)
    manifest_path = out_dir / "manifest.jsonl"

    with open(manifest_path, "w") as manifest:
        for trial in trials:
            df_adapt, df_test = split_adapt_test(df_data, target, adapt_fraction, split_method, split_seed)

            train_out = out_dir / f"{trial['trial_id']}_train.csv"
            test_out = out_dir / f"{trial['trial_id']}_test.csv"
            df_adapt.to_csv(train_out, index=False)
            df_test.to_csv(test_out, index=False)

            record = dict(trial)
            record["train_file"] = str(train_out)
            record["test_file"] = str(test_out)
            manifest.write(json.dumps(record) + "\n")

    print(f"Generated {len(trials)} trial(s) (Scenario 0, no perturbation) -> {manifest_path}")
    return manifest_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-csv", required=True, help="Table to split into D_adapt/D_test, unperturbed.")
    parser.add_argument("--target", default="branch")
    parser.add_argument("--adapt-fraction", type=float, default=0.5, help="Share of --data-csv that becomes D_adapt; the rest becomes D_test.")
    parser.add_argument(
        "--split-method", choices=["positional", "random", "case_chronological"], default="case_chronological",
        help="How --data-csv is cut into adapt/test. positional: first N rows by position. "
             "random: sklearn train_test_split, seeded by --split-seed. case_chronological (default): "
             "group by case_id, order CASES (not rows) by first-event timestamp, allocate whole cases "
             "to adapt/test -- matches grid_search_dp.py's split exactly, requires 'case_id'/"
             "'timestamp' columns.",
    )
    parser.add_argument("--split-seed", type=int, default=0, help="Only used when --split-method random")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    run_experiment(
        data_csv=args.data_csv, target=args.target,
        adapt_fraction=args.adapt_fraction, split_method=args.split_method, split_seed=args.split_seed,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
