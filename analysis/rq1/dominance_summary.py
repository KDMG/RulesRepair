import argparse
from pathlib import Path

import pandas as pd

from analysis.core.compare_rulesrepair_cart_dominance import DEFAULT_TOL
from analysis.rq1._common import (
    infer_dataset, BASELINE_ALGORITHMS, DATASET_DISPLAY_NAMES, DATASET_COLUMN_ORDER,
    load_dom_cov_raw, collect_multi_algorithm_dominance,
)


def _dominance_pct(d_i_values):
    vals = pd.Series(d_i_values).dropna()
    n = len(vals)
    if n == 0:
        return {"n_trials": 0, "dominance_pct": float("nan")}
    return {"n_trials": n, "dominance_pct": float((vals > 0).mean() * 100.0)}


def dominance_by_dataset(long_df):
    rows = []
    for (dataset, algorithm), group in long_df.groupby(["dataset", "algorithm"]):
        rows.append({"dataset": dataset, "algorithm": algorithm, **_dominance_pct(group["d_i"])})
    return pd.DataFrame(rows).sort_values(["dataset", "algorithm"]).reset_index(drop=True)


def dominance_by_decision_point_mutation(long_df):
    with_op = long_df.dropna(subset=["operator"])
    rows = []
    for (dataset, algorithm, decision_point, operator), group in with_op.groupby(
            ["dataset", "algorithm", "decision_point", "operator"]):
        rows.append({
            "dataset": dataset, "algorithm": algorithm,
            "decision_point": decision_point, "operator": operator,
            **_dominance_pct(group["d_i"]),
        })
    return pd.DataFrame(rows).sort_values(
        ["dataset", "algorithm", "decision_point", "operator"]
    ).reset_index(drop=True)


def dominance_by_mutation(long_df):
    with_op = long_df.dropna(subset=["operator"])
    rows = []
    for (dataset, algorithm, operator), group in with_op.groupby(["dataset", "algorithm", "operator"]):
        rows.append({"dataset": dataset, "algorithm": algorithm, "operator": operator, **_dominance_pct(group["d_i"])})
    return pd.DataFrame(rows).sort_values(["dataset", "algorithm", "operator"]).reset_index(drop=True)


def load_by_dataset_files(csv_paths):
    frames = []
    for p in csv_paths:
        df = pd.read_csv(p)
        missing = [c for c in ("dataset", "algorithm", "dominance_pct") if c not in df.columns]
        if missing:
            raise SystemExit(f"{p}: missing column(s) {missing} -- is this really a dominance_by_dataset.csv "
                              f"file from a previous dominance-summary run?")
        frames.append(df)
    if not frames:
        raise SystemExit("No usable dominance_by_dataset.csv files found among the given paths.")
    return pd.concat(frames, ignore_index=True)


def build_dominance_frequency_table(by_dataset_df):
    present_datasets = set(by_dataset_df["dataset"].dropna().unique())
    ordered_datasets = [d for d in DATASET_COLUMN_ORDER if d in present_datasets]
    ordered_datasets += sorted(d for d in present_datasets if d not in DATASET_COLUMN_ORDER)

    algo_displays = [a["display"] for a in BASELINE_ALGORITHMS]
    table = {}
    for algo in algo_displays:
        table[algo] = {}
        for dataset in ordered_datasets:
            rows = by_dataset_df[(by_dataset_df["dataset"] == dataset) & (by_dataset_df["algorithm"] == algo)]
            table[algo][dataset] = float(rows.iloc[0]["dominance_pct"]) if not rows.empty else None
    return table, ordered_datasets


def build_dominance_frequency_latex(table, ordered_datasets):
    backslash = chr(92)
    row_end = backslash + backslash
    newline = chr(10)
    dataset_labels = [DATASET_DISPLAY_NAMES.get(d, d) for d in ordered_datasets]
    col_spec = "l" + "c" * len(ordered_datasets)

    lines = []
    lines.append(backslash + "begin{table}[tb]")
    lines.append(backslash + "centering")
    lines.append(backslash + "large")
    lines.append(backslash + "resizebox{" + backslash + "textwidth}{!}{%")
    lines.append(backslash + "begin{tabular}{" + col_spec + "}")
    lines.append(backslash + "toprule")
    lines.append(backslash + "textsc{Mine}")
    for i, label in enumerate(dataset_labels):
        suffix = " " + row_end if i == len(dataset_labels) - 1 else ""
        lines.append("& " + backslash + "textit{" + label + "}" + suffix)
    lines.append(backslash + "midrule")

    algo_displays = [a["display"] for a in BASELINE_ALGORITHMS]
    for algo in algo_displays:
        lines.append(algo)
        for i, dataset in enumerate(ordered_datasets):
            val = table[algo][dataset]
            cell = f"{val:.2f}" + backslash + "%" if val is not None else "--"
            suffix = " " + row_end if i == len(ordered_datasets) - 1 else ""
            lines.append("& " + cell + suffix)

    lines.append(backslash + "bottomrule")
    lines.append(backslash + "end{tabular}%")
    lines.append("}")
    caption = (
        "Percentage of times in which " + backslash + "textsc{Mine} dominates at least one solution of the "
        "Pareto front produced by " + backslash + "textsc{RulesRepair}. We do not report the percentage of "
        "fronts entirely dominated by " + backslash + "textsc{Mine}, as it is always zero."
    )
    lines.append(backslash + "caption{" + caption + "}")
    lines.append(backslash + "label{tab:dominance_frequency}")
    lines.append(backslash + "end{table}")
    return newline.join(lines)


def main_dominance_summary(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv_paths", nargs="+")
    parser.add_argument("-tol", type=float, default=DEFAULT_TOL)
    parser.add_argument("-dom-cov-raw-out", default=None)
    parser.add_argument("-by-dataset-out", default=None)
    parser.add_argument("-by-dp-mutation-out", default=None)
    parser.add_argument("-by-mutation-out", default=None)
    parser.add_argument("-dom-cov-merge", nargs="+", default=None, metavar="RAW_CSV")
    parser.add_argument(
        "-combine", action="store_true",
        help="csv_paths are dominance_by_dataset.csv files (one per dataset) instead of "
             "pareto_per_trial.csv files; combine them into the single cross-dataset paper table.",
    )
    parser.add_argument("-combine-out-csv", default=None)
    parser.add_argument("-combine-out-tex", default=None)
    args = parser.parse_args(argv)

    if args.combine:
        _out_default_dir = Path("evaluation") / "quantitative" / "rq1" / "dominance_summary"
        out_csv = Path(args.combine_out_csv) if args.combine_out_csv else _out_default_dir / "dominance_frequency_table.csv"
        out_tex = Path(args.combine_out_tex) if args.combine_out_tex else _out_default_dir / "dominance_frequency_table.tex"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        out_tex.parent.mkdir(parents=True, exist_ok=True)

        print(f"Merging {len(args.csv_paths)} dominance_by_dataset.csv file(s) (-combine)...")
        by_dataset_df = load_by_dataset_files(args.csv_paths)
        table, ordered_datasets = build_dominance_frequency_table(by_dataset_df)

        rows = []
        for algo in [a["display"] for a in BASELINE_ALGORITHMS]:
            row = {"Mine": algo}
            for dataset in ordered_datasets:
                val = table[algo][dataset]
                row[DATASET_DISPLAY_NAMES.get(dataset, dataset)] = f"{val:.2f}" if val is not None else "--"
            rows.append(row)
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"Saved combined CSV table to {out_csv}")

        latex_text = build_dominance_frequency_latex(table, ordered_datasets)
        out_tex.write_text(latex_text + chr(10))
        print(f"Saved combined LaTeX table to {out_tex}")

        print("\ngenerated latex table\n")
        print(latex_text)
        return

    _datasets_hint = {infer_dataset(p) for p in args.csv_paths}
    if len(_datasets_hint) == 1:
        _default_dir = Path("evaluation") / "quantitative" / next(iter(_datasets_hint)) / "rq1" / "dominance_summary"
    else:
        _default_dir = Path("evaluation") / "quantitative" / "rq1" / "dominance_summary"
    _default_dir.mkdir(parents=True, exist_ok=True)

    if args.dom_cov_raw_out is None:
        args.dom_cov_raw_out = str(_default_dir / "dom_cov_raw.csv")
    if args.by_dataset_out is None:
        args.by_dataset_out = str(_default_dir / "dominance_by_dataset.csv")
    if args.by_dp_mutation_out is None:
        args.by_dp_mutation_out = str(_default_dir / "dominance_by_decision_point_mutation.csv")
    if args.by_mutation_out is None:
        args.by_mutation_out = str(_default_dir / "dominance_by_mutation.csv")

    if args.dom_cov_merge:
        print(f"Merging {len(args.dom_cov_merge)} -dom-cov-raw-out file(s) (-dom-cov-merge)...")
        multi_algo_df = load_dom_cov_raw(args.dom_cov_merge)
    else:
        print(f"Reading {len(args.csv_paths)} file(s)...")
        multi_algo_df = collect_multi_algorithm_dominance(args.csv_paths, algorithms=BASELINE_ALGORITHMS, tol=args.tol)

    if multi_algo_df.empty:
        raise SystemExit(
            "No usable trials found (missing baseline columns in every file, or -dom-cov-merge "
            "was needed instead)."
        )

    print(f"{multi_algo_df['trial_id'].nunique()} trial(s), across {multi_algo_df['dataset'].nunique()} "
          f"dataset(s), {multi_algo_df['algorithm'].nunique()} baseline algorithm(s).\n")

    for out_path in (Path(args.dom_cov_raw_out), Path(args.by_dataset_out),
                      Path(args.by_dp_mutation_out), Path(args.by_mutation_out)):
        out_path.parent.mkdir(parents=True, exist_ok=True)

    multi_algo_df.to_csv(args.dom_cov_raw_out, index=False)
    print(f"Saved raw per-trial-per-algorithm values to {args.dom_cov_raw_out}")

    by_dataset_df = dominance_by_dataset(multi_algo_df)
    print("\ndominance (%) by dataset x algorithm")
    print(by_dataset_df.to_string(index=False))
    by_dataset_df.to_csv(args.by_dataset_out, index=False)
    print(f"Saved {args.by_dataset_out}")

    by_dp_mut_df = dominance_by_decision_point_mutation(multi_algo_df)
    print("\ndominance (%) by decision point x mutation type x algorithm")
    print(by_dp_mut_df.to_string(index=False) if not by_dp_mut_df.empty else "(no operator column found, skipped)")
    by_dp_mut_df.to_csv(args.by_dp_mutation_out, index=False)
    print(f"Saved {args.by_dp_mutation_out}")

    by_mut_df = dominance_by_mutation(multi_algo_df)
    print("\ndominance (%) by mutation type x algorithm")
    print(by_mut_df.to_string(index=False) if not by_mut_df.empty else "(no operator column found, skipped)")
    by_mut_df.to_csv(args.by_mutation_out, index=False)
    print(f"Saved {args.by_mutation_out}")


if __name__ == "__main__":
    main_dominance_summary()
