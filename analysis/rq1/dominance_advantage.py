import argparse
from pathlib import Path

import pandas as pd

from analysis.core.compare_rulesrepair_cart_dominance import DEFAULT_TOL, DEFAULT_MAX_DEPTH, _dominates
from analysis.rq1._common import (
    infer_dataset, _expand_and_label_paths, _load_merge_csvs,
    DATASET_DISPLAY_NAMES, DATASET_COLUMN_ORDER, BASELINE_ALGORITHMS,
)


N_MAX_NODES = 2 ** (DEFAULT_MAX_DEPTH + 1) - 1

assert N_MAX_NODES == 31, f"N_MAX_NODES expected to be 31 (max_depth={DEFAULT_MAX_DEPTH}), got {N_MAX_NODES}"


METRIC_KEYS = ["accuracy", "simplicity", "similarity"]
METRIC_SUBSCRIPTS = {"accuracy": "fit", "simplicity": "simp", "similarity": "simi"}


RAW_COLUMNS = [
    "dataset", "mine_algorithm", "trial_id", "seed", "decision_point", "mutation_type",
    "repair_solution_id",
    "mine_accuracy", "repair_accuracy", "delta_accuracy",
    "mine_similarity", "repair_similarity", "delta_similarity",
    "mine_total_nodes", "repair_total_nodes",
    "mine_simplicity", "repair_simplicity", "delta_simplicity",
]


def collect_dominated_solution_deltas(csv_paths, algorithms=BASELINE_ALGORITHMS, tol=DEFAULT_TOL,
                                       n_max_nodes=N_MAX_NODES):
    rows = []
    warned = set()
    for path, dataset, seed, decision_point in _expand_and_label_paths(csv_paths, quiet=True):
        df = pd.read_csv(path)
        base_required = [
            "trial_id", "is_pareto", "acc_new_test", "total_nodes", "sim_old_new_jaccard",
            "w_simp", "w_simi", "t",
        ]
        if any(c not in df.columns for c in base_required):
            continue
        pareto_df = df[df["is_pareto"]]
        for trial_id, group in pareto_df.groupby("trial_id"):
            group = group.reset_index(drop=True)
            if "operator_detail" in group.columns:
                operator = group["operator_detail"].iloc[0]
            elif "operator" in group.columns:
                operator = group["operator"].iloc[0]
            else:
                operator = None

            for algo in algorithms:
                needed = [algo["acc_col"], algo["nodes_col"], algo["jaccard_col"]]
                missing = [c for c in needed if c not in group.columns]
                if missing:
                    warn_key = (str(path), algo["key"])
                    if warn_key not in warned:
                        print(f"  Warning: {path}: missing column(s) {missing} for algorithm "
                              f"{algo['display']}, skipping.")
                        warned.add(warn_key)
                    continue
                a_acc = float(group[algo["acc_col"]].iloc[0])
                a_nodes = float(group[algo["nodes_col"]].iloc[0])
                a_jaccard = float(group[algo["jaccard_col"]].iloc[0])
                a_simplicity = 1.0 - a_nodes / n_max_nodes

                for _, r in group.iterrows():
                    r_acc = float(r["acc_new_test"])
                    r_nodes = float(r["total_nodes"])
                    r_jaccard = float(r["sim_old_new_jaccard"])
                    if not _dominates(a_acc, a_nodes, a_jaccard, r_acc, r_nodes, r_jaccard, tol):
                        continue
                    r_simplicity = 1.0 - r_nodes / n_max_nodes
                    repair_solution_id = f"w_simp={r['w_simp']}_w_simi={r['w_simi']}_t={r['t']}"
                    rows.append({
                        "dataset": dataset, "mine_algorithm": algo["display"],
                        "trial_id": trial_id, "seed": seed, "decision_point": decision_point,
                        "mutation_type": operator, "repair_solution_id": repair_solution_id,
                        "mine_accuracy": a_acc, "repair_accuracy": r_acc,
                        "delta_accuracy": a_acc - r_acc,
                        "mine_similarity": a_jaccard, "repair_similarity": r_jaccard,
                        "delta_similarity": a_jaccard - r_jaccard,
                        "mine_total_nodes": a_nodes, "repair_total_nodes": r_nodes,
                        "mine_simplicity": a_simplicity, "repair_simplicity": r_simplicity,
                        "delta_simplicity": a_simplicity - r_simplicity,
                    })
    return pd.DataFrame(rows, columns=RAW_COLUMNS)


PER_TRIAL_COLUMNS = [
    "dataset", "mine_algorithm", "trial_id", "seed", "decision_point", "mutation_type",
    "n_dominated",
    "median_delta_accuracy", "median_delta_similarity", "median_delta_simplicity",
]


def build_dominated_deltas_per_trial(raw_df):
    if raw_df.empty:
        return pd.DataFrame(columns=PER_TRIAL_COLUMNS)
    group_cols = ["dataset", "mine_algorithm", "trial_id", "seed", "decision_point", "mutation_type"]
    rows = []
    for keys, group in raw_df.groupby(group_cols, dropna=False, sort=False):
        row = dict(zip(group_cols, keys))
        row["n_dominated"] = len(group)
        row["median_delta_accuracy"] = float(group["delta_accuracy"].median())
        row["median_delta_similarity"] = float(group["delta_similarity"].median())
        row["median_delta_simplicity"] = float(group["delta_simplicity"].median())
        rows.append(row)
    return pd.DataFrame(rows, columns=PER_TRIAL_COLUMNS)


def load_per_trial(csv_paths):
    return _load_merge_csvs(
        csv_paths, PER_TRIAL_COLUMNS,
        missing_column_hint="is this really a -per-trial-out file from a previous run of this script?",
        empty_message="No usable -per-trial-out CSV files found among the given paths.",
    )


def _aggregate_deltas(group):
    row = {"n_trials": len(group)}
    for key in METRIC_KEYS:
        vals = group[f"median_delta_{key}"].dropna()
        if vals.empty:
            row[f"median_delta_{key}"] = float("nan")
            row[f"q1_delta_{key}"] = float("nan")
            row[f"q3_delta_{key}"] = float("nan")
            continue
        q1, med, q3 = vals.quantile([0.25, 0.5, 0.75])
        row[f"median_delta_{key}"] = float(med)
        row[f"q1_delta_{key}"] = float(q1)
        row[f"q3_delta_{key}"] = float(q3)
    return row


_STAT_COLUMNS = ["n_trials"] + [
    f"{stat}_delta_{key}" for key in METRIC_KEYS for stat in ("median", "q1", "q3")
]


def advantage_by_dataset(per_trial_df):
    rows = []
    for (dataset, algorithm), group in per_trial_df.groupby(["dataset", "mine_algorithm"]):
        rows.append({"dataset": dataset, "algorithm": algorithm, **_aggregate_deltas(group)})
    if not rows:
        return pd.DataFrame(columns=["dataset", "algorithm"] + _STAT_COLUMNS)
    return pd.DataFrame(rows).sort_values(["dataset", "algorithm"]).reset_index(drop=True)


def advantage_by_decision_point_mutation(per_trial_df):
    with_op = per_trial_df.dropna(subset=["mutation_type"])
    rows = []
    for (dataset, algorithm, decision_point, mutation_type), group in with_op.groupby(
            ["dataset", "mine_algorithm", "decision_point", "mutation_type"]):
        rows.append({
            "dataset": dataset, "algorithm": algorithm,
            "decision_point": decision_point, "operator": mutation_type,
            **_aggregate_deltas(group),
        })
    if not rows:
        return pd.DataFrame(columns=["dataset", "algorithm", "decision_point", "operator"] + _STAT_COLUMNS)
    return pd.DataFrame(rows).sort_values(
        ["dataset", "algorithm", "decision_point", "operator"]
    ).reset_index(drop=True)


def advantage_by_mutation(per_trial_df):
    with_op = per_trial_df.dropna(subset=["mutation_type"])
    rows = []
    for (dataset, algorithm, mutation_type), group in with_op.groupby(["dataset", "mine_algorithm", "mutation_type"]):
        rows.append({"dataset": dataset, "algorithm": algorithm, "operator": mutation_type, **_aggregate_deltas(group)})
    if not rows:
        return pd.DataFrame(columns=["dataset", "algorithm", "operator"] + _STAT_COLUMNS)
    return pd.DataFrame(rows).sort_values(["dataset", "algorithm", "operator"]).reset_index(drop=True)


def load_by_dataset_files(csv_paths):
    frames = []
    path_datasets = set()
    for p in csv_paths:
        df = pd.read_csv(p)
        missing = [c for c in ("dataset", "algorithm") if c not in df.columns]
        if missing:
            raise SystemExit(f"{p}: missing column(s) {missing} -- is this really a "
                              f"dominance_advantage_by_dataset.csv file from a previous run?")
        frames.append(df)
        inferred = _infer_dataset_from_output_path(p)
        if inferred:
            path_datasets.add(inferred)
    if not frames:
        raise SystemExit("No usable dominance_advantage_by_dataset.csv files found among the given paths.")
    non_empty = [f for f in frames if not f.empty]
    combined = pd.concat(non_empty, ignore_index=True) if non_empty else frames[0]
    return combined, path_datasets


def _infer_dataset_from_output_path(path):
    parts = Path(path).parts
    if "quantitative" in parts:
        idx = parts.index("quantitative")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return None


def build_dominance_magnitude_table(by_dataset_df, extra_datasets=None):
    present_datasets = set(by_dataset_df["dataset"].dropna().unique())
    if extra_datasets:
        present_datasets |= set(extra_datasets)
    ordered_datasets = [d for d in DATASET_COLUMN_ORDER if d in present_datasets]
    ordered_datasets += sorted(d for d in present_datasets if d not in DATASET_COLUMN_ORDER)

    algo_displays = [a["display"] for a in BASELINE_ALGORITHMS]
    table = {}
    for algo in algo_displays:
        table[algo] = {}
        for metric_key in METRIC_KEYS:
            table[algo][metric_key] = {}
            for dataset in ordered_datasets:
                rows = by_dataset_df[(by_dataset_df["dataset"] == dataset) & (by_dataset_df["algorithm"] == algo)]
                if rows.empty:
                    table[algo][metric_key][dataset] = None
                    continue
                r = rows.iloc[0]
                med = r[f"median_delta_{metric_key}"]
                q1 = r[f"q1_delta_{metric_key}"]
                q3 = r[f"q3_delta_{metric_key}"]
                table[algo][metric_key][dataset] = None if pd.isna(med) else (float(med), float(q1), float(q3))
    return table, ordered_datasets


def build_dominance_magnitude_latex(table, ordered_datasets):
    backslash = chr(92)
    row_end = backslash + backslash
    newline = chr(10)
    dataset_labels = [DATASET_DISPLAY_NAMES.get(d, d) for d in ordered_datasets]
    col_spec = "ll" + "c" * len(ordered_datasets)

    lines = []
    lines.append(backslash + "begin{table}[tb]")
    lines.append(backslash + "centering")
    lines.append(backslash + "large")
    lines.append(backslash + "resizebox{" + backslash + "textwidth}{!}{%")
    lines.append(backslash + "begin{tabular}{" + col_spec + "}")
    lines.append(backslash + "toprule")
    lines.append(backslash + "textsc{Mine} & Gap")
    for i, label in enumerate(dataset_labels):
        suffix = " " + row_end if i == len(dataset_labels) - 1 else ""
        lines.append("& " + backslash + "textit{" + label + "}" + suffix)
    lines.append(backslash + "midrule")

    algo_displays = [a["display"] for a in BASELINE_ALGORITHMS]
    for ai, algo in enumerate(algo_displays):
        if ai > 0:
            lines.append(backslash + "midrule")
        lines.append(backslash + "multirow{3}{*}{" + algo + "}")
        for metric_key in METRIC_KEYS:
            subscript = METRIC_SUBSCRIPTS[metric_key]
            cells = []
            for dataset in ordered_datasets:
                cell = table[algo][metric_key][dataset]
                cells.append(f"{cell[0]:.2f} [{cell[1]:.2f}, {cell[2]:.2f}]" if cell is not None else "--")
            lines.append("& $" + backslash + "Delta_{" + subscript + "}$ & " + " & ".join(cells) + " " + row_end)

    lines.append(backslash + "bottomrule")
    lines.append(backslash + "end{tabular}%")
    lines.append("}")
    caption = (
        "Gaps in accuracy ($" + backslash + "Delta_{fit}$), simplicity ($" + backslash + "Delta_{simp}$), "
        "and similarity ($" + backslash + "Delta_{simi}$) for " + backslash + "textsc{RulesRepair} solutions "
        "dominated by " + backslash + "textsc{Mine} (Table " + backslash + "ref{tab:dominance_frequency}). "
        "-- indicates no trial in which Mine dominated RulesRepair."
    )
    lines.append(backslash + "caption{" + caption + "}")
    lines.append(backslash + "label{tab:dominance_magnitude}")
    lines.append(backslash + "end{table}")
    return newline.join(lines)


def main_dominance_advantage(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv_paths", nargs="+")
    parser.add_argument("-tol", type=float, default=DEFAULT_TOL)
    parser.add_argument("-max-depth", type=int, default=DEFAULT_MAX_DEPTH)
    parser.add_argument("-per-trial-out", default=None)
    parser.add_argument("-by-dataset-out", default=None)
    parser.add_argument("-by-dp-mutation-out", default=None)
    parser.add_argument("-by-mutation-out", default=None)
    parser.add_argument("-merge", nargs="+", default=None, metavar="PER_TRIAL_CSV")
    parser.add_argument(
        "-combine", action="store_true",
        help="csv_paths are dominance_advantage_by_dataset.csv files (one per dataset) instead of "
             "pareto_per_trial.csv files; combine them into the single cross-dataset paper table.",
    )
    parser.add_argument("-combine-out-csv", default=None)
    parser.add_argument("-combine-out-tex", default=None)
    args = parser.parse_args(argv)

    if args.combine:
        _out_default_dir = Path("evaluation") / "quantitative" / "rq1" / "dominance_advantage"
        out_csv = Path(args.combine_out_csv) if args.combine_out_csv else _out_default_dir / "dominance_magnitude_table.csv"
        out_tex = Path(args.combine_out_tex) if args.combine_out_tex else _out_default_dir / "dominance_magnitude_table.tex"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        out_tex.parent.mkdir(parents=True, exist_ok=True)

        print(f"Merging {len(args.csv_paths)} dominance_advantage_by_dataset.csv file(s) (-combine)...")
        by_dataset_df, path_datasets = load_by_dataset_files(args.csv_paths)
        table, ordered_datasets = build_dominance_magnitude_table(by_dataset_df, extra_datasets=path_datasets)

        rows = []
        for algo in [a["display"] for a in BASELINE_ALGORITHMS]:
            for metric_key in METRIC_KEYS:
                row = {"Mine": algo, "Gap": METRIC_SUBSCRIPTS[metric_key]}
                for dataset in ordered_datasets:
                    cell = table[algo][metric_key][dataset]
                    row[DATASET_DISPLAY_NAMES.get(dataset, dataset)] = (
                        f"{cell[0]:.2f} [{cell[1]:.2f}, {cell[2]:.2f}]" if cell is not None else "--"
                    )
                rows.append(row)
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"Saved combined CSV table to {out_csv}")

        latex_text = build_dominance_magnitude_latex(table, ordered_datasets)
        out_tex.write_text(latex_text + chr(10))
        print(f"Saved combined LaTeX table to {out_tex}")

        print("\ngenerated latex table\n")
        print(latex_text)
        return

    _datasets_hint = {infer_dataset(p) for p in args.csv_paths}
    if len(_datasets_hint) == 1:
        _default_dir = Path("evaluation") / "quantitative" / next(iter(_datasets_hint)) / "rq1" / "dominance_advantage"
    else:
        _default_dir = Path("evaluation") / "quantitative" / "rq1" / "dominance_advantage"
    _default_dir.mkdir(parents=True, exist_ok=True)

    if args.per_trial_out is None:
        args.per_trial_out = str(_default_dir / "per_trial.csv")
    if args.by_dataset_out is None:
        args.by_dataset_out = str(_default_dir / "dominance_advantage_by_dataset.csv")
    if args.by_dp_mutation_out is None:
        args.by_dp_mutation_out = str(_default_dir / "dominance_advantage_by_decision_point_mutation.csv")
    if args.by_mutation_out is None:
        args.by_mutation_out = str(_default_dir / "dominance_advantage_by_mutation.csv")

    n_max_nodes = 2 ** (args.max_depth + 1) - 1

    if args.merge:
        print(f"Merging {len(args.merge)} -per-trial-out file(s) (-merge)...")
        per_trial_df = load_per_trial(args.merge)
    else:
        print(f"Reading {len(args.csv_paths)} file(s)...")
        raw_df = collect_dominated_solution_deltas(args.csv_paths, tol=args.tol, n_max_nodes=n_max_nodes)
        per_trial_df = build_dominated_deltas_per_trial(raw_df)

    print(f"{len(per_trial_df)} trial x algorithm combination(s) with >=1 dominated solution.")
    print(f"Simplicity = 1 - total_nodes / {n_max_nodes} (max_depth={args.max_depth}).\n")

    for out_path in (Path(args.per_trial_out), Path(args.by_dataset_out),
                      Path(args.by_dp_mutation_out), Path(args.by_mutation_out)):
        out_path.parent.mkdir(parents=True, exist_ok=True)

    per_trial_df.to_csv(args.per_trial_out, index=False)
    print(f"Saved per-trial median deltas to {args.per_trial_out}")

    by_dataset_df = advantage_by_dataset(per_trial_df)
    print("\ndelta accuracy/similarity/simplicity (median [q1, q3]) by dataset x algorithm")
    print(by_dataset_df.to_string(index=False))
    by_dataset_df.to_csv(args.by_dataset_out, index=False)
    print(f"Saved {args.by_dataset_out}")

    by_dp_mut_df = advantage_by_decision_point_mutation(per_trial_df)
    print("\ndelta accuracy/similarity/simplicity (median [q1, q3]) by decision point x mutation type x algorithm")
    print(by_dp_mut_df.to_string(index=False) if not by_dp_mut_df.empty else "(no operator column found, skipped)")
    by_dp_mut_df.to_csv(args.by_dp_mutation_out, index=False)
    print(f"Saved {args.by_dp_mutation_out}")

    by_mut_df = advantage_by_mutation(per_trial_df)
    print("\ndelta accuracy/similarity/simplicity (median [q1, q3]) by mutation type x algorithm")
    print(by_mut_df.to_string(index=False) if not by_mut_df.empty else "(no operator column found, skipped)")
    by_mut_df.to_csv(args.by_mutation_out, index=False)
    print(f"Saved {args.by_mutation_out}")


if __name__ == "__main__":
    main_dominance_advantage()
