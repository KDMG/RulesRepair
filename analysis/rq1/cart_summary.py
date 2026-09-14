import argparse
from pathlib import Path

import pandas as pd

from analysis.core.compare_rulesrepair_cart_dominance import (
    DEFAULT_TOL, DEFAULT_MAX_DEPTH, DEFAULT_NODES_MIN, run_one, _dominates,
)
from analysis.rq1._common import (
    infer_dataset, infer_seed, _expand_and_label_paths, _load_merge_csvs, _quartile_row,
    OPERATOR_ORDER, OPERATOR_DISPLAY_NAMES, DATASET_DISPLAY_NAMES, _dataset_display,
    BASELINE_ALGORITHMS, DOM_COV_RAW_COLUMNS, load_dom_cov_raw, collect_multi_algorithm_dominance,
)


METRIC_COL = "frac_dominated_by_cart_strict"


def collect_trials(csv_paths, tol=DEFAULT_TOL, max_depth=DEFAULT_MAX_DEPTH, nodes_min=DEFAULT_NODES_MIN):
    frames = []
    for path, dataset, seed, decision_point in _expand_and_label_paths(csv_paths):
        result = run_one(path, label=f"{dataset}/{decision_point}", tol=tol, max_depth=max_depth, nodes_min=nodes_min)
        rep_df = result["_rep_df"].copy()
        rep_df["dataset"] = dataset
        rep_df["seed"] = seed
        rep_df["decision_point"] = decision_point
        rep_df["source_csv"] = str(path)
        frames.append(rep_df)

    if not frames:
        raise SystemExit("No usable pareto_per_trial.csv files found among the given paths.")

    combined = pd.concat(frames, ignore_index=True)
    combined[METRIC_COL] = combined["coverage_cart_over_rulesrepair"]
    return combined


def by_dataset_summary(all_trials, col=METRIC_COL):
    rows = [
        _quartile_row(group[col], dataset)
        for dataset, group in all_trials.groupby("dataset")
    ]
    return pd.DataFrame(rows).sort_values("group").reset_index(drop=True)


def by_operator_summary(all_trials, col=METRIC_COL):
    with_op = all_trials.dropna(subset=["operator"])
    if with_op.empty:
        return None
    rows = [
        _quartile_row(group[col], operator)
        for operator, group in with_op.groupby("operator")
    ]
    return pd.DataFrame(rows).sort_values("group").reset_index(drop=True)


def overall_summary(all_trials, col=METRIC_COL):
    return pd.DataFrame([_quartile_row(all_trials[col], "OVERALL")])


def by_decision_point_summary(all_trials, col=METRIC_COL):
    rows = [
        _quartile_row(group[col], dp)
        for dp, group in all_trials.groupby("decision_point")
    ]
    return pd.DataFrame(rows).sort_values("group").reset_index(drop=True)


def collect_trial_diagnostics(csv_paths, tol=DEFAULT_TOL):
    rows = []
    for path, dataset, seed, decision_point in _expand_and_label_paths(csv_paths, quiet=True):
        df = pd.read_csv(path)
        required = [
            "trial_id", "is_pareto", "acc_new_test", "acc_cart_test",
            "total_nodes", "cart_total_nodes", "sim_old_new_jaccard", "sim_old_cart_jaccard",
        ]
        if any(c not in df.columns for c in required):
            continue
        pareto_df = df[df["is_pareto"]]
        for trial_id, group in pareto_df.groupby("trial_id"):
            front = list(zip(
                group["acc_new_test"].astype(float),
                group["total_nodes"].astype(float),
                group["sim_old_new_jaccard"].astype(float),
            ))
            cart_acc = float(group["acc_cart_test"].iloc[0])
            cart_nodes = float(group["cart_total_nodes"].iloc[0])
            cart_jaccard = float(group["sim_old_cart_jaccard"].iloc[0])
            n_dominated = sum(
                1 for (acc, nodes, jaccard) in front
                if _dominates(cart_acc, cart_nodes, cart_jaccard, acc, nodes, jaccard, tol)
            )
            front_size = len(front)
            if "operator_detail" in group.columns:
                operator = group["operator_detail"].iloc[0]
            elif "operator" in group.columns:
                operator = group["operator"].iloc[0]
            else:
                operator = None
            rows.append({
                "dataset": dataset, "seed": seed, "decision_point": decision_point,
                "trial_id": trial_id, "operator": operator,
                "front_size": front_size, "n_dominated": n_dominated,
                "d_i": (n_dominated / front_size) if front_size else float("nan"),
            })
    return pd.DataFrame(rows)


def print_sanity_check(diag_df):
    def _report(df, label):
        n_total = len(df)
        n_nonzero = int((df["d_i"] > 0).sum())
        n_dominated_total = int(df["n_dominated"].sum())
        max_d = float(df["d_i"].max()) if n_total else float("nan")
        print(f"\nSanity check: {label}")
        print(f"  total trials: {n_total}")
        print(f"  trials with D_i > 0: {n_nonzero}")
        print(f"  total RulesRepair solutions strictly dominated by CART: {n_dominated_total}")
        print(f"  max D_i observed: {max_d:.6f}" if n_total else "  max D_i observed: n/a")
        nonzero = df[df["d_i"] > 0].sort_values("d_i", ascending=False)
        if len(nonzero):
            print(f"\n  {'trial_id':<22}{'decision_point':<16}{'operator':<20}{'front_size':<12}{'n_dominated':<13}{'d_i':<10}")
            for _, r in nonzero.iterrows():
                print(
                    f"  {str(r['trial_id']):<22}{str(r['decision_point']):<16}{str(r['operator']):<20}"
                    f"{r['front_size']:<12}{r['n_dominated']:<13}{r['d_i']:<10.4f}"
                )

    if diag_df.empty:
        print("\nSanity check: no diagnosable trials (original pareto_per_trial.csv files "
              "not available - this report needs the raw per-trial files, not -merge output)")
        return

    for dataset, group in diag_df.groupby("dataset"):
        _report(group, dataset)
    _report(diag_df, "OVERALL")


RAW_COLUMNS = ["dataset", "seed", "decision_point", "trial_id", "operator", METRIC_COL]


def save_raw_trials(all_trials, path):
    all_trials[RAW_COLUMNS].to_csv(path, index=False)


def load_raw_trials(csv_paths):
    return _load_merge_csvs(
        csv_paths, RAW_COLUMNS,
        missing_column_hint=(
            "is this really a -raw-out file from a previous run of this script, not a "
            "pareto_per_trial.csv? (pareto_per_trial.csv files need collect_trials(), "
            "i.e. run WITHOUT -merge)."
        ),
        empty_message="No usable raw-trials CSV files found among the given paths.",
    )


def _dominance_max_coverage(d_i_values):
    vals = pd.Series(d_i_values).dropna()
    n = len(vals)
    if n == 0:
        return {"n_trials": 0, "dominance_pct": float("nan"), "max_coverage_pct": float("nan")}
    return {
        "n_trials": n,
        "dominance_pct": float((vals > 0).mean() * 100.0),
        "max_coverage_pct": float(vals.max() * 100.0),
    }


def build_dominance_coverage_table(long_df):
    present_operators = set(long_df["operator"].dropna().unique())
    ordered_operators = [op for op in OPERATOR_ORDER if op in present_operators]
    ordered_operators += sorted(op for op in present_operators if op not in OPERATOR_ORDER)

    rows = []
    for (dataset, algorithm), group in long_df.groupby(["dataset", "algorithm"]):
        row = {"dataset": dataset, "algorithm": algorithm}
        overall = _dominance_max_coverage(group["d_i"])
        row["n_trials_overall"] = overall["n_trials"]
        row["dominance_overall"] = overall["dominance_pct"]
        row["max_coverage_overall"] = overall["max_coverage_pct"]
        for op in ordered_operators:
            stats = _dominance_max_coverage(group.loc[group["operator"] == op, "d_i"])
            row[f"n_trials_{op}"] = stats["n_trials"]
            row[f"dominance_{op}"] = stats["dominance_pct"]
            row[f"max_coverage_{op}"] = stats["max_coverage_pct"]
        rows.append(row)
    return pd.DataFrame(rows), ordered_operators


def print_dominance_coverage_table(table_df, ordered_operators):
    if table_df.empty:
        print("\n(no rows to display)")
        return
    present_datasets = list(table_df["dataset"].dropna().unique())
    ordered_datasets = [d for d in DATASET_DISPLAY_NAMES if d in present_datasets]
    ordered_datasets += sorted(d for d in present_datasets if d not in DATASET_DISPLAY_NAMES)

    col_keys = ["overall"] + ordered_operators
    col_labels = ["Overall"] + [OPERATOR_DISPLAY_NAMES.get(op, op.replace("_", " ").title()) for op in ordered_operators]

    dataset_col_width = max(len("Dataset"), max((len(_dataset_display(d)) for d in ordered_datasets), default=0)) + 2

    print()
    header1 = f"{'Dataset':<{dataset_col_width}}{'Mine':<9}"
    header2 = f"{'':<{dataset_col_width}}{'':<9}"
    for label in col_labels:
        header1 += f"{label:<22}"
        header2 += f"{'Dom.(%)':<11}{'Max cov.(%)':<11}"
    print(header1)
    print(header2)

    for dataset in ordered_datasets:
        d_rows = table_df[table_df["dataset"] == dataset]
        for algo in BASELINE_ALGORITHMS:
            algo_rows = d_rows[d_rows["algorithm"] == algo["display"]]
            if algo_rows.empty:
                continue
            r = algo_rows.iloc[0]
            line = f"{_dataset_display(dataset):<{dataset_col_width}}{algo['display']:<9}"
            for key in col_keys:
                dom = r.get(f"dominance_{key}", float("nan"))
                cov = r.get(f"max_coverage_{key}", float("nan"))
                dom_s = f"{dom:.2f}" if pd.notna(dom) else "-"
                cov_s = f"{cov:.2f}" if pd.notna(cov) else "-"
                line += f"{dom_s:<11}{cov_s:<11}"
            print(line)


def dominance_coverage_latex_table(
        table_df, ordered_operators,
        caption="Percentage of times in which \textsc{Mine} dominates at least one solution of the Pareto front "
                "produced by \textsc{RulesRepair}. We do not report the percentage of fronts entirely dominated by "
                "\textsc{Mine}, as it is always zero."):
    present_datasets = list(table_df["dataset"].dropna().unique())
    ordered_datasets = [d for d in DATASET_DISPLAY_NAMES if d in present_datasets]
    ordered_datasets += sorted(d for d in present_datasets if d not in DATASET_DISPLAY_NAMES)

    col_keys = ["overall"] + ordered_operators
    col_labels = ["Overall"] + [OPERATOR_DISPLAY_NAMES.get(op, op.replace("_", " ").title()) for op in ordered_operators]
    n_groups = len(col_keys)

    backslash = chr(92)
    row_end = backslash + backslash

    col_spec = "ll" + "cc" * n_groups
    top_header_cells = [backslash + "multicolumn{2}{c}{" + label + "}" for label in col_labels]
    top_header = " & & " + " & ".join(top_header_cells) + " " + row_end
    cmidrules = " ".join(
        backslash + "cmidrule(lr){" + str(3 + 2 * i) + "-" + str(4 + 2 * i) + "}"
        for i in range(n_groups)
    )
    sub_header = "Dataset & Mine & " + " & ".join(["Dom.(" + backslash + "%) & Max cov.(" + backslash + "%)"] * n_groups) + " " + row_end

    lines = []
    lines.append(backslash + "begin{table}[tb]")
    lines.append(backslash + "centering")
    lines.append(backslash + "resizebox{" + backslash + "textwidth}{!}{%")
    lines.append(backslash + "begin{tabular}{" + col_spec + "}")
    lines.append(backslash + "toprule")
    lines.append(top_header)
    lines.append(cmidrules)
    lines.append(sub_header)
    lines.append(backslash + "midrule")

    for di, dataset in enumerate(ordered_datasets):
        if di > 0:
            lines.append(backslash + "addlinespace")
        d_rows = table_df[table_df["dataset"] == dataset]
        display_name = _dataset_display(dataset)
        algo_rows_present = [a for a in BASELINE_ALGORITHMS if not d_rows[d_rows["algorithm"] == a["display"]].empty]
        for ai, algo in enumerate(algo_rows_present):
            r = d_rows[d_rows["algorithm"] == algo["display"]].iloc[0]
            cells = []
            for key in col_keys:
                dom = r.get(f"dominance_{key}", float("nan"))
                cov = r.get(f"max_coverage_{key}", float("nan"))
                dom_s = f"{dom:.2f}" if pd.notna(dom) else "-"
                cov_s = f"{cov:.2f}" if pd.notna(cov) else "-"
                cells.append(dom_s)
                cells.append(cov_s)
            dataset_cell = (backslash + "multirow{" + str(len(algo_rows_present)) + "}{*}{"
                             + backslash + "textit{" + display_name + "}}") if ai == 0 else ""
            lines.append(dataset_cell + " & " + algo["display"] + " & " + " & ".join(cells) + " " + row_end)
    lines.append(backslash + "bottomrule")
    lines.append(backslash + "end{tabular}")
    lines.append("}")
    lines.append(backslash + "vspace{2pt}")
    lines.append(backslash + "caption{" + caption + "}")
    lines.append(backslash + "label{" + label + "}")
    lines.append(backslash + "end{table}")
    return chr(10).join(lines)


def _format_median_iqr(vals, as_percent=True, decimals=1):
    vals = pd.Series(vals).dropna()
    if len(vals) == 0:
        return "-"
    scale = 100.0 if as_percent else 1.0
    q1, med, q3 = (vals * scale).quantile([0.25, 0.5, 0.75])
    return f"{med:.{decimals}f} [{q1:.{decimals}f}, {q3:.{decimals}f}]"


def dominance_probability_latex_table(
        all_trials, col=METRIC_COL, as_percent=True, decimals=1,
        caption="Percentage of times in which \textsc{Mine} dominates at least one solution of the Pareto front "
                "produced by \textsc{RulesRepair}. We do not report the percentage of fronts entirely dominated by "
                "\textsc{Mine}, as it is always zero.",
        label="tab:cart_point_dominance"):
    present_datasets = list(all_trials["dataset"].dropna().unique())
    ordered_datasets = [d for d in DATASET_DISPLAY_NAMES if d in present_datasets]
    ordered_datasets += sorted(d for d in present_datasets if d not in DATASET_DISPLAY_NAMES)

    present_operators = set(all_trials["operator"].dropna().unique())
    ordered_operators = [op for op in OPERATOR_ORDER if op in present_operators]
    ordered_operators += sorted(op for op in present_operators if op not in OPERATOR_ORDER)

    n_cols = 1 + 1 + len(ordered_operators)
    col_spec = "l" + "c" * (n_cols - 1)

    header_ops = " & ".join(OPERATOR_DISPLAY_NAMES.get(op, op.replace("_", " ").title()) for op in ordered_operators)

    backslash = chr(92)
    row_end = backslash + backslash

    lines = []
    lines.append(backslash + "begin{table}[tb]")
    lines.append(backslash + "centering")
    lines.append(backslash + "Huge")
    lines.append(backslash + "resizebox{" + backslash + "textwidth}{!}{%")
    lines.append(backslash + "begin{tabular}{" + col_spec + "}")
    lines.append(backslash + "toprule")
    lines.append(f"Dataset & Overall & {header_ops} {row_end}")
    lines.append(backslash + "midrule")
    for dataset in ordered_datasets:
        d_rows = all_trials[all_trials["dataset"] == dataset]
        overall_cell = _format_median_iqr(d_rows[col], as_percent, decimals)
        op_cells = []
        for op in ordered_operators:
            op_rows = d_rows[d_rows["operator"] == op]
            op_cells.append(_format_median_iqr(op_rows[col], as_percent, decimals))
        display_name = _dataset_display(dataset)
        lines.append(
            backslash + "textit{" + display_name + "} & " + overall_cell + " & " + " & ".join(op_cells) + " " + row_end
        )
    lines.append(backslash + "bottomrule")
    lines.append(backslash + "end{tabular}")
    lines.append("}")
    lines.append(backslash + "vspace{2pt}")
    lines.append(backslash + "caption{" + caption + "}")
    lines.append(backslash + "label{" + label + "}")
    lines.append(backslash + "end{table}")
    return chr(10).join(lines)


def _main_latex_hook(all_trials, latex_out, col=METRIC_COL):
    table_text = dominance_probability_latex_table(all_trials, col)
    Path(latex_out).write_text(table_text + chr(10))
    print(chr(10) + "Saved LaTeX table to " + str(latex_out))


def print_summary_table(df, title):
    print(f"\n{title}")
    print(f"{'group':<22}{'n_trials':<10}{'median':<10}{'q1':<10}{'q3':<10}{'iqr':<10}{'mean':<10}")
    for _, r in df.iterrows():
        print(
            f"{str(r['group']):<22}{r['n_trials']:<10}"
            f"{r['median']:<10.4f}{r['q1']:<10.4f}{r['q3']:<10.4f}{r['iqr']:<10.4f}{r['mean']:<10.4f}"
        )


def main_cart_summary(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv_paths", nargs="+")
    parser.add_argument("-tol", type=float, default=DEFAULT_TOL)
    parser.add_argument("-max-depth", type=int, default=DEFAULT_MAX_DEPTH)
    parser.add_argument("-nodes-min", type=int, default=DEFAULT_NODES_MIN)
    parser.add_argument("-out-csv", default=None)
    parser.add_argument("-raw-out", default=None)
    parser.add_argument("-merge", action="store_true")
    parser.add_argument("-latex-out", default=None)
    parser.add_argument("-dom-cov-raw-out", default=None)
    parser.add_argument("-dom-cov-csv-out", default=None)
    parser.add_argument("-dom-cov-latex-out", default=None)
    parser.add_argument("-dom-cov-merge", nargs="+", default=None, metavar="RAW_CSV")
    args = parser.parse_args(argv)

    _default_dir = Path("quantitative_evaluation") / "rq1" / "cart_summary"
    _default_dir.mkdir(parents=True, exist_ok=True)
    if args.out_csv is None:
        args.out_csv = str(_default_dir / "combined_summary.csv")
    if args.raw_out is None:
        args.raw_out = str(_default_dir / "raw_trials.csv")
    if args.latex_out is None:
        args.latex_out = str(_default_dir / "summary.tex")
    if args.dom_cov_raw_out is None:
        args.dom_cov_raw_out = str(_default_dir / "dom_cov_raw.csv")
    if args.dom_cov_csv_out is None:
        args.dom_cov_csv_out = str(_default_dir / "dom_cov_table.csv")
    if args.dom_cov_latex_out is None:
        args.dom_cov_latex_out = str(_default_dir / "dom_cov_table.tex")

    if args.merge:
        print(f"Merging {len(args.csv_paths)} raw-trials file(s) (-merge)...")
        all_trials = load_raw_trials(args.csv_paths)
    else:
        print(f"Reading {len(args.csv_paths)} file(s)...")
        all_trials = collect_trials(args.csv_paths, tol=args.tol, max_depth=args.max_depth, nodes_min=args.nodes_min)
    print(f"{len(all_trials)} trial(s) total, across {all_trials['dataset'].nunique()} dataset(s).\n")

    if args.raw_out:
        save_raw_trials(all_trials, args.raw_out)
        print(f"Saved raw per-trial values to {args.raw_out} (combine this with other datasets' -raw-out "
              f"files later via -merge).\n")

    col = METRIC_COL
    print(f"Metric: {col}, per trial, D_i = the fraction of that trial's own RulesRepair "
          f"Pareto-front points that are strictly dominated by CART's single point "
          f"(CART not worse on any objective and strictly better on at least one, "
          f"see _dominates() in compare_rulesrepair_cart_dominance.py). Exact ties (identical "
          f"objective vector) are explicitly not counted as dominated. This is exactly "
          f"\"if I pick one of my algorithm's front points, what's the probability CART "
          f"was actually better\", per trial.")

    dataset_df = by_dataset_summary(all_trials, col)
    print_summary_table(dataset_df, "by dataset")

    operator_df = by_operator_summary(all_trials, col)
    if operator_df is not None:
        print_summary_table(operator_df, "by mutation type (operator, pooled across datasets)")
    else:
        print("\nby mutation type: no 'operator'/'operator_detail' column found on any input "
              "CSV (none of these are Scenario 3 tree-mutation runs), skipped.")

    overall_df = overall_summary(all_trials, col)
    print_summary_table(overall_df, "overall (every trial, every dataset/operator pooled)")

    dp_df = by_decision_point_summary(all_trials, col)
    print_summary_table(dp_df, "by decision point (pooled across seeds/operators)")

    if args.merge:
        print("\nSanity check: skipped (-merge input carries only the aggregated "
              "per-trial fraction, not the original front points needed to recompute "
              "front_size/n_dominated)")
    else:
        diag_df = collect_trial_diagnostics(args.csv_paths, tol=args.tol)
        if not diag_df.empty:
            merged_check = diag_df.merge(
                all_trials[["dataset", "seed", "decision_point", "trial_id", col]],
                on=["dataset", "seed", "decision_point", "trial_id"], how="inner",
            )
            mismatches = merged_check[(merged_check["d_i"] - merged_check[col]).abs() > args.tol]
            if len(mismatches):
                print(f"\nwarning: sanity-check cross-check found {len(mismatches)} trial(s) where the "
                      f"independently-recomputed d_i disagrees with {col}, investigate before trusting "
                      f"either the RQ1 tables above or this sanity check.")
            else:
                print(f"\nsanity-check cross-check ok: independently-recomputed d_i matches {col} "
                      f"exactly on all {len(merged_check)} matched trial(s).")
        print_sanity_check(diag_df)

    if args.dom_cov_merge:
        print(f"\nMerging {len(args.dom_cov_merge)} -dom-cov-raw-out file(s) (-dom-cov-merge)...")
        multi_algo_df = load_dom_cov_raw(args.dom_cov_merge)
    elif args.merge:
        print("\ndataset x mine-algorithm dominance/max-coverage table: skipped (-merge "
              "input doesn't carry the original front points needed to recompute it, pass "
              "-dom-cov-merge with previously-saved -dom-cov-raw-out files instead)")
        multi_algo_df = pd.DataFrame()
    else:
        multi_algo_df = collect_multi_algorithm_dominance(args.csv_paths, tol=args.tol)

    if multi_algo_df.empty:
        print("\ndataset x mine-algorithm dominance/max-coverage table: no usable trials found "
              "(missing baseline columns in every file, or -merge was passed without "
              "-dom-cov-merge), skipped.")
    else:
        dom_cov_table, dom_cov_ops = build_dominance_coverage_table(multi_algo_df)
        print("\ndominance (%) / max coverage (%) by dataset x mine algorithm x mutation type")
        print_dominance_coverage_table(dom_cov_table, dom_cov_ops)

        if args.dom_cov_raw_out:
            multi_algo_df.to_csv(args.dom_cov_raw_out, index=False)
            print(f"\nSaved raw per-trial-per-algorithm values to {args.dom_cov_raw_out}")
        if args.dom_cov_csv_out:
            dom_cov_table.to_csv(args.dom_cov_csv_out, index=False)
            print(f"Saved aggregated Dominance/Max-coverage table to {args.dom_cov_csv_out}")
        if args.dom_cov_latex_out:
            latex_text = dominance_coverage_latex_table(dom_cov_table, dom_cov_ops)
            Path(args.dom_cov_latex_out).write_text(latex_text + chr(10))
            print(f"Saved LaTeX Dominance/Max-coverage table to {args.dom_cov_latex_out}")

    if args.out_csv:
        combined = pd.concat([
            dataset_df.assign(scope="dataset"),
            (operator_df.assign(scope="operator") if operator_df is not None else pd.DataFrame()),
            overall_df.assign(scope="overall"),
            dp_df.assign(scope="decision_point"),
        ], ignore_index=True)
        combined.to_csv(args.out_csv, index=False)
        print(f"\nSaved combined summary (dataset + operator + overall + decision_point rows) to {args.out_csv}")

    if args.latex_out:
        _main_latex_hook(all_trials, args.latex_out, col)
