import argparse
import re
import glob as glob_module
from pathlib import Path

import pandas as pd

from analysis.core.compare_rulesrepair_cart_dominance import DEFAULT_TOL, coverage_classical
from analysis.rq1._common import _load_merge_csvs, _quartile_row


BASELINE_PREFIXES = ["cart", "j48", "reptree"]


BASELINE_DISPLAY_NAMES = {"cart": "CART", "j48": "J48", "reptree": "REPTree"}


MB_OPERATOR_ORDER = [
    "prune", "change_label", "branch_swap",
    "change_threshold", "change_feature",
    "regrow_leaf", "regrow_internal",
]


MB_OPERATOR_DISPLAY_NAMES = {
    "prune": "Prune",
    "change_label": "Change label",
    "branch_swap": "Branch swap",
    "change_threshold": "Change threshold",
    "change_feature": "Change feature",
    "regrow_leaf": "Regrow leaf",
    "regrow_internal": "Regrow internal",
}


MB_DATASET_DISPLAY_NAMES = {
    "sepsis": "Sepsis",
    "road_traffic": "Road Traffic Fine",
    "production": "Production",
    "prepaid_travel_costs": "Prepaid Travel Costs",
    "hospital_billing": "Hospital Billing",
    "international_declarations": "International Declarations",
}


_MB_DATASET_RE = re.compile(r"experiments[/\\]([^/\\]+)[/\\]repair[/\\]")


_MB_SEED_RE = re.compile(r"seed_([^/\\]+)[/\\]")


MB_RAW_COLUMNS = ["dataset", "seed", "decision_point", "trial_id", "operator", "baseline", "coverage_classical_over_rulesrepair"]


def mb_infer_dataset(csv_path):
    m = _MB_DATASET_RE.search(str(csv_path))
    return m.group(1) if m else None


def mb_infer_seed(csv_path):
    m = _MB_SEED_RE.search(str(csv_path))
    return m.group(1) if m else None


def _baseline_columns(prefix):
    return f"acc_{prefix}_test", f"{prefix}_total_nodes", f"sim_old_{prefix}_jaccard"


def mb_collect_trials(csv_paths, baselines=BASELINE_PREFIXES, tol=DEFAULT_TOL):
    expanded_paths = []
    for raw_path in csv_paths:
        matches = sorted(glob_module.glob(str(raw_path)))
        expanded_paths.extend(matches if matches else [raw_path])

    rows = []
    n_unmatched = 0
    n_skipped_baseline = 0
    for raw_path in expanded_paths:
        path = Path(raw_path)
        if not path.exists():
            print(f"  {path}: file not found")
            continue
        dataset = mb_infer_dataset(path)
        seed = mb_infer_seed(path)
        try:
            decision_point = path.parents[2].name
        except IndexError:
            decision_point = path.parent.name
        if dataset is None:
            n_unmatched += 1
            print(f"  Warning: could not infer dataset from path, leaving as 'unknown': {path}")
            dataset = "unknown"

        df = pd.read_csv(path)
        if "is_pareto" in df.columns:
            pareto_df = df[df["is_pareto"]]
        else:
            pareto_df = df

        for trial_id, group in pareto_df.groupby("trial_id"):
            front_rows = list(zip(group["acc_new_test"], group["total_nodes"], group["sim_old_new_jaccard"]))
            if "operator_detail" in group.columns:
                operator = group["operator_detail"].iloc[0]
            elif "operator" in group.columns:
                operator = group["operator"].iloc[0]
            else:
                operator = None

            for prefix in baselines:
                acc_col, nodes_col, jac_col = _baseline_columns(prefix)
                if acc_col not in group.columns or nodes_col not in group.columns or jac_col not in group.columns:
                    n_skipped_baseline += 1
                    continue
                b_acc = group[acc_col].iloc[0]
                b_nodes = group[nodes_col].iloc[0]
                b_jac = group[jac_col].iloc[0]
                if pd.isna(b_acc) or pd.isna(b_nodes) or pd.isna(b_jac):
                    n_skipped_baseline += 1
                    continue
                frac = coverage_classical([(b_acc, b_nodes, b_jac)], front_rows, tol)
                rows.append({
                    "dataset": dataset, "seed": seed, "decision_point": decision_point,
                    "trial_id": trial_id, "operator": operator, "baseline": prefix,
                    "coverage_classical_over_rulesrepair": frac,
                })

    if n_unmatched:
        print(f"\n{n_unmatched} file(s) had a path that didn't match the expected "
              f"experiments/<dataset>/repair/... layout, kept under dataset='unknown'.\n")
    if n_skipped_baseline:
        print(f"\n{n_skipped_baseline} (trial, baseline) combination(s) skipped, missing/NaN "
              f"columns for that baseline on that trial (baseline not fit for this decision point).\n")

    if not rows:
        raise SystemExit("No usable (trial, baseline) rows found among the given paths.")
    return pd.DataFrame(rows)


def mb_save_raw_trials(all_trials, path):
    all_trials[MB_RAW_COLUMNS].to_csv(path, index=False)


def mb_load_raw_trials(csv_paths):
    return _load_merge_csvs(
        csv_paths, MB_RAW_COLUMNS,
        missing_column_hint="not a --raw-out file from this script?",
        empty_message="No usable raw-trials CSV files found among the given paths.",
    )


def mb_print_summary_table(df, title):
    print(f"\n{title}")
    print(f"{'group':<22}{'n_trials':<10}{'median':<10}{'q1':<10}{'q3':<10}{'iqr':<10}{'mean':<10}")
    for _, r in df.iterrows():
        print(f"{str(r['group']):<22}{r['n_trials']:<10}"
              f"{r['median']:<10.4f}{r['q1']:<10.4f}{r['q3']:<10.4f}{r['iqr']:<10.4f}{r['mean']:<10.4f}")


def by_dataset_baseline_summary(all_trials, col="coverage_classical_over_rulesrepair"):
    rows = [_quartile_row(group[col], f"{dataset}/{baseline}")
            for (dataset, baseline), group in all_trials.groupby(["dataset", "baseline"])]
    return pd.DataFrame(rows).sort_values("group").reset_index(drop=True)


def _mb_dataset_display(name):
    return MB_DATASET_DISPLAY_NAMES.get(name, str(name).replace("_", " ").title())


def _mb_format_median_iqr(vals, as_percent=True, decimals=1):
    vals = pd.Series(vals).dropna()
    if len(vals) == 0:
        return "--"
    scale = 100.0 if as_percent else 1.0
    q1, med, q3 = (vals * scale).quantile([0.25, 0.5, 0.75])
    return f"{med:.{decimals}f} [{q1:.{decimals}f}, {q3:.{decimals}f}]"


def multi_baseline_latex_table(
        all_trials, col="coverage_classical_over_rulesrepair", as_percent=True, decimals=1,
        baselines=BASELINE_PREFIXES,
        caption="Probability that a point on RulesRepair's own Pareto front is (weakly) "
                "dominated by each baseline's single point (CART, J48, REPTree), reported "
                "as median with interquartile range across trials, overall and by mutation "
                "operator.",
        label="tab:multi_baseline_point_dominance"):
    present_datasets = list(all_trials["dataset"].dropna().unique())
    ordered_datasets = [d for d in MB_DATASET_DISPLAY_NAMES if d in present_datasets]
    ordered_datasets += sorted(d for d in present_datasets if d not in MB_DATASET_DISPLAY_NAMES)

    present_operators = set(all_trials["operator"].dropna().unique())
    ordered_operators = [op for op in MB_OPERATOR_ORDER if op in present_operators]
    ordered_operators += sorted(op for op in present_operators if op not in MB_OPERATOR_ORDER)

    present_baselines = [b for b in baselines if b in set(all_trials["baseline"].dropna().unique())]

    col_spec = "l" + "l" + "c" * (1 + len(ordered_operators))

    backslash = chr(92)
    row_end = backslash + backslash

    header_ops = " & ".join(MB_OPERATOR_DISPLAY_NAMES.get(op, op.replace("_", " ").title()) for op in ordered_operators)

    lines = []
    lines.append(backslash + "begin{table}[tb]")
    lines.append(backslash + "centering")
    lines.append(backslash + "Huge")
    lines.append(backslash + "resizebox{" + backslash + "textwidth}{!}{%")
    lines.append(backslash + "begin{tabular}{" + col_spec + "}")
    lines.append(backslash + "toprule")
    lines.append("Dataset & vs. & Overall & " + header_ops + " " + row_end)
    lines.append(backslash + "midrule")

    for i, dataset in enumerate(ordered_datasets):
        d_rows = all_trials[all_trials["dataset"] == dataset]
        display_name = _mb_dataset_display(dataset)
        n_base = len(present_baselines)
        multirow_cell = (
            backslash + "multirow{" + str(n_base) + "}{*}{" + backslash
            + "rotatebox[origin=c]{0}{" + backslash + "textit{" + display_name + "}}}"
        )
        for j, baseline in enumerate(present_baselines):
            b_rows = d_rows[d_rows["baseline"] == baseline]
            overall_cell = _mb_format_median_iqr(b_rows[col], as_percent, decimals)
            op_cells = []
            for op in ordered_operators:
                op_rows = b_rows[b_rows["operator"] == op]
                op_cells.append(_mb_format_median_iqr(op_rows[col], as_percent, decimals))
            first_col = multirow_cell if j == 0 else ""
            baseline_name = BASELINE_DISPLAY_NAMES.get(baseline, baseline.upper())
            lines.append(
                first_col + " & " + baseline_name + " & " + overall_cell + " & "
                + " & ".join(op_cells) + " " + row_end
            )
        if i < len(ordered_datasets) - 1:
            lines.append(backslash + "midrule")

    lines.append(backslash + "bottomrule")
    lines.append(backslash + "end{tabular}")
    lines.append("}")
    lines.append(backslash + "vspace{2pt}")
    lines.append(backslash + "caption{" + caption + "}")
    lines.append(backslash + "label{" + label + "}")
    lines.append(backslash + "end{table}")
    return chr(10).join(lines)


def main_multi_baseline(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv_paths", nargs="+",
                         help="pareto_per_trial.csv files (glob-expanded), or --raw-out files with --merge")
    parser.add_argument("--tol", type=float, default=DEFAULT_TOL)
    parser.add_argument("--raw-out", default=None)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--latex-out", default=None)
    parser.add_argument("--out-csv", default=None)
    args = parser.parse_args(argv)

    _default_dir = Path("evaluation") / "quantitative" / "rq1" / "multi_baseline"
    _default_dir.mkdir(parents=True, exist_ok=True)
    if args.raw_out is None:
        args.raw_out = str(_default_dir / "raw_trials.csv")
    if args.out_csv is None:
        args.out_csv = str(_default_dir / "summary.csv")
    if args.latex_out is None:
        args.latex_out = str(_default_dir / "summary.tex")

    if args.merge:
        print(f"Merging {len(args.csv_paths)} raw-trials file(s) (--merge)...")
        all_trials = mb_load_raw_trials(args.csv_paths)
    else:
        print(f"Reading {len(args.csv_paths)} file(s)...")
        all_trials = mb_collect_trials(args.csv_paths, tol=args.tol)
    print(f"{len(all_trials)} (trial, baseline) row(s) total, across "
          f"{all_trials['dataset'].nunique()} dataset(s) and {all_trials['baseline'].nunique()} baseline(s).\n")

    if args.raw_out:
        mb_save_raw_trials(all_trials, args.raw_out)
        print(f"Saved raw per-(trial, baseline) values to {args.raw_out}.\n")

    summary_df = by_dataset_baseline_summary(all_trials)
    mb_print_summary_table(summary_df, "BY DATASET x BASELINE (overall, every operator pooled)")

    if args.out_csv:
        summary_df.to_csv(args.out_csv, index=False)
        print(f"\nSaved summary to {args.out_csv}")

    if args.latex_out:
        table_text = multi_baseline_latex_table(all_trials)
        Path(args.latex_out).write_text(table_text + chr(10))
        print(f"\nSaved LaTeX table to {args.latex_out}")
