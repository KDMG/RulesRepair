import argparse
from pathlib import Path

import pandas as pd

from analysis.core.compare_rulesrepair_cart_dominance import (
    DEFAULT_MAX_DEPTH, DEFAULT_NODES_MIN, DEFAULT_TOL, _latex_escape, infer_dp_label,
)
from analysis.core.mutation_types import MUTATION_TYPES
from analysis.rq2.statistical.igd_core import (
    build_seed_level_table, build_trial_level_table, per_trial_igd,
    decision_point_igd_test, decision_point_igd_test_trial_level,
)


def _rq12_view_result(csv_paths, mutation_type, max_depth, nodes_min, tol, alpha, sd_zero_tol,
                       baseline_acc_col="acc_cart_test", baseline_nodes_col="cart_total_nodes",
                       baseline_jaccard_col="sim_old_cart_jaccard", baseline_label="CART"):
    seed_table = build_seed_level_table(
        csv_paths, max_depth=max_depth, nodes_min=nodes_min, tol=tol, mutation_type=mutation_type,
        baseline_acc_col=baseline_acc_col, baseline_nodes_col=baseline_nodes_col,
        baseline_jaccard_col=baseline_jaccard_col,
    )
    total_trials = int(seed_table["n_trials"].sum()) if len(seed_table) else 0
    if total_trials == 0:
        return {
            "rq12_not_applicable": True, "rq12_n_seeds": 0, "wilcoxon_p": None, "p_holm": None,
            "hodges_lehmann": None, "hl_ci_low": None, "hl_ci_high": None,
            "winner": None,
        }
    result = decision_point_igd_test(seed_table, alpha=alpha, sd_zero_tol=sd_zero_tol, baseline_label=baseline_label)
    return {
        "rq12_not_applicable": False, "rq12_n_seeds": result["n_seeds"],
        "wilcoxon_p": result["wilcoxon_p"], "p_holm": result["p_holm"],
        "hodges_lehmann": result["hodges_lehmann"], "hl_ci_low": result["hl_ci_low"],
        "hl_ci_high": result["hl_ci_high"],
        "winner": result["winner"],
    }


def _rq12_view_result_from_trial_table(trial_table, alpha, sd_zero_tol, baseline_label):
    if len(trial_table) == 0:
        return {
            "rq12_not_applicable": True, "rq12_n_trials": 0, "wilcoxon_p": None, "p_holm": None,
            "hodges_lehmann": None, "hl_ci_low": None, "hl_ci_high": None,
            "winner": None,
        }
    result = decision_point_igd_test_trial_level(
        trial_table, alpha=alpha, sd_zero_tol=sd_zero_tol, baseline_label=baseline_label,
    )
    return {
        "rq12_not_applicable": False, "rq12_n_trials": result["n_trials"],
        "wilcoxon_p": result["wilcoxon_p"], "p_holm": result["p_holm"],
        "hodges_lehmann": result["hodges_lehmann"], "hl_ci_low": result["hl_ci_low"],
        "hl_ci_high": result["hl_ci_high"],
        "winner": result["winner"],
    }


def _rq12_view_result_trial_level(csv_paths, mutation_type, max_depth, nodes_min, tol, alpha, sd_zero_tol,
                                   baseline_acc_col="acc_cart_test", baseline_nodes_col="cart_total_nodes",
                                   baseline_jaccard_col="sim_old_cart_jaccard", baseline_label="CART"):
    trial_table = build_trial_level_table(
        csv_paths, max_depth=max_depth, nodes_min=nodes_min, tol=tol, mutation_type=mutation_type,
        baseline_acc_col=baseline_acc_col, baseline_nodes_col=baseline_nodes_col,
        baseline_jaccard_col=baseline_jaccard_col,
    )
    return _rq12_view_result_from_trial_table(trial_table, alpha, sd_zero_tol, baseline_label)


def _no_mutation_diff_for_dp(csv_paths, max_depth, nodes_min, tol,
                              baseline_acc_col="acc_cart_test", baseline_nodes_col="cart_total_nodes",
                              baseline_jaccard_col="sim_old_cart_jaccard"):
    if not csv_paths:
        return None
    marker = "/mutated/analysis/pareto_per_trial.csv"
    p = str(csv_paths[0])
    if marker not in p:
        return None
    baseline_csv = Path(p.replace(marker, "/baseline/results.csv"))
    if not baseline_csv.exists():
        return None
    df = pd.read_csv(baseline_csv)
    if df["trial_id"].nunique() != 1:
        return None
    per_trial = per_trial_igd(
        df, max_depth=max_depth, nodes_min=nodes_min, tol=tol,
        baseline_acc_col=baseline_acc_col, baseline_nodes_col=baseline_nodes_col,
        baseline_jaccard_col=baseline_jaccard_col,
    )
    if len(per_trial) != 1:
        return None
    row = per_trial.iloc[0]
    if pd.isna(row["igd_rulesrepair"]) or pd.isna(row["igd_baseline"]):
        return None
    return float(row["igd_baseline"] - row["igd_rulesrepair"])


DEFAULT_BASELINES = [
    {"label": "CART", "acc_col": "acc_cart_test", "nodes_col": "cart_total_nodes",
     "jaccard_col": "sim_old_cart_jaccard"},
]


def build_unified_results_table(dataset_csv_paths, mutation_types=MUTATION_TYPES, by_mutation_type=True,
                                 max_depth=DEFAULT_MAX_DEPTH, nodes_min=DEFAULT_NODES_MIN, tol=DEFAULT_TOL,
                                 alpha=0.05, sd_zero_tol=DEFAULT_TOL, baselines=None):
    if baselines is None:
        baselines = DEFAULT_BASELINES

    rows = []
    for dataset, csv_paths in dataset_csv_paths.items():
        by_dp = {}
        for p in csv_paths:
            by_dp.setdefault(infer_dp_label(p), []).append(p)

        for dp_label in sorted(by_dp):
            paths = by_dp[dp_label]
            views = [(None, "overall")]
            if by_mutation_type:
                views += [(mt, mt) for mt in mutation_types]

            no_mutation_by_baseline = {
                baseline["label"]: _no_mutation_diff_for_dp(
                    paths, max_depth, nodes_min, tol,
                    baseline_acc_col=baseline["acc_col"], baseline_nodes_col=baseline["nodes_col"],
                    baseline_jaccard_col=baseline["jaccard_col"],
                )
                for baseline in baselines
            }

            for mutation_type, tag in views:
                for baseline in baselines:
                    rq12 = _rq12_view_result(
                        paths, mutation_type, max_depth, nodes_min, tol, alpha, sd_zero_tol,
                        baseline_acc_col=baseline["acc_col"], baseline_nodes_col=baseline["nodes_col"],
                        baseline_jaccard_col=baseline["jaccard_col"], baseline_label=baseline["label"],
                    )
                    row = {
                        "dataset": dataset, "decision_point": dp_label, "view": tag,
                        "baseline": baseline["label"],
                        "no_mutation_diff": no_mutation_by_baseline[baseline["label"]],
                    }
                    row.update(rq12)
                    rows.append(row)

    return pd.DataFrame(rows)


def build_unified_results_table_trial_level(dataset_csv_paths, mutation_types=MUTATION_TYPES, by_mutation_type=True,
                                              max_depth=DEFAULT_MAX_DEPTH, nodes_min=DEFAULT_NODES_MIN,
                                              tol=DEFAULT_TOL, alpha=0.05, sd_zero_tol=DEFAULT_TOL, baselines=None):
    if baselines is None:
        baselines = DEFAULT_BASELINES

    rows = []
    for dataset, csv_paths in dataset_csv_paths.items():
        by_dp = {}
        for p in csv_paths:
            by_dp.setdefault(infer_dp_label(p), []).append(p)

        for dp_label in sorted(by_dp):
            paths = by_dp[dp_label]
            views = [(None, "overall")]
            if by_mutation_type:
                views += [(mt, mt) for mt in mutation_types]

            no_mutation_by_baseline = {
                baseline["label"]: _no_mutation_diff_for_dp(
                    paths, max_depth, nodes_min, tol,
                    baseline_acc_col=baseline["acc_col"], baseline_nodes_col=baseline["nodes_col"],
                    baseline_jaccard_col=baseline["jaccard_col"],
                )
                for baseline in baselines
            }

            for mutation_type, tag in views:
                for baseline in baselines:
                    rq12 = _rq12_view_result_trial_level(
                        paths, mutation_type, max_depth, nodes_min, tol, alpha, sd_zero_tol,
                        baseline_acc_col=baseline["acc_col"], baseline_nodes_col=baseline["nodes_col"],
                        baseline_jaccard_col=baseline["jaccard_col"], baseline_label=baseline["label"],
                    )
                    row = {
                        "dataset": dataset, "decision_point": dp_label, "view": tag,
                        "baseline": baseline["label"],
                        "no_mutation_diff": no_mutation_by_baseline[baseline["label"]],
                    }
                    row.update(rq12)
                    rows.append(row)

    return pd.DataFrame(rows)


def _build_dataset_pooled_trial_table(csv_paths, mutation_type, max_depth, nodes_min, tol,
                                       baseline_acc_col, baseline_nodes_col, baseline_jaccard_col):
    by_dp = {}
    for p in csv_paths:
        by_dp.setdefault(infer_dp_label(p), []).append(p)

    tables = []
    for dp_label in sorted(by_dp):
        t = build_trial_level_table(
            by_dp[dp_label], max_depth=max_depth, nodes_min=nodes_min, tol=tol, mutation_type=mutation_type,
            baseline_acc_col=baseline_acc_col, baseline_nodes_col=baseline_nodes_col,
            baseline_jaccard_col=baseline_jaccard_col,
        )
        if len(t):
            tables.append(t)
    if not tables:
        return pd.DataFrame(columns=["decision_point", "seed", "trial_id", "igd_rulesrepair_trial", "igd_baseline_trial"])
    return pd.concat(tables, ignore_index=True)


def _rq12_view_result_trial_level_pooled_dataset(csv_paths, mutation_type, max_depth, nodes_min, tol, alpha,
                                                   sd_zero_tol, baseline_acc_col="acc_cart_test",
                                                   baseline_nodes_col="cart_total_nodes",
                                                   baseline_jaccard_col="sim_old_cart_jaccard",
                                                   baseline_label="CART"):
    trial_table = _build_dataset_pooled_trial_table(
        csv_paths, mutation_type, max_depth, nodes_min, tol,
        baseline_acc_col, baseline_nodes_col, baseline_jaccard_col,
    )
    return _rq12_view_result_from_trial_table(trial_table, alpha, sd_zero_tol, baseline_label)


def build_unified_results_table_trial_level_by_dataset(dataset_csv_paths, mutation_types=MUTATION_TYPES,
                                                         by_mutation_type=True, max_depth=DEFAULT_MAX_DEPTH,
                                                         nodes_min=DEFAULT_NODES_MIN, tol=DEFAULT_TOL, alpha=0.05,
                                                         sd_zero_tol=DEFAULT_TOL, baselines=None):
    if baselines is None:
        baselines = DEFAULT_BASELINES

    rows = []
    for dataset, csv_paths in dataset_csv_paths.items():
        views = [(None, "overall")]
        if by_mutation_type:
            views += [(mt, mt) for mt in mutation_types]

        for mutation_type, tag in views:
            for baseline in baselines:
                rq12 = _rq12_view_result_trial_level_pooled_dataset(
                    csv_paths, mutation_type, max_depth, nodes_min, tol, alpha, sd_zero_tol,
                    baseline_acc_col=baseline["acc_col"], baseline_nodes_col=baseline["nodes_col"],
                    baseline_jaccard_col=baseline["jaccard_col"], baseline_label=baseline["label"],
                )
                row = {"dataset": dataset, "view": tag, "baseline": baseline["label"]}
                row.update(rq12)
                rows.append(row)

    return pd.DataFrame(rows)


VIEW_ORDER = ("overall",) + MUTATION_TYPES


VIEW_HEADERS = {
    "overall": "Overall", "prune": "Prune", "change_label": "Change label",
    "branch_swap": "Branch swap", "change_threshold": "Change threshold",
    "change_feature": "Change feature", "regrow_leaf": "Regrow leaf",
    "regrow_internal": "Regrow internal",
}


BASELINE_DISPLAY = {"J48-unbounded": "J48-unb."}


BASELINE_DISPLAY_BY_DATASET = {"J48": "C4.5", "J48-unbounded": "C4.5-unb."}


DATASET_DISPLAY = {
    "sepsis": "Sepsis",
    "hospital_billing": "Hospital Billing",
    "road_traffic": "Road Traffic Fine",
    "prepaid_travel_costs": "Prepaid Travel Costs",
    "production": "Production",
    "international_declarations": "International Declarations",
}


def _dataset_display(dataset):
    return DATASET_DISPLAY.get(dataset, dataset.replace("_", " ").title())


def _fmt2(x):
    return f"{x:+.2f}"


def _no_mutation_cell_text(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "-"
    return _fmt2(value)


def _hl_cell_text(row, alpha, include_ci=True):
    if row is None or row["rq12_not_applicable"]:
        return "-"
    hl = row["hodges_lehmann"]
    if pd.isna(hl):
        return "--"
    lo, hi = row["hl_ci_low"], row["hl_ci_high"]
    if not include_ci:
        return _fmt2(hl)
    text = _fmt2(hl) if pd.isna(lo) else f"{_fmt2(hl)} [{_fmt2(lo)}, {_fmt2(hi)}]"
    p = row["p_holm"]
    if pd.notna(p) and p < alpha:
        text += "*"
    return text


def unified_results_to_latex(df, alpha=0.05,
                              caption="RQ1+RQ2",
                              label="tab:rq12_unified"):
    n_view_cols = len(VIEW_ORDER)
    lines = []
    lines.append("% Requires \\usepackage{multirow} and \\usepackage{booktabs} in the preamble.")
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\Huge")
    lines.append("\\renewcommand{\\arraystretch}{0.85}")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append("\\begin{tabular}{lll" + "c" * n_view_cols + "}")
    lines.append("\\toprule")
    header_cols = " & ".join(VIEW_HEADERS[v] for v in VIEW_ORDER)
    lines.append(f"Dataset & D.p. & Baseline & {header_cols} \\\\")
    lines.append("\\midrule")

    for dataset, dataset_group in df.groupby("dataset", sort=False):
        dp_labels = list(dict.fromkeys(dataset_group["decision_point"]))
        first_dp_group = dataset_group[dataset_group["decision_point"] == dp_labels[0]]
        n_baselines = len(list(dict.fromkeys(first_dp_group["baseline"])))
        dataset_total_rows = len(dp_labels) * n_baselines
        dataset_row_seen = False

        for dp_idx, dp in enumerate(dp_labels):
            dp_group = dataset_group[dataset_group["decision_point"] == dp]
            baseline_labels = list(dict.fromkeys(dp_group["baseline"]))
            dp_rows = len(baseline_labels)

            for i, baseline in enumerate(baseline_labels):
                baseline_group = dp_group[dp_group["baseline"] == baseline]
                view_lookup = {r["view"]: r for _, r in baseline_group.iterrows()}

                dataset_cell = f"\\multirow{{{dataset_total_rows}}}{{*}}{{\\rotatebox[origin=c]{{90}}{{{_latex_escape(dataset)}}}}}" if not dataset_row_seen else ""
                dataset_row_seen = True
                dp_cell = f"\\multirow{{{dp_rows}}}{{*}}{{{_latex_escape(dp)}}}" if i == 0 else ""

                baseline_display = BASELINE_DISPLAY.get(baseline, baseline)
                baseline_text = f"{_latex_escape(baseline_display)} vs RulesRepair"

                cell_texts = [_hl_cell_text(view_lookup.get(v), alpha) for v in VIEW_ORDER]
                lines.append(
                    f"{dataset_cell} & {dp_cell} & {baseline_text} & " + " & ".join(cell_texts)
                    + " \\\\"
                )
            if dp_idx < len(dp_labels) - 1:
                lines.append(f"\\cmidrule(lr){{2-{3 + n_view_cols}}}")
        lines.append("\\midrule")

    if lines[-1] == "\\midrule":
        lines[-1] = "\\bottomrule"
    else:
        lines.append("\\bottomrule")

    lines.append("\\end{tabular}")
    lines.append("}")
    lines.append(
        f"\\vspace{{2pt}}\n{{\\footnotesize $^*$Holm-adjusted $p < {alpha:g}$ "
        "(RulesRepair vs. baseline significantly different on paired IGD+).}"
    )
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def unified_results_to_latex_by_dataset(df, alpha=0.05, caption=None, label="tab:hodges"):
    if caption is None:
        caption = (
            f"Hodges–Lehmann estimates. \\textsc{{RR}} denotes \\textsc{{RulesRepair}}."
        )
    n_view_cols = len(VIEW_ORDER)
    lines = []
    lines.append("\\begin{table}[tb]")
    lines.append("\\centering")
    lines.append("\\large")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append("\\begin{tabular}{ll" + "c" * n_view_cols + "}")
    lines.append("\\toprule")
    header_cols = " & ".join(VIEW_HEADERS[v] for v in VIEW_ORDER)
    lines.append(f"Dataset & Comparison & {header_cols} \\\\")
    lines.append("\\midrule")

    for dataset, dataset_group in df.groupby("dataset", sort=False):
        baseline_labels = list(dict.fromkeys(dataset_group["baseline"]))
        n_baselines = len(baseline_labels)

        for i, baseline in enumerate(baseline_labels):
            baseline_group = dataset_group[dataset_group["baseline"] == baseline]
            view_lookup = {r["view"]: r for _, r in baseline_group.iterrows()}

            dataset_cell = (
                f"\\multirow{{{n_baselines}}}{{*}}{{\\rotatebox[origin=c]{{0}}{{\\textit{{{_dataset_display(dataset)}}}}}}}"
                if i == 0 else ""
            )
            baseline_display = BASELINE_DISPLAY_BY_DATASET.get(baseline, baseline)
            baseline_text = f"\\textsc{{{_latex_escape(baseline_display)}}} vs \\textsc{{RR}}"

            cell_texts = [_hl_cell_text(view_lookup.get(v), alpha, include_ci=False) for v in VIEW_ORDER]
            lines.append(f"{dataset_cell} & {baseline_text} & " + " & ".join(cell_texts) + " \\\\")
        lines.append("\\midrule")

    if lines[-1] == "\\midrule":
        lines[-1] = "\\bottomrule"
    else:
        lines.append("\\bottomrule")

    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def main_rq_table(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "paths", nargs="+"
    )
    parser.add_argument(
        "--dataset", default=None
    )
    parser.add_argument(
        "--combine", action="store_true"
    )
    parser.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH)
    parser.add_argument("--nodes-min", type=int, default=DEFAULT_NODES_MIN)
    parser.add_argument("--tol", type=float, default=DEFAULT_TOL)
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance threshold for the RQ1/RQ2 'winner' column (applied to the Holm-adjusted p-value, same as igd_statistical_comparison.py).")
    parser.add_argument(
        "--sd-zero-tol", type=float, default=DEFAULT_TOL
    )
    parser.add_argument(
        "--baseline-acc-col", default="acc_cart_test"
    )
    parser.add_argument("--baseline-nodes-col", default="cart_total_nodes")
    parser.add_argument("--baseline-jaccard-col", default="sim_old_cart_jaccard")
    parser.add_argument(
        "--baseline-label", default="CART"
    )
    parser.add_argument(
        "--by-mutation-type", action="store_true", default=True
    )
    parser.add_argument(
        "--trial-level", action="store_true"
    )
    parser.add_argument(
        "--by-dataset", action="store_true"
    )
    parser.add_argument(
        "--out-dir", default=None
    )
    args = parser.parse_args(argv)

    if args.by_dataset and not args.combine and not args.trial_level:
        parser.error("--by-dataset requires --trial-level (in default, non --combine mode).")

    if args.out_dir is None:
        if args.dataset and not args.combine:
            args.out_dir = str(Path("evaluation") / "quantitative" / args.dataset / "rq2" / "rq_table")
        else:
            args.out_dir = str(Path("evaluation") / "quantitative" / "rq2" / "rq_table")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.by_dataset:
        csv_basename = "rq_unified_trial_level_by_dataset"
        tex_basename = "rq_unified_trial_level_by_dataset_results"
    else:
        csv_basename = "rq_unified_trial_level" if args.trial_level else "rq_unified"
        tex_basename = "rq_unified_trial_level_results" if args.trial_level else "rq_unified_results"
    caption = "RQ1+RQ2"

    if args.combine:
        df = pd.concat([pd.read_csv(p) for p in args.paths], ignore_index=True)
        print(f"Combined {len(args.paths)} summary file(s) into {len(df)} row(s).")
    else:
        if not args.dataset:
            parser.error("--dataset is required unless --combine is given")
        if args.by_dataset:
            build_fn = build_unified_results_table_trial_level_by_dataset
        else:
            build_fn = build_unified_results_table_trial_level if args.trial_level else build_unified_results_table
        df = build_fn(
            {args.dataset: args.paths}, by_mutation_type=args.by_mutation_type,
            max_depth=args.max_depth, nodes_min=args.nodes_min, tol=args.tol, alpha=args.alpha,
            sd_zero_tol=args.sd_zero_tol,
            baselines=[{
                "label": args.baseline_label, "acc_col": args.baseline_acc_col,
                "nodes_col": args.baseline_nodes_col, "jaccard_col": args.baseline_jaccard_col,
            }],
        )
        csv_out = out_dir / f"{csv_basename}_{args.dataset}.csv"
        df.to_csv(csv_out, index=False)
        print(f"Wrote {csv_out} ({len(df)} row(s)).")

    latex_fn = unified_results_to_latex_by_dataset if args.by_dataset else unified_results_to_latex
    tex = latex_fn(df, alpha=args.alpha) if args.by_dataset else latex_fn(df, alpha=args.alpha, caption=caption)
    tex_out = out_dir / f"{tex_basename}.tex"
    tex_out.write_text(tex)
    print(f"Wrote {tex_out}.")
