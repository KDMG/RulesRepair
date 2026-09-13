import argparse
from pathlib import Path

import pandas as pd

from analysis.core.pareto_analysis import compute_pareto_front, run_pareto_analysis

TRIAL_ID_COLS = ["trial_id", "scenario", "seed", "feature_type", "intensity", "features"]
OPTIONAL_ID_COLS = [
    "extent", "pair_seed", "perturbation_seed", "total_possible_pairs", "pair_selection",
    "operator", "mutation_index", "is_sound",
    "mutation_node_id", "operator_detail",
]

OTHER_BASELINE_VALUE_COLS = [
    "f1_j48_test", "j48_total_nodes", "pct_reaudit_j48", "sim_old_j48", "sim_old_j48_labeled",
    "sim_old_j48_jaccard", "acc_j48_test",
    "f1_j48_unbounded_test", "j48_unbounded_total_nodes", "pct_reaudit_j48_unbounded",
    "sim_old_j48_unbounded", "sim_old_j48_unbounded_labeled", "sim_old_j48_unbounded_jaccard",
    "acc_j48_unbounded_test",
    "f1_reptree_test", "reptree_total_nodes", "pct_reaudit_reptree", "sim_old_reptree",
    "sim_old_reptree_labeled", "sim_old_reptree_jaccard", "acc_reptree_test",
]

DEFAULT_VALUE_COLS = [
    "f1_new_test", "total_nodes", "pct_reaudit_new", "sim_old_new", "sim_old_new_labeled", "sim_old_new_jaccard",
    "acc_new_test",
    "f1_cart_test", "cart_total_nodes", "pct_reaudit_cart", "sim_old_cart", "sim_old_cart_labeled",
    "sim_old_cart_jaccard", "acc_cart_test",
] + OTHER_BASELINE_VALUE_COLS


def per_trial_pareto(df, out_csv, f1_col="acc_new_test", nodes_col="total_nodes", sim_col="pct_reaudit_new",
                      w_simp_col="w_simp", w_simi_col="w_simi", t_col="t", reaudit_maximize=False):
    id_cols = [c for c in TRIAL_ID_COLS + OPTIONAL_ID_COLS if c in df.columns]
    keep_cols = id_cols + [w_simp_col, w_simi_col, t_col, f1_col, nodes_col, sim_col]
    keep_cols += [c for c in (
        "pct_reaudit_new", "sim_old_new_labeled", "sim_old_new_jaccard", "acc_new_test",
        "f1_cart_test", "acc_cart_test", "cart_total_nodes",
        "pct_reaudit_cart", "sim_old_cart", "sim_old_cart_labeled", "sim_old_cart_jaccard",
        *OTHER_BASELINE_VALUE_COLS,
    ) if c in df.columns and c not in keep_cols]

    chunks = []
    for trial_id, group in df.groupby("trial_id", sort=False):
        group = group.reset_index(drop=True)
        group["is_pareto"] = compute_pareto_front(group, f1_col, nodes_col, sim_col, reaudit_maximize=reaudit_maximize)
        chunks.append(group[keep_cols + ["is_pareto"]])

    result = pd.concat(chunks, ignore_index=True)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_csv, index=False)
    n_trials = df["trial_id"].nunique()
    n_pareto = int(result["is_pareto"].sum())
    direction = "max" if reaudit_maximize else "min"
    print(f"Per-trial Pareto ({f1_col} max, nodes min, {sim_col} {direction}): {n_trials} trials, "
          f"{n_pareto} non-dominated rows total -> {out_csv}")
    return out_csv


def require_same_trial_sets(df, w_simp_col="w_simp", w_simi_col="w_simi", t_col="t"):
    trial_sets = df.groupby([w_simp_col, w_simi_col, t_col], dropna=False)["trial_id"].apply(lambda s: frozenset(s))
    reference_config = trial_sets.index[0]
    reference_set = trial_sets.iloc[0]
    mismatched = trial_sets[trial_sets != reference_set]
    if not mismatched.empty:
        example_config = mismatched.index[0]
        example_set = mismatched.iloc[0]
        raise ValueError(
            f"Configs do not share the exact same trial_id set: {len(mismatched)} of {len(trial_sets)} "
            f"(w_simp,w_simi,t) configs differ from the reference (even if some have the same COUNT).\n"
            f"  reference config {reference_config}: {len(reference_set)} trials, e.g. {sorted(reference_set)[:3]}\n"
            f"  mismatched config {example_config}: {len(example_set)} trials, e.g. {sorted(example_set)[:3]}\n"
            f"  only in reference: {sorted(reference_set - example_set)[:3]}\n"
            f"  only in mismatched: {sorted(example_set - reference_set)[:3]}"
        )
    return reference_set


def aggregate_by_config(df, out_csv, w_simp_col="w_simp", w_simi_col="w_simi", t_col="t", value_cols=None):
    if value_cols is None:
        value_cols = [c for c in DEFAULT_VALUE_COLS if c in df.columns]

    reference_set = require_same_trial_sets(df, w_simp_col, w_simi_col, t_col)

    grouped = df.groupby([w_simp_col, w_simi_col, t_col], dropna=False)
    agg = grouped.size().rename("n_trials").reset_index()
    for col in value_cols:
        g = grouped[col]
        stats = pd.DataFrame({
            f"{col}_median": g.median(),
            f"{col}_q1": g.quantile(0.25),
            f"{col}_q3": g.quantile(0.75),
        })
        agg = agg.merge(stats.reset_index(), on=[w_simp_col, w_simi_col, t_col])

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(out_csv, index=False)
    print(f"Aggregated by (w_simp,w_simi,t): {len(agg)} configs, all aggregated over the exact same "
          f"{len(reference_set)} trial_ids -> {out_csv}")
    return out_csv


def run_trial_pareto_analysis(
    csv_path, out_dir,
    f1_col="acc_new_test", nodes_col="total_nodes", sim_col="pct_reaudit_new",
    w_simp_col="w_simp", w_simi_col="w_simi", t_col="t",
    cart_f1_col="acc_cart_test", cart_nodes_col="cart_total_nodes", cart_sim_col="pct_reaudit_cart",
    dpi=300, metric_label="Accuracy", metric_header="Accuracy test",
    reaudit_maximize=False,
    reaudit_axis_label="Percentage of nodes needing re-audit",
    reaudit_title_phrase="re-audit percentage",
    reaudit_header="Reaudit %",
):
    csv_path = Path(csv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    required = ["trial_id", w_simp_col, w_simi_col, t_col, f1_col, nodes_col, sim_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Available: {list(df.columns)}")

    per_trial_pareto(
        df, out_dir / "pareto_per_trial.csv",
        f1_col=f1_col, nodes_col=nodes_col, sim_col=sim_col,
        w_simp_col=w_simp_col, w_simi_col=w_simi_col, t_col=t_col,
        reaudit_maximize=reaudit_maximize,
    )

    value_cols = [c for c in DEFAULT_VALUE_COLS if c in df.columns]
    aggregated_csv = aggregate_by_config(
        df, out_dir / "aggregated_by_config.csv",
        w_simp_col=w_simp_col, w_simi_col=w_simi_col, t_col=t_col, value_cols=value_cols,
    )

    has_cart = f"{cart_f1_col}_median" in pd.read_csv(aggregated_csv, nrows=0).columns

    print()
    run_pareto_analysis(
        csv_path=aggregated_csv, out_dir=out_dir / "aggregated",
        f1_col=f"{f1_col}_median", nodes_col=f"{nodes_col}_median", reaudit_col=f"{sim_col}_median",
        sim_col=f"{sim_col}_median",
        w_simp_col=w_simp_col, w_simi_col=w_simi_col, t_col=t_col,
        cart_f1_col=f"{cart_f1_col}_median" if has_cart else cart_f1_col,
        cart_nodes_col=f"{cart_nodes_col}_median" if has_cart else cart_nodes_col,
        cart_reaudit_col=f"{cart_sim_col}_median" if has_cart else cart_sim_col,
        dpi=dpi,
        reaudit_maximize=reaudit_maximize,
        reaudit_axis_label=reaudit_axis_label,
        reaudit_title_phrase=reaudit_title_phrase,
        reaudit_header=reaudit_header,
        metric_label=metric_label, metric_header=metric_header,
    )

    return out_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv_path", help="results.csv produced by run_perturbation_repair.py (many trials x w_simp/w_simi/t grid)")
    parser.add_argument("--out-dir", default=None, help="Output directory (default: <csv_dir>/trial_analysis)")
    parser.add_argument("--f1-col", default="acc_new_test", help="Primary metric column, plotted/optimized (default: acc_new_test, plain accuracy; pass f1_new_test for the old F1-macro behaviour and set --metric-label/--metric-header to match).")
    parser.add_argument("--nodes-col", default="total_nodes")
    parser.add_argument("--sim-col", default="pct_reaudit_new", help="Third Pareto objective (default: pct_reaudit_new, structural node-by-node re-audit percentage against T_old -- minimized. Pass sim_old_new_labeled/sim_old_new/sim_old_new_jaccard together with --reaudit-maximize to use a similarity score instead, maximized).")
    parser.add_argument("--w-simp-col", default="w_simp")
    parser.add_argument("--w-simi-col", default="w_simi")
    parser.add_argument("--t-col", default="t")
    parser.add_argument("--cart-f1-col", default="acc_cart_test")
    parser.add_argument("--cart-nodes-col", default="cart_total_nodes")
    parser.add_argument("--cart-sim-col", default="pct_reaudit_cart")
    parser.add_argument(
        "--reaudit-maximize", action="store_true", default=False,
        help="Maximize --sim-col instead of minimizing it (default: minimize, correct for pct_reaudit_new/cart; "
             "pass this flag when using a similarity column like sim_old_new_labeled instead).",
    )
    parser.add_argument("--reaudit-axis-label", default="Percentage of nodes needing re-audit")
    parser.add_argument("--reaudit-title-phrase", default="re-audit percentage")
    parser.add_argument("--reaudit-header", default="Reaudit %")
    parser.add_argument("--metric-label", default="Accuracy", help="Axis/title label for --f1-col (default 'Accuracy').")
    parser.add_argument("--metric-header", default="Accuracy test", help="Printed summary-table column header for --f1-col (default 'Accuracy test').")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    out_dir = args.out_dir or (csv_path.parent / "trial_analysis")

    run_trial_pareto_analysis(
        csv_path=csv_path, out_dir=out_dir,
        reaudit_maximize=args.reaudit_maximize,
        reaudit_axis_label=args.reaudit_axis_label,
        reaudit_title_phrase=args.reaudit_title_phrase,
        reaudit_header=args.reaudit_header,
        f1_col=args.f1_col, nodes_col=args.nodes_col, sim_col=args.sim_col,
        w_simp_col=args.w_simp_col, w_simi_col=args.w_simi_col, t_col=args.t_col,
        cart_f1_col=args.cart_f1_col, cart_nodes_col=args.cart_nodes_col, cart_sim_col=args.cart_sim_col,
        dpi=args.dpi, metric_label=args.metric_label, metric_header=args.metric_header,
    )


if __name__ == "__main__":
    main()
