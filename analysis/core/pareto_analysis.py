#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_F1_COL = "f1_test_after"
DEFAULT_NODES_COL = "total_nodes"
DEFAULT_REAUDIT_COL = "sim_old_new"
DEFAULT_W_SIMP_COL = "w_simp"
DEFAULT_W_SIMI_COL = "w_simi"
DEFAULT_T_COL = "t"
DEFAULT_CART_F1_COL = "f1_cart_test"
DEFAULT_CART_NODES_COL = "cart_total_nodes"
DEFAULT_CART_REAUDIT_COL = "sim_old_cart"


def load_results(csv_path, f1_col, nodes_col, reaudit_col, w_simp_col, w_simi_col, t_col):
    df = pd.read_csv(csv_path)

    required = [f1_col, nodes_col, reaudit_col, w_simp_col, w_simi_col, t_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Available: {list(df.columns)}")

    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    must_be_present = [c for c in required if c != t_col]
    invalid = df[must_be_present].isna().any(axis=1)
    if invalid.any():
        print(f"Warning: dropped {int(invalid.sum())} rows with missing/non-numeric values.")
        df = df.loc[~invalid].reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid rows in the CSV.")
    return df


def extract_cart_baseline(df, cart_f1_col, cart_nodes_col, cart_reaudit_col):
    required = [cart_f1_col, cart_nodes_col, cart_reaudit_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"Warning: CART baseline unavailable, missing columns: {missing}")
        return None

    values = df[required].apply(pd.to_numeric, errors="coerce").dropna().drop_duplicates().reset_index(drop=True)
    if values.empty:
        print("Warning: no valid CART values found.")
        return None
    if len(values) > 1:
        print("Warning: multiple distinct CART baselines found, using the first.")
    return values.iloc[0]


def build_distinct_objective_points(df, f1_col, nodes_col, reaudit_col):
    cols = [nodes_col, reaudit_col, f1_col]
    return df.groupby(cols, dropna=False).size().rename("n_configurations").reset_index()


def compute_pareto_front(df, f1_col, nodes_col, reaudit_col, tolerance=1e-9, reaudit_maximize=False):
    f1 = df[f1_col].to_numpy(dtype=float)
    nodes = df[nodes_col].to_numpy(dtype=float)
    raw_reaudit = df[reaudit_col].to_numpy(dtype=float)
    reaudit = -raw_reaudit if reaudit_maximize else raw_reaudit

    is_pareto = np.ones(len(df), dtype=bool)
    for i in range(len(df)):
        at_least_as_good = (
            (f1 >= f1[i] - tolerance) & (nodes <= nodes[i] + tolerance) & (reaudit <= reaudit[i] + tolerance)
        )
        strictly_better = (
            (f1 > f1[i] + tolerance) | (nodes < nodes[i] - tolerance) | (reaudit < reaudit[i] - tolerance)
        )
        dominated = at_least_as_good & strictly_better
        dominated[i] = False
        if dominated.any():
            is_pareto[i] = False
    return is_pareto


def assign_pareto_ids(pareto_df, nodes_col, reaudit_col, f1_col, reaudit_ascending=True):
    ordered = pareto_df.sort_values(
        by=[nodes_col, reaudit_col, f1_col], ascending=[True, reaudit_ascending, False], kind="stable"
    ).reset_index(drop=True)
    ordered.insert(0, "pareto_id", [f"P{i + 1}" for i in range(len(ordered))])
    return ordered


def find_solutions_dominating_cart(
    distinct_df, cart_baseline, f1_col, nodes_col, reaudit_col,
    cart_f1_col, cart_nodes_col, cart_reaudit_col, tolerance=1e-9, reaudit_maximize=False,
):
    if cart_baseline is None:
        return pd.DataFrame()

    cart_f1 = float(cart_baseline[cart_f1_col])
    cart_nodes = float(cart_baseline[cart_nodes_col])
    cart_reaudit = float(cart_baseline[cart_reaudit_col])

    if reaudit_maximize:
        reaudit_at_least_as_good = distinct_df[reaudit_col] >= cart_reaudit - tolerance
        reaudit_strictly_better = distinct_df[reaudit_col] > cart_reaudit + tolerance
    else:
        reaudit_at_least_as_good = distinct_df[reaudit_col] <= cart_reaudit + tolerance
        reaudit_strictly_better = distinct_df[reaudit_col] < cart_reaudit - tolerance

    return distinct_df.loc[
        (distinct_df[f1_col] >= cart_f1 - tolerance)
        & (distinct_df[nodes_col] <= cart_nodes + tolerance)
        & reaudit_at_least_as_good
        & (
            (distinct_df[f1_col] > cart_f1 + tolerance)
            | (distinct_df[nodes_col] < cart_nodes - tolerance)
            | reaudit_strictly_better
        )
    ].copy()


def build_pareto_summary(pareto_with_ids, pareto_configurations, w_simp_col, w_simi_col, t_col):
    representative = (
        pareto_configurations
        .sort_values(by=["pareto_id", w_simp_col, w_simi_col, t_col], kind="stable")
        .groupby("pareto_id", sort=False)
        .first()[[w_simp_col, w_simi_col, t_col]]
        .reset_index()
    )
    return pareto_with_ids.merge(representative, on="pareto_id")


def print_summary_table(
    summary_df, nodes_col, reaudit_col, f1_col, w_simp_col, w_simi_col, t_col,
    reaudit_header="Reaudit %", metric_header="F1 test",
):
    print(
        f"{'Pareto ID':<11}{'w_simp':<10}{'w_simi':<10}{'t':<8}"
        f"{'Nodes':<8}{reaudit_header:<12}{metric_header:<10}{'# configs'}"
    )
    for _, row in summary_df.iterrows():
        print(
            f"{row['pareto_id']:<11}{row[w_simp_col]:<10g}{row[w_simi_col]:<10g}{row[t_col]:<8g}"
            f"{int(row[nodes_col]):<8}{row[reaudit_col]:<12.1f}{row[f1_col]:<10.4f}{int(row['n_configurations'])}"
        )


def print_example_configurations(pareto_configurations, w_simp_col, w_simi_col, t_col, max_examples=3):
    print("\nExample configurations per Pareto point (full list in pareto_configurations.csv):")
    for pareto_id, group in pareto_configurations.groupby("pareto_id", sort=False):
        print(f"  {pareto_id} (n={len(group)}):")
        for _, row in group.head(max_examples).iterrows():
            print(f"    w_simp={row[w_simp_col]:g}, w_simi={row[w_simi_col]:g}, t={row[t_col]:g}")
        remaining = len(group) - max_examples
        if remaining > 0:
            print(f"    ... and {remaining} more configurations")


def run_pareto_analysis(
    csv_path, out_dir=None,
    f1_col=DEFAULT_F1_COL, nodes_col=DEFAULT_NODES_COL, reaudit_col=DEFAULT_REAUDIT_COL,
    w_simp_col=DEFAULT_W_SIMP_COL, w_simi_col=DEFAULT_W_SIMI_COL, t_col=DEFAULT_T_COL,
    cart_f1_col=DEFAULT_CART_F1_COL, cart_nodes_col=DEFAULT_CART_NODES_COL, cart_reaudit_col=DEFAULT_CART_REAUDIT_COL,
    reaudit_maximize=True,
    reaudit_header="Similarity",
    metric_header="F1 test",
):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")
    out_dir = Path(out_dir) if out_dir else csv_path.parent / "analysis"

    f1, nodes, reaudit = f1_col, nodes_col, reaudit_col
    w_simp, w_simi, t = w_simp_col, w_simi_col, t_col
    cart_f1, cart_nodes, cart_reaudit = cart_f1_col, cart_nodes_col, cart_reaudit_col
    reaudit_ascending = not reaudit_maximize

    df = load_results(csv_path, f1, nodes, reaudit, w_simp, w_simi, t)
    cart_baseline = extract_cart_baseline(df, cart_f1, cart_nodes, cart_reaudit)

    distinct_df = build_distinct_objective_points(df, f1, nodes, reaudit)
    distinct_df["is_pareto"] = compute_pareto_front(distinct_df, f1, nodes, reaudit, reaudit_maximize=reaudit_maximize)

    pareto_with_ids = assign_pareto_ids(
        distinct_df.loc[distinct_df["is_pareto"]], nodes, reaudit, f1, reaudit_ascending=reaudit_ascending,
    )
    id_map = pareto_with_ids.set_index([nodes, reaudit, f1])["pareto_id"]
    distinct_df = distinct_df.set_index([nodes, reaudit, f1], drop=False)
    distinct_df["pareto_id"] = id_map
    distinct_df = distinct_df.reset_index(drop=True)

    pareto_configurations = df.merge(
        pareto_with_ids[[nodes, reaudit, f1, "pareto_id", "n_configurations"]],
        on=[nodes, reaudit, f1], how="inner",
    ).sort_values(by=["pareto_id", w_simp, w_simi, t], kind="stable").reset_index(drop=True)

    cart_dominators = find_solutions_dominating_cart(
        distinct_df, cart_baseline, f1, nodes, reaudit, cart_f1, cart_nodes, cart_reaudit,
        reaudit_maximize=reaudit_maximize,
    )
    cart_is_dominated = not cart_dominators.empty

    out_dir.mkdir(parents=True, exist_ok=True)
    distinct_df[[nodes, reaudit, f1, "n_configurations", "is_pareto", "pareto_id"]].to_csv(
        out_dir / "all_distinct_objective_points.csv", index=False
    )
    pareto_with_ids[["pareto_id", nodes, reaudit, f1, "n_configurations"]].to_csv(
        out_dir / "pareto_front.csv", index=False
    )
    pareto_configurations[["pareto_id", w_simp, w_simi, t, nodes, reaudit, f1]].to_csv(
        out_dir / "pareto_configurations.csv", index=False
    )

    summary_df = build_pareto_summary(pareto_with_ids, pareto_configurations, w_simp, w_simi, t)
    print_summary_table(
        summary_df, nodes, reaudit, f1, w_simp, w_simi, t,
        reaudit_header=reaudit_header, metric_header=metric_header,
    )
    print_example_configurations(pareto_configurations, w_simp, w_simi, t)

    print()
    if cart_baseline is None:
        print("CART: baseline unavailable.")
    elif cart_is_dominated:
        print(f"CART is dominated by {len(cart_dominators)} distinct Keep-Regrow results.")
    else:
        print("CART is not dominated by Keep-Regrow.")

    print()
    print(f"Valid rows: {len(df)}")
    print(f"Distinct results: {len(distinct_df)}")
    print(f"Pareto solutions: {int(distinct_df['is_pareto'].sum())}")
    print(f"Output: {out_dir.resolve()}")

    return out_dir


def main():
    parser = argparse.ArgumentParser(
        description="Per-parameter effect (w_simp/w_simi/t), Pareto front and CART comparison."
    )
    parser.add_argument("csv_path", help="CSV with grid search results")
    parser.add_argument("--out-dir", default=None, help="Output directory")
    parser.add_argument("--f1-col", default=DEFAULT_F1_COL)
    parser.add_argument("--nodes-col", default=DEFAULT_NODES_COL)
    parser.add_argument("--reaudit-col", default=DEFAULT_REAUDIT_COL)
    parser.add_argument("--w-simp-col", default=DEFAULT_W_SIMP_COL)
    parser.add_argument("--w-simi-col", default=DEFAULT_W_SIMI_COL)
    parser.add_argument("--t-col", default=DEFAULT_T_COL)
    parser.add_argument("--cart-f1-col", default=DEFAULT_CART_F1_COL)
    parser.add_argument("--cart-nodes-col", default=DEFAULT_CART_NODES_COL)
    parser.add_argument("--cart-reaudit-col", default=DEFAULT_CART_REAUDIT_COL)
    parser.add_argument(
        "--reaudit-maximize", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--reaudit-header", default=None, help="Override third-objective column header in the printed summary table (default depends on --reaudit-maximize).")
    parser.add_argument("--metric-header", default="F1 test", help="Printed summary-table column header for --f1-col (default 'F1 test').")
    args = parser.parse_args()

    reaudit_header = args.reaudit_header or ("Similarity" if args.reaudit_maximize else "Reaudit %")

    run_pareto_analysis(
        csv_path=args.csv_path, out_dir=args.out_dir,
        f1_col=args.f1_col, nodes_col=args.nodes_col, reaudit_col=args.reaudit_col,
        w_simp_col=args.w_simp_col, w_simi_col=args.w_simi_col, t_col=args.t_col,
        cart_f1_col=args.cart_f1_col, cart_nodes_col=args.cart_nodes_col, cart_reaudit_col=args.cart_reaudit_col,
        reaudit_maximize=args.reaudit_maximize,
        reaudit_header=reaudit_header,
        metric_header=args.metric_header,
    )


if __name__ == "__main__":
    main()
