#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_F1_COL = "f1_test_after"
DEFAULT_NODES_COL = "total_nodes"
DEFAULT_REAUDIT_COL = "sim_old_new"
DEFAULT_SIM_COL = "sim_old_new"
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


def check_similarity_column(df, sim_col):
    if sim_col not in df.columns:
        print(f"Warning: similarity column '{sim_col}' not found, skipping similarity panels.")
        return False
    df[sim_col] = pd.to_numeric(df[sim_col], errors="coerce")
    return True


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


def _objective_plot_specs(
    f1_col, nodes_col, reaudit_col, cart_f1_col, cart_nodes_col, cart_reaudit_col,
    reaudit_axis_label="Percentage of nodes to re-audit", reaudit_title_phrase="re-audit cost",
    metric_label="F1-score",
):
    return [
        (reaudit_col, f1_col, reaudit_axis_label, metric_label,
         f"{metric_label} vs {reaudit_title_phrase}", cart_reaudit_col, cart_f1_col),
        (nodes_col, f1_col, "Total number of nodes", metric_label,
         f"{metric_label} vs simplicity", cart_nodes_col, cart_f1_col),
        (nodes_col, reaudit_col, "Total number of nodes", reaudit_axis_label,
         f"{reaudit_title_phrase[0].upper()}{reaudit_title_phrase[1:]} vs simplicity", cart_nodes_col, cart_reaudit_col),
    ]


def _parameter_colored_figure(
    df, parameter_col, parameter_label, output_stem, out_dir,
    f1_col, nodes_col, reaudit_col, cart_baseline,
    cart_f1_col, cart_nodes_col, cart_reaudit_col, dpi, cmap_name="viridis",
    reaudit_axis_label="Percentage of nodes to re-audit", reaudit_title_phrase="re-audit cost",
    metric_label="F1-score",
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = _objective_plot_specs(
        f1_col, nodes_col, reaudit_col, cart_f1_col, cart_nodes_col, cart_reaudit_col,
        reaudit_axis_label=reaudit_axis_label, reaudit_title_phrase=reaudit_title_phrase,
        metric_label=metric_label,
    )

    unique_values = np.sort(df[parameter_col].unique())
    n_values = len(unique_values)
    rank_of_value = {v: i for i, v in enumerate(unique_values)}
    ranks = df[parameter_col].map(rank_of_value).to_numpy()
    cmap = plt.get_cmap(cmap_name, max(n_values, 1))

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(19, 6.3))

    scatter = None
    for ax, (x_col, y_col, x_label, y_label, title, cart_x_col, cart_y_col) in zip(axes, specs):
        scatter = ax.scatter(
            df[x_col], df[y_col], c=ranks, cmap=cmap, vmin=-0.5, vmax=n_values - 0.5,
            s=26, alpha=0.75, edgecolors="none", zorder=2,
        )
        if cart_baseline is not None:
            ax.scatter(
                cart_baseline[cart_x_col], cart_baseline[cart_y_col], marker="*", s=220,
                color="black", edgecolors="white", linewidths=1.0, label="CART retraining", zorder=3,
            )
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)
        ax.margins(x=0.08, y=0.12)

    if cart_baseline is not None:
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01), fontsize=9)

    fig.suptitle(f"Objective space colored by {parameter_label}", fontsize=15, y=0.98)

    fig.subplots_adjust(
        left=0.05, right=0.90, top=0.86,
        bottom=0.20 if cart_baseline is not None else 0.10, wspace=0.28,
    )

    cbar_ax = fig.add_axes([0.925, 0.22, 0.015, 0.58])
    cbar = fig.colorbar(scatter, cax=cbar_ax, ticks=range(n_values))
    cbar.ax.set_yticklabels([f"{v:g}" for v in unique_values], fontsize=7)
    cbar.set_label(parameter_label)

    fig.savefig(out_dir / f"{output_stem}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(out_dir / f"{output_stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def exploratory_parameter_plots(
    df, out_dir, cart_baseline, f1_col, nodes_col, reaudit_col,
    w_simp_col, w_simi_col, t_col, cart_f1_col, cart_nodes_col, cart_reaudit_col, dpi=300,
    reaudit_axis_label="Percentage of nodes to re-audit", reaudit_title_phrase="re-audit cost",
    metric_label="F1-score",
):
    specs = [(w_simp_col, "w_simp", "effect_w_simp"), (w_simi_col, "w_simi", "effect_w_simi"), (t_col, r"$t$", "effect_t")]
    for parameter_col, label, stem in specs:
        _parameter_colored_figure(
            df, parameter_col, label, stem, out_dir, f1_col, nodes_col, reaudit_col,
            cart_baseline, cart_f1_col, cart_nodes_col, cart_reaudit_col, dpi,
            reaudit_axis_label=reaudit_axis_label, reaudit_title_phrase=reaudit_title_phrase,
            metric_label=metric_label,
        )


def _metric_specs(f1_col, nodes_col, sim_col, has_similarity, metric_label="F1-score"):
    specs = [(f1_col, metric_label, metric_label), (nodes_col, "Simplicity", "Total number of nodes")]
    if has_similarity:
        specs.append((sim_col, "Similarity to T_old", "Rule-set similarity"))
    return specs


def _marginal_effect_figure(df, parameter_col, parameter_label, output_stem, out_dir, metric_specs, dpi):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x = np.sort(df[parameter_col].unique())
    fig, axes = plt.subplots(nrows=1, ncols=len(metric_specs), figsize=(6.3 * len(metric_specs), 5.6))
    axes = np.atleast_1d(axes)

    for ax, (metric_col, title, ylabel) in zip(axes, metric_specs):
        grouped = df.groupby(parameter_col, dropna=False)[metric_col]
        median = grouped.median().reindex(x).to_numpy()
        q1 = grouped.quantile(0.25).reindex(x).to_numpy()
        q3 = grouped.quantile(0.75).reindex(x).to_numpy()

        ax.fill_between(x, q1, q3, alpha=0.25, label="IQR (Q1-Q3)")
        ax.plot(x, median, marker="o", linewidth=1.8, label="Median")

        ax.set_xlabel(parameter_label)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)
        ax.set_xticks(x)
        ax.tick_params(axis="x", labelrotation=45)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01), fontsize=9)
    fig.suptitle(f"Marginal effect of {parameter_label} (other parameters aggregated out)", fontsize=14, y=0.98)
    fig.subplots_adjust(bottom=0.28, top=0.84, wspace=0.32)

    fig.savefig(out_dir / f"{output_stem}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(out_dir / f"{output_stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def marginal_effect_plots(df, out_dir, metric_specs, w_simp_col, w_simi_col, t_col, dpi=300):
    specs = [(w_simp_col, "w_simp", "marginal_w_simp"), (w_simi_col, "w_simi", "marginal_w_simi"), (t_col, r"$t$", "marginal_t")]
    for parameter_col, label, stem in specs:
        if df[parameter_col].nunique(dropna=True) < 2:
            continue
        _marginal_effect_figure(df, parameter_col, label, stem, out_dir, metric_specs, dpi)


def pick_reference_configuration(
    pareto_configurations, f1_col, nodes_col, reaudit_col, w_simp_col, w_simi_col, t_col, reaudit_ascending=True,
):
    best = pareto_configurations.sort_values(
        by=[f1_col, nodes_col, reaudit_col, w_simp_col, w_simi_col, t_col],
        ascending=[False, True, reaudit_ascending, True, True, True],
    ).iloc[0]
    return float(best[w_simp_col]), float(best[w_simi_col]), float(best[t_col])


def controlled_effect_plot(
    df, w_simp0, w_simi0, t0, w_simp_col, w_simi_col, t_col, out_dir, metric_specs, dpi=300, tolerance=1e-6,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reference = {w_simp_col: w_simp0, w_simi_col: w_simi0, t_col: t0}
    rows = [
        (w_simp_col, "w_simp", [w_simi_col, t_col]),
        (w_simi_col, "w_simi", [w_simp_col, t_col]),
        (t_col, r"$t$", [w_simp_col, w_simi_col]),
    ]

    n_cols = len(metric_specs)
    fig, axes = plt.subplots(nrows=3, ncols=n_cols, figsize=(6.3 * n_cols, 4.6 * 3))

    for row_idx, (varied_col, varied_label, fixed_cols) in enumerate(rows):
        varies = df[varied_col].nunique(dropna=True) >= 2
        mask = np.ones(len(df), dtype=bool)
        for fc in fixed_cols:
            mask &= np.isclose(df[fc].to_numpy(dtype=float), reference[fc], atol=tolerance, equal_nan=True)
        subset = df.loc[mask].sort_values(varied_col) if varies else df.iloc[0:0]
        x_values = np.sort(df[varied_col].unique()) if varies else np.array([])

        for col_idx, (metric_col, title, ylabel) in enumerate(metric_specs):
            ax = axes[row_idx, col_idx]
            if not varies:
                ax.text(0.5, 0.5, "constant", ha="center", va="center", transform=ax.transAxes)
            elif subset.empty:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            else:
                ax.plot(subset[varied_col], subset[metric_col], marker="o", linewidth=1.8)
            ax.set_xlabel(varied_label)
            ax.set_ylabel(ylabel)
            if row_idx == 0:
                ax.set_title(title)
            ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)
            if varies:
                ax.set_xticks(x_values)
            ax.tick_params(axis="x", labelrotation=45, labelsize=8)

    fig.suptitle(f"One-factor-at-a-time (reference: w_simp={w_simp0:g}, w_simi={w_simi0:g}, t={t0:g})", fontsize=14, y=0.995)
    fig.subplots_adjust(top=0.92, hspace=0.45, wspace=0.32)

    fig.savefig(out_dir / "one_factor_at_a_time.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(out_dir / "one_factor_at_a_time.pdf", bbox_inches="tight")
    plt.close(fig)


def _annotate_pareto_points(ax, pareto_df, x_col, y_col):
    for i, (_, row) in enumerate(pareto_df.iterrows()):
        text = f"{row['pareto_id']}\nn={int(row['n_configurations'])}"
        dy = 9 if i % 2 == 0 else -21
        ax.annotate(
            text, (row[x_col], row[y_col]), xytext=(7, dy), textcoords="offset points",
            fontsize=7, alpha=0.9, arrowprops=dict(arrowstyle="-", color="gray", lw=0.4, alpha=0.6),
        )


def pareto_plots(
    distinct_df, out_dir, cart_baseline, cart_is_dominated, f1_col, nodes_col, reaudit_col,
    cart_f1_col, cart_nodes_col, cart_reaudit_col, dpi=300,
    reaudit_axis_label="Percentage of nodes to re-audit", reaudit_title_phrase="re-audit cost",
    error_bars_df=None, col_key_map=None, metric_label="F1-score",
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pareto_df = distinct_df.loc[distinct_df["is_pareto"]].copy()
    dominated_df = distinct_df.loc[~distinct_df["is_pareto"]].copy()
    if error_bars_df is not None and "pareto_id" in pareto_df.columns:
        pareto_df = pareto_df.merge(error_bars_df, on="pareto_id", how="left")
    specs = _objective_plot_specs(
        f1_col, nodes_col, reaudit_col, cart_f1_col, cart_nodes_col, cart_reaudit_col,
        reaudit_axis_label=reaudit_axis_label, reaudit_title_phrase=reaudit_title_phrase,
        metric_label=metric_label,
    )

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(19, 6.3))

    cart_color = "lightgray" if cart_is_dominated else "black"
    cart_edge = "gray" if cart_is_dominated else "white"
    cart_text = "gray" if cart_is_dominated else "black"
    cart_label = f"CART retraining ({'dominated' if cart_is_dominated else 'non-dominated'})"

    for ax, (x_col, y_col, x_label, y_label, title, cart_x_col, cart_y_col) in zip(axes, specs):
        ax.scatter(
            dominated_df[x_col], dominated_df[y_col], color="lightgray", edgecolors="none",
            alpha=0.55, s=30, label="Dominated solutions", zorder=1,
        )

        ordered = pareto_df.sort_values(by=[x_col, y_col], kind="stable")
        if len(ordered) >= 2:
            ax.plot(
                ordered[x_col], ordered[y_col], linestyle="--", linewidth=1.3, alpha=0.80,
                label="Pareto front line", zorder=2,
            )

        if error_bars_df is not None and col_key_map is not None:
            x_key = col_key_map.get(x_col)
            y_key = col_key_map.get(y_col)
            xerr = None
            yerr = None
            if x_key and f"{x_key}_q1" in pareto_df.columns:
                xerr = np.vstack([
                    (pareto_df[x_col] - pareto_df[f"{x_key}_q1"]).clip(lower=0).to_numpy(),
                    (pareto_df[f"{x_key}_q3"] - pareto_df[x_col]).clip(lower=0).to_numpy(),
                ])
            if y_key and f"{y_key}_q1" in pareto_df.columns:
                yerr = np.vstack([
                    (pareto_df[y_col] - pareto_df[f"{y_key}_q1"]).clip(lower=0).to_numpy(),
                    (pareto_df[f"{y_key}_q3"] - pareto_df[y_col]).clip(lower=0).to_numpy(),
                ])
            if xerr is not None or yerr is not None:
                ax.errorbar(
                    pareto_df[x_col], pareto_df[y_col], xerr=xerr, yerr=yerr,
                    fmt="none", ecolor="black", elinewidth=1.0, capsize=3, alpha=0.5, zorder=3,
                    label="Variability across mutations (IQR Q1-Q3)",
                )

        ax.scatter(
            pareto_df[x_col], pareto_df[y_col], edgecolors="black", linewidths=0.5, s=65,
            label="Pareto-optimal solutions", zorder=4,
        )
        _annotate_pareto_points(ax, pareto_df, x_col, y_col)

        if cart_baseline is not None:
            ax.scatter(
                cart_baseline[cart_x_col], cart_baseline[cart_y_col], marker="*", s=280,
                color=cart_color, edgecolors=cart_edge, linewidths=1.0, label=cart_label, zorder=5,
            )
            ax.annotate(
                "CART", (cart_baseline[cart_x_col], cart_baseline[cart_y_col]), xytext=(7, 7),
                textcoords="offset points", fontsize=8, fontweight="bold", color=cart_text,
            )

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)
        ax.margins(x=0.08, y=0.12)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=len(handles), fontsize=9)
    fig.suptitle(f"Pareto front: {metric_label}, simplicity and {reaudit_title_phrase}", fontsize=15, y=0.98)
    fig.text(
        0.5, 0.075, "The front is computed jointly over all three objectives; each panel shows a 2D projection.",
        ha="center", va="center", fontsize=9,
    )
    fig.subplots_adjust(left=0.055, right=0.985, top=0.86, bottom=0.20, wspace=0.28)

    fig.savefig(out_dir / "pareto_combined.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(out_dir / "pareto_combined.pdf", bbox_inches="tight")
    plt.close(fig)


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
    sim_col=DEFAULT_SIM_COL, w_simp_col=DEFAULT_W_SIMP_COL, w_simi_col=DEFAULT_W_SIMI_COL, t_col=DEFAULT_T_COL,
    cart_f1_col=DEFAULT_CART_F1_COL, cart_nodes_col=DEFAULT_CART_NODES_COL, cart_reaudit_col=DEFAULT_CART_REAUDIT_COL,
    ref_w_simp=None, ref_w_simi=None, ref_t=None, dpi=300,
    reaudit_maximize=True,
    reaudit_axis_label="Rule-set similarity to $T_{old}$",
    reaudit_title_phrase="similarity to T_old",
    reaudit_header="Similarity",
    f1_q1_col=None, f1_q3_col=None, nodes_q1_col=None, nodes_q3_col=None,
    reaudit_q1_col=None, reaudit_q3_col=None,
    metric_label="F1-score", metric_header="F1 test",
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
    has_similarity = check_similarity_column(df, sim_col)
    metric_specs = _metric_specs(f1, nodes, sim_col, has_similarity, metric_label=metric_label)

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

    col_key_map = {f1: "f1", nodes: "nodes", reaudit: "reaudit"}
    q1q3_specs = [
        ("f1", f1_q1_col, f1_q3_col), ("nodes", nodes_q1_col, nodes_q3_col), ("reaudit", reaudit_q1_col, reaudit_q3_col),
    ]
    available_q1q3 = [(k, q1c, q3c) for k, q1c, q3c in q1q3_specs if q1c and q3c and q1c in df.columns and q3c in df.columns]
    error_bars_df = None
    if available_q1q3:
        agg_kwargs = {}
        for k, q1c, q3c in available_q1q3:
            agg_kwargs[f"{k}_q1"] = (q1c, "mean")
            agg_kwargs[f"{k}_q3"] = (q3c, "mean")
        error_bars_df = pareto_configurations.groupby("pareto_id").agg(**agg_kwargs).reset_index()

    exploratory_parameter_plots(
        df, out_dir / "parameter_effects", cart_baseline, f1, nodes, reaudit,
        w_simp, w_simi, t, cart_f1, cart_nodes, cart_reaudit, dpi=dpi,
        reaudit_axis_label=reaudit_axis_label, reaudit_title_phrase=reaudit_title_phrase,
        metric_label=metric_label,
    )
    marginal_effect_plots(df, out_dir / "marginal_effects", metric_specs, w_simp, w_simi, t, dpi=dpi)

    if ref_w_simp is not None and ref_w_simi is not None and ref_t is not None:
        w_simp0, w_simi0, t0 = ref_w_simp, ref_w_simi, ref_t
    else:
        w_simp0, w_simi0, t0 = pick_reference_configuration(
            pareto_configurations, f1, nodes, reaudit, w_simp, w_simi, t, reaudit_ascending=reaudit_ascending,
        )
    controlled_effect_plot(df, w_simp0, w_simi0, t0, w_simp, w_simi, t, out_dir, metric_specs, dpi=dpi)

    pareto_plots(
        distinct_df, out_dir / "pareto", cart_baseline, cart_is_dominated, f1, nodes, reaudit,
        cart_f1, cart_nodes, cart_reaudit, dpi=dpi,
        reaudit_axis_label=reaudit_axis_label, reaudit_title_phrase=reaudit_title_phrase,
        error_bars_df=error_bars_df, col_key_map=col_key_map, metric_label=metric_label,
    )

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
    print(f"One-factor-at-a-time reference: w_simp={w_simp0:g}, w_simi={w_simi0:g}, t={t0:g}")

    print()
    if cart_baseline is None:
        print("CART: baseline unavailable.")
    elif cart_is_dominated:
        print(f"CART is dominated by {len(cart_dominators)} distinct Keep-Regrow results.")
        print("In the Pareto plots CART is shown as a light gray star.")
    else:
        print("CART is not dominated by Keep-Regrow.")
        print("In the Pareto plots CART is shown as a black star.")

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
    parser.add_argument("--sim-col", default=DEFAULT_SIM_COL, help="Rule-set similarity column (optional)")
    parser.add_argument("--w-simp-col", default=DEFAULT_W_SIMP_COL)
    parser.add_argument("--w-simi-col", default=DEFAULT_W_SIMI_COL)
    parser.add_argument("--t-col", default=DEFAULT_T_COL)
    parser.add_argument("--cart-f1-col", default=DEFAULT_CART_F1_COL)
    parser.add_argument("--cart-nodes-col", default=DEFAULT_CART_NODES_COL)
    parser.add_argument("--cart-reaudit-col", default=DEFAULT_CART_REAUDIT_COL)
    parser.add_argument(
        "--reaudit-maximize", action=argparse.BooleanOptionalAction, default=True,
        help="Direction for --reaudit-col: maximize (default -- e.g. sim_old_new, rule-set similarity "
             "to T_old, matching grid_search_dp.py's own default analysis and "
             "trial_pareto_analysis.py/run_baseline_repair.py) or minimize (e.g. pct_to_reaudit, the "
             "original re-audit-cost analysis). To reproduce the original re-audit-cost Pareto "
             "analysis, pass --reaudit-col pct_to_reaudit --cart-reaudit-col cart_pct_to_reaudit "
             "--no-reaudit-maximize.",
    )
    parser.add_argument("--reaudit-axis-label", default=None, help="Override third-objective axis label (default depends on --reaudit-maximize).")
    parser.add_argument("--reaudit-title-phrase", default=None, help="Override third-objective phrase used in plot titles (default depends on --reaudit-maximize).")
    parser.add_argument("--reaudit-header", default=None, help="Override third-objective column header in the printed summary table (default depends on --reaudit-maximize).")
    parser.add_argument("--metric-label", default="F1-score", help="Axis/title label for --f1-col (default 'F1-score'; pass 'Accuracy' when --f1-col points at an accuracy column).")
    parser.add_argument("--metric-header", default="F1 test", help="Printed summary-table column header for --f1-col (default 'F1 test').")
    parser.add_argument("--ref-w-simp", type=float, default=None, help="Override reference w_simp for one-factor-at-a-time")
    parser.add_argument("--ref-w-simi", type=float, default=None, help="Override reference w_simi for one-factor-at-a-time")
    parser.add_argument("--ref-t", type=float, default=None, help="Override reference t for one-factor-at-a-time")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    reaudit_axis_label = args.reaudit_axis_label
    reaudit_title_phrase = args.reaudit_title_phrase
    reaudit_header = args.reaudit_header
    if args.reaudit_maximize:
        reaudit_axis_label = reaudit_axis_label or "Rule-set similarity to $T_{old}$"
        reaudit_title_phrase = reaudit_title_phrase or "similarity to T_old"
        reaudit_header = reaudit_header or "Similarity"
    else:
        reaudit_axis_label = reaudit_axis_label or "Percentage of nodes to re-audit"
        reaudit_title_phrase = reaudit_title_phrase or "re-audit cost"
        reaudit_header = reaudit_header or "Reaudit %"

    run_pareto_analysis(
        csv_path=args.csv_path, out_dir=args.out_dir,
        f1_col=args.f1_col, nodes_col=args.nodes_col, reaudit_col=args.reaudit_col,
        sim_col=args.sim_col, w_simp_col=args.w_simp_col, w_simi_col=args.w_simi_col, t_col=args.t_col,
        cart_f1_col=args.cart_f1_col, cart_nodes_col=args.cart_nodes_col, cart_reaudit_col=args.cart_reaudit_col,
        ref_w_simp=args.ref_w_simp, ref_w_simi=args.ref_w_simi, ref_t=args.ref_t, dpi=args.dpi,
        reaudit_maximize=args.reaudit_maximize,
        reaudit_axis_label=reaudit_axis_label, reaudit_title_phrase=reaudit_title_phrase,
        reaudit_header=reaudit_header,
        metric_label=args.metric_label, metric_header=args.metric_header,
    )


if __name__ == "__main__":
    main()
