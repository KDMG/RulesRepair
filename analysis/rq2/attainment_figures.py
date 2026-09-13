import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from analysis.core.compare_kr_cart_dominance import (
    DEFAULT_MAX_DEPTH,
    DEFAULT_NODES_MIN,
    DEFAULT_TOL,
    infer_dp_label,
    infer_seed_from_path,
)
from analysis.core.mutation_types import MUTATION_TYPES, filter_by_mutation_type
from analysis.rq2.statistical.igd_core import per_trial_normalized_sets


def weakly_dominates_matrix(points, z_grid, tol=DEFAULT_TOL):
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    z = np.atleast_2d(np.asarray(z_grid, dtype=float))
    if pts.size == 0 or z.size == 0:
        return np.zeros(len(z), dtype=bool)
    ge = pts[:, None, :] >= (z[None, :, :] - tol)
    dominates = ge.all(axis=2)
    return dominates.any(axis=0)


def seed_attainment_matrix(trial_sets, z_grid, tol=DEFAULT_TOL):
    z = np.atleast_2d(np.asarray(z_grid, dtype=float))
    if not trial_sets:
        return np.zeros(len(z), dtype=float)
    total = np.zeros(len(z), dtype=float)
    for points in trial_sets:
        total += weakly_dominates_matrix(points, z, tol=tol).astype(float)
    return total / len(trial_sets)


def seed_difference_matrix(kr_trial_sets, baseline_trial_sets, z_grid, tol=DEFAULT_TOL):
    if len(kr_trial_sets) != len(baseline_trial_sets):
        raise ValueError(
            f"kr_trial_sets ({len(kr_trial_sets)}) and baseline_trial_sets ({len(baseline_trial_sets)}) "
            f"must have the same length -- one paired entry per trial of this seed's view, both from the "
            f"SAME trial (see module docstring: paired means paired at the trial level)."
        )
    a_kr = seed_attainment_matrix(kr_trial_sets, z_grid, tol=tol)
    a_baseline = seed_attainment_matrix(baseline_trial_sets, z_grid, tol=tol)
    return a_kr - a_baseline


def attainment_difference(per_seed_trials, z_grid, tol=DEFAULT_TOL):
    z = np.atleast_2d(np.asarray(z_grid, dtype=float))
    seeds = list(per_seed_trials.keys())
    if not seeds:
        return np.full(len(z), np.nan), 0
    total = np.zeros(len(z), dtype=float)
    for seed, trials in per_seed_trials.items():
        if not trials:
            raise ValueError(
                f"seed {seed} has an empty trial list -- build_per_seed_trials() must OMIT zero-trial "
                f"seeds entirely rather than including them as empty lists (a seed with no data for this "
                f"view does not contribute to the D(z) average at all, per Step 4 of the design; see that "
                f"function's docstring)."
            )
        kr_sets = [t[0] for t in trials]
        baseline_sets = [[t[1]] for t in trials]
        total += seed_difference_matrix(kr_sets, baseline_sets, z, tol=tol)
    return total / len(seeds), len(seeds)


def build_per_seed_trials(csv_paths, mutation_type=None, max_depth=DEFAULT_MAX_DEPTH, nodes_min=DEFAULT_NODES_MIN,
                           tol=DEFAULT_TOL, baseline_acc_col="acc_cart_test", baseline_nodes_col="cart_total_nodes",
                           baseline_jaccard_col="sim_old_cart_jaccard"):
    per_seed = {}
    dp_labels = set()
    for p in csv_paths:
        seed = infer_seed_from_path(p)
        if seed is None:
            raise ValueError(f"{p}: no 'seed_<N>' segment found in the path -- every input must be seed-tagged.")
        dp_labels.add(infer_dp_label(p))
        df = pd.read_csv(p)
        if mutation_type is not None:
            df = filter_by_mutation_type(df, mutation_type, source=str(p))
        if len(df) == 0:
            continue
        trial_sets = per_trial_normalized_sets(
            df, max_depth=max_depth, nodes_min=nodes_min, tol=tol,
            baseline_acc_col=baseline_acc_col, baseline_nodes_col=baseline_nodes_col,
            baseline_jaccard_col=baseline_jaccard_col,
        )
        if trial_sets:
            per_seed[seed] = list(trial_sets.values())
    if len(dp_labels) > 1:
        raise ValueError(
            f"Input paths span {len(dp_labels)} different decision points ({sorted(dp_labels)}) -- "
            f"pass paths for exactly one decision point at a time."
        )
    return per_seed


def build_per_seed_trials_by_mutation_type(csv_paths, mutation_types=MUTATION_TYPES, **kwargs):
    tables = {}
    for mt in mutation_types:
        per_seed = build_per_seed_trials(csv_paths, mutation_type=mt, **kwargs)
        tables[mt] = per_seed if per_seed else None
    return tables


def observed_breakpoints(per_seed_trials):
    all_points = []
    for trials in per_seed_trials.values():
        for kr_front, baseline_point in trials:
            all_points.extend(kr_front)
            all_points.append(baseline_point)
    if not all_points:
        return np.array([]), np.array([]), np.array([])
    arr = np.asarray(all_points, dtype=float)
    return np.unique(arr[:, 0]), np.unique(arr[:, 1]), np.unique(arr[:, 2])


def build_breakpoint_grid(per_seed_trials):
    acc_values, nodes_values, jaccard_values = observed_breakpoints(per_seed_trials)
    if len(acc_values) == 0:
        return np.empty((0, 3)), acc_values, nodes_values, jaccard_values
    grid = np.array(np.meshgrid(acc_values, nodes_values, jaccard_values, indexing="ij")).reshape(3, -1).T
    return grid, acc_values, nodes_values, jaccard_values


def compute_attainment_field(per_seed_trials, tol=DEFAULT_TOL):
    grid, acc_values, nodes_values, jaccard_values = build_breakpoint_grid(per_seed_trials)
    if len(grid) == 0:
        return pd.DataFrame(columns=["acc", "nodes", "jaccard", "D", "n_seeds"])
    d_values, n_seeds = attainment_difference(per_seed_trials, grid, tol=tol)
    return pd.DataFrame({
        "acc": grid[:, 0], "nodes": grid[:, 1], "jaccard": grid[:, 2],
        "D": d_values, "n_seeds": n_seeds,
    })


def attainment_components(per_seed_trials, z_grid, tol=DEFAULT_TOL):
    z = np.atleast_2d(np.asarray(z_grid, dtype=float))
    seeds = list(per_seed_trials.keys())
    if not seeds:
        return np.full(len(z), np.nan), np.full(len(z), np.nan), 0
    total_kr = np.zeros(len(z), dtype=float)
    total_baseline = np.zeros(len(z), dtype=float)
    for seed, trials in per_seed_trials.items():
        if not trials:
            raise ValueError(
                f"seed {seed} has an empty trial list -- build_per_seed_trials() must OMIT zero-trial "
                f"seeds entirely (see attainment_difference()'s identical check)."
            )
        kr_sets = [t[0] for t in trials]
        baseline_sets = [[t[1]] for t in trials]
        total_kr += seed_attainment_matrix(kr_sets, z, tol=tol)
        total_baseline += seed_attainment_matrix(baseline_sets, z, tol=tol)
    return total_kr / len(seeds), total_baseline / len(seeds), len(seeds)


def compute_attainment_components_field(per_seed_trials, tol=DEFAULT_TOL):
    grid, acc_values, nodes_values, jaccard_values = build_breakpoint_grid(per_seed_trials)
    if len(grid) == 0:
        return pd.DataFrame(columns=["acc", "nodes", "jaccard", "A_KR", "A_baseline", "D", "n_seeds"])
    a_kr, a_baseline, n_seeds = attainment_components(per_seed_trials, grid, tol=tol)
    return pd.DataFrame({
        "acc": grid[:, 0], "nodes": grid[:, 1], "jaccard": grid[:, 2],
        "A_KR": a_kr, "A_baseline": a_baseline, "D": a_kr - a_baseline, "n_seeds": n_seeds,
    })


POOLED_BUCKET_KEY = "__pooled__"


def build_pooled_trials(csv_paths, mutation_type=None, max_depth=DEFAULT_MAX_DEPTH, nodes_min=DEFAULT_NODES_MIN,
                         tol=DEFAULT_TOL, baseline_acc_col="acc_cart_test", baseline_nodes_col="cart_total_nodes",
                         baseline_jaccard_col="sim_old_cart_jaccard"):
    per_seed = build_per_seed_trials(
        csv_paths, mutation_type=mutation_type, max_depth=max_depth, nodes_min=nodes_min, tol=tol,
        baseline_acc_col=baseline_acc_col, baseline_nodes_col=baseline_nodes_col,
        baseline_jaccard_col=baseline_jaccard_col,
    )
    pooled = [trial for trials in per_seed.values() for trial in trials]
    if not pooled:
        return {}
    return {POOLED_BUCKET_KEY: pooled}


def build_moocore_input(per_seed_points):
    rows = []
    for i, (seed, points) in enumerate(per_seed_points.items(), start=1):
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        for row in pts:
            rows.append(list(row) + [i])
    return np.asarray(rows, dtype=float)


def eafdiff_maximise_crosscheck(kr_per_seed_points, baseline_per_seed_points, tol=DEFAULT_TOL):
    import moocore

    kr_seeds = set(kr_per_seed_points.keys())
    baseline_seeds = set(baseline_per_seed_points.keys())
    if kr_seeds != baseline_seeds:
        raise ValueError(
            f"Unbalanced design: KR seeds {sorted(kr_seeds)} != baseline seeds {sorted(baseline_seeds)} -- "
            f"this cross-check requires the SAME seed set (paired, one set per seed) on both sides."
        )
    n_seeds = len(kr_seeds)

    x = build_moocore_input(kr_per_seed_points)
    y = build_moocore_input(baseline_per_seed_points)
    diff = moocore.eafdiff(x, y, maximise=True, intervals=None)
    transition_points = diff[:, :-1]
    moocore_raw = diff[:, -1]
    moocore_D = moocore_raw / n_seeds

    per_seed_trials = {
        seed: [(list(np.atleast_2d(kr_per_seed_points[seed])), tuple(np.atleast_2d(baseline_per_seed_points[seed])[0]))]
        for seed in kr_seeds
    }
    our_D, our_n_seeds = attainment_difference(per_seed_trials, transition_points, tol=tol)
    if our_n_seeds != n_seeds:
        raise ValueError(f"internal check failed: our_n_seeds={our_n_seeds} != n_seeds={n_seeds}")

    return transition_points, our_D, moocore_D, n_seeds


def eaf_single_side_crosscheck(per_seed_points, z_points, tol=DEFAULT_TOL):
    import moocore

    z = np.atleast_2d(np.asarray(z_points, dtype=float))

    seeds = list(per_seed_points.keys())
    our_A = np.zeros(len(z), dtype=float)
    for seed in seeds:
        our_A += weakly_dominates_matrix(per_seed_points[seed], z, tol=tol).astype(float)
    our_A /= len(seeds)

    x_neg = build_moocore_input({s: -np.atleast_2d(np.asarray(p, dtype=float)) for s, p in per_seed_points.items()})
    eaf_result = moocore.eaf(x_neg[:, :-1], x_neg[:, -1])
    eaf_points = eaf_result[:, :-1]
    eaf_pct = eaf_result[:, -1]

    moocore_A = np.zeros(len(z), dtype=float)
    for i, zi in enumerate(z):
        zi_neg = -zi
        attained_mask = np.all(eaf_points <= zi_neg + tol, axis=1)
        moocore_A[i] = eaf_pct[attained_mask].max() / 100.0 if attained_mask.any() else 0.0

    return our_A, moocore_A


AXIS_LABELS = {
    "acc": "Accuracy",
    "nodes": "Simplicity",
    "jaccard": "Jaccard similarity",
}

COLOR_RANGE = 1.0

COLORBAR_LABEL = r"$D(z) = \mathrm{EAF}_{KR}(z) - \mathrm{EAF}_{baseline}(z)$"

DEFAULT_SLICE_PERCENTILES = (25, 50, 75)
DEFAULT_SLICE_LABELS = ("Low", "Medium", "High")


def select_representative_accuracy_slices(acc_values, percentiles=DEFAULT_SLICE_PERCENTILES):
    unique_acc = np.unique(np.asarray(acc_values, dtype=float))
    if len(unique_acc) == 0:
        return np.array([])
    chosen = np.atleast_1d(np.percentile(unique_acc, percentiles, method="nearest"))
    deduped = []
    for v in chosen:
        v = float(v)
        if not any(abs(v - seen) < 1e-12 for seen in deduped):
            deduped.append(v)
    return np.array(deduped)


def _cell_edges(values, lo=0.0):
    values = np.sort(np.asarray(values, dtype=float))
    if len(values) == 0:
        return np.array([lo, 1.0])
    return np.concatenate(([lo], values))


def plot_attainment_difference_slices(
    field_df, percentiles=DEFAULT_SLICE_PERCENTILES, slice_labels=DEFAULT_SLICE_LABELS,
    panel_size=(3.4, 3.0),
):
    if len(field_df) == 0:
        raise ValueError(
            "field_df is empty -- nothing to plot. This view (dataset/decision point/mutation type "
            "combination) had zero trials in every seed (see analysis.attainment_field's 'not applicable' "
            "convention); check for this case before calling plot_attainment_difference_slices(), do not "
            "plot an empty field silently."
        )

    slice_values = select_representative_accuracy_slices(field_df["acc"].to_numpy(), percentiles=percentiles)
    n_panels = len(slice_values)

    fig, axes = plt.subplots(
        1, n_panels, figsize=(panel_size[0] * n_panels, panel_size[1]), squeeze=False, sharey=True,
    )
    axes = axes[0]

    HI = 1.0
    mesh = None
    for ax, acc_v, label in zip(axes, slice_values, slice_labels):
        sub = field_df[np.isclose(field_df["acc"].to_numpy(), acc_v, atol=1e-9)]
        nodes_values = np.sort(sub["nodes"].unique())
        jaccard_values = np.sort(sub["jaccard"].unique())
        d_grid = (
            sub.pivot(index="jaccard", columns="nodes", values="D")
            .reindex(index=jaccard_values, columns=nodes_values)
            .to_numpy()
        )
        x_edges = _cell_edges(nodes_values)
        y_edges = _cell_edges(jaccard_values)

        if x_edges[-1] < HI:
            x_edges = np.append(x_edges, HI)
            d_grid = np.concatenate([d_grid, np.zeros((d_grid.shape[0], 1))], axis=1)
        if y_edges[-1] < HI:
            y_edges = np.append(y_edges, HI)
            d_grid = np.concatenate([d_grid, np.zeros((1, d_grid.shape[1]))], axis=0)

        mesh = ax.pcolormesh(
            x_edges, y_edges, d_grid, cmap="RdBu_r", vmin=-COLOR_RANGE, vmax=COLOR_RANGE, shading="flat",
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xlabel(AXIS_LABELS["nodes"], fontsize=10)
        ax.set_title(f"{label} Accuracy (={acc_v:.3f})", fontsize=9)
        ax.tick_params(labelsize=8)

    axes[0].set_ylabel(AXIS_LABELS["jaccard"], fontsize=10)

    fig.colorbar(mesh, ax=axes.tolist(), shrink=0.85, pad=0.02, label=COLORBAR_LABEL)
    return fig


ACCURACY_SLICE_PERCENTILES = (0, 50, 100)
ACCURACY_SLICE_LABELS = ("Min", "Median", "Max")


def select_min_median_max_accuracy_slices(acc_values):
    return select_representative_accuracy_slices(acc_values, percentiles=ACCURACY_SLICE_PERCENTILES)


BASELINE_EAF_CMAP = "Blues"

RULESREPAIR_LABEL = "RulesRepair"


def _latex_available():
    try:
        with plt.rc_context({"text.usetex": True}):
            fig = plt.figure()
            fig.text(0.5, 0.5, r"\textsc{test}", usetex=True)
            fig.canvas.draw()
        plt.close(fig)
        return True
    except Exception:
        return False


_LATEX_AVAILABLE_CACHE = None


def _smallcaps(text):
    global _LATEX_AVAILABLE_CACHE
    if _LATEX_AVAILABLE_CACHE is None:
        _LATEX_AVAILABLE_CACHE = _latex_available()
    if _LATEX_AVAILABLE_CACHE:
        return rf"\textsc{{{text}}}", True
    return text, False


TITLE_FONTSIZE = 11
TICK_LABELSIZE = 9
AXIS_LABEL_FONTSIZE = 12
COLORBAR_LABEL_FONTSIZE = 11
COLORBAR_TICK_LABELSIZE = 9


def plot_eaf_accuracy_slices_figure(
    components_field_df, percentiles=ACCURACY_SLICE_PERCENTILES, slice_labels=ACCURACY_SLICE_LABELS,
    baseline_label="CART", panel_size=(3.8, 2.2),
):
    if len(components_field_df) == 0:
        raise ValueError(
            "components_field_df is empty -- nothing to plot. This view (dataset/decision point/mutation "
            "type combination) had zero trials in every seed; check for this case before calling "
            "plot_eaf_accuracy_slices_figure(), do not plot an empty field silently."
        )

    slice_values = select_representative_accuracy_slices(
        components_field_df["acc"].to_numpy(), percentiles=percentiles,
    )
    n_cols = len(slice_values)

    nodes_values = np.sort(components_field_df["nodes"].unique())
    jaccard_values = np.sort(components_field_df["jaccard"].unique())
    x_edges_base = _cell_edges(nodes_values)
    y_edges_base = _cell_edges(jaccard_values)

    HI = 1.0
    kr_label, kr_usetex = _smallcaps(RULESREPAIR_LABEL)
    baseline_label_sc, baseline_usetex = _smallcaps(baseline_label)
    row_specs = [
        ("A_KR", BASELINE_EAF_CMAP, 0.0, 1.0),
        ("A_baseline", EAF_VALUE_CMAP, 0.0, 1.0),
    ]
    n_rows = len(row_specs)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(panel_size[0] * n_cols, panel_size[1] * n_rows), squeeze=False,
        gridspec_kw={"wspace": 0.25, "hspace": 0.35},
    )

    mesh_kr = None
    mesh_baseline = None
    for col_idx, (acc_v, label) in enumerate(zip(slice_values, slice_labels)):
        sub = components_field_df[np.isclose(components_field_df["acc"].to_numpy(), acc_v, atol=1e-9)]

        for row_idx, (value_col, cmap, vmin, vmax) in enumerate(row_specs):
            grid = (
                sub.pivot(index="jaccard", columns="nodes", values=value_col)
                .reindex(index=jaccard_values, columns=nodes_values)
                .to_numpy()
            )
            x_edges, y_edges, g = x_edges_base, y_edges_base, grid
            if x_edges[-1] < HI:
                x_edges = np.append(x_edges, HI)
                g = np.concatenate([g, np.zeros((g.shape[0], 1))], axis=1)
            if y_edges[-1] < HI:
                y_edges = np.append(y_edges, HI)
                g = np.concatenate([g, np.zeros((1, g.shape[1]))], axis=0)

            ax = axes[row_idx][col_idx]
            mesh = ax.pcolormesh(x_edges, y_edges, g, cmap=cmap, vmin=vmin, vmax=vmax, shading="flat")
            if value_col == "A_KR":
                mesh_kr = mesh
            else:
                mesh_baseline = mesh
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("auto")
            ax.tick_params(labelsize=TICK_LABELSIZE)
            if row_idx == 0:
                ax.set_title(f"{label} Accuracy (={acc_v:.3f})", fontsize=TITLE_FONTSIZE)
            if col_idx == 0:
                ax.set_ylabel(AXIS_LABELS["jaccard"], fontsize=AXIS_LABEL_FONTSIZE)

    fig.subplots_adjust(bottom=0.08)
    fig.text(0.5, 0.0, AXIS_LABELS["nodes"], ha="center", va="top", fontsize=AXIS_LABEL_FONTSIZE)

    cbar_kr = fig.colorbar(
        mesh_kr, ax=[axes[0][c] for c in range(n_cols)], shrink=0.8, pad=0.02,
        label=rf"{kr_label} EAF(z)" if not kr_usetex else rf"\textsc{{{RULESREPAIR_LABEL}}} EAF(z)",
    )
    cbar_baseline_label = f"{baseline_label} EAF(z)" if not baseline_usetex else rf"\textsc{{{baseline_label}}} EAF(z)"
    cbar_baseline = fig.colorbar(
        mesh_baseline, ax=[axes[1][c] for c in range(n_cols)], shrink=0.8, pad=0.02, label=cbar_baseline_label,
    )
    for cbar in (cbar_kr, cbar_baseline):
        cbar.ax.yaxis.label.set_size(COLORBAR_LABEL_FONTSIZE)
        cbar.ax.tick_params(labelsize=COLORBAR_TICK_LABELSIZE)
    return fig


EAF_VALUE_CMAP = "Reds"
HELD_OUT_AXIS_DEFAULT = "acc"
ROTATED_AXES_DEFAULT = ("nodes", "jaccard")
DEFAULT_SLICE_ANGLES = (5, 45, 85)

MIP_ROTATED_AXES_DEFAULT = ("nodes", "acc")
MIP_HELD_OUT_AXIS_DEFAULT = "jaccard"


def _prosection_cells(bp_i, bp_j, phi_deg, hi=1.0):
    phi = np.radians(phi_deg)
    cos_p, sin_p = np.cos(phi), np.sin(phi)
    bp_i = np.sort(np.asarray(bp_i, dtype=float))
    bp_j = np.sort(np.asarray(bp_j, dtype=float))
    t_max = min(hi / cos_p, hi / sin_p)
    t_i = bp_i / cos_p
    t_j = bp_j / sin_p
    candidates = np.concatenate(([0.0], t_i, t_j, [t_max]))
    candidates = candidates[(candidates >= -1e-12) & (candidates <= t_max + 1e-9)]
    candidates = np.clip(candidates, 0.0, t_max)
    t_edges = np.unique(np.round(candidates, 10))

    cells = []
    for k in range(len(t_edges) - 1):
        t_mid = (t_edges[k] + t_edges[k + 1]) / 2.0
        zi, zj = t_mid * cos_p, t_mid * sin_p
        i_ge = bp_i[bp_i >= zi - 1e-9]
        j_ge = bp_j[bp_j >= zj - 1e-9]
        i_cell = float(i_ge[0]) if len(i_ge) else None
        j_cell = float(j_ge[0]) if len(j_ge) else None
        cells.append((i_cell, j_cell))
    return t_edges, cells


def _prosection_grid(components_field_df, value_col, held_out_axis, i_axis, j_axis, phi_deg):
    bp_i = np.sort(components_field_df[i_axis].unique())
    bp_j = np.sort(components_field_df[j_axis].unique())
    bp_k = np.sort(components_field_df[held_out_axis].unique())
    HI = 1.0

    t_edges, cells = _prosection_cells(bp_i, bp_j, phi_deg, hi=HI)

    k_edges = _cell_edges(bp_k)
    k_bands = list(bp_k)
    if k_edges[-1] < HI:
        k_edges = np.append(k_edges, HI)
        k_bands = k_bands + [None]

    lookup = components_field_df.set_index([i_axis, j_axis, held_out_axis])[value_col]
    grid = np.zeros((len(k_bands), len(cells)))
    for row, k_val in enumerate(k_bands):
        if k_val is None:
            continue
        for col, (i_cell, j_cell) in enumerate(cells):
            if i_cell is None or j_cell is None:
                continue
            grid[row, col] = lookup.loc[(i_cell, j_cell, k_val)]
    return t_edges, k_edges, grid


def plot_eaf_slicing_figure(
    components_field_df, angles=DEFAULT_SLICE_ANGLES, held_out_axis=HELD_OUT_AXIS_DEFAULT,
    rotated_axes=ROTATED_AXES_DEFAULT, panel_size=(3.6, 3.0),
):
    if len(components_field_df) == 0:
        raise ValueError(
            "components_field_df is empty -- nothing to plot. This view had zero trials in every seed; "
            "check for this case before calling plot_eaf_slicing_figure()."
        )
    i_axis, j_axis = rotated_axes
    n_angles = len(angles)
    fig, axes = plt.subplots(
        n_angles, 3, figsize=(panel_size[0] * 3, panel_size[1] * n_angles), squeeze=False,
    )

    panel_specs = [
        ("A_KR", "KR EAF", EAF_VALUE_CMAP, 0.0, 1.0),
        ("A_baseline", "Baseline EAF", EAF_VALUE_CMAP, 0.0, 1.0),
        ("D", "EAF difference", "RdBu_r", -COLOR_RANGE, COLOR_RANGE),
    ]
    mesh_eaf = None
    mesh_diff = None
    for row_idx, phi in enumerate(angles):
        for col_idx, (value_col, title, cmap, vmin, vmax) in enumerate(panel_specs):
            ax = axes[row_idx][col_idx]
            t_edges, k_edges, grid = _prosection_grid(components_field_df, value_col, held_out_axis, i_axis, j_axis, phi)
            mesh = ax.pcolormesh(t_edges, k_edges, grid, cmap=cmap, vmin=vmin, vmax=vmax, shading="flat")
            if value_col == "D":
                mesh_diff = mesh
            else:
                mesh_eaf = mesh
            ax.set_ylim(0, 1)
            ax.set_xlim(0, t_edges[-1])
            if row_idx == 0:
                ax.set_title(title, fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(f"{AXIS_LABELS[held_out_axis]}\nφ={phi}°", fontsize=9)
            if row_idx == n_angles - 1:
                ax.set_xlabel(f"{AXIS_LABELS[i_axis]}–{AXIS_LABELS[j_axis]} (rotated)", fontsize=8)
            ax.tick_params(labelsize=7)

    fig.colorbar(
        mesh_eaf, ax=[axes[r][c] for r in range(n_angles) for c in (0, 1)],
        shrink=0.8, pad=0.02, label="EAF(z)",
    )
    fig.colorbar(mesh_diff, ax=[axes[r][2] for r in range(n_angles)], shrink=0.8, pad=0.02, label=COLORBAR_LABEL)
    return fig


def plot_eaf_mip_figure(
    components_field_df, held_out_axis=MIP_HELD_OUT_AXIS_DEFAULT, rotated_axes=MIP_ROTATED_AXES_DEFAULT,
    panel_size=(4.2, 3.6),
):
    if len(components_field_df) == 0:
        raise ValueError(
            "components_field_df is empty -- nothing to plot. This view had zero trials in every seed; "
            "check for this case before calling plot_eaf_mip_figure()."
        )
    i_axis, j_axis = rotated_axes
    i_vals = np.sort(components_field_df[i_axis].unique())
    j_vals = np.sort(components_field_df[j_axis].unique())
    HI = 1.0

    max_d = (
        components_field_df.pivot_table(index=j_axis, columns=i_axis, values="D", aggfunc="max")
        .reindex(index=j_vals, columns=i_vals)
        .to_numpy()
    )
    min_d = (
        components_field_df.pivot_table(index=j_axis, columns=i_axis, values="D", aggfunc="min")
        .reindex(index=j_vals, columns=i_vals)
        .to_numpy()
    )
    kr_favoring = np.clip(max_d, 0.0, None)
    baseline_favoring = np.clip(-min_d, 0.0, None)

    x_edges = _cell_edges(i_vals)
    y_edges = _cell_edges(j_vals)
    if x_edges[-1] < HI:
        x_edges = np.append(x_edges, HI)
        kr_favoring = np.concatenate([kr_favoring, np.zeros((kr_favoring.shape[0], 1))], axis=1)
        baseline_favoring = np.concatenate([baseline_favoring, np.zeros((baseline_favoring.shape[0], 1))], axis=1)
    if y_edges[-1] < HI:
        y_edges = np.append(y_edges, HI)
        kr_favoring = np.concatenate([kr_favoring, np.zeros((1, kr_favoring.shape[1]))], axis=0)
        baseline_favoring = np.concatenate([baseline_favoring, np.zeros((1, baseline_favoring.shape[1]))], axis=0)

    fig, (ax_kr, ax_base) = plt.subplots(1, 2, figsize=(panel_size[0] * 2, panel_size[1]))
    m1 = ax_kr.pcolormesh(x_edges, y_edges, kr_favoring, cmap="Reds", vmin=0.0, vmax=1.0, shading="flat")
    m2 = ax_base.pcolormesh(x_edges, y_edges, baseline_favoring, cmap="Blues", vmin=0.0, vmax=1.0, shading="flat")

    for ax, title in ((ax_kr, "KR − baseline (MIP)"), (ax_base, "Baseline − KR (MIP)")):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xlabel(AXIS_LABELS[i_axis], fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.tick_params(labelsize=8)
    ax_kr.set_ylabel(AXIS_LABELS[j_axis], fontsize=10)

    fig.colorbar(m1, ax=ax_kr, shrink=0.85, pad=0.02, label="max(D, 0)")
    fig.colorbar(m2, ax=ax_base, shrink=0.85, pad=0.02, label="max(−D, 0)")
    return fig


def plot_attainment_field_3d(field_df, marker_size=14, figsize=(6.4, 5.2), elev=22, azim=-60):
    if len(field_df) == 0:
        raise ValueError(
            "field_df is empty -- nothing to plot. This view (dataset/decision point/mutation type "
            "combination) had zero trials in every seed (see analysis.attainment_field's 'not applicable' "
            "convention); check for this case before calling plot_attainment_field_3d(), do not plot an "
            "empty field silently."
        )

    d_values = field_df["D"].to_numpy()

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=elev, azim=azim)

    abs_norm = np.clip(np.abs(d_values), 0.0, COLOR_RANGE) / COLOR_RANGE
    MIN_ALPHA, MAX_ALPHA = 0.15, 0.95
    MIN_SIZE = marker_size * 0.35
    alpha_arr = MIN_ALPHA + (MAX_ALPHA - MIN_ALPHA) * abs_norm
    size_arr = MIN_SIZE + (marker_size - MIN_SIZE) * abs_norm

    sc = ax.scatter(
        field_df["acc"], field_df["nodes"], field_df["jaccard"],
        c=d_values, cmap="RdBu_r", vmin=-COLOR_RANGE, vmax=COLOR_RANGE,
        s=size_arr, alpha=alpha_arr, edgecolors=(0.35, 0.35, 0.35, 0.45), linewidths=0.25,
    )

    ax.set_xlabel(AXIS_LABELS["acc"], labelpad=6, fontsize=10)
    ax.set_ylabel(AXIS_LABELS["nodes"], labelpad=6, fontsize=10)
    ax.set_zlabel(AXIS_LABELS["jaccard"], labelpad=6, fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.tick_params(labelsize=8)

    cbar = fig.colorbar(sc, ax=ax, shrink=0.65, pad=0.1)
    cbar.set_label(COLORBAR_LABEL, fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.tight_layout()
    return fig


def save_attainment_field_png(fig, out_path, dpi=300):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    return str(out_path)


@dataclass
class FieldResult:
    dataset: str
    decision_point: str
    view: str
    mutation_type: str
    field_df: pd.DataFrame
    acc_values: np.ndarray
    nodes_values: np.ndarray
    jaccard_values: np.ndarray
    seeds_used: list
    trial_counts: dict
    tol: float
    max_depth: int
    nodes_min: int
    baseline_acc_col: str
    baseline_nodes_col: str
    baseline_jaccard_col: str


def compute_view_field(csv_paths, dataset, mutation_type=None, max_depth=DEFAULT_MAX_DEPTH,
                        nodes_min=DEFAULT_NODES_MIN, tol=DEFAULT_TOL, baseline_acc_col="acc_cart_test",
                        baseline_nodes_col="cart_total_nodes", baseline_jaccard_col="sim_old_cart_jaccard"):
    per_seed = build_per_seed_trials(
        csv_paths, mutation_type=mutation_type, max_depth=max_depth, nodes_min=nodes_min, tol=tol,
        baseline_acc_col=baseline_acc_col, baseline_nodes_col=baseline_nodes_col,
        baseline_jaccard_col=baseline_jaccard_col,
    )
    if not per_seed:
        return None

    field_df = compute_attainment_field(per_seed, tol=tol)
    acc_values, nodes_values, jaccard_values = observed_breakpoints(per_seed)
    trial_counts = {seed: len(trials) for seed, trials in per_seed.items()}

    return FieldResult(
        dataset=dataset,
        decision_point=infer_dp_label(csv_paths[0]),
        view=("overall" if mutation_type is None else "mutation_type"),
        mutation_type=(mutation_type or ""),
        field_df=field_df,
        acc_values=acc_values, nodes_values=nodes_values, jaccard_values=jaccard_values,
        seeds_used=sorted(trial_counts), trial_counts=trial_counts,
        tol=tol, max_depth=max_depth, nodes_min=nodes_min,
        baseline_acc_col=baseline_acc_col, baseline_nodes_col=baseline_nodes_col,
        baseline_jaccard_col=baseline_jaccard_col,
    )


def view_tag(result):
    return result.mutation_type if result.mutation_type else "overall"


def field_result_to_dataframe(result):
    df = result.field_df.copy()
    df.insert(0, "dataset", result.dataset)
    df.insert(1, "decision_point", result.decision_point)
    df.insert(2, "view", result.view)
    df.insert(3, "mutation_type", result.mutation_type)
    df["seeds_used"] = ";".join(str(s) for s in result.seeds_used)
    df["n_trials_total"] = sum(result.trial_counts.values())
    df["tol"] = result.tol
    df["max_depth"] = result.max_depth
    df["nodes_min"] = result.nodes_min
    df["baseline_acc_col"] = result.baseline_acc_col
    df["baseline_nodes_col"] = result.baseline_nodes_col
    df["baseline_jaccard_col"] = result.baseline_jaccard_col
    return df


def write_field_csv(result, out_path):
    field_result_to_dataframe(result).to_csv(out_path, index=False)
    return str(out_path)


def write_field_npz(result, out_path):
    d_grid = result.field_df["D"].to_numpy().reshape(
        len(result.acc_values), len(result.nodes_values), len(result.jaccard_values)
    )
    seeds_arr = np.array(result.seeds_used, dtype=np.int64)
    trial_counts_arr = np.array([result.trial_counts[s] for s in result.seeds_used], dtype=np.int64)
    out_path = Path(out_path)
    np.savez(
        out_path,
        acc_values=result.acc_values, nodes_values=result.nodes_values, jaccard_values=result.jaccard_values,
        D=d_grid, n_seeds=np.array(len(result.seeds_used)),
        seeds_used=seeds_arr, trial_counts=trial_counts_arr,
        dataset=np.array(result.dataset), decision_point=np.array(result.decision_point),
        view=np.array(result.view), mutation_type=np.array(result.mutation_type),
        tol=np.array(result.tol), max_depth=np.array(result.max_depth), nodes_min=np.array(result.nodes_min),
        baseline_acc_col=np.array(result.baseline_acc_col), baseline_nodes_col=np.array(result.baseline_nodes_col),
        baseline_jaccard_col=np.array(result.baseline_jaccard_col),
    )
    return str(out_path) if str(out_path).endswith(".npz") else str(out_path) + ".npz"


def load_field_npz(path):
    data = np.load(path)
    acc_values = data["acc_values"]
    nodes_values = data["nodes_values"]
    jaccard_values = data["jaccard_values"]
    d_grid = data["D"]

    grid = np.array(np.meshgrid(acc_values, nodes_values, jaccard_values, indexing="ij")).reshape(3, -1).T
    d_flat = d_grid.reshape(-1)
    n_seeds = int(data["n_seeds"])
    field_df = pd.DataFrame({
        "acc": grid[:, 0], "nodes": grid[:, 1], "jaccard": grid[:, 2], "D": d_flat, "n_seeds": n_seeds,
    })

    seeds_used = data["seeds_used"].tolist()
    trial_counts_arr = data["trial_counts"].tolist()
    metadata = {
        "dataset": str(data["dataset"]), "decision_point": str(data["decision_point"]),
        "view": str(data["view"]), "mutation_type": str(data["mutation_type"]),
        "tol": float(data["tol"]), "max_depth": int(data["max_depth"]), "nodes_min": int(data["nodes_min"]),
        "baseline_acc_col": str(data["baseline_acc_col"]), "baseline_nodes_col": str(data["baseline_nodes_col"]),
        "baseline_jaccard_col": str(data["baseline_jaccard_col"]),
        "seeds_used": seeds_used, "trial_counts": dict(zip(seeds_used, trial_counts_arr)),
    }
    return field_df, metadata


def field_summary_row(dataset, decision_point, mutation_type, result):
    if result is None:
        return {
            "dataset": dataset, "decision_point": decision_point,
            "view": ("overall" if not mutation_type else "mutation_type"),
            "mutation_type": (mutation_type or ""), "not_applicable": True,
            "n_breakpoints": 0, "n_seeds": 0, "n_trials_total": 0,
            "D_min": None, "D_max": None, "D_mean": None,
        }
    d = result.field_df["D"]
    return {
        "dataset": result.dataset, "decision_point": result.decision_point, "view": result.view,
        "mutation_type": result.mutation_type, "not_applicable": False,
        "n_breakpoints": len(result.field_df), "n_seeds": len(result.seeds_used),
        "n_trials_total": sum(result.trial_counts.values()),
        "D_min": float(d.min()), "D_max": float(d.max()), "D_mean": float(d.mean()),
    }


def main_outputs(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "csv_paths", nargs="+",
        help="pareto_per_trial.csv paths, one or more decision points x seeds (glob-expanded by your "
             "shell) -- each path must contain a 'seed_<N>' segment (same contract as "
             "igd_statistical_comparison.py / attainment_field.py). Point this at the NON-fixed "
             "'mutated/analysis/pareto_per_trial.csv' files unless you deliberately want the --fixed run's "
             "'mutated_fixed/analysis/pareto_per_trial.csv' instead.",
    )
    parser.add_argument(
        "--dataset", required=True,
        help="Dataset label (e.g. sepsis, hospital_billing, road_traffic) -- stored in every output "
             "row/file for traceability; NOT inferred from the path (paths don't reliably carry it).",
    )
    parser.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH)
    parser.add_argument("--nodes-min", type=int, default=DEFAULT_NODES_MIN)
    parser.add_argument("--tol", type=float, default=DEFAULT_TOL)
    parser.add_argument("--baseline-acc-col", default="acc_cart_test")
    parser.add_argument("--baseline-nodes-col", default="cart_total_nodes")
    parser.add_argument("--baseline-jaccard-col", default="sim_old_cart_jaccard")
    parser.add_argument(
        "--out-dir", default=None,
        help="Directory to write attainment_field_<dataset>_<dp>_<view>.csv/.npz (one pair per "
             "decision-point x view) plus attainment_field_summary.csv (one row per view, across every "
             "decision point processed) into -- created if missing. Default: "
             "quantitative_evaluation/rq2/attainment_outputs/.",
    )
    parser.add_argument(
        "--by-mutation-type", action="store_true",
        help="Also compute+write one view per canonical mutation type (analysis.mutation_types."
             "MUTATION_TYPES), in addition to the overall view (always computed). A type with zero "
             "trials at a decision point is 'not applicable' -- reported in the summary CSV with "
             "not_applicable=True, no per-view CSV/NPZ written for it (same convention as "
             "igd_statistical_comparison.py's --by-mutation-type).",
    )
    args = parser.parse_args(argv)

    if args.out_dir is None:
        args.out_dir = str(Path("quantitative_evaluation") / "rq2" / "attainment_outputs")

    by_dp = {}
    for p in args.csv_paths:
        by_dp.setdefault(infer_dp_label(p), []).append(p)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for dp_label in sorted(by_dp):
        paths = by_dp[dp_label]
        views = [(None, "overall")]
        if args.by_mutation_type:
            views += [(mt, mt) for mt in MUTATION_TYPES]

        for mutation_type, tag in views:
            result = compute_view_field(
                paths, dataset=args.dataset, mutation_type=mutation_type,
                max_depth=args.max_depth, nodes_min=args.nodes_min, tol=args.tol,
                baseline_acc_col=args.baseline_acc_col, baseline_nodes_col=args.baseline_nodes_col,
                baseline_jaccard_col=args.baseline_jaccard_col,
            )
            if result is None:
                print(f"[{dp_label}] {tag}: not applicable (zero trials in this view)")
                summary_rows.append(field_summary_row(args.dataset, dp_label, mutation_type, None))
                continue

            csv_path = out_dir / f"attainment_field_{args.dataset}_{dp_label}_{tag}.csv"
            npz_path = out_dir / f"attainment_field_{args.dataset}_{dp_label}_{tag}.npz"
            write_field_csv(result, csv_path)
            write_field_npz(result, npz_path)
            d = result.field_df["D"]
            print(
                f"[{dp_label}] {tag}: {len(result.field_df)} breakpoints, n_seeds={len(result.seeds_used)}, "
                f"D range=[{d.min():+.4f}, {d.max():+.4f}] -> {csv_path.name}, {npz_path.name}"
            )
            summary_rows.append(field_summary_row(args.dataset, dp_label, mutation_type, result))

    summary_csv = out_dir / "attainment_field_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    print(f"\nWrote {summary_csv} ({len(summary_rows)} view(s)).")


TRIAL_LEVEL_CAVEAT = (
    "SUPPLEMENTARY, TRIAL-LEVEL -- every trial from every seed pooled with equal weight "
    "(no per-seed aggregation). NOT the primary RQ3 result: trials within a seed are not "
    "independent (same T_old, same adapt/test split, same mutation-sampling random stream), "
    "so a seed with more accepted trials counts more here. See the seed-weighted figure for "
    "the validated RQ3 view."
)


def select_representative_decision_points(df, baseline_label="CART", view="overall"):
    subset = df[(df["view"] == view) & (df["baseline"] == baseline_label) & (~df["rq12_not_applicable"])]
    subset = subset.dropna(subset=["hodges_lehmann"])

    result = {}
    all_datasets = list(dict.fromkeys(df["dataset"]))
    for dataset in all_datasets:
        g = subset[subset["dataset"] == dataset]
        if len(g) == 0:
            raise ValueError(
                f"dataset={dataset!r}: no applicable rows for view={view!r}, baseline={baseline_label!r} "
                f"(either that baseline was never computed for this dataset, every decision point was "
                f"'not applicable', or every hodges_lehmann value was NaN) -- cannot select a strongest/"
                f"weakest decision point."
            )
        strongest = g.loc[g["hodges_lehmann"].idxmax()]
        weakest = g.loc[g["hodges_lehmann"].idxmin()]
        result[dataset] = {
            "strongest_dp": str(strongest["decision_point"]), "strongest_hl": float(strongest["hodges_lehmann"]),
            "weakest_dp": str(weakest["decision_point"]), "weakest_hl": float(weakest["hodges_lehmann"]),
        }
    return result


def render_selected_figures(
    selection, csv_paths, dataset, out_dir, baseline_label="CART",
    max_depth=DEFAULT_MAX_DEPTH, nodes_min=DEFAULT_NODES_MIN, tol=DEFAULT_TOL,
    baseline_acc_col="acc_cart_test", baseline_nodes_col="cart_total_nodes", baseline_jaccard_col="sim_old_cart_jaccard",
    trial_level=False,
):
    if dataset not in selection:
        raise ValueError(f"dataset={dataset!r} not found in `selection` (available: {sorted(selection)}).")
    sel = selection[dataset]

    by_dp = {}
    for p in csv_paths:
        by_dp.setdefault(infer_dp_label(p), []).append(p)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if sel["strongest_dp"] == sel["weakest_dp"]:
        dps_to_render = [(sel["strongest_dp"], "strongest_and_weakest")]
    else:
        dps_to_render = [(sel["strongest_dp"], "strongest"), (sel["weakest_dp"], "weakest")]

    written = []
    for dp_label, tag in dps_to_render:
        paths = by_dp.get(dp_label)
        if not paths:
            raise FileNotFoundError(
                f"No pareto_per_trial.csv paths found for decision_point={dp_label!r} among the "
                f"{len(csv_paths)} csv_paths given for dataset={dataset!r} -- pass every seed's "
                f"'mutated/analysis/pareto_per_trial.csv' for this dataset (see module docstring's USAGE), "
                f"this function never renders from a partial/wrong set of paths."
            )
        build_fn = build_pooled_trials if trial_level else build_per_seed_trials
        per_seed = build_fn(
            paths, max_depth=max_depth, nodes_min=nodes_min, tol=tol,
            baseline_acc_col=baseline_acc_col, baseline_nodes_col=baseline_nodes_col,
            baseline_jaccard_col=baseline_jaccard_col,
        )
        components_field_df = compute_attainment_components_field(per_seed, tol=tol)
        fig = plot_eaf_accuracy_slices_figure(components_field_df, baseline_label=baseline_label)
        if trial_level:
            filename = f"rq3_figure_trial_level_{dataset}_{dp_label}_{tag}_{baseline_label}.png"
        else:
            filename = f"rq3_figure_{dataset}_{dp_label}_{tag}_{baseline_label}.png"
        png_path = out_dir / filename
        save_attainment_field_png(fig, png_path)
        plt.close(fig)
        written.append({"dataset": dataset, "decision_point": dp_label, "tag": tag, "png_path": str(png_path)})
    return written


def render_all_decision_points(
    csv_paths, dataset, out_dir, baseline_label="CART",
    max_depth=DEFAULT_MAX_DEPTH, nodes_min=DEFAULT_NODES_MIN, tol=DEFAULT_TOL,
    baseline_acc_col="acc_cart_test", baseline_nodes_col="cart_total_nodes", baseline_jaccard_col="sim_old_cart_jaccard",
    trial_level=False,
):
    by_dp = {}
    for path in csv_paths:
        by_dp.setdefault(infer_dp_label(path), []).append(path)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for dp_label in sorted(by_dp):
        paths = by_dp[dp_label]
        build_fn = build_pooled_trials if trial_level else build_per_seed_trials
        per_seed = build_fn(
            paths, max_depth=max_depth, nodes_min=nodes_min, tol=tol,
            baseline_acc_col=baseline_acc_col, baseline_nodes_col=baseline_nodes_col,
            baseline_jaccard_col=baseline_jaccard_col,
        )
        if not per_seed:
            print(f"  skip {dp_label}: zero trials in the 'overall' view, nothing to plot.")
            continue
        components_field_df = compute_attainment_components_field(per_seed, tol=tol)
        fig = plot_eaf_accuracy_slices_figure(components_field_df, baseline_label=baseline_label)
        if trial_level:
            filename = f"rq3_figure_trial_level_{dataset}_{dp_label}_all_{baseline_label}.png"
        else:
            filename = f"rq3_figure_{dataset}_{dp_label}_all_{baseline_label}.png"
        png_path = out_dir / filename
        save_attainment_field_png(fig, png_path)
        plt.close(fig)
        written.append({"dataset": dataset, "decision_point": dp_label, "tag": "all", "png_path": str(png_path)})
    return written


def render_dataset_level(
    csv_paths, dataset, out_dir, baseline_label="CART",
    max_depth=DEFAULT_MAX_DEPTH, nodes_min=DEFAULT_NODES_MIN, tol=DEFAULT_TOL,
    baseline_acc_col="acc_cart_test", baseline_nodes_col="cart_total_nodes", baseline_jaccard_col="sim_old_cart_jaccard",
):
    by_dp = {}
    for path in csv_paths:
        by_dp.setdefault(infer_dp_label(path), []).append(path)

    pooled_trials = []
    for dp_label in sorted(by_dp):
        per_dp = build_pooled_trials(
            by_dp[dp_label], max_depth=max_depth, nodes_min=nodes_min, tol=tol,
            baseline_acc_col=baseline_acc_col, baseline_nodes_col=baseline_nodes_col,
            baseline_jaccard_col=baseline_jaccard_col,
        )
        if per_dp:
            pooled_trials.extend(per_dp[POOLED_BUCKET_KEY])

    if not pooled_trials:
        print(f"  skip {dataset}: zero trials in the 'overall' view across every decision point, nothing to plot.")
        return []

    per_seed = {POOLED_BUCKET_KEY: pooled_trials}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    components_field_df = compute_attainment_components_field(per_seed, tol=tol)
    fig = plot_eaf_accuracy_slices_figure(
        components_field_df, baseline_label=baseline_label,
        percentiles=(25, 50, 75), slice_labels=("Q1", "Median", "Q3"),
    )
    filename = f"rq3_figure_trial_level_{dataset}_ALL_DPS_all_{baseline_label}.png"
    png_path = out_dir / filename
    save_attainment_field_png(fig, png_path)
    plt.close(fig)
    written.append({"dataset": dataset, "decision_point": "ALL", "tag": "all", "png_path": str(png_path)})

    diff_field_df = compute_attainment_field(per_seed, tol=tol)
    diff_fig = plot_attainment_difference_slices(
        diff_field_df, percentiles=(25, 50, 75), slice_labels=("Q1", "Median", "Q3"),
    )
    diff_filename = f"rq3_figure_trial_level_{dataset}_ALL_DPS_diff_{baseline_label}.png"
    diff_png_path = out_dir / diff_filename
    save_attainment_field_png(diff_fig, diff_png_path)
    plt.close(diff_fig)
    written.append({"dataset": dataset, "decision_point": "ALL", "tag": "diff", "png_path": str(diff_png_path)})

    return written


def main_render_figures(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "rq_table_csvs", nargs="*",
        help="rq_unified_<dataset>.csv path(s) (analysis/rq_results_latex.py's RQ1/RQ2 table output) -- "
             "one or more, concatenated if several; only the row(s) matching --dataset are used for "
             "decision-point selection. NOT needed with --all-dps (no selection happens in that mode) -- "
             "omit it entirely in that case.",
    )
    parser.add_argument(
        "--dataset", required=True,
        help="Which dataset to render RQ3 figures for -- must match a 'dataset' value present in "
             "rq_table_csvs. ONE dataset per invocation (see module docstring's USAGE) -- run this script "
             "once per dataset to regenerate all of them.",
    )
    parser.add_argument(
        "--pareto-csvs", nargs="+", required=True,
        help="Raw pareto_per_trial.csv paths for --dataset (glob-expanded by your shell), one or more "
             "seeds x one or more decision points -- point this at the NON-fixed "
             "'mutated/analysis/pareto_per_trial.csv' files (same contract as attainment_field_outputs.py).",
    )
    parser.add_argument(
        "--baseline-label", default="CART",
        help="Which baseline's RQ1/RQ2 numbers to use for decision-point selection, and which baseline_* "
             "columns to read from --pareto-csvs -- default CART (explicit user decision, 2026, see "
             "select_representative_decision_points()'s docstring).",
    )
    parser.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH)
    parser.add_argument("--nodes-min", type=int, default=DEFAULT_NODES_MIN)
    parser.add_argument("--tol", type=float, default=DEFAULT_TOL)
    parser.add_argument("--baseline-acc-col", default="acc_cart_test")
    parser.add_argument("--baseline-nodes-col", default="cart_total_nodes")
    parser.add_argument("--baseline-jaccard-col", default="sim_old_cart_jaccard")
    parser.add_argument(
        "--out-dir", default=None,
        help="Directory to write the rendered PNG figures into. Default: "
             "quantitative_evaluation/rq2/render_figures/.",
    )
    parser.add_argument(
        "--all-dps", action="store_true",
        help="2026 addition, explicit user request (\"print them for every decision point of every "
             "dataset\"): render EVERY decision point found in --pareto-csvs instead of only the "
             "strongest/weakest 2 -- no rq_table_csvs/Hodges-Lehmann-based selection at all in this mode "
             "(rq_table_csvs may be omitted). Each figure's field pools EVERY accepted mutation trial of "
             "that decision point (every operator, every mutant, every seed, 'overall' view) -- there is "
             "no single 'the mutant used' for any of these figures, same as the strongest/weakest ones.",
    )
    parser.add_argument(
        "--dataset-level", action="store_true",
        help="2026 addition, explicit user request (\"don't split by decision point at all... do it "
             "per dataset\"): render ONE figure for the WHOLE dataset, pooling every "
             "decision point found in --pareto-csvs together (not one figure per dp) -- rq_table_csvs is "
             "not needed and is ignored. ALWAYS trial-level (no seed averaging, no --trial-level needed/"
             "accepted with this flag) and ALWAYS uses Q1/Median/Q3 observed-Accuracy slices instead of "
             "Min/Median/Max (see render_dataset_level()'s docstring). Mutually exclusive with --all-dps "
             "in practice (dataset-level takes priority if both are given).",
    )
    parser.add_argument(
        "--trial-level", action="store_true",
        help="2026 addition, explicit user request. SUPPLEMENTARY ONLY -- renders the attainment field "
             "pooling every trial from every seed with EQUAL weight (no per-seed aggregation) instead of "
             "this pipeline's primary equal-per-seed-weighted average. Decision-point SELECTION (strongest/"
             "weakest) is unaffected -- it still uses the validated seed-level RQ1/RQ2 Hodges-Lehmann value; "
             "only the rendered FIELD itself changes. Output filenames get a '_trial_level' tag so they "
             "never collide with the primary PNGs, and the figure itself is stamped with a caveat. "
             "CAVEAT: trials within a seed are not independent (same T_old, same adapt/test split, same "
             "mutation-sampling random stream) -- a seed with more accepted trials counts more here. See "
             "attainment_field.py's build_pooled_trials() docstring for the full rationale. The seed-"
             "weighted figure (no --trial-level) remains the PRIMARY, validated RQ3 result.",
    )
    args = parser.parse_args(argv)

    if args.out_dir is None:
        args.out_dir = str(Path("quantitative_evaluation") / "rq2" / "render_figures")

    if args.dataset_level:
        print(
            f"[{args.dataset}] rendering ONE pooled figure for the WHOLE dataset (every decision point "
            f"pooled together) [baseline={args.baseline_label}]  [TRIAL-LEVEL, Q1/Median/Q3 Accuracy slices]"
        )
        written = render_dataset_level(
            args.pareto_csvs, args.dataset, out_dir=args.out_dir, baseline_label=args.baseline_label,
            max_depth=args.max_depth, nodes_min=args.nodes_min, tol=args.tol,
            baseline_acc_col=args.baseline_acc_col, baseline_nodes_col=args.baseline_nodes_col,
            baseline_jaccard_col=args.baseline_jaccard_col,
        )
    elif args.all_dps:
        print(
            f"[{args.dataset}] rendering ALL decision points found in --pareto-csvs "
            f"[baseline={args.baseline_label}]"
            + ("  (Supplementary, TRIAL-LEVEL FIELD)" if args.trial_level else "")
        )
        written = render_all_decision_points(
            args.pareto_csvs, args.dataset, out_dir=args.out_dir, baseline_label=args.baseline_label,
            max_depth=args.max_depth, nodes_min=args.nodes_min, tol=args.tol,
            baseline_acc_col=args.baseline_acc_col, baseline_nodes_col=args.baseline_nodes_col,
            baseline_jaccard_col=args.baseline_jaccard_col, trial_level=args.trial_level,
        )
    else:
        if not args.rq_table_csvs:
            raise SystemExit("rq_table_csvs is required unless --all-dps is given.")
        df = pd.concat([pd.read_csv(p) for p in args.rq_table_csvs], ignore_index=True)
        selection = select_representative_decision_points(df, baseline_label=args.baseline_label, view="overall")

        if args.dataset not in selection:
            raise SystemExit(
                f"--dataset={args.dataset!r} not found among the datasets in rq_table_csvs "
                f"({sorted(selection)}) -- check the input files."
            )
        sel = selection[args.dataset]
        print(
            f"[{args.dataset}] strongest={sel['strongest_dp']} (Hodges-Lehmann={sel['strongest_hl']:+.4f})  "
            f"weakest={sel['weakest_dp']} (Hodges-Lehmann={sel['weakest_hl']:+.4f})  [baseline={args.baseline_label}]"
            + ("  (Supplementary, TRIAL-LEVEL FIELD -- selection itself still seed-level)" if args.trial_level else "")
        )

        written = render_selected_figures(
            selection, args.pareto_csvs, args.dataset, out_dir=args.out_dir, baseline_label=args.baseline_label,
            max_depth=args.max_depth, nodes_min=args.nodes_min, tol=args.tol,
            baseline_acc_col=args.baseline_acc_col, baseline_nodes_col=args.baseline_nodes_col,
            baseline_jaccard_col=args.baseline_jaccard_col, trial_level=args.trial_level,
        )
    for w in written:
        print(f"  Wrote {w['png_path']}")
    print(f"\n{len(written)} figure(s) written for dataset={args.dataset}.")


COMMANDS = {
    "outputs": main_outputs,
    "render-figures": main_render_figures,
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print("Usage: python -m analysis.rq2.attainment_figures <command> [arguments...]")
        print(f"Available commands: {', '.join(COMMANDS)}")
        sys.exit(1)
    command = sys.argv[1]
    argv = sys.argv[2:]
    COMMANDS[command](argv)


if __name__ == "__main__":
    main()
