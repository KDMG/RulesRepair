import math

import numpy as np
import pandas as pd
from scipy.stats import norm, wilcoxon

from analysis.core.compare_rulesrepair_cart_dominance import (
    DEFAULT_MAX_DEPTH, DEFAULT_NODES_MIN, DEFAULT_TOL,
    _effect_size_label, _holm_bonferroni, infer_dp_label, infer_seed_from_path,
    nodes_bounds, normalize_point, rank_biserial_correlation,
)
from analysis.core.mutation_types import MUTATION_TYPES, filter_by_mutation_type
from analysis.core.pareto_analysis import compute_pareto_front


REQUIRED_COLUMNS = [
    "trial_id", "w_simp", "w_simi",
    "acc_new_test", "total_nodes", "sim_old_new_jaccard",
    "acc_cart_test", "cart_total_nodes", "sim_old_cart_jaccard",
]


def non_dominated_mask(points, tol=DEFAULT_TOL):
    arr = np.asarray(points, dtype=float)
    n = len(arr)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        diffs = arr - arr[i]
        weakly_better = np.all(diffs >= -tol, axis=1)
        strictly_better = np.any(diffs > tol, axis=1)
        dominated_by_someone = weakly_better & strictly_better
        dominated_by_someone[i] = False
        if dominated_by_someone.any():
            mask[i] = False
    return mask


def deduplicate_points(points):
    return list(dict.fromkeys(points))


def igd_plus(front_points, reference_points):
    front = list(front_points)
    reference = list(reference_points)
    if not front or not reference:
        return float("nan")
    total = 0.0
    for r in reference:
        best = min(
            math.sqrt(sum(max(r_i - p_i, 0.0) ** 2 for r_i, p_i in zip(r, p)))
            for p in front
        )
        total += best
    return total / len(reference)


def hodges_lehmann_paired(diffs, confidence_level=0.95):
    d = [x for x in diffs if not (isinstance(x, float) and math.isnan(x))]
    n = len(d)
    if n == 0:
        return None, None, None, 0

    walsh = sorted((d[i] + d[j]) / 2.0 for i in range(n) for j in range(i, n))
    estimate = float(np.median(walsh))
    if n < 2:
        return estimate, None, None, n

    N = len(walsh)
    alpha = 1.0 - confidence_level
    z = norm.ppf(1.0 - alpha / 2.0)
    mu = n * (n + 1) / 4.0
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    k = int(math.floor(mu - z * sigma + 0.5))
    k = max(1, min(k, N))
    ci_low = walsh[k - 1]
    ci_high = walsh[N - k]
    if ci_low > ci_high:
        ci_low, ci_high = ci_high, ci_low
    return estimate, float(ci_low), float(ci_high), n


def cohens_d_z(diffs):
    d = [x for x in diffs if not (isinstance(x, float) and math.isnan(x))]
    n = len(d)
    if n < 2:
        return None
    std = float(np.std(d, ddof=1))
    if std == 0:
        return None
    return float(np.mean(d) / std)


def cohens_d_z_diagnostics(diffs, sd_zero_tol=DEFAULT_TOL):
    d = [x for x in diffs if not (isinstance(x, float) and math.isnan(x))]
    n = len(d)
    if n < 2:
        return {"cohens_dz_raw": None, "cohens_dz_status": None, "diff_mean": None, "diff_sd": None}
    mean = float(np.mean(d))
    std = float(np.std(d, ddof=1))
    status = "near_zero_sd" if std <= sd_zero_tol else "regular"
    return {"cohens_dz_raw": cohens_d_z(d), "cohens_dz_status": status, "diff_mean": mean, "diff_sd": std}


def per_trial_normalized_sets(df, max_depth=DEFAULT_MAX_DEPTH, nodes_min=DEFAULT_NODES_MIN, tol=DEFAULT_TOL,
                               baseline_acc_col="acc_cart_test", baseline_nodes_col="cart_total_nodes",
                               baseline_jaccard_col="sim_old_cart_jaccard"):
    required = list(REQUIRED_COLUMNS) + [
        c for c in (baseline_acc_col, baseline_nodes_col, baseline_jaccard_col)
        if c not in REQUIRED_COLUMNS
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"missing required column(s) {missing}; available: {list(df.columns)}")

    nodes_min_b, nodes_max_b = nodes_bounds(max_depth, nodes_min)
    result = {}
    for trial_id, group in df.groupby("trial_id", sort=False):
        group = group.reset_index(drop=True)
        is_pareto = compute_pareto_front(
            group, f1_col="acc_new_test", nodes_col="total_nodes", reaudit_col="sim_old_new_jaccard",
            tolerance=tol, reaudit_maximize=True,
        )
        rulesrepair_rows = group[is_pareto]
        rulesrepair_front_norm = deduplicate_points([
            normalize_point(acc, nodes, jac, nodes_min_b, nodes_max_b)
            for acc, nodes, jac in zip(rulesrepair_rows["acc_new_test"], rulesrepair_rows["total_nodes"], rulesrepair_rows["sim_old_new_jaccard"])
        ])
        baseline_norm = normalize_point(
            float(group[baseline_acc_col].iloc[0]), float(group[baseline_nodes_col].iloc[0]),
            float(group[baseline_jaccard_col].iloc[0]), nodes_min_b, nodes_max_b,
        )
        result[trial_id] = (rulesrepair_front_norm, baseline_norm)
    return result


def per_trial_igd(df, max_depth=DEFAULT_MAX_DEPTH, nodes_min=DEFAULT_NODES_MIN, tol=DEFAULT_TOL,
                   baseline_acc_col="acc_cart_test", baseline_nodes_col="cart_total_nodes",
                   baseline_jaccard_col="sim_old_cart_jaccard"):
    trial_sets = per_trial_normalized_sets(
        df, max_depth=max_depth, nodes_min=nodes_min, tol=tol,
        baseline_acc_col=baseline_acc_col, baseline_nodes_col=baseline_nodes_col,
        baseline_jaccard_col=baseline_jaccard_col,
    )
    rows = []
    for trial_id, (rulesrepair_front_norm, cart_norm) in trial_sets.items():
        pooled = deduplicate_points(rulesrepair_front_norm + [cart_norm])
        ref_mask = non_dominated_mask(pooled, tol=tol)
        reference_front = [p for p, keep in zip(pooled, ref_mask) if keep]

        rows.append({
            "trial_id": trial_id,
            "igd_rulesrepair": igd_plus(rulesrepair_front_norm, reference_front),
            "igd_baseline": igd_plus([cart_norm], reference_front),
            "n_reference": len(reference_front),
            "n_rulesrepair_front": len(rulesrepair_front_norm),
        })
    return pd.DataFrame(rows)


def build_seed_level_table(csv_paths, max_depth=DEFAULT_MAX_DEPTH, nodes_min=DEFAULT_NODES_MIN, tol=DEFAULT_TOL,
                            mutation_type=None, baseline_acc_col="acc_cart_test",
                            baseline_nodes_col="cart_total_nodes", baseline_jaccard_col="sim_old_cart_jaccard"):
    rows = []
    dp_labels = set()
    for p in csv_paths:
        seed = infer_seed_from_path(p)
        if seed is None:
            raise ValueError(f"{p}: no 'seed_<N>' segment found in the path -- every input must be seed-tagged.")
        dp_labels.add(infer_dp_label(p))
        df = pd.read_csv(p)
        if mutation_type is not None:
            df = filter_by_mutation_type(df, mutation_type, source=str(p))
        per_trial = per_trial_igd(
            df, max_depth=max_depth, nodes_min=nodes_min, tol=tol,
            baseline_acc_col=baseline_acc_col, baseline_nodes_col=baseline_nodes_col,
            baseline_jaccard_col=baseline_jaccard_col,
        )
        rows.append({
            "decision_point": infer_dp_label(p),
            "seed": seed,
            "n_trials": len(per_trial),
            "igd_rulesrepair_seed": float(per_trial["igd_rulesrepair"].median()) if len(per_trial) else float("nan"),
            "igd_baseline_seed": float(per_trial["igd_baseline"].median()) if len(per_trial) else float("nan"),
        })
    if len(dp_labels) > 1:
        raise ValueError(
            f"Input paths span {len(dp_labels)} different decision points ({sorted(dp_labels)}) -- "
            f"pass paths for exactly one decision point at a time (build_seed_level_table is called "
            f"once per decision point by main()'s grouping loop)."
        )
    return pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)


def build_trial_level_table(csv_paths, max_depth=DEFAULT_MAX_DEPTH, nodes_min=DEFAULT_NODES_MIN, tol=DEFAULT_TOL,
                             mutation_type=None, baseline_acc_col="acc_cart_test",
                             baseline_nodes_col="cart_total_nodes", baseline_jaccard_col="sim_old_cart_jaccard"):
    rows = []
    dp_labels = set()
    for p in csv_paths:
        seed = infer_seed_from_path(p)
        if seed is None:
            raise ValueError(f"{p}: no 'seed_<N>' segment found in the path -- every input must be seed-tagged.")
        dp_label = infer_dp_label(p)
        dp_labels.add(dp_label)
        df = pd.read_csv(p)
        if mutation_type is not None:
            df = filter_by_mutation_type(df, mutation_type, source=str(p))
        per_trial = per_trial_igd(
            df, max_depth=max_depth, nodes_min=nodes_min, tol=tol,
            baseline_acc_col=baseline_acc_col, baseline_nodes_col=baseline_nodes_col,
            baseline_jaccard_col=baseline_jaccard_col,
        )
        for _, prow in per_trial.iterrows():
            rows.append({
                "decision_point": dp_label, "seed": seed, "trial_id": prow["trial_id"],
                "igd_rulesrepair_trial": float(prow["igd_rulesrepair"]), "igd_baseline_trial": float(prow["igd_baseline"]),
            })
    if len(dp_labels) > 1:
        raise ValueError(
            f"Input paths span {len(dp_labels)} different decision points ({sorted(dp_labels)}) -- "
            f"pass paths for exactly one decision point at a time."
        )
    return pd.DataFrame(rows, columns=["decision_point", "seed", "trial_id", "igd_rulesrepair_trial", "igd_baseline_trial"])


def build_trial_level_tables_by_mutation_type(csv_paths, mutation_types=MUTATION_TYPES,
                                               max_depth=DEFAULT_MAX_DEPTH, nodes_min=DEFAULT_NODES_MIN,
                                               tol=DEFAULT_TOL, baseline_acc_col="acc_cart_test",
                                               baseline_nodes_col="cart_total_nodes",
                                               baseline_jaccard_col="sim_old_cart_jaccard"):
    tables = {}
    for mt in mutation_types:
        trial_table = build_trial_level_table(
            csv_paths, max_depth=max_depth, nodes_min=nodes_min, tol=tol, mutation_type=mt,
            baseline_acc_col=baseline_acc_col, baseline_nodes_col=baseline_nodes_col,
            baseline_jaccard_col=baseline_jaccard_col,
        )
        tables[mt] = trial_table if len(trial_table) > 0 else None
    return tables


def build_seed_level_tables_by_mutation_type(csv_paths, mutation_types=MUTATION_TYPES,
                                              max_depth=DEFAULT_MAX_DEPTH, nodes_min=DEFAULT_NODES_MIN,
                                              tol=DEFAULT_TOL, baseline_acc_col="acc_cart_test",
                                              baseline_nodes_col="cart_total_nodes",
                                              baseline_jaccard_col="sim_old_cart_jaccard"):
    tables = {}
    for mt in mutation_types:
        seed_table = build_seed_level_table(
            csv_paths, max_depth=max_depth, nodes_min=nodes_min, tol=tol, mutation_type=mt,
            baseline_acc_col=baseline_acc_col, baseline_nodes_col=baseline_nodes_col,
            baseline_jaccard_col=baseline_jaccard_col,
        )
        total_trials = int(seed_table["n_trials"].sum()) if len(seed_table) else 0
        tables[mt] = seed_table if total_trials > 0 else None
    return tables


def decision_point_igd_test(seed_table, alpha=0.05, sd_zero_tol=DEFAULT_TOL, baseline_label="CART"):
    seed_table = seed_table.dropna(subset=["igd_rulesrepair_seed", "igd_baseline_seed"])
    diffs = (seed_table["igd_baseline_seed"] - seed_table["igd_rulesrepair_seed"]).tolist()
    n = len(diffs)

    wilcoxon_stat, p_raw = None, None
    if n >= 1 and any(d != 0 for d in diffs):
        result = wilcoxon(diffs, alternative="two-sided", zero_method="wilcox")
        wilcoxon_stat, p_raw = float(result.statistic), float(result.pvalue)
    p_holm = _holm_bonferroni([p_raw])[0] if p_raw is not None else None

    r, _r_n = rank_biserial_correlation(diffs)
    hl_estimate, hl_low, hl_high, _hl_n = hodges_lehmann_paired(diffs)
    d_z = cohens_d_z(diffs)
    dz_diag = cohens_d_z_diagnostics(diffs, sd_zero_tol=sd_zero_tol)

    if p_holm is not None and p_holm < alpha and hl_estimate is not None:
        winner = "RulesRepair" if hl_estimate > 0 else baseline_label
    else:
        winner = "no significant difference"

    return {
        "comparison": f"RulesRepair vs {baseline_label}",
        "n_seeds": n,
        "wilcoxon_statistic": wilcoxon_stat,
        "wilcoxon_p": p_raw,
        "p_holm": p_holm,
        "rank_biserial_r": r,
        "rank_biserial_label": _effect_size_label(r),
        "hodges_lehmann": hl_estimate,
        "hl_ci_low": hl_low,
        "hl_ci_high": hl_high,
        "cohens_d_z": d_z,
        "cohens_dz_status": dz_diag["cohens_dz_status"],
        "diff_mean": dz_diag["diff_mean"],
        "diff_sd": dz_diag["diff_sd"],
        "winner": winner,
        "median_igd_rulesrepair": float(seed_table["igd_rulesrepair_seed"].median()) if n else None,
        "median_igd_baseline": float(seed_table["igd_baseline_seed"].median()) if n else None,
    }


def print_decision_point_result(dp_label, result, trial_level=False):
    unit = "trials" if trial_level else "seeds"
    n = result["n_trials"] if trial_level else result["n_seeds"]
    banner = (
        f"  (supplementary, trial-level, not the primary seed-level result; "
        f"trials within a seed are not independent, see "
        f"decision_point_igd_test_trial_level()'s docstring)"
        if trial_level else ""
    )
    print(f"decision point: {dp_label}{banner}")
    if n == 0:
        print(f"  0 usable {unit} (every seed had zero trials or a NaN IGD+), not testable.")
        return
    baseline_label = result["comparison"].split(" vs ", 1)[1] if "comparison" in result else "CART"
    print(
        f"  n_{unit}={n}  median IGD+ RulesRepair={result['median_igd_rulesrepair']:.4f}  "
        f"median IGD+ {baseline_label}={result['median_igd_baseline']:.4f}"
    )
    if result["wilcoxon_p"] is None:
        print(f"  Wilcoxon signed-rank: not testable (no non-zero diffs across {unit})")
        return
    hl_text = f"{result['hodges_lehmann']:+.4f}" if result["hodges_lehmann"] is not None else "n/a"
    ci_text = (
        f"[{result['hl_ci_low']:+.4f}, {result['hl_ci_high']:+.4f}]"
        if result["hl_ci_low"] is not None else "n/a"
    )
    dz_text = f"{result['cohens_d_z']:+.4f}" if result["cohens_d_z"] is not None else "n/a"
    r_text = (
        f"{result['rank_biserial_r']:+.3f} ({result['rank_biserial_label']})"
        if result["rank_biserial_r"] is not None else "n/a"
    )
    print(
        f"  comparison={result['comparison']}  Wilcoxon p={result['wilcoxon_p']:.4g}  Holm p={result['p_holm']:.4g}  "
        f"r_rb={r_text}  Hodges-Lehmann={hl_text}  HL 95% CI={ci_text}  Cohen's d_z={dz_text}  "
        f"winner={result['winner']}"
    )


def decision_point_igd_test_trial_level(trial_table, alpha=0.05, sd_zero_tol=DEFAULT_TOL, baseline_label="CART"):
    trial_table = trial_table.dropna(subset=["igd_rulesrepair_trial", "igd_baseline_trial"])
    diffs = (trial_table["igd_baseline_trial"] - trial_table["igd_rulesrepair_trial"]).tolist()
    n = len(diffs)

    wilcoxon_stat, p_raw = None, None
    if n >= 1 and any(d != 0 for d in diffs):
        result = wilcoxon(diffs, alternative="two-sided", zero_method="wilcox")
        wilcoxon_stat, p_raw = float(result.statistic), float(result.pvalue)
    p_holm = _holm_bonferroni([p_raw])[0] if p_raw is not None else None

    r, _r_n = rank_biserial_correlation(diffs)
    hl_estimate, hl_low, hl_high, _hl_n = hodges_lehmann_paired(diffs)
    d_z = cohens_d_z(diffs)
    dz_diag = cohens_d_z_diagnostics(diffs, sd_zero_tol=sd_zero_tol)

    if p_holm is not None and p_holm < alpha and hl_estimate is not None:
        winner = "RulesRepair" if hl_estimate > 0 else baseline_label
    else:
        winner = "no significant difference"

    return {
        "comparison": f"RulesRepair vs {baseline_label}",
        "unit": "trial",
        "n_trials": n,
        "wilcoxon_statistic": wilcoxon_stat,
        "wilcoxon_p": p_raw,
        "p_holm": p_holm,
        "rank_biserial_r": r,
        "rank_biserial_label": _effect_size_label(r),
        "hodges_lehmann": hl_estimate,
        "hl_ci_low": hl_low,
        "hl_ci_high": hl_high,
        "cohens_d_z": d_z,
        "cohens_dz_status": dz_diag["cohens_dz_status"],
        "diff_mean": dz_diag["diff_mean"],
        "diff_sd": dz_diag["diff_sd"],
        "winner": winner,
        "median_igd_rulesrepair": float(trial_table["igd_rulesrepair_trial"].median()) if n else None,
        "median_igd_baseline": float(trial_table["igd_baseline_trial"].median()) if n else None,
    }


def build_summary_dataframe(dp_labels, results):
    rows = []
    for dp_label, result in zip(dp_labels, results):
        row = {"decision_point": dp_label}
        row.update(result)
        rows.append(row)
    return pd.DataFrame(rows)
