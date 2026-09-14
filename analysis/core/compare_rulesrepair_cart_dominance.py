from pathlib import Path

import pandas as pd
from scipy.stats import binomtest, rankdata

DEFAULT_TOL = 1e-9
DEFAULT_MAX_DEPTH = 4
DEFAULT_NODES_MIN = 1
DEFAULT_ACC_MATCH_TOL = 0.01


def _dominates(a_acc, a_nodes, a_jaccard, b_acc, b_nodes, b_jaccard, tol=DEFAULT_TOL):
    weakly_better = (a_acc >= b_acc - tol) and (a_nodes <= b_nodes + tol) and (a_jaccard >= b_jaccard - tol)
    strictly_better = (a_acc > b_acc + tol) or (a_nodes < b_nodes - tol) or (a_jaccard > b_jaccard + tol)
    return weakly_better and strictly_better


def _weakly_dominates(a_acc, a_nodes, a_jaccard, b_acc, b_nodes, b_jaccard, tol=DEFAULT_TOL):
    return (a_acc >= b_acc - tol) and (a_nodes <= b_nodes + tol) and (a_jaccard >= b_jaccard - tol)


def classify_trial(front_rows, cart_acc, cart_nodes, cart_jaccard, tol=DEFAULT_TOL):
    rulesrepair_wins = any(
        _dominates(acc, nodes, jaccard, cart_acc, cart_nodes, cart_jaccard, tol)
        for acc, nodes, jaccard in front_rows
    )
    if rulesrepair_wins:
        return "rulesrepair"
    cart_wins = all(
        _dominates(cart_acc, cart_nodes, cart_jaccard, acc, nodes, jaccard, tol)
        for acc, nodes, jaccard in front_rows
    ) if front_rows else False
    if cart_wins:
        return "cart"
    return "incomparable"


def coverage(set_a, set_b, tol=DEFAULT_TOL):
    set_b = list(set_b)
    if not set_b:
        return float("nan")
    set_a = list(set_a)
    n_covered = sum(
        1 for (b_acc, b_nodes, b_jaccard) in set_b
        if any(
            _dominates(a_acc, a_nodes, a_jaccard, b_acc, b_nodes, b_jaccard, tol)
            for (a_acc, a_nodes, a_jaccard) in set_a
        )
    )
    return n_covered / len(set_b)


def coverage_classical(set_a, set_b, tol=DEFAULT_TOL):
    set_b = list(set_b)
    if not set_b:
        return float("nan")
    set_a = list(set_a)
    n_covered = sum(
        1 for (b_acc, b_nodes, b_jaccard) in set_b
        if any(
            _weakly_dominates(a_acc, a_nodes, a_jaccard, b_acc, b_nodes, b_jaccard, tol)
            for (a_acc, a_nodes, a_jaccard) in set_a
        )
    )
    return n_covered / len(set_b)


def epsilon_indicator(set_a, set_b):
    set_a = list(set_a)
    set_b = list(set_b)
    if not set_a or not set_b:
        return float("nan")
    return max(
        min(
            max(b_i - a_i for a_i, b_i in zip(a, b))
            for a in set_a
        )
        for b in set_b
    )


def nodes_bounds(max_depth, nodes_min):
    nodes_max = 2 ** (max_depth + 1) - 1
    return nodes_min, nodes_max


def normalize_point(acc, nodes, jaccard, nodes_min, nodes_max, acc_scale=1.0):
    acc_norm = acc / acc_scale
    nodes_norm = 1 - nodes / nodes_max if nodes_max > 0 else float("nan")
    jaccard_norm = jaccard
    return (
        min(max(acc_norm, 0.0), 1.0),
        min(max(nodes_norm, 0.0), 1.0),
        min(max(jaccard_norm, 0.0), 1.0),
    )


def _insert_nondominated_2d(staircase, point):
    py, pz = point
    for (y, z) in staircase:
        if y >= py and z >= pz:
            return staircase  # point is dominated (or an exact duplicate); no change
    kept = [(y, z) for (y, z) in staircase if not (py >= y and pz >= z)]
    kept.append(point)
    kept.sort(key=lambda yz: yz[0], reverse=True)
    return kept


def _staircase_area(staircase):
    if not staircase:
        return 0.0
    ordered = sorted(staircase, key=lambda yz: yz[1])
    area = 0.0
    prev_z = 0.0
    for y, z in ordered:
        area += y * (z - prev_z)
        prev_z = z
    return area


def hypervolume(points, ref=(0.0, 0.0, 0.0)):
    ref_x, ref_y, ref_z = ref
    pts = [(x - ref_x, y - ref_y, z - ref_z) for x, y, z in points]
    pts = [(max(x, 0.0), max(y, 0.0), max(z, 0.0)) for x, y, z in pts]
    if not pts:
        return 0.0
    pts.sort(key=lambda p: p[0], reverse=True)

    hv = 0.0
    staircase = []
    n = len(pts)
    for i in range(n):
        x_i, y_i, z_i = pts[i]
        staircase = _insert_nondominated_2d(staircase, (y_i, z_i))
        area_i = _staircase_area(staircase)
        next_x = pts[i + 1][0] if i + 1 < n else 0.0
        hv += (x_i - next_x) * area_i
    return hv


def pick_lexicographic_solution(front_rows_full, tol=DEFAULT_TOL):
    candidates = front_rows_full
    best_acc = max(r["acc_new_test"] for r in candidates)
    candidates = [r for r in candidates if r["acc_new_test"] >= best_acc - tol]
    best_nodes = min(r["total_nodes"] for r in candidates)
    candidates = [r for r in candidates if r["total_nodes"] <= best_nodes + tol]
    best_jaccard = max(r["sim_old_new_jaccard"] for r in candidates)
    candidates = [r for r in candidates if r["sim_old_new_jaccard"] >= best_jaccard - tol]
    return sorted(candidates, key=lambda r: (r["w_simp"], r["w_simi"], r["t"]))[0]


def find_accuracy_matched_point(front_rows, cart_acc, acc_match_tol=DEFAULT_ACC_MATCH_TOL):
    candidates = [
        (acc, nodes, jaccard) for acc, nodes, jaccard in front_rows
        if abs(acc - cart_acc) <= acc_match_tol + DEFAULT_TOL
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: (abs(p[0] - cart_acc), p[1], -p[2]))[0]


def _proportion_ci_text(n_success, n_trials, confidence_level=0.95):
    if n_trials == 0:
        return "n/a"
    result = binomtest(n_success, n_trials, p=0.5)
    ci = result.proportion_ci(confidence_level=confidence_level, method="exact")
    return f"[{ci.low:.3f}, {ci.high:.3f}]"


def run_one(csv_path, label=None, tol=DEFAULT_TOL, max_depth=DEFAULT_MAX_DEPTH, nodes_min=DEFAULT_NODES_MIN,
            acc_scale=1.0, acc_match_tol=DEFAULT_ACC_MATCH_TOL):
    df = pd.read_csv(csv_path)
    required = [
        "trial_id", "is_pareto", "acc_new_test", "acc_cart_test",
        "total_nodes", "cart_total_nodes", "sim_old_new_jaccard", "sim_old_cart_jaccard",
        "w_simp", "w_simi", "t",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path}: missing columns {missing}. Available: {list(df.columns)}")

    pareto_df = df[df["is_pareto"]] if "is_pareto" in df.columns else df

    rep_rows = []
    for trial_id, group in pareto_df.groupby("trial_id"):
        front_rows = list(zip(group["acc_new_test"], group["total_nodes"], group["sim_old_new_jaccard"]))
        cart_acc = group["acc_cart_test"].iloc[0]
        cart_nodes = group["cart_total_nodes"].iloc[0]
        cart_jaccard = group["sim_old_cart_jaccard"].iloc[0]
        outcome = classify_trial(front_rows, cart_acc, cart_nodes, cart_jaccard, tol)

        cart_front = [(cart_acc, cart_nodes, cart_jaccard)]
        coverage_rulesrepair_over_cart = coverage(front_rows, cart_front, tol)
        coverage_cart_over_rulesrepair = coverage(cart_front, front_rows, tol)

        coverage_classical_rulesrepair_over_cart = coverage_classical(front_rows, cart_front, tol)
        coverage_classical_cart_over_rulesrepair = coverage_classical(cart_front, front_rows, tol)

        nodes_min_b, nodes_max_b = nodes_bounds(max_depth, nodes_min)
        front_norm = [
            normalize_point(acc, nodes, jaccard, nodes_min_b, nodes_max_b, acc_scale)
            for acc, nodes, jaccard in front_rows
        ]
        cart_norm = normalize_point(cart_acc, cart_nodes, cart_jaccard, nodes_min_b, nodes_max_b, acc_scale)
        hv_rulesrepair = hypervolume(front_norm)
        hv_cart = hypervolume([cart_norm])

        eps_rulesrepair_over_cart = epsilon_indicator(front_norm, [cart_norm])
        eps_cart_over_rulesrepair = epsilon_indicator([cart_norm], front_norm)

        acc_matched_point = find_accuracy_matched_point(front_rows, cart_acc, acc_match_tol)
        if acc_matched_point is not None:
            m_acc, m_nodes, m_jaccard = acc_matched_point
            acc_matched_found = True
            nodes_gap_matched = cart_nodes - m_nodes
            jaccard_gap_matched = m_jaccard - cart_jaccard
        else:
            m_acc = float("nan")
            acc_matched_found = False
            nodes_gap_matched = float("nan")
            jaccard_gap_matched = float("nan")

        rep = pick_lexicographic_solution(group.to_dict("records"), tol)
        if "operator_detail" in group.columns:
            operator = group["operator_detail"].iloc[0]
        elif "operator" in group.columns:
            operator = group["operator"].iloc[0]
        else:
            operator = None

        rep_rows.append({
            "trial_id": trial_id,
            "operator": operator,
            "outcome": outcome,
            "coverage_rulesrepair_over_cart": coverage_rulesrepair_over_cart,
            "coverage_cart_over_rulesrepair": coverage_cart_over_rulesrepair,
            "coverage_classical_rulesrepair_over_cart": coverage_classical_rulesrepair_over_cart,
            "coverage_classical_cart_over_rulesrepair": coverage_classical_cart_over_rulesrepair,
            "hv_rulesrepair": hv_rulesrepair,
            "hv_cart": hv_cart,
            "eps_rulesrepair_over_cart": eps_rulesrepair_over_cart,
            "eps_cart_over_rulesrepair": eps_cart_over_rulesrepair,
            "acc_matched_found": acc_matched_found,
            "acc_matched_acc": m_acc,
            "nodes_gap_matched": nodes_gap_matched,
            "jaccard_gap_matched": jaccard_gap_matched,
            "lex_w_simp": rep["w_simp"], "lex_w_simi": rep["w_simi"], "lex_t": rep["t"],
            "acc_rulesrepair": rep["acc_new_test"], "acc_cart": cart_acc,
            "acc_improvement": rep["acc_new_test"] - cart_acc,
            "nodes_rulesrepair": rep["total_nodes"], "nodes_cart": cart_nodes,
            "nodes_improvement": cart_nodes - rep["total_nodes"],  # positive = RulesRepair simpler
            "jaccard_rulesrepair": rep["sim_old_new_jaccard"], "jaccard_cart": cart_jaccard,
            "jaccard_improvement": rep["sim_old_new_jaccard"] - cart_jaccard,  # positive = RulesRepair more similar to T_old
        })

    rep_df = pd.DataFrame(rep_rows)
    return summarize(rep_df, label or str(csv_path))


def summarize(rep_df, label):
    n_total = len(rep_df)
    n_rulesrepair = int((rep_df["outcome"] == "rulesrepair").sum())
    n_cart = int((rep_df["outcome"] == "cart").sum())
    n_incomparable = int((rep_df["outcome"] == "incomparable").sum())
    n_resolved = n_rulesrepair + n_cart

    binom_result = binomtest(n_rulesrepair, n_resolved, p=0.5, alternative="greater") if n_resolved else None

    return {
        "label": label,
        "n_total": n_total,
        "n_rulesrepair": n_rulesrepair,
        "n_cart": n_cart,
        "n_incomparable": n_incomparable,
        "n_resolved": n_resolved,
        "r_inc": n_incomparable / n_total if n_total else float("nan"),
        "p_hat_resolved": n_rulesrepair / n_resolved if n_resolved else float("nan"),
        "p_hat_resolved_ci": _proportion_ci_text(n_rulesrepair, n_resolved),
        "ndr": (n_rulesrepair - n_cart) / n_total if n_total else float("nan"),
        "p_value_one_sided": binom_result.pvalue if binom_result is not None else None,
        "_rep_df": rep_df,
    }


def _latex_escape(s):
    return str(s).replace("_", "\\_")


def infer_dp_label(csv_path):
    path = Path(csv_path)
    if path.parent.name == "analysis" and path.parent.parent.name in ("mutated", "baseline"):
        return path.parent.parent.parent.name
    return str(path)


def infer_seed_from_path(csv_path):
    import re
    match = re.search(r"seed_(\d+)", str(csv_path))
    return int(match.group(1)) if match else None


def rank_biserial_correlation(diffs):
    nonzero = [d for d in diffs if d != 0]
    n = len(nonzero)
    if n == 0:
        return None, 0
    abs_ranks = rankdata([abs(d) for d in nonzero], method="average")
    r_plus = float(sum(rk for rk, d in zip(abs_ranks, nonzero) if d > 0))
    r_minus = float(sum(rk for rk, d in zip(abs_ranks, nonzero) if d < 0))
    r = (r_plus - r_minus) / (r_plus + r_minus)
    return r, n


def _effect_size_label(r):
    if r is None:
        return "n/a"
    a = abs(r)
    if a >= 0.5:
        return "large"
    if a >= 0.3:
        return "medium"
    if a >= 0.1:
        return "small"
    return "negligible"


def _holm_bonferroni(pvalues):
    m = len(pvalues)
    indexed = sorted(enumerate(pvalues), key=lambda pair: pair[1])
    adjusted = [None] * m
    running_max = 0.0
    for rank, (original_idx, p) in enumerate(indexed):
        adj = min((m - rank) * p, 1.0)
        running_max = max(running_max, adj)
        adjusted[original_idx] = running_max
    return adjusted
