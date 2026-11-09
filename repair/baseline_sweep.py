import pickle
import sys
import time

from keep_remine_prune.param_grid import MIN_SAMPLES_LEAF_PCT_GRID, CART_CRITERIA, to_min_samples_leaf
from keep_remine_prune.tree_ruleset_conversion import tuple_tree_conversion
from keep_remine_prune.similar_tree import rule_set_similarity, rule_set_similarity_labeled, jaccard_rule_set_similarity
from keep_remine_prune.reaudit import mark_reaudit_nodes, reaudit_summary
from repair.run_perturbation_repair import build_cart_baseline, build_j48_baseline, build_reptree_baseline

SWEEP_COLUMNS = [
    "trial_id", "operator", "mutation_index",
    "algorithm", "criterion", "min_samples_leaf_pct", "min_samples_leaf",
    "f1_adapt", "f1_test", "acc_adapt", "acc_test", "total_nodes", "pct_reaudit",
    "sim_old", "sim_old_labeled", "sim_old_jaccard", "train_time_sec",
]


def _config(algorithm, criterion, pct, min_samples_leaf, result, train_time_sec):
    tree, f1_adapt, f1_test, acc_adapt, acc_test, total_nodes = result[:6]
    return {
        "algorithm": algorithm, "criterion": criterion,
        "min_samples_leaf_pct": pct, "min_samples_leaf": min_samples_leaf,
        "tree": tree, "f1_adapt": f1_adapt, "f1_test": f1_test,
        "acc_adapt": acc_adapt, "acc_test": acc_test, "total_nodes": total_nodes,
        "train_time_sec": train_time_sec,
    }


def fit_baseline_sweep(df_adapt_raw, df_test_raw, max_depth, rs_old):
    configs = []
    for pct in MIN_SAMPLES_LEAF_PCT_GRID:
        min_samples_leaf = to_min_samples_leaf(pct)

        for criterion in CART_CRITERIA:
            start = time.perf_counter()
            result = build_cart_baseline(
                df_adapt_raw, df_test_raw, max_depth, rs_old,
                criterion=criterion, min_samples_leaf=min_samples_leaf,
            )
            configs.append(_config("cart", criterion, pct, min_samples_leaf, result, time.perf_counter() - start))

        start = time.perf_counter()
        result = build_j48_baseline(
            df_adapt_raw, df_test_raw, max_depth, rs_old, min_samples_leaf=min_samples_leaf,
        )
        configs.append(_config("j48", None, pct, min_samples_leaf, result, time.perf_counter() - start))

        start = time.perf_counter()
        result = build_reptree_baseline(df_adapt_raw, df_test_raw, max_depth, rs_old, min_samples_leaf=min_samples_leaf)
        configs.append(_config("reptree", None, pct, min_samples_leaf, result, time.perf_counter() - start))

    print(f"Baseline sweep: fitted {len(configs)} baseline trees "
          f"({len(MIN_SAMPLES_LEAF_PCT_GRID)} min_samples_leaf values, CART criteria {list(CART_CRITERIA)}).", flush=True)
    return configs


def fit_or_load_baseline_sweep(df_adapt_raw, df_test_raw, max_depth, rs_old, cache_path):
    if cache_path is not None and cache_path.exists():
        old_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(max(old_limit, 20000))
        try:
            with open(cache_path, "rb") as f:
                configs = pickle.load(f)
        finally:
            sys.setrecursionlimit(old_limit)
        print(f"Baseline sweep cache hit: {cache_path}", flush=True)
        return configs

    configs = fit_baseline_sweep(df_adapt_raw, df_test_raw, max_depth, rs_old)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        old_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(max(old_limit, 20000))
        try:
            with open(tmp_path, "wb") as f:
                pickle.dump(configs, f)
        finally:
            sys.setrecursionlimit(old_limit)
        tmp_path.replace(cache_path)
        print(f"Baseline sweep cache miss, saved to {cache_path}.", flush=True)

    return configs


def sweep_rows_for_trial(configs, old_tree, rs_old, trial_meta):
    rows = []
    for config in configs:
        tree = config["tree"]
        sim_old = sim_old_labeled = sim_old_jaccard = pct_reaudit = None
        if tree is not None:
            if "rs" not in config:
                config["rs"] = tuple_tree_conversion(tree)
            sim_old = round(rule_set_similarity(rs_old, config["rs"]), 4)
            sim_old_labeled = round(rule_set_similarity_labeled(rs_old, config["rs"]), 4)
            sim_old_jaccard = round(jaccard_rule_set_similarity(rs_old, config["rs"]), 4)
            mark_reaudit_nodes(old_tree, tree)
            _, _, pct_reaudit = reaudit_summary(tree)
        rows.append({
            "trial_id": trial_meta.get("trial_id"),
            "operator": trial_meta.get("operator"),
            "mutation_index": trial_meta.get("mutation_index"),
            "algorithm": config["algorithm"],
            "criterion": config["criterion"],
            "min_samples_leaf_pct": config["min_samples_leaf_pct"],
            "min_samples_leaf": config["min_samples_leaf"],
            "f1_adapt": config["f1_adapt"],
            "f1_test": config["f1_test"],
            "acc_adapt": config["acc_adapt"],
            "acc_test": config["acc_test"],
            "total_nodes": config["total_nodes"],
            "pct_reaudit": pct_reaudit,
            "sim_old": sim_old,
            "sim_old_labeled": sim_old_labeled,
            "sim_old_jaccard": sim_old_jaccard,
            "train_time_sec": round(config["train_time_sec"], 6),
        })
    return rows
