import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import pandas as pd

from keep_remine_prune import keep_remine_prune_alg
from keep_remine_prune.tree_ruleset_conversion import tuple_tree_conversion
from keep_remine_prune.similar_tree import rule_set_similarity, rule_set_similarity_labeled, jaccard_rule_set_similarity
from keep_remine_prune.reaudit import mark_reaudit_nodes, reaudit_summary
from repair.run_perturbation_experiment import split_adapt_test
from repair.run_perturbation_repair import (
    SKLEARN_GROW_FUNC_MIN_SAMPLES_LEAF,
    predict, f1_macro, acc_score, count_nodes, count_new_nodes, load_xy,
    build_cart_baseline, build_chefboost_baseline, extend_columns_for_regrow,
    build_j48_pair_baseline, build_reptree_baseline,
)
from mutations.mutation_sampling import (
    generate_mutants, print_mutation_summary,
    DEFAULT_P, DEFAULT_DEGRADATION_THRESHOLD,
    DEFAULT_MAX_REGROW_ATTEMPTS, DEFAULT_REGROW_MAX_DEPTH, DEFAULT_REGROW_SPLIT_PROB,
)
from mutations.tree_mutations import root_to_leaf_paths
from analysis.core.trial_pareto_analysis import run_trial_pareto_analysis

RESULTS_COLUMNS = [
    "trial_id", "scenario", "operator", "mutation_index",
    "mutation_node_id", "mutation_description",
    "mutation_acc_before", "mutation_acc_after", "mutation_degradation",
    "n_alternatives_tried",
    "w_simp", "w_simi", "t", "delta_f1_old", "delta_acc_old",
    "f1_old_adapt", "f1_old_test", "acc_old_adapt", "acc_old_test",
    "f1_new_adapt", "f1_new_test", "acc_new_adapt", "acc_new_test",
    "total_nodes", "new_nodes", "pct_to_reaudit",
    "sim_old_new", "sim_old_new_labeled", "sim_old_new_jaccard",
    "f1_cart_adapt", "f1_cart_test", "acc_cart_adapt", "acc_cart_test",
    "cart_total_nodes", "cart_pct_to_reaudit",
    "sim_old_cart", "sim_old_cart_labeled", "sim_old_cart_jaccard",
    "cart_train_time_sec", "kr_grow_time_sec",
    "f1_cart_entropy_adapt", "f1_cart_entropy_test", "acc_cart_entropy_adapt", "acc_cart_entropy_test",
    "cart_entropy_total_nodes", "cart_entropy_pct_to_reaudit",
    "sim_old_cart_entropy", "sim_old_cart_entropy_labeled", "sim_old_cart_entropy_jaccard",
    "cart_entropy_train_time_sec",
    "f1_c45_adapt", "f1_c45_test", "acc_c45_adapt", "acc_c45_test",
    "c45_total_nodes", "c45_train_time_sec",
    "f1_j48_adapt", "f1_j48_test", "acc_j48_adapt", "acc_j48_test",
    "j48_total_nodes", "j48_pct_to_reaudit", "pct_reaudit_j48",
    "sim_old_j48", "sim_old_j48_labeled", "sim_old_j48_jaccard", "j48_train_time_sec",
    "f1_j48_unbounded_adapt", "f1_j48_unbounded_test", "acc_j48_unbounded_adapt", "acc_j48_unbounded_test",
    "j48_unbounded_total_nodes", "j48_unbounded_pct_to_reaudit", "pct_reaudit_j48_unbounded",
    "sim_old_j48_unbounded", "sim_old_j48_unbounded_labeled", "sim_old_j48_unbounded_jaccard",
    "f1_reptree_adapt", "f1_reptree_test", "acc_reptree_adapt", "acc_reptree_test",
    "reptree_total_nodes", "reptree_pct_to_reaudit", "pct_reaudit_reptree",
    "sim_old_reptree", "sim_old_reptree_labeled", "sim_old_reptree_jaccard", "reptree_train_time_sec",
]


def _fit_or_load_baseline_trees(df_adapt_raw, df_test_raw, max_depth, rs_true_old, cache_path):
    if cache_path is not None and cache_path.exists():
        old_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(max(old_limit, 20000))
        try:
            with open(cache_path, "rb") as f:
                payload = pickle.load(f)
        finally:
            sys.setrecursionlimit(old_limit)
        print(f"Baseline-tree cache hit at {cache_path}, reusing CART/CART-entropy/C4.5/"
              f"J48(bounded+unbounded)/REPTree fits instead of refitting (identical "
              f"across every --seed for this dp).", flush=True)
        return payload

    cart_start = time.perf_counter()
    (
        cart_tree, f1_cart_adapt, f1_cart_test, acc_cart_adapt, acc_cart_test, cart_total_nodes,
        _unused_sim, _unused_sim_labeled, _unused_sim_jaccard, _unused_pct_reaudit_cart,
    ) = build_cart_baseline(df_adapt_raw, df_test_raw, max_depth, rs_true_old)
    cart_train_time_sec = time.perf_counter() - cart_start

    cart_entropy_start = time.perf_counter()
    (
        cart_entropy_tree, f1_cart_entropy_adapt, f1_cart_entropy_test,
        acc_cart_entropy_adapt, acc_cart_entropy_test, cart_entropy_total_nodes,
        _unused_sim_e, _unused_sim_e_labeled, _unused_sim_e_jaccard, _unused_pct_reaudit_cart_e,
    ) = build_cart_baseline(df_adapt_raw, df_test_raw, max_depth, rs_true_old, criterion="entropy")
    cart_entropy_train_time_sec = time.perf_counter() - cart_entropy_start

    c45_start = time.perf_counter()
    (c45_tree, f1_c45_adapt, f1_c45_test, acc_c45_adapt, acc_c45_test,
     c45_total_nodes) = build_chefboost_baseline("C4.5", df_adapt_raw, df_test_raw, max_depth)
    c45_train_time_sec = time.perf_counter() - c45_start

    j48_start = time.perf_counter()
    (
        (j48_tree, f1_j48_adapt, f1_j48_test, acc_j48_adapt, acc_j48_test, j48_total_nodes,
         _unused_sim_j48, _unused_sim_j48_labeled, _unused_sim_j48_jaccard, _unused_pct_reaudit_j48),
        (j48_unbounded_tree, f1_j48_unbounded_adapt, f1_j48_unbounded_test, acc_j48_unbounded_adapt,
         acc_j48_unbounded_test, j48_unbounded_total_nodes, _unused_sim_j48u, _unused_sim_j48u_labeled,
         _unused_sim_j48u_jaccard, _unused_pct_reaudit_j48u),
    ) = build_j48_pair_baseline(df_adapt_raw, df_test_raw, max_depth, rs_true_old)
    j48_train_time_sec = time.perf_counter() - j48_start

    reptree_start = time.perf_counter()
    (
        reptree_tree, f1_reptree_adapt, f1_reptree_test, acc_reptree_adapt, acc_reptree_test, reptree_total_nodes,
        _unused_sim_rt, _unused_sim_rt_labeled, _unused_sim_rt_jaccard, _unused_pct_reaudit_rt,
    ) = build_reptree_baseline(df_adapt_raw, df_test_raw, max_depth, rs_true_old)
    reptree_train_time_sec = time.perf_counter() - reptree_start

    payload = {
        "cart_tree": cart_tree, "f1_cart_adapt": f1_cart_adapt, "f1_cart_test": f1_cart_test,
        "acc_cart_adapt": acc_cart_adapt, "acc_cart_test": acc_cart_test, "cart_total_nodes": cart_total_nodes,
        "cart_train_time_sec": cart_train_time_sec,
        "cart_entropy_tree": cart_entropy_tree, "f1_cart_entropy_adapt": f1_cart_entropy_adapt,
        "f1_cart_entropy_test": f1_cart_entropy_test, "acc_cart_entropy_adapt": acc_cart_entropy_adapt,
        "acc_cart_entropy_test": acc_cart_entropy_test, "cart_entropy_total_nodes": cart_entropy_total_nodes,
        "cart_entropy_train_time_sec": cart_entropy_train_time_sec,
        "f1_c45_adapt": f1_c45_adapt, "f1_c45_test": f1_c45_test,
        "acc_c45_adapt": acc_c45_adapt, "acc_c45_test": acc_c45_test, "c45_total_nodes": c45_total_nodes,
        "c45_train_time_sec": c45_train_time_sec,
        "j48_tree": j48_tree, "f1_j48_adapt": f1_j48_adapt, "f1_j48_test": f1_j48_test,
        "acc_j48_adapt": acc_j48_adapt, "acc_j48_test": acc_j48_test, "j48_total_nodes": j48_total_nodes,
        "j48_unbounded_tree": j48_unbounded_tree, "f1_j48_unbounded_adapt": f1_j48_unbounded_adapt,
        "f1_j48_unbounded_test": f1_j48_unbounded_test, "acc_j48_unbounded_adapt": acc_j48_unbounded_adapt,
        "acc_j48_unbounded_test": acc_j48_unbounded_test, "j48_unbounded_total_nodes": j48_unbounded_total_nodes,
        "j48_train_time_sec": j48_train_time_sec,
        "reptree_tree": reptree_tree, "f1_reptree_adapt": f1_reptree_adapt, "f1_reptree_test": f1_reptree_test,
        "acc_reptree_adapt": acc_reptree_adapt, "acc_reptree_test": acc_reptree_test,
        "reptree_total_nodes": reptree_total_nodes, "reptree_train_time_sec": reptree_train_time_sec,
    }

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        old_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(max(old_limit, 20000))
        try:
            with open(tmp_path, "wb") as f:
                pickle.dump(payload, f)
        finally:
            sys.setrecursionlimit(old_limit)
        tmp_path.replace(cache_path)
        print(f"Baseline-tree cache miss, fit CART/CART-entropy/C4.5/J48(bounded+unbounded)/"
              f"REPTree fresh and saved to {cache_path} for reuse by later --seed runs "
              f"of this same dp.", flush=True)

    return payload


def run_mutated_repair(
    normative_model_path, dp, data_csv, target, adapt_fraction, split_method, split_seed,
    out_dir, operator_names=None,
    w_simps=(5.0,), w_simis=(1.0,), max_depth=4,
    p=DEFAULT_P, degradation_threshold=DEFAULT_DEGRADATION_THRESHOLD, seed=0,
    n_thresholds=10, max_regrow_attempts=DEFAULT_MAX_REGROW_ATTEMPTS,
    regrow_max_depth=DEFAULT_REGROW_MAX_DEPTH, regrow_split_prob=DEFAULT_REGROW_SPLIT_PROB,
    fixed=False, baseline_tree_cache_dir=None, max_alternatives_per_node=None,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(normative_model_path, "rb") as f:
        normative_model = pickle.load(f)
    if dp not in normative_model:
        raise SystemExit(f"'{dp}' not found in {normative_model_path}. Available: {sorted(normative_model.keys())}")
    dp_entry = normative_model[dp]
    true_old_tree, onehot_map, columns = dp_entry["tree"], dp_entry["onehot_map"], dp_entry["columns"]
    cat_cols = list({orig for orig, _ in onehot_map.values()})

    pkl_max_depth = dp_entry.get("max_depth")
    pkl_min_samples_leaf = dp_entry.get("min_samples_leaf")
    if pkl_max_depth is None or pkl_min_samples_leaf is None:
        print(f"Warning: '{dp}' in {normative_model_path} has no 'max_depth'/'min_samples_leaf' "
              f"(old pickle?) -- cannot verify that Keep-Regrow's sklearn_grow_func "
              f"(max_depth={max_depth}, min_samples_leaf={SKLEARN_GROW_FUNC_MIN_SAMPLES_LEAF}) "
              f"matches how T_old was mined. Rebuild with build_normative_model.py to enable this check.")
    else:
        if pkl_max_depth != max_depth:
            print(f"Warning: T_old for '{dp}' was mined with max_depth={pkl_max_depth}, but "
                  f"run_mutated_repair() is using max_depth={max_depth} -- pass --max-depth "
                  f"{pkl_max_depth} to keep the regrow candidates consistent with T_old.")
        if pkl_min_samples_leaf != SKLEARN_GROW_FUNC_MIN_SAMPLES_LEAF:
            print(f"Warning: T_old for '{dp}' was mined with min_samples_leaf={pkl_min_samples_leaf}, "
                  f"but keep_remine_prune_alg.sklearn_grow_func hardcodes "
                  f"min_samples_leaf={SKLEARN_GROW_FUNC_MIN_SAMPLES_LEAF} -- regrow candidates are "
                  f"NOT consistent with how T_old was mined.")

    df_data = pd.read_csv(data_csv)

    df_adapt_raw, df_test_raw = split_adapt_test(df_data, target, adapt_fraction, split_method, split_seed)
    adapt_out = out_dir / "shared_adapt.csv"
    test_out = out_dir / "shared_test.csv"
    df_adapt_raw.to_csv(adapt_out, index=False)
    df_test_raw.to_csv(test_out, index=False)

    columns = extend_columns_for_regrow(df_adapt_raw, cat_cols, columns)
    X_adapt, y_adapt = load_xy(df_adapt_raw, cat_cols, columns)
    X_test, y_test = load_xy(df_test_raw, cat_cols, columns)

    pred_true_old_test = predict(true_old_tree, X_test)
    baseline_f1 = f1_macro(y_test, pred_true_old_test)
    baseline_acc = acc_score(y_test, pred_true_old_test)
    print(f"Unperturbed test f1_macro of the true (unmutated) T_old: {baseline_f1:.3f} "
          f"(accuracy: {baseline_acc:.3f})", flush=True)

    rs_true_old = tuple_tree_conversion(true_old_tree)

    baseline_tree_cache_path = (
        Path(baseline_tree_cache_dir) / (
            f"{dp}_af{adapt_fraction}_{split_method}_ss{split_seed}_md{max_depth}.pkl"
        ) if baseline_tree_cache_dir else None
    )
    _bt = _fit_or_load_baseline_trees(df_adapt_raw, df_test_raw, max_depth, rs_true_old, baseline_tree_cache_path)
    cart_tree, f1_cart_adapt, f1_cart_test, acc_cart_adapt, acc_cart_test, cart_total_nodes, cart_train_time_sec = (
        _bt["cart_tree"], _bt["f1_cart_adapt"], _bt["f1_cart_test"], _bt["acc_cart_adapt"],
        _bt["acc_cart_test"], _bt["cart_total_nodes"], _bt["cart_train_time_sec"],
    )
    rs_cart = tuple_tree_conversion(cart_tree) if cart_tree is not None else None

    (cart_entropy_tree, f1_cart_entropy_adapt, f1_cart_entropy_test, acc_cart_entropy_adapt,
     acc_cart_entropy_test, cart_entropy_total_nodes, cart_entropy_train_time_sec) = (
        _bt["cart_entropy_tree"], _bt["f1_cart_entropy_adapt"], _bt["f1_cart_entropy_test"],
        _bt["acc_cart_entropy_adapt"], _bt["acc_cart_entropy_test"], _bt["cart_entropy_total_nodes"],
        _bt["cart_entropy_train_time_sec"],
    )
    rs_cart_entropy = tuple_tree_conversion(cart_entropy_tree) if cart_entropy_tree is not None else None

    (f1_c45_adapt, f1_c45_test, acc_c45_adapt, acc_c45_test, c45_total_nodes, c45_train_time_sec) = (
        _bt["f1_c45_adapt"], _bt["f1_c45_test"], _bt["acc_c45_adapt"],
        _bt["acc_c45_test"], _bt["c45_total_nodes"], _bt["c45_train_time_sec"],
    )

    (j48_tree, f1_j48_adapt, f1_j48_test, acc_j48_adapt, acc_j48_test, j48_total_nodes,
     j48_unbounded_tree, f1_j48_unbounded_adapt, f1_j48_unbounded_test, acc_j48_unbounded_adapt,
     acc_j48_unbounded_test, j48_unbounded_total_nodes, j48_train_time_sec) = (
        _bt["j48_tree"], _bt["f1_j48_adapt"], _bt["f1_j48_test"], _bt["acc_j48_adapt"], _bt["acc_j48_test"],
        _bt["j48_total_nodes"], _bt["j48_unbounded_tree"], _bt["f1_j48_unbounded_adapt"],
        _bt["f1_j48_unbounded_test"], _bt["acc_j48_unbounded_adapt"], _bt["acc_j48_unbounded_test"],
        _bt["j48_unbounded_total_nodes"], _bt["j48_train_time_sec"],
    )
    rs_j48 = tuple_tree_conversion(j48_tree) if j48_tree is not None else None
    rs_j48_unbounded = tuple_tree_conversion(j48_unbounded_tree) if j48_unbounded_tree is not None else None

    (reptree_tree, f1_reptree_adapt, f1_reptree_test, acc_reptree_adapt, acc_reptree_test,
     reptree_total_nodes, reptree_train_time_sec) = (
        _bt["reptree_tree"], _bt["f1_reptree_adapt"], _bt["f1_reptree_test"], _bt["acc_reptree_adapt"],
        _bt["acc_reptree_test"], _bt["reptree_total_nodes"], _bt["reptree_train_time_sec"],
    )
    rs_reptree = tuple_tree_conversion(reptree_tree) if reptree_tree is not None else None

    X_train_full, y_train_full = load_xy(df_data, cat_cols, columns)
    classes = sorted(set(y_adapt) | set(y_test))

    print(f"Generating mutants: p={p:.0%} of compatible nodes per operator (ceil-rounded), "
          f"degradation_threshold={degradation_threshold} (plain accuracy on D_adapt vs the TRUE "
          f"T_old on D_adapt -- D_test plays no role in mutant acceptance), "
          f"seed={seed}"
          + (f", max_alternatives_per_node={max_alternatives_per_node} (opt-in cap on "
                                            f"change_threshold/change_feature search)"
             if max_alternatives_per_node is not None else ""),
          flush=True)
    mutation_results, mutation_summary = generate_mutants(
        true_old_tree, X_train_full, columns, classes, X_adapt, y_adapt,
        p=p, degradation_threshold=degradation_threshold, seed=seed,
        n_thresholds=n_thresholds, max_regrow_attempts=max_regrow_attempts,
        regrow_max_depth=regrow_max_depth, regrow_split_prob=regrow_split_prob,
        max_alternatives_per_node=max_alternatives_per_node,
    )
    if operator_names is not None:
        mutation_results = {k: v for k, v in mutation_results.items() if k in operator_names}
        mutation_summary = {k: v for k, v in mutation_summary.items() if k in operator_names}
    print_mutation_summary(mutation_summary)

    trials = []
    for operator, records in mutation_results.items():
        for i, rec in enumerate(records):
            trials.append({
                "trial_id": f"s3_{operator}_{i}",
                "operator": operator,
                "mutation_index": i,
                "mutant_tree": rec["tree"],
                "mutation_node_id": int(rec["node_id"]),
                "mutation_description": rec["description"],
                "mutation_acc_before": float(rec["acc_before"]),
                "mutation_acc_after": float(rec["acc_after"]),
                "mutation_degradation": float(rec["degradation"]),
                "n_alternatives_tried": int(rec["n_alternatives_tried"]),
            })
    n_by_operator = {op: len(recs) for op, recs in mutation_results.items()}
    print(f"Generated {len(trials)} mutant trial(s) across operators: {n_by_operator}", flush=True)
    if not trials:
        print(f"Warning: '{dp}'s T_old produced zero accepted mutants across all operators, "
              f"Scenario 3 has nothing to repair for this decision point. Writing an empty "
              f"(header-only) results.csv.", flush=True)

    manifest_path = out_dir / "manifest.jsonl"
    out_csv = out_dir / "results.csv"

    if fixed:
        path_lists_by_trial = [root_to_leaf_paths(t["mutant_tree"]) for t in trials]
    else:
        path_lists_by_trial = [[(None, None)] for _ in trials]

    n_combos = len(w_simps) * len(w_simis)
    total_evals = sum(len(pl) for pl in path_lists_by_trial) * n_combos
    print(f"Grid: {len(w_simps)} w_simps x {len(w_simis)} w_simis = {n_combos} combos/trial"
          + (f" x (path count varies per mutant, {sum(len(pl) for pl in path_lists_by_trial)} total)"
             if fixed else "")
          + f" -> {total_evals} total evaluations.", flush=True)
    progress_every = max(1, total_evals // 100)
    start_time = time.time()
    done = 0

    rows = []
    tree_records = []
    with open(manifest_path, "w") as manifest_f:
        for trial_idx, trial in enumerate(trials, start=1):
            mutant_tree = trial["mutant_tree"]

            mutant_tree_file = out_dir / f"{trial['trial_id']}_mutant_tree.pkl"
            with open(mutant_tree_file, "wb") as f:
                pickle.dump(mutant_tree, f)

            manifest_record = {
                "trial_id": trial["trial_id"],
                "scenario": 3,
                "operator": trial["operator"],
                "mutation_index": trial["mutation_index"],
                "mutation_node_id": trial["mutation_node_id"],
                "mutation_description": trial["mutation_description"],
                "mutation_acc_before": trial["mutation_acc_before"],
                "mutation_acc_after": trial["mutation_acc_after"],
                "mutation_degradation": trial["mutation_degradation"],
                "n_alternatives_tried": trial["n_alternatives_tried"],
                "mutant_n_nodes": count_nodes(mutant_tree),
                "adapt_file": str(adapt_out),
                "test_file": str(test_out),
                "mutant_tree_file": str(mutant_tree_file),
            }
            manifest_f.write(json.dumps(manifest_record) + "\n")

            pred_old_adapt = predict(mutant_tree, X_adapt)
            pred_old_test = predict(mutant_tree, X_test)
            f1_old_adapt = f1_macro(y_adapt, pred_old_adapt)
            f1_old_test = f1_macro(y_test, pred_old_test)
            acc_old_adapt = acc_score(y_adapt, pred_old_adapt)
            acc_old_test = acc_score(y_test, pred_old_test)

            rs_mutant = tuple_tree_conversion(mutant_tree)
            sim_old_cart = rule_set_similarity(rs_mutant, rs_cart) if rs_cart is not None else None
            sim_old_cart_labeled = rule_set_similarity_labeled(rs_mutant, rs_cart) if rs_cart is not None else None
            sim_old_cart_jaccard = jaccard_rule_set_similarity(rs_mutant, rs_cart) if rs_cart is not None else None

            pct_reaudit_cart = None
            if cart_tree is not None:
                mark_reaudit_nodes(mutant_tree, cart_tree)
                _, _, pct_reaudit_cart = reaudit_summary(cart_tree)

            sim_old_cart_entropy = rule_set_similarity(rs_mutant, rs_cart_entropy) if rs_cart_entropy is not None else None
            sim_old_cart_entropy_labeled = rule_set_similarity_labeled(rs_mutant, rs_cart_entropy) if rs_cart_entropy is not None else None
            sim_old_cart_entropy_jaccard = jaccard_rule_set_similarity(rs_mutant, rs_cart_entropy) if rs_cart_entropy is not None else None
            pct_reaudit_cart_entropy = None
            if cart_entropy_tree is not None:
                mark_reaudit_nodes(mutant_tree, cart_entropy_tree)
                _, _, pct_reaudit_cart_entropy = reaudit_summary(cart_entropy_tree)

            sim_old_j48 = rule_set_similarity(rs_mutant, rs_j48) if rs_j48 is not None else None
            sim_old_j48_labeled = rule_set_similarity_labeled(rs_mutant, rs_j48) if rs_j48 is not None else None
            sim_old_j48_jaccard = jaccard_rule_set_similarity(rs_mutant, rs_j48) if rs_j48 is not None else None
            pct_reaudit_j48 = None
            if j48_tree is not None:
                mark_reaudit_nodes(mutant_tree, j48_tree)
                _, _, pct_reaudit_j48 = reaudit_summary(j48_tree)

            sim_old_j48_unbounded = rule_set_similarity(rs_mutant, rs_j48_unbounded) if rs_j48_unbounded is not None else None
            sim_old_j48_unbounded_labeled = rule_set_similarity_labeled(rs_mutant, rs_j48_unbounded) if rs_j48_unbounded is not None else None
            sim_old_j48_unbounded_jaccard = jaccard_rule_set_similarity(rs_mutant, rs_j48_unbounded) if rs_j48_unbounded is not None else None
            pct_reaudit_j48_unbounded = None
            if j48_unbounded_tree is not None:
                mark_reaudit_nodes(mutant_tree, j48_unbounded_tree)
                _, _, pct_reaudit_j48_unbounded = reaudit_summary(j48_unbounded_tree)

            sim_old_reptree = rule_set_similarity(rs_mutant, rs_reptree) if rs_reptree is not None else None
            sim_old_reptree_labeled = rule_set_similarity_labeled(rs_mutant, rs_reptree) if rs_reptree is not None else None
            sim_old_reptree_jaccard = jaccard_rule_set_similarity(rs_mutant, rs_reptree) if rs_reptree is not None else None
            pct_reaudit_reptree = None
            if reptree_tree is not None:
                mark_reaudit_nodes(mutant_tree, reptree_tree)
                _, _, pct_reaudit_reptree = reaudit_summary(reptree_tree)

            kr_cache = {}

            path_list = path_lists_by_trial[trial_idx - 1]

            for leaf_id, forced_ids in path_list:
                for w_simp in w_simps:
                    for w_simi in w_simis:
                        kr_start = time.perf_counter()
                        new_tree = keep_remine_prune_alg.grow_tree(
                            X_adapt, y_adapt, old_tree=mutant_tree,
                            w_simp=w_simp, w_simi=w_simi,
                            max_depth=max_depth, grow_func=keep_remine_prune_alg.sklearn_grow_func,
                            cache=kr_cache, forced_keep_ids=forced_ids,
                        )
                        kr_grow_time_sec = time.perf_counter() - kr_start

                        pred_new_adapt = predict(new_tree, X_adapt, cache=kr_cache)
                        pred_new_test = predict(new_tree, X_test, cache=kr_cache)
                        f1_new_adapt = f1_macro(y_adapt, pred_new_adapt)
                        f1_new_test = f1_macro(y_test, pred_new_test)
                        acc_new_adapt = acc_score(y_adapt, pred_new_adapt)
                        acc_new_test = acc_score(y_test, pred_new_test)

                        n_total = count_nodes(new_tree)
                        n_new = count_new_nodes(new_tree)

                        rs_new = tuple_tree_conversion(new_tree)
                        sim_old_new = rule_set_similarity(rs_mutant, rs_new)
                        sim_old_new_labeled = rule_set_similarity_labeled(rs_mutant, rs_new)
                        sim_old_new_jaccard = jaccard_rule_set_similarity(rs_mutant, rs_new)

                        mark_reaudit_nodes(mutant_tree, new_tree)
                        _, _, pct_reaudit_new = reaudit_summary(new_tree)

                        row = {
                            "trial_id": f"{trial['trial_id']}_fixedleaf{leaf_id}" if fixed else trial["trial_id"],
                            "scenario": 3,
                            "operator": trial["operator"],
                            "mutation_index": trial["mutation_index"],
                            "mutation_node_id": trial["mutation_node_id"],
                            "mutation_description": trial["mutation_description"],
                            "mutation_acc_before": trial["mutation_acc_before"],
                            "mutation_acc_after": trial["mutation_acc_after"],
                            "mutation_degradation": trial["mutation_degradation"],
                            "n_alternatives_tried": trial["n_alternatives_tried"],
                            "w_simp": w_simp,
                            "w_simi": w_simi,
                            "t": None,  # kept for CSV-schema compat; no longer swept
                            "delta_f1_old": baseline_f1 - f1_old_test,
                            "delta_acc_old": baseline_acc - acc_old_test,
                            "f1_old_adapt": f1_old_adapt,
                            "f1_old_test": f1_old_test,
                            "acc_old_adapt": acc_old_adapt,
                            "acc_old_test": acc_old_test,
                            "f1_new_adapt": f1_new_adapt,
                            "f1_new_test": f1_new_test,
                            "acc_new_adapt": acc_new_adapt,
                            "acc_new_test": acc_new_test,
                            "total_nodes": n_total,
                            "new_nodes": n_new,
                            "pct_to_reaudit": round(100 * n_new / n_total, 1) if n_total else 0.0,
                            "pct_reaudit_new": pct_reaudit_new,
                            "sim_old_new": round(sim_old_new, 4),
                            "sim_old_new_labeled": round(sim_old_new_labeled, 4),
                            "sim_old_new_jaccard": round(sim_old_new_jaccard, 4),
                            "f1_cart_adapt": f1_cart_adapt,
                            "f1_cart_test": f1_cart_test,
                            "acc_cart_adapt": acc_cart_adapt,
                            "acc_cart_test": acc_cart_test,
                            "cart_total_nodes": cart_total_nodes,
                            "cart_pct_to_reaudit": 100.0 if cart_tree is not None else None,
                            "pct_reaudit_cart": pct_reaudit_cart,
                            "sim_old_cart": round(sim_old_cart, 4) if sim_old_cart is not None else None,
                            "sim_old_cart_labeled": round(sim_old_cart_labeled, 4) if sim_old_cart_labeled is not None else None,
                            "sim_old_cart_jaccard": round(sim_old_cart_jaccard, 4) if sim_old_cart_jaccard is not None else None,
                            "cart_train_time_sec": round(cart_train_time_sec, 6),
                            "kr_grow_time_sec": round(kr_grow_time_sec, 6),
                            "f1_cart_entropy_adapt": f1_cart_entropy_adapt,
                            "f1_cart_entropy_test": f1_cart_entropy_test,
                            "acc_cart_entropy_adapt": acc_cart_entropy_adapt,
                            "acc_cart_entropy_test": acc_cart_entropy_test,
                            "cart_entropy_total_nodes": cart_entropy_total_nodes,
                            "cart_entropy_pct_to_reaudit": 100.0 if cart_entropy_tree is not None else None,
                            "pct_reaudit_cart_entropy": pct_reaudit_cart_entropy,
                            "sim_old_cart_entropy": round(sim_old_cart_entropy, 4) if sim_old_cart_entropy is not None else None,
                            "sim_old_cart_entropy_labeled": round(sim_old_cart_entropy_labeled, 4) if sim_old_cart_entropy_labeled is not None else None,
                            "sim_old_cart_entropy_jaccard": round(sim_old_cart_entropy_jaccard, 4) if sim_old_cart_entropy_jaccard is not None else None,
                            "cart_entropy_train_time_sec": round(cart_entropy_train_time_sec, 6),
                            "f1_c45_adapt": f1_c45_adapt,
                            "f1_c45_test": f1_c45_test,
                            "acc_c45_adapt": acc_c45_adapt,
                            "acc_c45_test": acc_c45_test,
                            "c45_total_nodes": c45_total_nodes,
                            "c45_train_time_sec": round(c45_train_time_sec, 6),
                            "f1_j48_adapt": f1_j48_adapt,
                            "f1_j48_test": f1_j48_test,
                            "acc_j48_adapt": acc_j48_adapt,
                            "acc_j48_test": acc_j48_test,
                            "j48_total_nodes": j48_total_nodes,
                            "j48_pct_to_reaudit": 100.0 if j48_tree is not None else None,
                            "pct_reaudit_j48": pct_reaudit_j48,
                            "sim_old_j48": round(sim_old_j48, 4) if sim_old_j48 is not None else None,
                            "sim_old_j48_labeled": round(sim_old_j48_labeled, 4) if sim_old_j48_labeled is not None else None,
                            "sim_old_j48_jaccard": round(sim_old_j48_jaccard, 4) if sim_old_j48_jaccard is not None else None,
                            "j48_train_time_sec": round(j48_train_time_sec, 6),
                            "f1_j48_unbounded_adapt": f1_j48_unbounded_adapt,
                            "f1_j48_unbounded_test": f1_j48_unbounded_test,
                            "acc_j48_unbounded_adapt": acc_j48_unbounded_adapt,
                            "acc_j48_unbounded_test": acc_j48_unbounded_test,
                            "j48_unbounded_total_nodes": j48_unbounded_total_nodes,
                            "j48_unbounded_pct_to_reaudit": 100.0 if j48_unbounded_tree is not None else None,
                            "pct_reaudit_j48_unbounded": pct_reaudit_j48_unbounded,
                            "sim_old_j48_unbounded": round(sim_old_j48_unbounded, 4) if sim_old_j48_unbounded is not None else None,
                            "sim_old_j48_unbounded_labeled": round(sim_old_j48_unbounded_labeled, 4) if sim_old_j48_unbounded_labeled is not None else None,
                            "sim_old_j48_unbounded_jaccard": round(sim_old_j48_unbounded_jaccard, 4) if sim_old_j48_unbounded_jaccard is not None else None,
                            "f1_reptree_adapt": f1_reptree_adapt,
                            "f1_reptree_test": f1_reptree_test,
                            "acc_reptree_adapt": acc_reptree_adapt,
                            "acc_reptree_test": acc_reptree_test,
                            "reptree_total_nodes": reptree_total_nodes,
                            "reptree_pct_to_reaudit": 100.0 if reptree_tree is not None else None,
                            "pct_reaudit_reptree": pct_reaudit_reptree,
                            "sim_old_reptree": round(sim_old_reptree, 4) if sim_old_reptree is not None else None,
                            "sim_old_reptree_labeled": round(sim_old_reptree_labeled, 4) if sim_old_reptree_labeled is not None else None,
                            "sim_old_reptree_jaccard": round(sim_old_reptree_jaccard, 4) if sim_old_reptree_jaccard is not None else None,
                            "reptree_train_time_sec": round(reptree_train_time_sec, 6),
                        }
                        if fixed:
                            row["fixed_leaf_id"] = leaf_id
                        rows.append(row)

                        tree_record = {
                            "trial_id": trial["trial_id"],
                            "operator": trial["operator"],
                            "mutation_index": trial["mutation_index"],
                            "w_simp": w_simp,
                            "w_simi": w_simi,
                            "t": None,
                            "old_tree": mutant_tree,
                            "new_tree": new_tree,
                            "cart_tree": cart_tree,
                        }
                        if fixed:
                            tree_record["fixed_leaf_id"] = leaf_id
                        tree_records.append(tree_record)

                        done += 1
                        if done % progress_every == 0 or done == total_evals:
                            elapsed = time.time() - start_time
                            rate = done / elapsed if elapsed > 0 else 0
                            eta_min = (total_evals - done) / rate / 60 if rate > 0 else float("inf")
                            print(
                                f"  {done}/{total_evals} ({100 * done / total_evals:.1f}%) -- "
                                f"trial {trial_idx}/{len(trials)} -- "
                                f"elapsed {elapsed / 60:.1f} min, ETA {eta_min:.1f} min",
                                flush=True,
                            )

    empty_columns = RESULTS_COLUMNS + ["fixed_leaf_id"] if fixed else RESULTS_COLUMNS
    out = pd.DataFrame(rows) if rows else pd.DataFrame(columns=empty_columns)
    out.to_csv(out_csv, index=False)

    trees_pkl = out_dir / "trees.pkl"
    with open(trees_pkl, "wb") as f:
        pickle.dump(tree_records, f)

    elapsed_total = time.time() - start_time
    print(f"Done in {elapsed_total / 60:.1f} min: {len(trials)} trials x {len(w_simps)}x{len(w_simis)} "
          f"(w_simp,w_simi) -> {len(out)} rows -> {out_csv}", flush=True)
    print(f"Saved {len(tree_records)} tree record(s) (old_tree/new_tree/cart_tree) -> {trees_pkl}", flush=True)

    if not trials:
        print(f"Skipping Pareto analysis: '{dp}' had zero mutant trials, nothing to analyze.", flush=True)
        analysis_dir = None
    else:
        print(f"Scenario 3: Pareto analysis (per trial + aggregated by config) -> {out_dir / 'analysis'}", flush=True)
        analysis_dir = run_trial_pareto_analysis(
            csv_path=out_csv, out_dir=out_dir / "analysis",
        )

    return manifest_path, out_csv, trees_pkl, analysis_dir

run_tree_mutation_repair = run_mutated_repair


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--normative-model", default="normative_model.pkl")
    parser.add_argument("--dp", required=True)
    parser.add_argument("--data-csv", required=True)
    parser.add_argument("--target", default="branch")
    parser.add_argument("--adapt-fraction", type=float, default=0.5)
    parser.add_argument("--split-method", choices=["positional", "random", "case_chronological"], default="case_chronological")
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument(
        "--operators", default=None,
        help="Comma-separated subset of prune,change_label,branch_swap,change_threshold,"
             "change_feature,regrow_leaf,regrow_internal (default: all 7).",
    )
    parser.add_argument("--w-simps", default="0.0,0.001,0.005,0.01,0.02,0.05,0.1,0.2,0.3,0.5,0.75,1.0,1.5,2.0,3.0,5.0", help="Comma-separated list of w_simp values to sweep. Normalized scale (complexity term divided by nodes_max=2**(max_depth+1)-1). Extra resolution below 0.02 since some hospital_billing dps collapse to a trivial 1-node tree already at the first coarser nonzero value.")
    parser.add_argument("--w-simis", default="0.0,0.001,0.005,0.01,0.02,0.05,0.1,0.2,0.3,0.5,0.75,1.0,1.5", help="Comma-separated list of w_simi values to sweep. Normalized scale (audit term divided by nodes_max). Narrower top end than --w-simps: w_simi saturates by ~0.5-1 in every decision point checked.")
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--p", type=float, default=DEFAULT_P, help="Fraction of compatible nodes to target per operator, ceil-rounded ('at least p%%'), default 0.2.")
    parser.add_argument("--degradation-threshold", type=float, default=DEFAULT_DEGRADATION_THRESHOLD, help="Minimum plain-accuracy drop (vs the TRUE unmutated T_old, on the test split) required to accept a mutation, default 0.2.")
    parser.add_argument("--seed", type=int, default=0, help="Single global seed for all mutation sampling/randomization (node visit order, label/threshold/feature choices, regrow subtrees) -- for full reproducibility.")
    parser.add_argument("--n-thresholds", type=int, default=10, help="Number of n-tiles per feature in the threshold pool (computed from --data-csv, i.e. the train split -- 10 = deciles).")
    parser.add_argument("--max-regrow-attempts", type=int, default=DEFAULT_MAX_REGROW_ATTEMPTS, help="regrow has no finite alternative enumeration -- max random attempts per node before giving up on it.")
    parser.add_argument("--regrow-max-depth", type=int, default=DEFAULT_REGROW_MAX_DEPTH)
    parser.add_argument("--regrow-split-prob", type=float, default=DEFAULT_REGROW_SPLIT_PROB)
    parser.add_argument(
        "--max-alternatives-per-node", type=int, default=None,
        help="2026 opt-in addition, default None = unchanged exhaustive behaviour. Caps how "
             "many alternatives change_threshold/change_feature try per node before giving up "
             "on it (a random subsample of the shuffled alternative list, not a biased "
             "truncation). change_feature in particular can otherwise enumerate "
             "(n_columns-1)*n_thresholds candidates per node, each requiring a tree deepcopy + "
             "a full accuracy pass over D_adapt -- on datasets with many one-hot columns "
             "and/or large D_adapt this can take hours per decision point. Does not affect "
             "prune/change_label (already tiny, bounded by len(classes)), branch_swap (1 "
             "alternative), or regrow (already capped by --max-regrow-attempts).",
    )
    parser.add_argument(
        "--fixed", action=argparse.BooleanOptionalAction, default=False,
        help="'Expert forces keep' experiment: mutant generation is completely unchanged (same "
             "seed -> same mutants). For EACH accepted mutant, instead of one w_simp/w_simi sweep, "
             "run one full sweep PER root-to-leaf path of that mutant, with that path's nodes "
             "forced to keep, never regrown. results.csv/trees.pkl gain a fixed_leaf_id "
             "column/field and trial_id is suffixed with the protected leaf's id. Point --out-dir "
             "at a separate directory (e.g. mutated_fixed/) -- this flag does not rename it.",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--baseline-tree-cache-dir", default=None,
        help="2026 addition: directory to cache the once-per-dp CART/CART-entropy/C4.5/"
             "J48(bounded+unbounded)/REPTree fits in (one <dp>.pkl per decision point). "
             "These fits depend only on --data-csv/--split-seed, never on --seed -- when "
             "sweeping many --seed values for the SAME dp (run_repair_all_seeds.py), pass the "
             "SAME cache dir to every seed so only the first seed actually fits them and every "
             "later seed reuses the cached result instead of refitting. Omit for standalone runs "
             "(no caching, identical to the pre-2026 behaviour).",
    )
    args = parser.parse_args()

    w_simps = [float(a) for a in args.w_simps.split(",")]
    w_simis = [float(b) for b in args.w_simis.split(",")]
    operator_names = args.operators.split(",") if args.operators else None

    run_mutated_repair(
        normative_model_path=args.normative_model, dp=args.dp, data_csv=args.data_csv,
        target=args.target, adapt_fraction=args.adapt_fraction, split_method=args.split_method,
        split_seed=args.split_seed, out_dir=args.out_dir, operator_names=operator_names,
        w_simps=w_simps, w_simis=w_simis, max_depth=args.max_depth,
        p=args.p, degradation_threshold=args.degradation_threshold, seed=args.seed,
        n_thresholds=args.n_thresholds, max_regrow_attempts=args.max_regrow_attempts,
        regrow_max_depth=args.regrow_max_depth, regrow_split_prob=args.regrow_split_prob,
        fixed=args.fixed, baseline_tree_cache_dir=args.baseline_tree_cache_dir,
        max_alternatives_per_node=args.max_alternatives_per_node,
    )


if __name__ == "__main__":
    main()
