import argparse
import ast
import fcntl
import json
import os
import pickle
import re
import time
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from chefboost import Chefboost as chef
from chefboost.commons import functions as chef_functions

from mining.cart_training import build_tree, encode_records
from mining.weka_baselines import build_j48_trees, build_reptree_tree
from keep_remine_prune import keep_remine_prune_alg
from keep_remine_prune.tree_ruleset_conversion import tuple_tree_conversion
from keep_remine_prune.similar_tree import rule_set_similarity, rule_set_similarity_labeled, jaccard_rule_set_similarity
from keep_remine_prune.reaudit import mark_reaudit_nodes, reaudit_summary
from repair.run_perturbation_experiment import split_adapt_test
from mutations.selection import IDENTIFIER_COLUMNS
from mutations.tree_mutations import root_to_leaf_paths

NON_FEATURE_COLUMNS = IDENTIFIER_COLUMNS | {"branch_index", "branch_label"}

SKLEARN_GROW_FUNC_MIN_SAMPLES_LEAF = 0.02


def encode(df, cat_cols, columns):
    df = df.copy()
    missing_cat_cols = [c for c in cat_cols if c not in df.columns]
    if missing_cat_cols:
        for col in missing_cat_cols:
            df[col] = None
        print(f"Warning: {missing_cat_cols} not seen in this split, treated as all-missing.")
    for col in df.columns:
        if col not in cat_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median()).fillna(0)
    df = pd.get_dummies(df, columns=cat_cols)
    return df.reindex(columns=columns, fill_value=0)


def extend_columns_for_regrow(df, cat_cols, base_columns):
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS and c != "branch"]
    df = df[feature_cols].copy()
    missing_cat_cols = [c for c in cat_cols if c not in df.columns]
    for col in missing_cat_cols:
        df[col] = None
    dummies = pd.get_dummies(df, columns=cat_cols)
    novel = sorted(c for c in dummies.columns if c not in base_columns)
    return list(base_columns) + novel


def predict(node, X, cache=None):
    if cache is not None:
        bucket = cache.setdefault('X_numpy', {})
        key = id(X)
        X_np = bucket.get(key)
        if X_np is None:
            X_np = X.to_numpy()
            bucket[key] = X_np
    else:
        X_np = X.to_numpy()
    return [node.predict(row) for row in X_np]


def f1_macro(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def acc_score(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


def count_nodes(node):
    return 1 + sum(count_nodes(c) for c in node.children)


def count_new_nodes(node):
    is_new = getattr(node, "is_new", True)
    return (1 if is_new else 0) + sum(count_new_nodes(c) for c in node.children)


def load_xy(df, cat_cols, columns):
    y = df["branch"]
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS and c != "branch"]
    X = encode(df[feature_cols], cat_cols, columns)
    return X, y


def build_cart_baseline(df_adapt_raw, df_test_raw, max_depth, rs_old, old_tree=None, criterion="gini"):
    cart_records = df_adapt_raw.drop(
        columns=[c for c in NON_FEATURE_COLUMNS if c in df_adapt_raw.columns]
    ).to_dict("records")
    cart_tree, f1_cart_adapt, cart_onehot_map, cart_columns = build_tree(cart_records, max_depth=max_depth, criterion=criterion)
    if cart_tree is None:
        return cart_tree, f1_cart_adapt, None, None, None, None, None, None, None, None
    cart_cat_cols = list({orig for orig, _ in cart_onehot_map.values()})
    X_test_cart, y_test_cart = load_xy(df_test_raw, cart_cat_cols, cart_columns)
    pred_cart_test = predict(cart_tree, X_test_cart)
    f1_cart_test = f1_macro(y_test_cart, pred_cart_test)
    acc_cart_test = acc_score(y_test_cart, pred_cart_test)
    X_adapt_cart, y_adapt_cart = load_xy(df_adapt_raw, cart_cat_cols, cart_columns)
    acc_cart_adapt = acc_score(y_adapt_cart, predict(cart_tree, X_adapt_cart))
    cart_total_nodes = count_nodes(cart_tree)
    rs_cart = tuple_tree_conversion(cart_tree)
    sim_old_cart = rule_set_similarity(rs_old, rs_cart)
    sim_old_cart_labeled = rule_set_similarity_labeled(rs_old, rs_cart)
    sim_old_cart_jaccard = jaccard_rule_set_similarity(rs_old, rs_cart)
    pct_reaudit_cart = None
    if old_tree is not None:
        mark_reaudit_nodes(old_tree, cart_tree)
        _, _, pct_reaudit_cart = reaudit_summary(cart_tree)
    return (
        cart_tree, f1_cart_adapt, f1_cart_test, acc_cart_adapt, acc_cart_test, cart_total_nodes,
        sim_old_cart, sim_old_cart_labeled, sim_old_cart_jaccard, pct_reaudit_cart,
    )


def _evaluate_binary_tree(tree, X_adapt, y_adapt, onehot_map, columns, df_test_raw, rs_old, old_tree=None):
    if tree is None:
        return None, None, None, None, None, None, None, None, None, None

    f1_adapt = f1_macro(y_adapt, predict(tree, X_adapt))
    acc_adapt = acc_score(y_adapt, predict(tree, X_adapt))

    cat_cols = list({orig for orig, _ in onehot_map.values()})
    X_test, y_test = load_xy(df_test_raw, cat_cols, columns)
    pred_test = predict(tree, X_test)
    f1_test = f1_macro(y_test, pred_test)
    acc_test = acc_score(y_test, pred_test)

    total_nodes = count_nodes(tree)
    rs_tree = tuple_tree_conversion(tree)
    sim_old = rule_set_similarity(rs_old, rs_tree)
    sim_old_labeled = rule_set_similarity_labeled(rs_old, rs_tree)
    sim_old_jaccard = jaccard_rule_set_similarity(rs_old, rs_tree)
    pct_reaudit = None
    if old_tree is not None:
        mark_reaudit_nodes(old_tree, tree)
        _, _, pct_reaudit = reaudit_summary(tree)

    return (
        tree, f1_adapt, f1_test, acc_adapt, acc_test, total_nodes,
        sim_old, sim_old_labeled, sim_old_jaccard, pct_reaudit,
    )


def build_binary_baseline(build_tree_func, label, df_adapt_raw, df_test_raw, max_depth, rs_old, old_tree=None):
    records = df_adapt_raw.drop(
        columns=[c for c in NON_FEATURE_COLUMNS if c in df_adapt_raw.columns]
    ).to_dict("records")
    X_adapt, y_adapt, onehot_map, columns = encode_records(records)
    if X_adapt is None:
        print(f"Warning: {label} baseline got zero feature columns for this trial's D_adapt, "
              f"reporting None for every {label} column on this row.")
        return None, None, None, None, None, None, None, None, None, None

    tree = build_tree_func(X_adapt, y_adapt, columns, max_depth)
    return _evaluate_binary_tree(tree, X_adapt, y_adapt, onehot_map, columns, df_test_raw, rs_old, old_tree)


def build_j48_pair_baseline(df_adapt_raw, df_test_raw, max_depth, rs_old, old_tree=None):
    records = df_adapt_raw.drop(
        columns=[c for c in NON_FEATURE_COLUMNS if c in df_adapt_raw.columns]
    ).to_dict("records")
    X_adapt, y_adapt, onehot_map, columns = encode_records(records)
    if X_adapt is None:
        print("Warning: J48 baseline got zero feature columns for this trial's D_adapt, "
              "reporting None for every j48/j48_unbounded column on this row.")
        empty = (None, None, None, None, None, None, None, None, None, None)
        return empty, empty

    bounded_tree, unbounded_tree = build_j48_trees(X_adapt, y_adapt, columns, max_depth)
    bounded = _evaluate_binary_tree(bounded_tree, X_adapt, y_adapt, onehot_map, columns, df_test_raw, rs_old, old_tree)
    unbounded = _evaluate_binary_tree(unbounded_tree, X_adapt, y_adapt, onehot_map, columns, df_test_raw, rs_old, old_tree)
    return bounded, unbounded


def build_j48_baseline(df_adapt_raw, df_test_raw, max_depth, rs_old, old_tree=None):
    bounded, _unbounded = build_j48_pair_baseline(df_adapt_raw, df_test_raw, max_depth, rs_old, old_tree)
    return bounded


def build_reptree_baseline(df_adapt_raw, df_test_raw, max_depth, rs_old, old_tree=None):
    return build_binary_baseline(build_reptree_tree, "REPTree", df_adapt_raw, df_test_raw, max_depth, rs_old, old_tree)


# chefboost always writes its fitted model to <cwd>/outputs/rules/rules.py --
# that path is hardcoded inside the library and not configurable. To avoid
# littering the repo root with it, we chdir into a dedicated, gitignored
# scratch folder for the duration of the fit call instead.
REPO_ROOT = Path(__file__).resolve().parent.parent
CHEFBOOST_SCRATCH_DIR = REPO_ROOT / ".chefboost_scratch"
CHEFBOOST_RULES_FILE = CHEFBOOST_SCRATCH_DIR / "outputs" / "rules" / "rules.py"

CHEFBOOST_FIT_LOCK_FILE = CHEFBOOST_SCRATCH_DIR / "chefboost_fit.lock"


@contextmanager
def _chefboost_fit_lock():
    CHEFBOOST_SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHEFBOOST_FIT_LOCK_FILE, "a+") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        prev_cwd = os.getcwd()
        os.chdir(CHEFBOOST_SCRATCH_DIR)
        try:
            yield
        finally:
            os.chdir(prev_cwd)
            fcntl.flock(lock_f, fcntl.LOCK_UN)


def _count_chefboost_nodes(source_text):
    n_splits = len(re.findall(r'^\s*#\s*\{"feature":', source_text, flags=re.MULTILINE))
    n_leaves = sum(1 for node in ast.walk(ast.parse(source_text)) if isinstance(node, ast.Return))
    return n_splits + n_leaves


def build_chefboost_baseline(algorithm, df_adapt_raw, df_test_raw, max_depth):
    def _to_chefboost_records(df_raw):
        records = df_raw.drop(
            columns=[c for c in NON_FEATURE_COLUMNS if c in df_raw.columns]
        ).rename(columns={"branch": "Decision"})
        if records.columns[-1] != "Decision":
            new_order = [c for c in records.columns if c != "Decision"] + ["Decision"]
            records = records[new_order]
        return records

    config = {"algorithm": algorithm, "max_depth": max_depth, "enableParallelism": False}
    try:
        with _chefboost_fit_lock():
            model = chef.fit(_to_chefboost_records(df_adapt_raw), config, target_label="Decision", silent=True)
            with open(CHEFBOOST_RULES_FILE) as f:
                total_nodes = _count_chefboost_nodes(f.read())
    except Exception as ex:  # pragma: no cover -- mirrors build_cart_baseline()'s fail-soft convention
        print(f"Warning: chefboost failed to fit '{algorithm}' on this trial's D_adapt ({ex}), "
              f"reporting None for every {algorithm} column on this row.")
        return None, None, None, None, None, None

    adapt_records = _to_chefboost_records(df_adapt_raw)
    test_records = _to_chefboost_records(df_test_raw)
    chef_functions.bulk_prediction(adapt_records, model)
    chef_functions.bulk_prediction(test_records, model)

    f1_adapt = f1_macro(adapt_records["Decision"], adapt_records["Prediction"])
    f1_test = f1_macro(test_records["Decision"], test_records["Prediction"])
    acc_adapt = acc_score(adapt_records["Decision"], adapt_records["Prediction"])
    acc_test = acc_score(test_records["Decision"], test_records["Prediction"])

    return model, f1_adapt, f1_test, acc_adapt, acc_test, total_nodes


def run_repair(
    normative_model_path, dp, data_csv, target, adapt_fraction, split_method, split_seed,
    manifest_path, out_csv,
    w_simps=(0.01,), w_simis=(0.01,), max_depth=4,
    fixed=False,
):
    with open(normative_model_path, "rb") as f:
        normative_model = pickle.load(f)
    if dp not in normative_model:
        raise SystemExit(f"'{dp}' not found in {normative_model_path}. Available: {sorted(normative_model.keys())}")
    dp_entry = normative_model[dp]
    old_tree, onehot_map, columns = dp_entry["tree"], dp_entry["onehot_map"], dp_entry["columns"]
    cat_cols = list({orig for orig, _ in onehot_map.values()})

    pkl_max_depth = dp_entry.get("max_depth")
    pkl_min_samples_leaf = dp_entry.get("min_samples_leaf")
    if pkl_max_depth is None or pkl_min_samples_leaf is None:
        print(f"Warning: '{dp}' has no max_depth/min_samples_leaf recorded, "
              f"cannot verify it matches how T_old was mined. Rebuild with build_normative_model.py.")
    else:
        if pkl_max_depth != max_depth:
            print(f"Warning: T_old for '{dp}' was mined with max_depth={pkl_max_depth}, "
                  f"but max_depth={max_depth} is being used now. Pass --max-depth {pkl_max_depth}.")
        if pkl_min_samples_leaf != SKLEARN_GROW_FUNC_MIN_SAMPLES_LEAF:
            print(f"Warning: T_old for '{dp}' was mined with min_samples_leaf={pkl_min_samples_leaf}, "
                  f"but regrow candidates use min_samples_leaf={SKLEARN_GROW_FUNC_MIN_SAMPLES_LEAF}.")

    df_data = pd.read_csv(data_csv)

    rs_old = tuple_tree_conversion(old_tree)

    print("Loading data and evaluating T_old on the unperturbed baseline...", flush=True)
    _, unperturbed_test = split_adapt_test(df_data, target, adapt_fraction, split_method, split_seed)
    X_base_test, y_base_test = load_xy(unperturbed_test, cat_cols, columns)
    pred_base_test = predict(old_tree, X_base_test)
    baseline_f1 = f1_macro(y_base_test, pred_base_test)
    baseline_acc = acc_score(y_base_test, pred_base_test)
    print(f"T_old test accuracy: {baseline_acc:.3f}", flush=True)

    with open(manifest_path) as f:
        trials = [json.loads(line) for line in f]

    if fixed:
        path_list = root_to_leaf_paths(old_tree)
        print(f"--fixed: T_old has {len(path_list)} root-to-leaf path(s), "
              f"one full repair sweep per protected path.", flush=True)
    else:
        path_list = [(None, None)]

    n_combos = len(w_simps) * len(w_simis)
    total_evals = len(trials) * len(path_list) * n_combos
    print(f"Loaded {len(trials)} trials. Grid: {len(w_simps)} w_simps x {len(w_simis)} w_simis "
          f"= {n_combos} combos/trial" + (f" x {len(path_list)} protected path(s)" if fixed else "") +
          f" -> {total_evals} total evaluations.", flush=True)
    progress_every = max(1, total_evals // 100)
    start_time = time.time()
    done = 0

    rows = []
    for trial_idx, trial in enumerate(trials, start=1):
        df_adapt_raw = pd.read_csv(trial["adapt_file"])
        df_test_raw = pd.read_csv(trial["test_file"])

        columns_for_trial = extend_columns_for_regrow(df_adapt_raw, cat_cols, columns)
        X_adapt, y_adapt = load_xy(df_adapt_raw, cat_cols, columns_for_trial)
        X_test, y_test = load_xy(df_test_raw, cat_cols, columns_for_trial)

        pred_old_adapt = predict(old_tree, X_adapt)
        pred_old_test = predict(old_tree, X_test)
        f1_old_adapt = f1_macro(y_adapt, pred_old_adapt)
        f1_old_test = f1_macro(y_test, pred_old_test)
        acc_old_adapt = acc_score(y_adapt, pred_old_adapt)
        acc_old_test = acc_score(y_test, pred_old_test)

        cart_start = time.perf_counter()
        (
            cart_tree, f1_cart_adapt, f1_cart_test, acc_cart_adapt, acc_cart_test, cart_total_nodes,
            sim_old_cart, sim_old_cart_labeled, sim_old_cart_jaccard, pct_reaudit_cart,
        ) = build_cart_baseline(
            df_adapt_raw, df_test_raw, max_depth, rs_old, old_tree=old_tree,
        )
        cart_train_time_sec = time.perf_counter() - cart_start

        cart_entropy_start = time.perf_counter()
        (
            cart_entropy_tree, f1_cart_entropy_adapt, f1_cart_entropy_test,
            acc_cart_entropy_adapt, acc_cart_entropy_test, cart_entropy_total_nodes,
            sim_old_cart_entropy, sim_old_cart_entropy_labeled, sim_old_cart_entropy_jaccard,
            pct_reaudit_cart_entropy,
        ) = build_cart_baseline(
            df_adapt_raw, df_test_raw, max_depth, rs_old, old_tree=old_tree, criterion="entropy",
        )
        cart_entropy_train_time_sec = time.perf_counter() - cart_entropy_start

        c45_start = time.perf_counter()
        (c45_tree, f1_c45_adapt, f1_c45_test, acc_c45_adapt, acc_c45_test,
         c45_total_nodes) = build_chefboost_baseline("C4.5", df_adapt_raw, df_test_raw, max_depth)
        c45_train_time_sec = time.perf_counter() - c45_start

        j48_start = time.perf_counter()
        (
            (j48_tree, f1_j48_adapt, f1_j48_test, acc_j48_adapt, acc_j48_test, j48_total_nodes,
             sim_old_j48, sim_old_j48_labeled, sim_old_j48_jaccard, pct_reaudit_j48),
            (j48_unbounded_tree, f1_j48_unbounded_adapt, f1_j48_unbounded_test, acc_j48_unbounded_adapt,
             acc_j48_unbounded_test, j48_unbounded_total_nodes, sim_old_j48_unbounded,
             sim_old_j48_unbounded_labeled, sim_old_j48_unbounded_jaccard, pct_reaudit_j48_unbounded),
        ) = build_j48_pair_baseline(df_adapt_raw, df_test_raw, max_depth, rs_old, old_tree=old_tree)
        j48_train_time_sec = time.perf_counter() - j48_start

        reptree_start = time.perf_counter()
        (
            reptree_tree, f1_reptree_adapt, f1_reptree_test, acc_reptree_adapt, acc_reptree_test, reptree_total_nodes,
            sim_old_reptree, sim_old_reptree_labeled, sim_old_reptree_jaccard, pct_reaudit_reptree,
        ) = build_reptree_baseline(df_adapt_raw, df_test_raw, max_depth, rs_old, old_tree=old_tree)
        reptree_train_time_sec = time.perf_counter() - reptree_start

        rulesrepair_cache = {}

        for leaf_id, forced_ids in path_list:
            for w_simp in w_simps:
                for w_simi in w_simis:
                    rulesrepair_start = time.perf_counter()
                    new_tree = keep_remine_prune_alg.grow_tree(
                        X_adapt, y_adapt, old_tree=old_tree,
                        w_simp=w_simp, w_simi=w_simi,
                        max_depth=max_depth, grow_func=keep_remine_prune_alg.sklearn_grow_func,
                        cache=rulesrepair_cache, forced_keep_ids=forced_ids,
                    )
                    rulesrepair_grow_time_sec = time.perf_counter() - rulesrepair_start

                    pred_new_adapt = predict(new_tree, X_adapt, cache=rulesrepair_cache)
                    pred_new_test = predict(new_tree, X_test, cache=rulesrepair_cache)
                    f1_new_adapt = f1_macro(y_adapt, pred_new_adapt)
                    f1_new_test = f1_macro(y_test, pred_new_test)
                    acc_new_adapt = acc_score(y_adapt, pred_new_adapt)
                    acc_new_test = acc_score(y_test, pred_new_test)

                    n_total = count_nodes(new_tree)
                    n_new = count_new_nodes(new_tree)

                    rs_new = tuple_tree_conversion(new_tree)
                    sim_old_new = rule_set_similarity(rs_old, rs_new)
                    sim_old_new_labeled = rule_set_similarity_labeled(rs_old, rs_new)
                    sim_old_new_jaccard = jaccard_rule_set_similarity(rs_old, rs_new)

                    mark_reaudit_nodes(old_tree, new_tree)
                    _, _, pct_reaudit_new = reaudit_summary(new_tree)

                    row = {
                        "trial_id": f"{trial['trial_id']}_fixedleaf{leaf_id}" if fixed else trial["trial_id"],
                        "scenario": trial["scenario"],
                        "seed": trial.get("seed"),
                        "feature_type": trial["feature_type"],
                        "intensity": trial["intensity"],
                        "extent": trial.get("extent"),  # scenario 1 only
                        "pair_seed": trial.get("pair_seed"),  # scenario 2 only
                        "perturbation_seed": trial.get("perturbation_seed"),  # scenario 2 only
                        "total_possible_pairs": trial.get("total_possible_pairs"),  # scenario 2 only
                        "pair_selection": trial.get("pair_selection"),  # scenario 2 only
                        "features": ",".join(trial["features"]),
                        "w_simp": w_simp,
                        "w_simi": w_simi,
                        "t": None,  # kept for CSV-schema compat with pareto/dominance tooling; no longer swept
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
                        "rulesrepair_grow_time_sec": round(rulesrepair_grow_time_sec, 6),
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

                    done += 1
                    if done % progress_every == 0 or done == total_evals:
                        elapsed = time.time() - start_time
                        rate = done / elapsed if elapsed > 0 else 0
                        eta_min = (total_evals - done) / rate / 60 if rate > 0 else float("inf")
                        print(
                            f"  {done}/{total_evals} ({100 * done / total_evals:.1f}%), "
                            f"trial {trial_idx}/{len(trials)}, "
                            f"elapsed {elapsed / 60:.1f} min, ETA {eta_min:.1f} min",
                            flush=True,
                        )

    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)
    elapsed_total = time.time() - start_time
    print(f"Done in {elapsed_total / 60:.1f} min: {len(trials)} trials x {len(w_simps)}x{len(w_simis)} "
          f"(w_simp,w_simi) -> {len(out)} rows -> {out_csv}", flush=True)
    return out_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--normative-model", default="normative_model.pkl", help="Pickle produced by build_normative_model.py, containing T_old for every decision point.")
    parser.add_argument("--dp", required=True, help="Which decision point's entry to use from --normative-model, e.g. 'n5'.")
    parser.add_argument("--data-csv", required=True, help="Table that was perturbed to produce the manifest's trials. Must match --data-csv passed to run_perturbation_experiment.py.")
    parser.add_argument("--target", default="branch")
    parser.add_argument("--adapt-fraction", type=float, default=0.5)
    parser.add_argument(
        "--split-method", choices=["positional", "random", "case_chronological"], default="case_chronological",
        help="Must match --split-method passed to run_perturbation_experiment.py for this manifest -- "
             "used here only to rebuild the unperturbed D_test baseline the same way.",
    )
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--w-simps", default="0.0,0.001,0.005,0.01,0.02,0.05,0.1,0.2,0.3,0.5,0.75,1.0,1.5,2.0,3.0,5.0", help="Comma-separated list of w_simp values to sweep. Normalized scale (complexity term divided by nodes_max=2**(max_depth+1)-1): range covers no-penalty through full collapse-to-root, calibrated across sepsis/hospital_billing/road_traffic decision points spanning n_train 154-37592. Extra resolution below 0.02 (0.001/0.005/0.01) since some hospital_billing dps collapse to a trivial 1-node tree already at the first coarser nonzero value.")
    parser.add_argument("--w-simis", default="0.0,0.001,0.005,0.01,0.02,0.05,0.1,0.2,0.3,0.5,0.75,1.0,1.5", help="Comma-separated list of w_simi values to sweep. Normalized scale (audit term divided by nodes_max). Narrower top end than --w-simps (1.5 vs 5.0): w_simi saturates (n_new=0, fully reverts to keep) by ~0.5-1 in every decision point checked, so values above 1.5 are redundant.")
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args()

    w_simps = [float(a) for a in args.w_simps.split(",")]
    w_simis = [float(b) for b in args.w_simis.split(",")]

    run_repair(
        normative_model_path=args.normative_model, dp=args.dp, data_csv=args.data_csv,
        target=args.target, adapt_fraction=args.adapt_fraction, split_method=args.split_method,
        split_seed=args.split_seed, manifest_path=args.manifest, out_csv=args.out_csv,
        w_simps=w_simps, w_simis=w_simis, max_depth=args.max_depth,
    )


if __name__ == "__main__":
    main()
