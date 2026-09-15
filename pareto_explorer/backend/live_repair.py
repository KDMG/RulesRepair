import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

from ._paths import REPO_ROOT
from keep_remine_prune import keep_remine_prune_alg
from keep_remine_prune.tree_ruleset_conversion import tuple_tree_conversion
from keep_remine_prune.similar_tree import jaccard_rule_set_similarity
from repair.run_perturbation_repair import (
    load_xy, predict, acc_score, count_nodes,
    build_cart_baseline, build_j48_baseline, build_reptree_baseline,
)
from repair.run_perturbation_experiment import split_adapt_test
from analysis.core.pareto_analysis import compute_pareto_front

from .tree_diff import forced_ids_for_leaves

DEFAULT_MAX_DEPTH = 4
DEFAULT_ALPHAS = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
DEFAULT_BETAS = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5]


class MissingDataError(Exception):
    """Raised when one of the 3 required inputs (model / log observations)
    isn't loaded yet, or doesn't cover the decision point being computed.
    """


BASELINE_LABELS = {"cart": "CART", "j48": "C4.5", "reptree": "REPTree"}
BASELINE_BUILDERS = {"cart": build_cart_baseline, "j48": build_j48_baseline, "reptree": build_reptree_baseline}



def compute_baseline_point(prefix, base_tree, rs_old, df_adapt_raw, df_test_raw=None, max_depth=DEFAULT_MAX_DEPTH):
    builder = BASELINE_BUILDERS[prefix]
    nodes_max = 2 ** (max_depth + 1) - 1
    (
        tree, f1_adapt, f1_test, acc_adapt, acc_test,
        total_nodes, sim_old, sim_old_labeled, sim_old_jaccard, pct_reaudit,
    ) = builder(df_adapt_raw, df_test_raw if df_test_raw is not None else df_adapt_raw, max_depth, rs_old, old_tree=base_tree)
    if tree is None:
        return None
    return {
        "accuracy": acc_adapt,  # TRAIN/adapt accuracy ("Fitness") -- unchanged semantics/callers
        "test_accuracy": acc_test if df_test_raw is not None else None,  # 2026 addition, real held-out accuracy
        "simplicity": 1 - total_nodes / nodes_max,
        "jaccard": sim_old_jaccard,
        "tree": tree,
    }


def compute_baseline_point_for_tree(prefix, base_tree, df_adapt_raw, df_test_raw=None, max_depth=DEFAULT_MAX_DEPTH):
    rs_old = tuple_tree_conversion(base_tree)
    return compute_baseline_point(prefix, base_tree, rs_old, df_adapt_raw, df_test_raw=df_test_raw, max_depth=max_depth)


def compute_baseline_point_for_tree_isolated(prefix, base_tree, df_adapt_raw, df_test_raw=None, max_depth=DEFAULT_MAX_DEPTH, timeout=180):
    if prefix == "cart":
        return compute_baseline_point_for_tree(prefix, base_tree, df_adapt_raw, df_test_raw=df_test_raw, max_depth=max_depth)

    label = BASELINE_LABELS.get(prefix, prefix)
    script_path = Path(__file__).resolve().parent / "_baseline_subprocess.py"

    with tempfile.TemporaryDirectory() as tmp_dir:
        input_path = Path(tmp_dir) / "input.pkl"
        output_path = Path(tmp_dir) / "output.pkl"
        with open(input_path, "wb") as f:
            pickle.dump((prefix, base_tree, df_adapt_raw, df_test_raw, max_depth), f)

        try:
            result = subprocess.run(
                [sys.executable, str(script_path), str(input_path), str(output_path)],
                cwd=str(REPO_ROOT), timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Fitting {label} timed out after {timeout}s.")

        if result.returncode != 0:
            raise RuntimeError(
                f"Fitting {label} crashed the worker process (exit code {result.returncode}) -- this "
                f"usually means the local Java/JVM used by python-weka-wrapper3 is incompatible "
                f"(needs Java 9+). The main app was not affected. See the terminal output above "
                f"for the worker process's own error details."
            )

        if not output_path.exists():
            raise RuntimeError(f"Fitting {label} produced no result.")
        with open(output_path, "rb") as f:
            status, payload = pickle.load(f)

    if status == "error":
        raise RuntimeError(payload)
    return payload



def _find_offline_shared_train(dp_name):
    matches = sorted(REPO_ROOT.glob(f"experiments/*/repair/seed_*/{dp_name}/mutated/shared_train.csv"))
    if not matches:
        return None
    datasets = {m.relative_to(REPO_ROOT / "experiments").parts[0] for m in matches}
    if len(datasets) != 1:
        return None
    return matches[0]



def _find_offline_shared_test(dp_name):
    matches = sorted(REPO_ROOT.glob(f"experiments/*/repair/seed_*/{dp_name}/mutated/shared_test.csv"))
    if not matches:
        return None
    datasets = {m.relative_to(REPO_ROOT / "experiments").parts[0] for m in matches}
    if len(datasets) != 1:
        return None
    return matches[0]



def prepare_normative_base(dp_name, model, observations):
    if model is None:
        raise MissingDataError("No model loaded -- open a trees (.pkl) file first.")
    entry = model.get(dp_name)
    if entry is None or entry.get("tree") is None:
        raise MissingDataError(f"'{dp_name}' has no tree in the loaded model.")

    offline_path = _find_offline_shared_train(dp_name)
    if offline_path is not None:
        df_adapt = pd.read_csv(offline_path)
        return entry["tree"], df_adapt

    if observations is None:
        raise MissingDataError("No log loaded -- open a log (.xes) file first.")
    records = observations.get(dp_name)
    if not records:
        raise MissingDataError(f"The loaded log has zero observations for '{dp_name}'.")

    df_full = pd.DataFrame(records)
    df_adapt, _df_test = split_log_train_test(df_full)
    return entry["tree"], df_adapt



def split_log_train_test(df_full, adapt_fraction=0.7, split_method="case_chronological", split_seed=0):
    return split_adapt_test(df_full, target="branch", adapt_fraction=adapt_fraction,
                             split_method=split_method, split_seed=split_seed)



def compute_live_pareto_front(
    dp_name, model, mandatory_leaf_ids, base_tree, df_adapt_raw, df_test_raw=None,
    alphas=DEFAULT_ALPHAS, betas=DEFAULT_BETAS, max_depth=DEFAULT_MAX_DEPTH,
    progress_cb=None,
):
    onehot_map, columns = model[dp_name]["onehot_map"], model[dp_name]["columns"]
    cat_cols = list({orig for orig, _ in onehot_map.values()})
    forced_ids = forced_ids_for_leaves(base_tree, mandatory_leaf_ids)

    X_adapt, y_adapt = load_xy(df_adapt_raw, cat_cols, columns)
    X_test, y_test = (load_xy(df_test_raw, cat_cols, columns) if df_test_raw is not None else (None, None))

    rs_old = tuple_tree_conversion(base_tree)
    nodes_max = 2 ** (max_depth + 1) - 1

    pred_old_adapt = predict(base_tree, X_adapt, cache={})
    n_nodes_old = count_nodes(base_tree)
    train_f1 = model.get(dp_name, {}).get("f1_train")
    normative_test_accuracy = acc_score(y_test, predict(base_tree, X_test, cache={})) if X_test is not None else None
    normative_point = {
        "accuracy": acc_score(y_adapt, pred_old_adapt),
        "test_accuracy": normative_test_accuracy,
        "accuracy_display": normative_test_accuracy if normative_test_accuracy is not None else acc_score(y_adapt, pred_old_adapt),
        "simplicity": 1 - n_nodes_old / nodes_max,
        "jaccard": 1.0,
        "tree": base_tree,
        "train_f1": train_f1,
    }

    (
        cart_tree, f1_cart_adapt, f1_cart_test, acc_cart_adapt, acc_cart_test,
        cart_total_nodes, sim_old_cart, sim_old_cart_labeled, sim_old_cart_jaccard, pct_reaudit_cart,
    ) = build_cart_baseline(df_adapt_raw, df_test_raw if df_test_raw is not None else df_adapt_raw, max_depth, rs_old, old_tree=base_tree)
    cart_point = None
    if cart_tree is not None:
        cart_test_accuracy = acc_cart_test if df_test_raw is not None else None
        cart_point = {
            "accuracy": acc_cart_adapt,
            "test_accuracy": cart_test_accuracy,
            "accuracy_display": cart_test_accuracy if cart_test_accuracy is not None else acc_cart_adapt,
            "simplicity": 1 - cart_total_nodes / nodes_max,
            "jaccard": sim_old_cart_jaccard,
            "tree": cart_tree,
        }

    rows = []
    kr_cache = {}
    total = len(alphas) * len(betas)
    done = 0
    for alpha in alphas:
        for beta in betas:
            new_tree = keep_remine_prune_alg.grow_tree(
                X_adapt, y_adapt, old_tree=base_tree,
                w_simp=alpha, w_simi=beta,
                max_depth=max_depth, grow_func=keep_remine_prune_alg.sklearn_grow_func,
                cache=kr_cache, forced_keep_ids=forced_ids,
            )
            pred_adapt = predict(new_tree, X_adapt, cache=kr_cache)
            rs_new = tuple_tree_conversion(new_tree)
            n_nodes = count_nodes(new_tree)
            train_accuracy = acc_score(y_adapt, pred_adapt)
            test_accuracy = None
            if X_test is not None:
                pred_test = predict(new_tree, X_test, cache=kr_cache)
                test_accuracy = acc_score(y_test, pred_test)
            rows.append({
                "accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
                "accuracy_display": test_accuracy if test_accuracy is not None else train_accuracy,
                "total_nodes": n_nodes,
                "simplicity": 1 - n_nodes / nodes_max,
                "jaccard": jaccard_rule_set_similarity(rs_old, rs_new),
                "tree": new_tree,
            })
            done += 1
            if progress_cb:
                progress_cb(done, total)

    df = pd.DataFrame(rows)
    df["is_pareto"] = compute_pareto_front(
        df, f1_col="accuracy_display", nodes_col="total_nodes", reaudit_col="jaccard", reaudit_maximize=True,
    )
    return df, cart_point, normative_point
