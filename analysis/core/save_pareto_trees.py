import argparse
import json
import pickle
import time
from pathlib import Path

import pandas as pd

from keep_remine_prune import keep_remine_prune_alg
from keep_remine_prune.tree_ruleset_conversion import tuple_tree_conversion
from keep_remine_prune.similar_tree import rule_set_similarity, rule_set_similarity_labeled, jaccard_rule_set_similarity
from keep_remine_prune.reaudit import mark_reaudit_nodes, reaudit_summary
from repair.run_perturbation_repair import (
    load_xy, predict, count_nodes, count_new_nodes, f1_macro, acc_score,
    SKLEARN_GROW_FUNC_MIN_SAMPLES_LEAF, build_cart_baseline, extend_columns_for_regrow,
)

TRIAL_CONTEXT_COLS = [
    "feature_type", "features", "intensity", "extent",
    "pair_seed", "perturbation_seed", "total_possible_pairs", "pair_selection",
    # Scenario 3 (tree mutation) only:
    "operator", "mutation_index", "is_sound",
]


def save_pareto_trees(normative_model_path, dp, manifest_path, pareto_csv, out_pkl, max_depth=4):
    with open(normative_model_path, "rb") as f:
        normative_model = pickle.load(f)
    if dp not in normative_model:
        raise SystemExit(f"'{dp}' not found in {normative_model_path}. Available: {sorted(normative_model.keys())}")
    dp_entry = normative_model[dp]
    old_tree, onehot_map, columns = dp_entry["tree"], dp_entry["onehot_map"], dp_entry["columns"]
    cat_cols = list({orig for orig, _ in onehot_map.values()})
    positive_class = dp_entry.get("positive_class")

    pkl_max_depth = dp_entry.get("max_depth")
    pkl_min_samples_leaf = dp_entry.get("min_samples_leaf")
    if pkl_max_depth is not None and pkl_max_depth != max_depth:
        print(f"Warning: T_old for '{dp}' was mined with max_depth={pkl_max_depth}, but this run uses "
              f"max_depth={max_depth} -- pass max_depth={pkl_max_depth} to stay consistent with T_old.")
    if pkl_min_samples_leaf is not None and pkl_min_samples_leaf != SKLEARN_GROW_FUNC_MIN_SAMPLES_LEAF:
        print(f"Warning: T_old for '{dp}' was mined with min_samples_leaf={pkl_min_samples_leaf}, but "
              f"keep_remine_prune_alg.sklearn_grow_func hardcodes min_samples_leaf={SKLEARN_GROW_FUNC_MIN_SAMPLES_LEAF} "
              f"-- not consistent with T_old.")

    pareto_df = pd.read_csv(pareto_csv)
    if "is_pareto" not in pareto_df.columns:
        raise SystemExit(f"'is_pareto' column not found in {pareto_csv}. Available: {list(pareto_df.columns)}")
    pareto_df = pareto_df[pareto_df["is_pareto"]].reset_index(drop=True)

    n_trials = pareto_df["trial_id"].nunique()
    print(f"{len(pareto_df)} Pareto-optimal (trial_id, w_simp, w_simi, t) rows across {n_trials} trials "
          f"-> regenerating trees from {manifest_path}", flush=True)

    trials_by_id = {}
    with open(manifest_path) as f:
        for line in f:
            trial = json.loads(line)
            trials_by_id[trial["trial_id"]] = trial
    missing = set(pareto_df["trial_id"]) - set(trials_by_id)
    if missing:
        raise SystemExit(f"{len(missing)} trial_id(s) from {pareto_csv} not found in {manifest_path}, "
                          f"e.g. {sorted(missing)[:5]}")

    records = []
    progress_every = max(1, len(pareto_df) // 100)
    start_time = time.time()
    done = 0

    for trial_id, group in pareto_df.groupby("trial_id", sort=False):
        trial = trials_by_id[trial_id]
        df_adapt_raw = pd.read_csv(trial["adapt_file"])
        df_test_raw = pd.read_csv(trial["test_file"])
        columns_for_trial = extend_columns_for_regrow(df_adapt_raw, cat_cols, columns)
        X_adapt, y_adapt = load_xy(df_adapt_raw, cat_cols, columns_for_trial)
        X_test, y_test = load_xy(df_test_raw, cat_cols, columns_for_trial)
        context = {c: trial.get(c) for c in TRIAL_CONTEXT_COLS}

        if trial.get("scenario") == 3 and trial.get("mutant_tree_file"):
            with open(trial["mutant_tree_file"], "rb") as f:
                trial_old_tree = pickle.load(f)
        else:
            trial_old_tree = old_tree
        rs_old = tuple_tree_conversion(trial_old_tree)

        (
            cart_tree, f1_cart_adapt, f1_cart_test, acc_cart_adapt, acc_cart_test, cart_total_nodes,
            sim_old_cart, sim_old_cart_labeled, sim_old_cart_jaccard, pct_reaudit_cart,
        ) = build_cart_baseline(
            df_adapt_raw, df_test_raw, max_depth, rs_old, old_tree=trial_old_tree,
        )
        cart_fields = {
            "cart_tree": cart_tree, "cart_total_nodes": cart_total_nodes,
            "f1_cart_adapt": f1_cart_adapt, "f1_cart_test": f1_cart_test,
            "acc_cart_adapt": acc_cart_adapt, "acc_cart_test": acc_cart_test,
            "sim_old_cart": sim_old_cart,
            "sim_old_cart_labeled": sim_old_cart_labeled, "sim_old_cart_jaccard": sim_old_cart_jaccard,
            "pct_reaudit_cart": pct_reaudit_cart,
        }

        kr_cache = {}

        for _, row in group.iterrows():
            w_simp, w_simi, t = row["w_simp"], row["w_simi"], row["t"]
            t = None if pd.isna(t) else t

            new_tree = keep_remine_prune_alg.grow_tree(
                X_adapt, y_adapt, old_tree=trial_old_tree,
                w_simp=w_simp, w_simi=w_simi, t=t, positive_class=positive_class,
                max_depth=max_depth, grow_func=keep_remine_prune_alg.sklearn_grow_func,
                cache=kr_cache,
            )

            pred_new_adapt = predict(new_tree, X_adapt)
            pred_new_test = predict(new_tree, X_test)
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

            mark_reaudit_nodes(trial_old_tree, new_tree)
            _, _, pct_reaudit_new = reaudit_summary(new_tree)

            records.append({
                "trial_id": trial_id, "w_simp": w_simp, "w_simi": w_simi, "t": t,
                "tree": new_tree,
                "total_nodes": n_total, "new_nodes": n_new,
                "pct_to_reaudit": round(100 * n_new / n_total, 1) if n_total else 0.0,
                "pct_reaudit_new": pct_reaudit_new,
                "f1_new_adapt": f1_new_adapt, "f1_new_test": f1_new_test,
                "acc_new_adapt": acc_new_adapt, "acc_new_test": acc_new_test,
                "sim_old_new": round(sim_old_new, 4),
                "sim_old_new_labeled": round(sim_old_new_labeled, 4),
                "sim_old_new_jaccard": round(sim_old_new_jaccard, 4),
                **cart_fields,
                **context,
            })

            done += 1
            if done % progress_every == 0 or done == len(pareto_df):
                elapsed = time.time() - start_time
                rate = done / elapsed if elapsed > 0 else 0
                eta_min = (len(pareto_df) - done) / rate / 60 if rate > 0 else float("inf")
                print(f"  {done}/{len(pareto_df)} ({100 * done / len(pareto_df):.1f}%), "
                      f"elapsed {elapsed / 60:.1f} min, ETA {eta_min:.1f} min", flush=True)

    out_pkl = Path(out_pkl)
    out_pkl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_pkl, "wb") as f:
        pickle.dump(records, f)
    print(f"Saved {len(records)} Pareto-optimal tree(s) -> {out_pkl}", flush=True)
    return out_pkl


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--normative-model", default="normative_model.pkl", help="Pickle produced by build_normative_model.py, containing T_old for every decision point.")
    parser.add_argument("--dp", required=True, help="Which decision point's entry to use, e.g. 'n5'.")
    parser.add_argument("--manifest", required=True, help="manifest.jsonl written by run_perturbation_experiment.py for this scenario.")
    parser.add_argument("--pareto-csv", required=True, help="pareto_per_trial.csv written by trial_pareto_analysis.py for this scenario's results.csv.")
    parser.add_argument("--out-pkl", required=True)
    parser.add_argument("--max-depth", type=int, default=4)
    args = parser.parse_args()

    save_pareto_trees(
        normative_model_path=args.normative_model, dp=args.dp, manifest_path=args.manifest,
        pareto_csv=args.pareto_csv, out_pkl=args.out_pkl, max_depth=args.max_depth,
    )


if __name__ == "__main__":
    main()
