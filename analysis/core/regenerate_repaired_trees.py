import argparse
import pickle
from pathlib import Path

import pandas as pd

from keep_remine_prune import keep_remine_prune_alg
from keep_remine_prune.param_grid import GROW_CONFIGS, grow_kwargs
from repair.run_perturbation_repair import load_xy, predict, count_nodes, acc_score, extend_columns_for_regrow

CONFIG_BY_NAME = {c["name"]: c for c in GROW_CONFIGS}
SELECTION_COLUMNS = ["trial_id", "mining_config", "w_simp", "w_simi"]


def _load_context(normative_model_path, dp, mutated_dir):
    mutated_dir = Path(mutated_dir)
    with open(normative_model_path, "rb") as f:
        normative_model = pickle.load(f)
    if dp not in normative_model:
        raise SystemExit(f"'{dp}' not found in {normative_model_path}. Available: {sorted(normative_model.keys())}")
    entry = normative_model[dp]
    cat_cols = list({orig for orig, _ in entry["onehot_map"].values()})

    df_adapt_raw = pd.read_csv(mutated_dir / "shared_train.csv")
    df_test_raw = pd.read_csv(mutated_dir / "shared_test.csv")
    columns = extend_columns_for_regrow(df_adapt_raw, cat_cols, entry["columns"])
    X_adapt, y_adapt = load_xy(df_adapt_raw, cat_cols, columns)
    X_test, y_test = load_xy(df_test_raw, cat_cols, columns)
    return {
        "mutated_dir": mutated_dir, "max_depth": entry.get("max_depth", 4),
        "X_adapt": X_adapt, "y_adapt": y_adapt, "X_test": X_test, "y_test": y_test,
    }


def _load_mutant_tree(mutated_dir, trial_id):
    with open(Path(mutated_dir) / f"{trial_id}_mutant_tree.pkl", "rb") as f:
        return pickle.load(f)


def _find_result_row(mutated_dir, mining_config, trial_id, w_simp, w_simi):
    results_path = Path(mutated_dir) / "configs" / mining_config / "results.csv"
    if not results_path.exists():
        return None
    results = pd.read_csv(results_path)
    match = results[
        (results["trial_id"] == trial_id)
        & ((results["w_simp"] - w_simp).abs() < 1e-12)
        & ((results["w_simi"] - w_simi).abs() < 1e-12)
    ]
    return match.iloc[0] if len(match) else None


def regenerate_repaired_trees(normative_model_path, dp, mutated_dir, selections, max_depth=None, verify=True):
    ctx = _load_context(normative_model_path, dp, mutated_dir)
    depth = max_depth if max_depth is not None else ctx["max_depth"]
    mutant_cache = {}
    records = []
    n_mismatch = 0

    for sel in selections:
        trial_id, config_name = sel["trial_id"], sel["mining_config"]
        w_simp, w_simi = float(sel["w_simp"]), float(sel["w_simi"])
        if config_name not in CONFIG_BY_NAME:
            raise SystemExit(f"Unknown mining_config '{config_name}'. Known: {sorted(CONFIG_BY_NAME)}")
        if trial_id not in mutant_cache:
            mutant_cache[trial_id] = _load_mutant_tree(ctx["mutated_dir"], trial_id)
        mutant_tree = mutant_cache[trial_id]

        cache = {}
        new_tree = keep_remine_prune_alg.grow_tree(
            ctx["X_adapt"], ctx["y_adapt"], old_tree=mutant_tree,
            w_simp=w_simp, w_simi=w_simi,
            max_depth=depth, grow_func=keep_remine_prune_alg.sklearn_grow_func,
            cache=cache, forced_keep_ids=None,
            grow_params=grow_kwargs(CONFIG_BY_NAME[config_name]),
        )
        pred_test = predict(new_tree, ctx["X_test"], cache=cache)
        acc_test = acc_score(ctx["y_test"], pred_test)
        n_nodes = count_nodes(new_tree)

        record = {
            "trial_id": trial_id, "mining_config": config_name, "w_simp": w_simp, "w_simi": w_simi,
            "old_tree": mutant_tree, "new_tree": new_tree,
            "acc_new_test": acc_test, "total_nodes": n_nodes,
        }

        if verify:
            row = _find_result_row(ctx["mutated_dir"], config_name, trial_id, w_simp, w_simi)
            if row is None:
                record["verified"] = None
                print(f"  {trial_id} {config_name} w_simp={w_simp} w_simi={w_simi}: no matching row in results.csv, not verified.")
            else:
                ok = abs(float(row["acc_new_test"]) - acc_test) < 1e-9 and int(row["total_nodes"]) == n_nodes
                record["verified"] = bool(ok)
                if not ok:
                    n_mismatch += 1
                    print(f"  MISMATCH {trial_id} {config_name} w_simp={w_simp} w_simi={w_simi}: "
                          f"results.csv acc_new_test={row['acc_new_test']}, total_nodes={row['total_nodes']} "
                          f"vs regenerated acc_new_test={acc_test}, total_nodes={n_nodes}")
        records.append(record)

    n_ok = sum(1 for r in records if r.get("verified") is True)
    print(f"Regenerated {len(records)} tree(s); {n_ok} matched results.csv, {n_mismatch} mismatched.")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--normative-model", required=True)
    parser.add_argument("--dp", required=True)
    parser.add_argument("--mutated-dir", required=True,
                        help="experiments/<dataset>/repair/seed_<N>/<dp>/mutated")
    parser.add_argument("--selection-csv", default=None,
                        help="CSV with columns trial_id, mining_config, w_simp, w_simi (e.g. rows of a results.csv).")
    parser.add_argument("--trial-id", default=None)
    parser.add_argument("--mining-config", default=None)
    parser.add_argument("--w-simp", type=float, default=None)
    parser.add_argument("--w-simi", type=float, default=None)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--out-pkl", required=True)
    args = parser.parse_args()

    if args.selection_csv:
        selection_df = pd.read_csv(args.selection_csv)
        missing = [c for c in SELECTION_COLUMNS if c not in selection_df.columns]
        if missing:
            raise SystemExit(f"{args.selection_csv}: missing column(s) {missing}")
        selections = selection_df[SELECTION_COLUMNS].to_dict("records")
    else:
        if None in (args.trial_id, args.mining_config, args.w_simp, args.w_simi):
            raise SystemExit("Pass --selection-csv, or all of --trial-id, --mining-config, --w-simp, --w-simi.")
        selections = [{
            "trial_id": args.trial_id, "mining_config": args.mining_config,
            "w_simp": args.w_simp, "w_simi": args.w_simi,
        }]

    records = regenerate_repaired_trees(
        args.normative_model, args.dp, args.mutated_dir, selections,
        max_depth=args.max_depth, verify=not args.no_verify,
    )
    out_pkl = Path(args.out_pkl)
    out_pkl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_pkl, "wb") as f:
        pickle.dump(records, f)
    print(f"Saved {len(records)} tree(s) -> {out_pkl}")


if __name__ == "__main__":
    main()
