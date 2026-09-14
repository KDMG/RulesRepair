import argparse
import pickle

import pandas as pd

from repair.run_baseline_repair import run_baseline_repair
from repair.run_mutated_repair import (
    run_mutated_repair,
    DEFAULT_P, DEFAULT_DEGRADATION_THRESHOLD,
    DEFAULT_MAX_REGROW_ATTEMPTS, DEFAULT_REGROW_MAX_DEPTH, DEFAULT_REGROW_SPLIT_PROB,
    DEFAULT_MAX_ALTERNATIVES_PER_NODE,
)
from repair.run_perturbation_experiment import split_adapt_test


def _parse_floats(csv_string):
    return [float(x) for x in csv_string.split(",")]


def run_both(
    dp, data_csv, out_dir,
    normative_model="normative_model.pkl", target="branch",
    adapt_fraction=0.7, split_method="case_chronological", split_seed=0, max_depth=4,
    w_simps=(0.0, 5.0), w_simis=(0.0, 1.0),
    save_pareto_trees=False,
    operator_names=None,
    p=DEFAULT_P, degradation_threshold=DEFAULT_DEGRADATION_THRESHOLD, seed=0,
    n_thresholds=10, max_regrow_attempts=DEFAULT_MAX_REGROW_ATTEMPTS,
    regrow_max_depth=DEFAULT_REGROW_MAX_DEPTH, regrow_split_prob=DEFAULT_REGROW_SPLIT_PROB,
    fixed=False, skip_baseline=False, baseline_tree_cache_dir=None,
    max_alternatives_per_node=DEFAULT_MAX_ALTERNATIVES_PER_NODE,
):
    df_data = pd.read_csv(data_csv)
    preflight_adapt, preflight_test = split_adapt_test(df_data, target, adapt_fraction, split_method, split_seed)
    if len(preflight_adapt) == 0 or len(preflight_test) == 0:
        print(
            f"Skipping '{dp}': only {len(df_data)} row(s) in {data_csv}, "
            f"splitting would produce an empty "
            f"{'D_adapt' if len(preflight_adapt) == 0 else 'D_test'} "
            f"({len(preflight_adapt)} adapt / {len(preflight_test)} test rows). "
            f"Too little data, no results.csv written for it."
        )
        return None

    with open(normative_model, "rb") as f:
        available_dps = set(pickle.load(f).keys())
    if dp not in available_dps:
        print(
            f"Skipping '{dp}': no tree for this decision point in {normative_model} "
            f"(excluded during mining, no usable feature columns). "
            f"No results.csv written for it."
        )
        return None

    baseline_outputs = None
    if skip_baseline:
        print(f"baseline repair skipped (skip_baseline=True, reused from elsewhere): {dp}")
    else:
        print(f"baseline repair (no perturbation){' [--fixed]' if fixed else ''}: {dp}")
        baseline_outputs = run_baseline_repair(
            dp=dp, data_csv=data_csv, out_dir=out_dir,
            normative_model=normative_model, target=target,
            adapt_fraction=adapt_fraction, split_method=split_method, split_seed=split_seed,
            max_depth=max_depth, w_simps=w_simps, w_simis=w_simis,
            save_pareto_trees=save_pareto_trees,
            fixed=fixed,
        )

    print(f"\nmutated repair (tree mutation){' [--fixed]' if fixed else ''}: {dp}")
    mutated_dir = f"{out_dir}/mutated_fixed" if fixed else f"{out_dir}/mutated"
    manifest_path, results_csv, trees_pkl, analysis_dir = run_mutated_repair(
        normative_model_path=normative_model, dp=dp, data_csv=data_csv, target=target,
        adapt_fraction=adapt_fraction, split_method=split_method, split_seed=split_seed,
        out_dir=mutated_dir, operator_names=operator_names,
        w_simps=w_simps, w_simis=w_simis, max_depth=max_depth,
        p=p, degradation_threshold=degradation_threshold, seed=seed,
        n_thresholds=n_thresholds, max_regrow_attempts=max_regrow_attempts,
        regrow_max_depth=regrow_max_depth, regrow_split_prob=regrow_split_prob,
        fixed=fixed, baseline_tree_cache_dir=baseline_tree_cache_dir,
        max_alternatives_per_node=max_alternatives_per_node,
    )
    mutated_outputs = {
        "manifest": manifest_path, "results_csv": results_csv,
        "trees_pkl": trees_pkl, "analysis_dir": analysis_dir,
    }

    print(f"\nRepair complete for '{dp}'. Output under: {out_dir}")
    if not skip_baseline:
        print(f"  Baseline -> {out_dir}/{'baseline_fixed' if fixed else 'baseline'}")
    print(f"  Mutated  -> {mutated_dir}")

    return {
        "baseline": baseline_outputs["baseline"] if baseline_outputs is not None else None,
        "mutated": mutated_outputs,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dp", required=True, help="Decision point name, e.g. 'n5'. Must be a key in --normative-model.")
    parser.add_argument("--data-csv", required=True, help="Table to split into D_adapt/D_test, unperturbed, shared by both runs.")
    parser.add_argument("--normative-model", default="normative_model.pkl")
    parser.add_argument("--target", default="branch")
    parser.add_argument("--adapt-fraction", type=float, default=0.7)
    parser.add_argument(
        "--split-method", choices=["positional", "random", "case_chronological"], default="case_chronological",
    )
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--max-depth", type=int, default=4)

    parser.add_argument("--w-simps", default="0.0,0.001,0.005,0.01,0.02,0.05,0.1,0.2,0.3,0.5,0.75,1.0,1.5,2.0,3.0,5.0", help="Comma-separated w_simp grid, shared by both runs. Normalized scale (complexity term / nodes_max).")
    parser.add_argument("--w-simis", default="0.0,0.001,0.005,0.01,0.02,0.05,0.1,0.2,0.3,0.5,0.75,1.0,1.5", help="Comma-separated w_simi grid, shared by both runs. Normalized scale (audit term / nodes_max).")
    parser.add_argument(
        "--save-pareto-trees", action=argparse.BooleanOptionalAction, default=False,
        help="Baseline only: regenerate/pickle every per-trial Pareto-optimal repaired tree to baseline/pareto_trees.pkl.",
    )

    parser.add_argument(
        "--operators", default=None,
        help="Mutated repair only: comma-separated subset of prune,change_label,branch_swap,"
             "change_threshold,change_feature,regrow_leaf,regrow_internal (default: all 7).",
    )
    parser.add_argument("--p", type=float, default=DEFAULT_P, help="Mutated repair only: fraction of compatible nodes to target per operator, ceil-rounded, default 0.2.")
    parser.add_argument("--degradation-threshold", type=float, default=DEFAULT_DEGRADATION_THRESHOLD, help="Mutated repair only: minimum plain-accuracy drop required to accept a mutation, default 0.2.")
    parser.add_argument("--seed", type=int, default=0, help="Mutated repair only: single global seed for mutation sampling.")
    parser.add_argument("--n-thresholds", type=int, default=10, help="Mutated repair only: number of n-tiles per feature in the threshold pool.")
    parser.add_argument("--max-regrow-attempts", type=int, default=DEFAULT_MAX_REGROW_ATTEMPTS)
    parser.add_argument("--regrow-max-depth", type=int, default=DEFAULT_REGROW_MAX_DEPTH)
    parser.add_argument("--regrow-split-prob", type=float, default=DEFAULT_REGROW_SPLIT_PROB)
    parser.add_argument(
        "--max-alternatives-per-node", type=int, default=DEFAULT_MAX_ALTERNATIVES_PER_NODE,
        help="Mutated repair only, 2026 addition, default 100. Caps how many alternatives "
             "change_threshold/change_feature try per node (random subsample of the shuffled "
             "list) before giving up on it, useful if mutant generation is taking hours on a "
             "decision point with many one-hot columns and/or large D_adapt. Pass a very large "
             "value (e.g. 999999) for unchanged exhaustive behaviour. Forwarded as-is to "
             "run_mutated_repair.py.",
    )

    parser.add_argument(
        "--fixed", action=argparse.BooleanOptionalAction, default=False,
        help="'Expert forces keep' experiment (both baseline and mutated repair, scope confirmed "
             "with the user): same seed/split/grid as a normal run, but repeated once per "
             "root-to-leaf path (T_old's for baseline, each mutant's own for mutated), with that "
             "path's nodes forced to keep. Output goes to baseline_fixed/ and mutated_fixed/ "
             "instead of baseline/ and mutated/ -- never touches a normal run's output. Not "
             "compatible with --save-pareto-trees yet.",
    )
    parser.add_argument("--out-dir", required=True, help="Base output directory; baseline/ and mutated/ (or baseline_fixed/ and mutated_fixed/ with --fixed) subfolders are created inside it.")
    parser.add_argument(
        "--skip-baseline", action=argparse.BooleanOptionalAction, default=False,
        help="2026 addition: skip baseline repair entirely (only run mutated repair). Baseline "
             "depends only on data-csv/split-seed, never on --seed -- use this for every seed "
             "after the first in a --seed sweep over the SAME dp, and copy the first seed's "
             "baseline/ output into the others instead (see run_repair_all_seeds.py).",
    )
    parser.add_argument(
        "--baseline-tree-cache-dir", default=None,
        help="2026 addition: forwarded to run_mutated_repair.py's --baseline-tree-cache-dir -- "
             "caches the once-per-dp CART/CART-entropy/C4.5/J48(bounded+unbounded)/REPTree "
             "fits so a --seed sweep over the same dp fits them only once. See that script's "
             "own --help for details.",
    )
    args = parser.parse_args()

    operator_names = args.operators.split(",") if args.operators else None

    run_both(
        dp=args.dp, data_csv=args.data_csv, out_dir=args.out_dir,
        normative_model=args.normative_model, target=args.target,
        adapt_fraction=args.adapt_fraction, split_method=args.split_method, split_seed=args.split_seed,
        max_depth=args.max_depth,
        w_simps=_parse_floats(args.w_simps), w_simis=_parse_floats(args.w_simis),
        save_pareto_trees=args.save_pareto_trees,
        operator_names=operator_names,
        p=args.p, degradation_threshold=args.degradation_threshold, seed=args.seed,
        n_thresholds=args.n_thresholds, max_regrow_attempts=args.max_regrow_attempts,
        regrow_max_depth=args.regrow_max_depth, regrow_split_prob=args.regrow_split_prob,
        fixed=args.fixed, skip_baseline=args.skip_baseline,
        baseline_tree_cache_dir=args.baseline_tree_cache_dir,
        max_alternatives_per_node=args.max_alternatives_per_node,
    )


if __name__ == "__main__":
    main()
