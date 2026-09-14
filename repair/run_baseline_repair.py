import argparse
import pickle
from pathlib import Path

from repair.run_perturbation_experiment import run_experiment
from repair.run_perturbation_repair import run_repair
from analysis.core.trial_pareto_analysis import run_trial_pareto_analysis
from analysis.core.save_pareto_trees import save_pareto_trees as save_pareto_trees_fn


def check_prerequisites(data_csv, normative_model_path, dp):
    problems = []

    if not Path(data_csv).exists():
        problems.append(f"Data CSV not found: {data_csv}")

    normative_model_path = Path(normative_model_path)
    if not normative_model_path.exists():
        problems.append(
            f"Normative model not found: {normative_model_path}\n"
            f"Generate it with:\n"
            f"poetry run python build_normative_model.py --dp-dir <dir_of_dp_*.csv> --save {normative_model_path}"
        )
    else:
        with open(normative_model_path, "rb") as f:
            normative_model = pickle.load(f)
        if dp not in normative_model:
            problems.append(
                f"Decision point '{dp}' not found in {normative_model_path}. "
                f"Available: {sorted(normative_model.keys())}\n"
                f"Rebuild with:\n"
                f"poetry run python build_normative_model.py --dp-dir <dir_of_dp_*.csv> --save {normative_model_path}"
            )

    if problems:
        raise SystemExit(
            "Pipeline cannot start -- missing prerequisite(s):\n\n" + "\n\n".join(problems)
        )


def run_baseline_repair(
    dp, data_csv, out_dir,
    normative_model="normative_model.pkl", target="branch",
    adapt_fraction=0.7, split_method="case_chronological", split_seed=0, max_depth=4,
    w_simps=(0.0, 5.0), w_simis=(0.0, 1.0),
    save_pareto_trees=False,
    fixed=False,
):
    check_prerequisites(data_csv, normative_model, dp)

    out_dir = Path(out_dir)
    baseline_dir = out_dir / ("baseline_fixed" if fixed else "baseline")
    print(f"baseline repair{' (--fixed)' if fixed else ''}: {baseline_dir}")

    manifest_path = run_experiment(
        data_csv=data_csv, target=target,
        adapt_fraction=adapt_fraction, split_method=split_method, split_seed=split_seed,
        out_dir=baseline_dir,
    )

    print(f"Baseline: repairing (w_simp x w_simi grid: {len(w_simps)}x{len(w_simis)})")
    results_csv = baseline_dir / "results.csv"
    run_repair(
        normative_model_path=normative_model, dp=dp, data_csv=data_csv, target=target,
        adapt_fraction=adapt_fraction, split_method=split_method, split_seed=split_seed,
        manifest_path=manifest_path, out_csv=results_csv,
        w_simps=w_simps, w_simis=w_simis, max_depth=max_depth,
        fixed=fixed,
    )

    print(f"Baseline: Pareto analysis (per trial + aggregated by config) -> {baseline_dir / 'analysis'}")
    analysis_dir = run_trial_pareto_analysis(
        csv_path=results_csv, out_dir=baseline_dir / "analysis",
    )

    outputs = {
        "baseline": {
            "manifest": manifest_path, "results_csv": results_csv, "analysis_dir": analysis_dir,
        }
    }

    if save_pareto_trees and fixed:
        raise NotImplementedError(
            "--save-pareto-trees is not supported together with --fixed yet "
            "(manifest.jsonl entries aren't keyed by protected path)."
        )

    if save_pareto_trees:
        print(f"Baseline: regenerating + saving per-trial Pareto-optimal tree(s) -> {baseline_dir / 'pareto_trees.pkl'}")
        pareto_trees_pkl = save_pareto_trees_fn(
            normative_model_path=normative_model, dp=dp, manifest_path=manifest_path,
            pareto_csv=analysis_dir / "pareto_per_trial.csv",
            out_pkl=baseline_dir / "pareto_trees.pkl", max_depth=max_depth,
        )
        outputs["baseline"]["pareto_trees_pkl"] = pareto_trees_pkl

    print(f"Baseline repair complete. Output under: {out_dir.resolve()}")
    return outputs


run_pipeline = run_baseline_repair


def _parse_floats(csv_string):
    return [float(x) for x in csv_string.split(",")]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dp", required=True, help="Decision point name, e.g. 'n5'. Must be a key in --normative-model.")
    parser.add_argument("--data-csv", required=True, help="Table to split into D_adapt/D_test, unperturbed.")
    parser.add_argument("--normative-model", default="normative_model.pkl", help="Pickle produced by build_normative_model.py, containing T_old for every decision point.")
    parser.add_argument("--target", default="branch")
    parser.add_argument("--adapt-fraction", type=float, default=0.7)
    parser.add_argument(
        "--split-method", choices=["positional", "random", "case_chronological"], default="case_chronological",
        help="positional: first N rows by position. random: sklearn train_test_split, seeded by "
             "--split-seed. case_chronological: group by case_id, order CASES (not rows) by "
             "first-event timestamp, allocate whole cases to adapt/test -- matches grid_search_dp.py's "
             "split exactly.",
    )
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--max-depth", type=int, default=4)

    parser.add_argument("--w-simps", default="0.0,0.001,0.005,0.01,0.02,0.05,0.1,0.2,0.3,0.5,0.75,1.0,1.5,2.0,3.0,5.0", help="Comma-separated w_simp grid, applied to every trial. Normalized scale (complexity term / nodes_max).")
    parser.add_argument("--w-simis", default="0.0,0.001,0.005,0.01,0.02,0.05,0.1,0.2,0.3,0.5,0.75,1.0,1.5", help="Comma-separated w_simi grid, applied to every trial. Normalized scale (audit term / nodes_max).")

    parser.add_argument(
        "--save-pareto-trees", action=argparse.BooleanOptionalAction, default=False,
        help="Regenerate and pickle the actual repaired tree object for every PER-TRIAL "
             "Pareto-optimal (trial_id, w_simp, w_simi) combination (see "
             "trial_pareto_analysis.py's pareto_per_trial.csv), to <baseline_dir>/pareto_trees.pkl.",
    )
    parser.add_argument(
        "--fixed", action=argparse.BooleanOptionalAction, default=False,
        help="'Expert forces keep' experiment: same seed/split/grid as a normal run, but for "
             "EACH of T_old's root-to-leaf paths (one at a time), run the full repair sweep "
             "with that path's nodes forced to keep, never regrown. Output goes to a separate "
             "baseline_fixed/ folder (never baseline/); results.csv gains a fixed_leaf_id column "
             "and trial_id is suffixed with the protected leaf's id.",
    )
    parser.add_argument("--out-dir", required=True, help="Base output directory; a baseline/ (or baseline_fixed/ with --fixed) subfolder is created inside it.")
    args = parser.parse_args()

    run_baseline_repair(
        dp=args.dp, data_csv=args.data_csv, out_dir=args.out_dir,
        normative_model=args.normative_model, target=args.target,
        adapt_fraction=args.adapt_fraction, split_method=args.split_method, split_seed=args.split_seed,
        max_depth=args.max_depth,
        w_simps=_parse_floats(args.w_simps), w_simis=_parse_floats(args.w_simis),
        save_pareto_trees=args.save_pareto_trees,
        fixed=args.fixed,
    )


if __name__ == "__main__":
    main()
