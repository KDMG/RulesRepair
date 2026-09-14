import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.core.compare_rulesrepair_cart_dominance import DEFAULT_TOL, DEFAULT_MAX_DEPTH, _dominates
from analysis.rq1._common import (
    infer_dataset, _expand_and_label_paths,
    OPERATOR_ORDER, OPERATOR_DISPLAY_NAMES, DATASET_DISPLAY_NAMES, _dataset_display,
    BASELINE_ALGORITHMS, collect_multi_algorithm_dominance,
)


N_MAX_NODES = 2 ** (DEFAULT_MAX_DEPTH + 1) - 1


assert N_MAX_NODES == 31, f"N_MAX_NODES expected to be 31 (max_depth={DEFAULT_MAX_DEPTH}), got {N_MAX_NODES}"


BOOTSTRAP_N_RESAMPLES = 10000


BOOTSTRAP_SEED = 12345


MIN_N_FOR_BOOTSTRAP = 10


RQ2_RAW_COLUMNS = [
    "dataset", "mine_algorithm", "trial_id", "seed", "decision_point", "mutation_type",
    "repair_solution_id",
    "mine_accuracy", "repair_accuracy", "delta_accuracy",
    "mine_similarity", "repair_similarity", "delta_similarity",
    "mine_total_nodes", "repair_total_nodes",
    "mine_simplicity", "repair_simplicity", "delta_simplicity",
    "source_csv",
]


def collect_dominated_solution_deltas(csv_paths, algorithms=BASELINE_ALGORITHMS, tol=DEFAULT_TOL,
                                       n_max_nodes=N_MAX_NODES):
    rows = []
    warned = set()
    for path, dataset, seed, decision_point in _expand_and_label_paths(csv_paths, quiet=True):
        df = pd.read_csv(path)
        base_required = [
            "trial_id", "is_pareto", "acc_new_test", "total_nodes", "sim_old_new_jaccard",
            "w_simp", "w_simi", "t",
        ]
        if any(c not in df.columns for c in base_required):
            continue
        pareto_df = df[df["is_pareto"]]
        for trial_id, group in pareto_df.groupby("trial_id"):
            group = group.reset_index(drop=True)
            if "operator_detail" in group.columns:
                operator = group["operator_detail"].iloc[0]
            elif "operator" in group.columns:
                operator = group["operator"].iloc[0]
            else:
                operator = None

            for algo in algorithms:
                needed = [algo["acc_col"], algo["nodes_col"], algo["jaccard_col"]]
                missing = [c for c in needed if c not in group.columns]
                if missing:
                    warn_key = (str(path), algo["key"])
                    if warn_key not in warned:
                        print(f"  Warning: {path}: missing column(s) {missing} for algorithm "
                              f"{algo['display']}, skipping.")
                        warned.add(warn_key)
                    continue
                a_acc = float(group[algo["acc_col"]].iloc[0])
                a_nodes = float(group[algo["nodes_col"]].iloc[0])
                a_jaccard = float(group[algo["jaccard_col"]].iloc[0])
                a_simplicity = 1.0 - a_nodes / n_max_nodes

                for _, r in group.iterrows():
                    r_acc = float(r["acc_new_test"])
                    r_nodes = float(r["total_nodes"])
                    r_jaccard = float(r["sim_old_new_jaccard"])
                    if not _dominates(a_acc, a_nodes, a_jaccard, r_acc, r_nodes, r_jaccard, tol):
                        continue
                    r_simplicity = 1.0 - r_nodes / n_max_nodes
                    repair_solution_id = f"w_simp={r['w_simp']}_w_simi={r['w_simi']}_t={r['t']}"
                    rows.append({
                        "dataset": dataset, "mine_algorithm": algo["display"],
                        "trial_id": trial_id, "seed": seed, "decision_point": decision_point,
                        "mutation_type": operator, "repair_solution_id": repair_solution_id,
                        "mine_accuracy": a_acc, "repair_accuracy": r_acc,
                        "delta_accuracy": a_acc - r_acc,
                        "mine_similarity": a_jaccard, "repair_similarity": r_jaccard,
                        "delta_similarity": a_jaccard - r_jaccard,
                        "mine_total_nodes": a_nodes, "repair_total_nodes": r_nodes,
                        "mine_simplicity": a_simplicity, "repair_simplicity": r_simplicity,
                        "delta_simplicity": a_simplicity - r_simplicity,
                        "source_csv": str(path),
                    })
    return pd.DataFrame(rows, columns=RQ2_RAW_COLUMNS)


def build_dominated_deltas_per_trial(raw_df):
    if raw_df.empty:
        return pd.DataFrame(columns=[
            "dataset", "mine_algorithm", "trial_id", "seed", "decision_point", "mutation_type",
            "n_dominated",
            "median_delta_accuracy", "max_delta_accuracy",
            "median_delta_similarity", "max_delta_similarity",
            "median_delta_simplicity", "max_delta_simplicity",
        ])
    group_cols = ["dataset", "mine_algorithm", "trial_id", "seed", "decision_point", "mutation_type"]
    rows = []
    for keys, group in raw_df.groupby(group_cols, dropna=False, sort=False):
        row = dict(zip(group_cols, keys))
        row["n_dominated"] = len(group)
        row["median_delta_accuracy"] = float(group["delta_accuracy"].median())
        row["max_delta_accuracy"] = float(group["delta_accuracy"].max())
        row["median_delta_similarity"] = float(group["delta_similarity"].median())
        row["max_delta_similarity"] = float(group["delta_similarity"].max())
        row["median_delta_simplicity"] = float(group["delta_simplicity"].median())
        row["max_delta_simplicity"] = float(group["delta_simplicity"].max())
        rows.append(row)
    return pd.DataFrame(rows)


def run_rq2_sanity_checks(raw_df, per_trial_df, multi_algo_df, tol=DEFAULT_TOL, n_max_nodes=N_MAX_NODES):
    ok = True

    def report(label, passed, detail=""):
        nonlocal ok
        status = "pass" if passed else "fail"
        if not passed:
            ok = False
        print(f"  [{status}] {label}{(': ' + detail) if detail else ''}")

    if raw_df.empty:
        print("  (raw_df is empty, no dominated solutions found in the given input, "
              "nothing further to check)")
        return True

    trial_keys = per_trial_df[["dataset", "mine_algorithm", "trial_id", "seed", "decision_point"]].drop_duplicates()
    merged = trial_keys.merge(
        multi_algo_df.rename(columns={"algorithm": "mine_algorithm"}),
        on=["dataset", "mine_algorithm", "trial_id", "seed", "decision_point"], how="left",
    )
    missing_in_rq1 = merged["d_i"].isna().sum()
    nonpositive_d_i = (merged["d_i"] <= 0).sum()
    report(
        "every included trial has D_i > 0 per RQ1's collect_multi_algorithm_dominance()",
        missing_in_rq1 == 0 and nonpositive_d_i == 0,
        f"{missing_in_rq1} trial(s) not found in RQ1 output, {nonpositive_d_i} with D_i <= 0 "
        f"(out of {len(trial_keys)} included trial x algorithm combinations)",
    )

    def _is_dominated_row(row):
        return _dominates(
            row["mine_accuracy"], row["mine_total_nodes"], row["mine_similarity"],
            row["repair_accuracy"], row["repair_total_nodes"], row["repair_similarity"],
            tol,
        )
    dominated_mask = raw_df.apply(_is_dominated_row, axis=1)
    report(
        "every raw_df row is strictly dominated by its mine_algorithm point (_dominates())",
        bool(dominated_mask.all()),
        f"{(~dominated_mask).sum()} / {len(raw_df)} row(s) failed re-verification",
    )

    delta_cols = ["delta_accuracy", "delta_similarity", "delta_simplicity"]
    below_tol = (raw_df[delta_cols] < -tol).any(axis=1)
    report(
        "all oriented deltas >= 0 (within tol)",
        bool((~below_tol).all()),
        f"{int(below_tol.sum())} / {len(raw_df)} row(s) have a delta < -{tol}",
    )

    all_near_zero = (raw_df[delta_cols].abs() <= tol).all(axis=1)
    report(
        "at least one delta is strictly > tol for every dominated pair",
        bool((~all_near_zero).all()),
        f"{int(all_near_zero.sum())} / {len(raw_df)} row(s) have all three deltas within tol of 0",
    )

    report(
        "no exact ties (all three objectives within tol) present in raw_df",
        bool((~all_near_zero).all()),
        f"{int(all_near_zero.sum())} exact-tie row(s) found (should be 0, identical to check above)",
    )

    cross = per_trial_df.merge(
        multi_algo_df.rename(columns={"algorithm": "mine_algorithm"}),
        on=["dataset", "mine_algorithm", "trial_id", "seed", "decision_point"],
        how="left", suffixes=("_rq2", "_rq1"),
    )
    mismatched_n = cross[cross["n_dominated_rq2"] != cross["n_dominated_rq1"]]
    report(
        "n_dominated (RQ2, this module) exactly matches n_dominated (RQ1, collect_multi_algorithm_dominance())",
        len(mismatched_n) == 0,
        f"{len(mismatched_n)} / {len(cross)} trial x algorithm combination(s) disagree",
    )

    dup_rows = raw_df.duplicated(
        subset=["dataset", "mine_algorithm", "trial_id", "seed", "decision_point", "repair_solution_id"]
    ).sum()
    source_per_trial = raw_df.groupby(
        ["dataset", "mine_algorithm", "trial_id", "seed", "decision_point"]
    )["source_csv"].nunique()
    multi_source_trials = int((source_per_trial > 1).sum())
    report(
        "no duplicate (dataset, algorithm, trial, repair_solution_id) rows, and every trial "
        "maps to exactly one source_csv",
        dup_rows == 0 and multi_source_trials == 0,
        f"{dup_rows} duplicate row(s), {multi_source_trials} trial(s) spanning >1 source file",
    )

    expected_mine_simplicity = 1.0 - raw_df["mine_total_nodes"] / n_max_nodes
    expected_repair_simplicity = 1.0 - raw_df["repair_total_nodes"] / n_max_nodes
    mismatch_mine = (raw_df["mine_simplicity"] - expected_mine_simplicity).abs() > tol
    mismatch_repair = (raw_df["repair_simplicity"] - expected_repair_simplicity).abs() > tol
    report(
        f"simplicity = 1 - total_nodes / {n_max_nodes} applied correctly to Mine and RulesRepair",
        bool((~mismatch_mine).all() and (~mismatch_repair).all()),
        f"{int(mismatch_mine.sum())} mine_simplicity mismatch(es), "
        f"{int(mismatch_repair.sum())} repair_simplicity mismatch(es) (out of {len(raw_df)} rows)",
    )

    expected_delta_simplicity = (raw_df["repair_total_nodes"] - raw_df["mine_total_nodes"]) / n_max_nodes
    mismatch_delta = (raw_df["delta_simplicity"] - expected_delta_simplicity).abs() > tol
    report(
        f"delta_simplicity == (repair_total_nodes - mine_total_nodes) / {n_max_nodes}",
        bool((~mismatch_delta).all()),
        f"{int(mismatch_delta.sum())} / {len(raw_df)} row(s) mismatch",
    )

    return ok


def print_rq2_diagnostics(per_trial_df, multi_algo_df):
    if multi_algo_df.empty:
        print("\n(no data)")
        return
    present_datasets = list(multi_algo_df["dataset"].dropna().unique())
    ordered_datasets = [d for d in DATASET_DISPLAY_NAMES if d in present_datasets]
    ordered_datasets += sorted(d for d in present_datasets if d not in DATASET_DISPLAY_NAMES)

    for dataset in ordered_datasets:
        for algo in BASELINE_ALGORITHMS:
            algo_display = algo["display"]
            base = multi_algo_df[(multi_algo_df["dataset"] == dataset) & (multi_algo_df["algorithm"] == algo_display)]
            if base.empty:
                continue
            n_total = len(base)
            n_dom_trials = int((base["d_i"] > 0).sum())
            pct_dom_trials = 100.0 * n_dom_trials / n_total if n_total else float("nan")

            pt = per_trial_df[(per_trial_df["dataset"] == dataset) & (per_trial_df["mine_algorithm"] == algo_display)]
            n_pairs = int(pt["n_dominated"].sum()) if len(pt) else 0

            print(f"\n{_dataset_display(dataset)} / {algo_display}")
            print(f"  total trials: {n_total}")
            print(f"  trials with dominance (D_i > 0): {n_dom_trials} ({pct_dom_trials:.2f}%)")
            print(f"  dominated solution pairs (sum of n_dominated across those trials): {n_pairs}")
            if len(pt) == 0:
                print("  (no dominated trials, nothing further to report for this cell)")
                continue
            for label, col in (
                ("accuracy", "median_delta_accuracy"),
                ("similarity", "median_delta_similarity"),
                ("simplicity", "median_delta_simplicity"),
            ):
                vals = pt[col]
                pp_note = f"  ({vals.median() * 100:.2f} pp)" if label == "accuracy" else ""
                print(f"  per-trial median Delta_{label}: median={vals.median():.4f}, "
                      f"range=[{vals.min():.4f}, {vals.max():.4f}] (n={len(vals)} trials){pp_note}")


def bootstrap_median_ci(values, n_resamples=BOOTSTRAP_N_RESAMPLES, seed=BOOTSTRAP_SEED, ci=0.95):
    vals = pd.Series(values).dropna().to_numpy()
    n = len(vals)
    if n < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    boot_medians = np.median(vals[idx], axis=1)
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(boot_medians, [alpha, 1.0 - alpha])
    return float(lo), float(hi)


def build_rq2_group_summary(per_trial_df, n_resamples=BOOTSTRAP_N_RESAMPLES, seed=BOOTSTRAP_SEED):
    rows = []
    for (dataset, algorithm), group in per_trial_df.groupby(["dataset", "mine_algorithm"], sort=False):
        row = {"dataset": dataset, "mine_algorithm": algorithm, "n_trials_with_dominance": len(group)}
        for label, col in (
            ("accuracy", "median_delta_accuracy"),
            ("similarity", "median_delta_similarity"),
            ("simplicity", "median_delta_simplicity"),
        ):
            vals = group[col]
            q1, med, q3 = vals.quantile([0.25, 0.5, 0.75])
            ci_lo, ci_hi = bootstrap_median_ci(vals, n_resamples=n_resamples, seed=seed)
            row[f"median_of_median_delta_{label}"] = float(med)
            row[f"q1_of_median_delta_{label}"] = float(q1)
            row[f"q3_of_median_delta_{label}"] = float(q3)
            row[f"boot_ci95_lo_delta_{label}"] = ci_lo
            row[f"boot_ci95_hi_delta_{label}"] = ci_hi
        rows.append(row)
    return pd.DataFrame(rows)


def print_rq2_group_summary(group_summary_df):
    if group_summary_df.empty:
        print("\n(no data)")
        return
    present_datasets = list(group_summary_df["dataset"].dropna().unique())
    ordered_datasets = [d for d in DATASET_DISPLAY_NAMES if d in present_datasets]
    ordered_datasets += sorted(d for d in present_datasets if d not in DATASET_DISPLAY_NAMES)

    for dataset in ordered_datasets:
        d_rows = group_summary_df[group_summary_df["dataset"] == dataset]
        for algo in BASELINE_ALGORITHMS:
            algo_display = algo["display"]
            algo_rows = d_rows[d_rows["mine_algorithm"] == algo_display]
            if algo_rows.empty:
                continue
            r = algo_rows.iloc[0]
            print(f"\n{_dataset_display(dataset)} / {algo_display} (n = {int(r['n_trials_with_dominance'])} trials with dominance)")
            for label in ("accuracy", "similarity", "simplicity"):
                med = r[f"median_of_median_delta_{label}"]
                q1 = r[f"q1_of_median_delta_{label}"]
                q3 = r[f"q3_of_median_delta_{label}"]
                ci_lo = r[f"boot_ci95_lo_delta_{label}"]
                ci_hi = r[f"boot_ci95_hi_delta_{label}"]
                pp_note = f"  ({med * 100:.2f} pp)" if label == "accuracy" else ""
                print(f"  {label.capitalize():<11}median [Q1, Q3]        = {med:.4f} [{q1:.4f}, {q3:.4f}]{pp_note}")
                print(f"  {'':<11}median [95% bootstrap CI] = {med:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]")


def build_rq2_mutation_summary(per_trial_df, n_resamples=BOOTSTRAP_N_RESAMPLES, seed=BOOTSTRAP_SEED,
                                min_n_for_bootstrap=MIN_N_FOR_BOOTSTRAP):
    rows = []
    for (dataset, algorithm, mutation_type), group in per_trial_df.groupby(
        ["dataset", "mine_algorithm", "mutation_type"], sort=False, dropna=False
    ):
        n = len(group)
        flagged = n < min_n_for_bootstrap
        row = {
            "dataset": dataset, "mine_algorithm": algorithm, "mutation_type": mutation_type,
            "n_trials_with_dominance": n, "bootstrap_flagged_low_n": flagged,
        }
        for label, col in (
            ("accuracy", "median_delta_accuracy"),
            ("similarity", "median_delta_similarity"),
            ("simplicity", "median_delta_simplicity"),
        ):
            vals = group[col]
            q1, med, q3 = vals.quantile([0.25, 0.5, 0.75])
            row[f"median_of_median_delta_{label}"] = float(med)
            row[f"q1_of_median_delta_{label}"] = float(q1)
            row[f"q3_of_median_delta_{label}"] = float(q3)
            if flagged:
                row[f"boot_ci95_lo_delta_{label}"] = float("nan")
                row[f"boot_ci95_hi_delta_{label}"] = float("nan")
            else:
                ci_lo, ci_hi = bootstrap_median_ci(vals, n_resamples=n_resamples, seed=seed)
                row[f"boot_ci95_lo_delta_{label}"] = ci_lo
                row[f"boot_ci95_hi_delta_{label}"] = ci_hi
        rows.append(row)
    return pd.DataFrame(rows)


def print_rq2_mutation_summary(mutation_summary_df, min_n_for_bootstrap=MIN_N_FOR_BOOTSTRAP):
    if mutation_summary_df.empty:
        print("\n(no data)")
        return
    present_datasets = list(mutation_summary_df["dataset"].dropna().unique())
    ordered_datasets = [d for d in DATASET_DISPLAY_NAMES if d in present_datasets]
    ordered_datasets += sorted(d for d in present_datasets if d not in DATASET_DISPLAY_NAMES)

    for dataset in ordered_datasets:
        d_rows = mutation_summary_df[mutation_summary_df["dataset"] == dataset]
        for algo in BASELINE_ALGORITHMS:
            algo_display = algo["display"]
            a_rows = d_rows[d_rows["mine_algorithm"] == algo_display]
            if a_rows.empty:
                continue
            present_ops = set(a_rows["mutation_type"].dropna().unique())
            ordered_ops = [op for op in OPERATOR_ORDER if op in present_ops]
            ordered_ops += sorted(op for op in present_ops if op not in OPERATOR_ORDER)
            print(f"\n{_dataset_display(dataset)} / {algo_display}, by mutation type")
            for op in ordered_ops:
                r_rows = a_rows[a_rows["mutation_type"] == op]
                if r_rows.empty:
                    continue
                r = r_rows.iloc[0]
                op_label = OPERATOR_DISPLAY_NAMES.get(op, str(op).replace("_", " ").title())
                n = int(r["n_trials_with_dominance"])
                print(f"  {op_label} (n = {n} trials with dominance)")
                for label in ("accuracy", "similarity", "simplicity"):
                    med = r[f"median_of_median_delta_{label}"]
                    q1 = r[f"q1_of_median_delta_{label}"]
                    q3 = r[f"q3_of_median_delta_{label}"]
                    pp_note = f"  ({med * 100:.2f} pp)" if label == "accuracy" else ""
                    print(f"    {label.capitalize():<11}median [Q1, Q3] = {med:.4f} [{q1:.4f}, {q3:.4f}]{pp_note}")
                    if r["bootstrap_flagged_low_n"]:
                        print(f"    {'':<11}bootstrap CI: flagged, n={n} < min_n_for_bootstrap={min_n_for_bootstrap}, "
                              f"not computed/interpreted")
                    else:
                        ci_lo = r[f"boot_ci95_lo_delta_{label}"]
                        ci_hi = r[f"boot_ci95_hi_delta_{label}"]
                        print(f"    {'':<11}median [95% bootstrap CI] = {med:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]")


def print_rq2_distributions(raw_df, per_trial_df):
    if raw_df.empty:
        return
    present_datasets = list(raw_df["dataset"].dropna().unique())
    ordered_datasets = [d for d in DATASET_DISPLAY_NAMES if d in present_datasets]
    ordered_datasets += sorted(d for d in present_datasets if d not in DATASET_DISPLAY_NAMES)

    def _describe(vals):
        vals = pd.Series(vals).dropna()
        if len(vals) == 0:
            return "n=0"
        q1, med, q3 = vals.quantile([0.25, 0.5, 0.75])
        return (f"n={len(vals)}  min={vals.min():.4f}  q1={q1:.4f}  median={med:.4f}  "
                f"q3={q3:.4f}  max={vals.max():.4f}  mean={vals.mean():.4f}")

    for dataset in ordered_datasets:
        for algo in BASELINE_ALGORITHMS:
            algo_display = algo["display"]
            raw_slice = raw_df[(raw_df["dataset"] == dataset) & (raw_df["mine_algorithm"] == algo_display)]
            if raw_slice.empty:
                continue
            pt_slice = per_trial_df[(per_trial_df["dataset"] == dataset) & (per_trial_df["mine_algorithm"] == algo_display)]
            print(f"\n{_dataset_display(dataset)} / {algo_display}: distributions")
            for label, raw_col, pt_col in (
                ("accuracy", "delta_accuracy", "median_delta_accuracy"),
                ("similarity", "delta_similarity", "median_delta_similarity"),
                ("simplicity", "delta_simplicity", "median_delta_simplicity"),
            ):
                print(f"  Delta_{label}, raw (every dominated solution, {len(raw_slice)} row(s), "
                      f"pseudoreplicated, descriptive only): {_describe(raw_slice[raw_col])}")
                print(f"  Delta_{label}, per-trial median ({len(pt_slice)} trial(s), the correct unit): "
                      f"{_describe(pt_slice[pt_col])}")


def main_dominance_advantage(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv_paths", nargs="+",
                         help="pareto_per_trial.csv files (glob-expanded by your shell), any mix of datasets")
    parser.add_argument("--tol", type=float, default=DEFAULT_TOL)
    parser.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH,
                         help="Used only to derive N_MAX_NODES = 2**(max_depth+1)-1 for the simplicity "
                              "transform (1 - total_nodes/N_MAX_NODES) -- does NOT affect dominance itself.")
    parser.add_argument("--n-resamples", type=int, default=BOOTSTRAP_N_RESAMPLES,
                         help="Number of percentile-bootstrap replications for the 95%% CI of the median "
                              "(resampling unit: trial).")
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED,
                         help="Fixed random seed for the bootstrap, for reproducibility.")
    parser.add_argument("--raw-out", default=None, help="Save the raw per-dominated-solution CSV to this path")
    parser.add_argument("--per-trial-out", default=None, help="Save the per-trial aggregated CSV to this path")
    parser.add_argument("--group-summary-out", default=None,
                         help="Save the per (dataset, Mine algorithm) summary (median/Q1/Q3 and 95%% bootstrap "
                              "CI of the per-trial median deltas) to this CSV.")
    parser.add_argument("--mutation-summary-out", default=None,
                         help="Save the same descriptive statistics broken down further by mutation type "
                              "(dataset x Mine algorithm x mutation type) to this CSV. Exploratory only -- "
                              "not proposed for the main paper table yet.")
    parser.add_argument("--min-n-bootstrap", type=int, default=MIN_N_FOR_BOOTSTRAP,
                         help="Below this many trials-with-dominance, a subgroup's bootstrap CI is not "
                              "computed and is flagged instead (median/Q1/Q3 are still reported regardless).")
    args = parser.parse_args(argv)

    _default_dir = Path("quantitative_evaluation") / "rq1" / "dominance_advantage"
    _default_dir.mkdir(parents=True, exist_ok=True)
    _dataset = infer_dataset(args.csv_paths[0])
    if args.raw_out is None:
        args.raw_out = str(_default_dir / "raw_dominated_solutions.csv")
    if args.per_trial_out is None:
        args.per_trial_out = str(_default_dir / "per_trial.csv")
    if args.group_summary_out is None:
        args.group_summary_out = str(_default_dir / f"rq2_group_summary_{_dataset}.csv")
    if args.mutation_summary_out is None:
        args.mutation_summary_out = str(_default_dir / f"rq2_mutation_summary_{_dataset}.csv")

    n_max_nodes = 2 ** (args.max_depth + 1) - 1

    print(f"Reading {len(args.csv_paths)} file(s)...")
    raw_df = collect_dominated_solution_deltas(args.csv_paths, tol=args.tol, n_max_nodes=n_max_nodes)
    per_trial_df = build_dominated_deltas_per_trial(raw_df)
    multi_algo_df = collect_multi_algorithm_dominance(args.csv_paths, tol=args.tol)

    print(f"\n{len(raw_df)} dominated (trial, algorithm, solution) row(s), "
          f"{len(per_trial_df)} trial x algorithm combination(s) with >=1 dominated solution.")
    print(f"Simplicity = 1 - total_nodes / {n_max_nodes} (max_depth={args.max_depth}).")

    print("\nsanity checks")
    ok = run_rq2_sanity_checks(raw_df, per_trial_df, multi_algo_df, tol=args.tol, n_max_nodes=n_max_nodes)
    print(f"\noverall: {'all checks passed' if ok else 'some checks failed, see above'}")

    print("\ndiagnostic summary (per dataset x mine algorithm, not aggregated across datasets)")
    print_rq2_diagnostics(per_trial_df, multi_algo_df)

    print("\nper (dataset x mine algorithm) summary of per-trial median deltas")
    print(f"(percentile bootstrap, n_resamples={args.n_resamples}, seed={args.bootstrap_seed})")
    group_summary_df = build_rq2_group_summary(per_trial_df, n_resamples=args.n_resamples, seed=args.bootstrap_seed)
    print_rq2_group_summary(group_summary_df)

    print("\nDISTRIBUTIONS")
    print_rq2_distributions(raw_df, per_trial_df)

    if args.group_summary_out:
        group_summary_df.to_csv(args.group_summary_out, index=False)
        print(f"\nSaved per (dataset, Mine algorithm) summary to {args.group_summary_out}")

    print("\nper (dataset x mine algorithm x mutation type) summary (exploratory, not for the main "
          "paper table yet)")
    mutation_summary_df = build_rq2_mutation_summary(
        per_trial_df, n_resamples=args.n_resamples, seed=args.bootstrap_seed,
        min_n_for_bootstrap=args.min_n_bootstrap,
    )
    print_rq2_mutation_summary(mutation_summary_df, min_n_for_bootstrap=args.min_n_bootstrap)
    if args.mutation_summary_out:
        mutation_summary_df.to_csv(args.mutation_summary_out, index=False)
        print(f"\nSaved per (dataset, Mine algorithm, mutation type) summary to {args.mutation_summary_out}")

    if args.raw_out:
        raw_df.to_csv(args.raw_out, index=False)
        print(f"\nSaved raw per-dominated-solution CSV to {args.raw_out}")
    if args.per_trial_out:
        per_trial_df.to_csv(args.per_trial_out, index=False)
        print(f"Saved per-trial aggregated CSV to {args.per_trial_out}")
