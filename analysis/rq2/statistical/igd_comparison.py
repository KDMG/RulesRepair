import argparse
from pathlib import Path

import pandas as pd

from analysis.core.compare_rulesrepair_cart_dominance import DEFAULT_MAX_DEPTH, DEFAULT_NODES_MIN, DEFAULT_TOL, infer_dp_label
from analysis.rq2.statistical.igd_core import (
    build_seed_level_table, build_trial_level_table,
    build_seed_level_tables_by_mutation_type, build_trial_level_tables_by_mutation_type,
    decision_point_igd_test, decision_point_igd_test_trial_level,
    print_decision_point_result, build_summary_dataframe,
)


def main_igd_comparison(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "csv_paths", nargs="+"
    )
    parser.add_argument(
        "--max-depth", type=int, default=DEFAULT_MAX_DEPTH
    )
    parser.add_argument("--nodes-min", type=int, default=DEFAULT_NODES_MIN)
    parser.add_argument("--tol", type=float, default=DEFAULT_TOL)
    parser.add_argument(
        "--alpha", type=float, default=0.05
    )
    parser.add_argument(
        "--out-dir", type=str, default=None
    )
    parser.add_argument(
        "--by-mutation-type", action="store_true"
    )
    parser.add_argument(
        "--baseline-acc-col", type=str, default="acc_cart_test"
    )
    parser.add_argument("--baseline-nodes-col", type=str, default="cart_total_nodes")
    parser.add_argument("--baseline-jaccard-col", type=str, default="sim_old_cart_jaccard")
    parser.add_argument(
        "--baseline-label", type=str, default="CART"
    )
    parser.add_argument(
        "--trial-level", action="store_true"
    )
    args = parser.parse_args(argv)

    if args.out_dir is None:
        args.out_dir = str(Path("quantitative_evaluation") / "rq2" / "igd_comparison")

    by_dp = {}
    for p in args.csv_paths:
        by_dp.setdefault(infer_dp_label(p), []).append(p)

    dp_labels_sorted = sorted(by_dp)
    print(f"Loaded {len(args.csv_paths)} input CSV(s) across {len(dp_labels_sorted)} decision point(s).")
    if args.trial_level:
        print("--trial-level: supplementary analysis, trials pooled with no per-seed aggregation, "
              "not the primary RQ1/RQ2 result. See decision_point_igd_test_trial_level()'s docstring.")

    if args.trial_level:
        build_table_fn = build_trial_level_table
        build_by_mt_fn = build_trial_level_tables_by_mutation_type
        test_fn = decision_point_igd_test_trial_level
        print_fn = lambda dp_label, result: print_decision_point_result(dp_label, result, trial_level=True)
        table_csv_prefix = "igd_by_trial"
        summary_csv_name = "igd_summary_trial_level.csv"
        mt_summary_csv_name = "igd_summary_by_mutation_type_trial_level.csv"
    else:
        build_table_fn = build_seed_level_table
        build_by_mt_fn = build_seed_level_tables_by_mutation_type
        test_fn = decision_point_igd_test
        print_fn = print_decision_point_result
        table_csv_prefix = "igd_by_seed"
        summary_csv_name = "igd_summary.csv"
        mt_summary_csv_name = "igd_summary_by_mutation_type.csv"

    results = []
    mt_summary_rows = []
    for dp_label in dp_labels_sorted:
        unit_table = build_table_fn(
            by_dp[dp_label], max_depth=args.max_depth, nodes_min=args.nodes_min, tol=args.tol,
            baseline_acc_col=args.baseline_acc_col, baseline_nodes_col=args.baseline_nodes_col,
            baseline_jaccard_col=args.baseline_jaccard_col,
        )
        result = test_fn(unit_table, alpha=args.alpha, baseline_label=args.baseline_label)
        print()
        print_fn(dp_label, result)
        results.append(result)

        if args.out_dir:
            out_dir = Path(args.out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            unit_csv = out_dir / f"{table_csv_prefix}_{dp_label}.csv"
            unit_table.to_csv(unit_csv, index=False)
            print(f"  Wrote {unit_csv}")

        if args.by_mutation_type:
            mt_tables = build_by_mt_fn(
                by_dp[dp_label], max_depth=args.max_depth, nodes_min=args.nodes_min, tol=args.tol,
                baseline_acc_col=args.baseline_acc_col, baseline_nodes_col=args.baseline_nodes_col,
                baseline_jaccard_col=args.baseline_jaccard_col,
            )
            for mt, mt_unit_table in mt_tables.items():
                print()
                if mt_unit_table is None:
                    print(f"  [{mt}] not applicable (zero trials of this mutation type at this decision point)")
                    mt_summary_rows.append({
                        "decision_point": dp_label, "mutation_type": mt, "not_applicable": True,
                    })
                    continue
                mt_result = test_fn(mt_unit_table, alpha=args.alpha, baseline_label=args.baseline_label)
                print_fn(f"{dp_label} [{mt}]", mt_result)
                mt_row = {"decision_point": dp_label, "mutation_type": mt, "not_applicable": False}
                mt_row.update(mt_result)
                mt_summary_rows.append(mt_row)
                if args.out_dir:
                    mt_csv = Path(args.out_dir) / f"{table_csv_prefix}_{dp_label}_{mt}.csv"
                    mt_unit_table.to_csv(mt_csv, index=False)
                    print(f"    Wrote {mt_csv}")

    if args.out_dir:
        summary_df = build_summary_dataframe(dp_labels_sorted, results)
        summary_csv = Path(args.out_dir) / summary_csv_name
        summary_df.to_csv(summary_csv, index=False)
        print(f"\nWrote {summary_csv} ({len(summary_df)} decision point(s)).")

        if args.by_mutation_type:
            mt_summary_df = pd.DataFrame(mt_summary_rows)
            mt_summary_csv = Path(args.out_dir) / mt_summary_csv_name
            mt_summary_df.to_csv(mt_summary_csv, index=False)
            print(f"Wrote {mt_summary_csv} ({len(mt_summary_df)} row(s), "
                  f"{sum(1 for r in mt_summary_rows if not r['not_applicable'])} applicable).")
