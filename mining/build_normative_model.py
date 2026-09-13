import argparse
import pickle
from pathlib import Path

import pandas as pd

from mining.cart_training import build_tree, print_tree
from mutations.selection import IDENTIFIER_COLUMNS

NON_FEATURE_COLUMNS = IDENTIFIER_COLUMNS | {"branch_index", "branch_label"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dp-dir", default="decision_points/sepsis/normative",
                         help="Directory of dp_<place>.csv files, e.g. produced by regenerate_decision_points.py")
    parser.add_argument("--initial-fraction", type=float, default=1,
                         help="Share of each table used as D_initial (guard mining), taken from the start, in order")
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--min-samples-leaf", type=float, default=0.02)
    parser.add_argument("--save", default=None, help="Save the built normative model to this file (pickle)")
    parser.add_argument("--quiet", action="store_true", help="Don't print each tree, just the summary table")
    args = parser.parse_args()

    dp_dir = Path(args.dp_dir)
    csv_files = sorted(dp_dir.glob("dp_*.csv"))
    if not csv_files:
        raise SystemExit(f"No dp_*.csv files found in {dp_dir}")

    model = {}
    summary_rows = []

    for path in csv_files:
        place_name = path.stem[len("dp_"):]
        df_full = pd.read_csv(path)

        if "branch" not in df_full.columns:
            raise SystemExit(
                f"ERROR: {path} has no 'branch' column (columns found: "
                f"{list(df_full.columns)}). Likely cause: the file was written by an earlier/"
                f"incomplete run of regenerate_decision_points.py, or is a different format. "
                f"Regenerate this file (or the whole --dp-dir folder) with "
                f"mining/regenerate_decision_points.py before rerunning."
            )

        if "timestamp" in df_full.columns:
            df_full["timestamp"] = pd.to_datetime(
                df_full["timestamp"],
                errors="coerce"
            )

            sort_columns = ["timestamp"]

            if "case_id" in df_full.columns:
                sort_columns.append("case_id")

            df_full = (
                df_full
                .sort_values(sort_columns, kind="stable")
                .reset_index(drop=True)
            )

        n_initial = round(
            len(df_full) * args.initial_fraction
        )

        df_initial = (
            df_full.iloc[:n_initial]
            .reset_index(drop=True)
        )

        raw_feature_cols = [
            c
            for c in df_initial.columns
            if c not in NON_FEATURE_COLUMNS and c != "branch"
        ]
        n_raw_features = len(raw_feature_cols)

        n_classes = df_initial["branch"].nunique() if "branch" in df_initial.columns and n_initial > 0 else 0
        positive_class = df_initial["branch"].value_counts().idxmin() if n_classes > 0 else None
        drop_cols = [c for c in NON_FEATURE_COLUMNS if c in df_initial.columns] + ["branch"]
        records = df_initial.drop(columns=drop_cols).to_dict("records") if n_initial > 0 else []
        for r, b in zip(records, df_initial["branch"] if n_initial > 0 else []):
            r["branch"] = b

        buildable = n_initial > 0
        tree = f1 = onehot_map = columns = None
        if buildable:
            tree, f1, onehot_map, columns = build_tree(
                records, max_depth=args.max_depth, min_samples_leaf=args.min_samples_leaf,
            )
            buildable = tree is not None  # only fails if there are literally zero feature columns

        if buildable:
            model[place_name] = {
                "tree": tree,
                "f1_train": f1,
                "onehot_map": onehot_map,
                "columns": columns,
                "n_initial": n_initial,
                "n_raw_features": n_raw_features,
                "n_encoded_features": len(columns),
                "n_classes": n_classes,
                "positive_class": positive_class,
                "max_depth": args.max_depth,
                "min_samples_leaf": args.min_samples_leaf,
            }
            trivial_flag = "  (single class -- trivial guard, not a useful drift-injection candidate)" if n_classes < 2 else ""
            summary_rows.append(
                (
                    place_name,
                    n_initial,
                    n_raw_features,
                    len(columns),
                    n_classes,
                    round(f1, 3),
                )
            )
            if not args.quiet:
                print(f"\n{place_name} (D_initial: {n_initial} rows, f1_train={f1:.3f}){trivial_flag}")
                print_tree(tree, onehot_map, depth=1)
        else:
            summary_rows.append(
                (
                    place_name,
                    n_initial,
                    n_raw_features,
                    0,
                    n_classes,
                    None,
                )
            )
            print(f"\n{place_name}: not buildable (n_initial={n_initial}, no feature columns available)")

    print("\nsummary")
    print(
        f"{'place':<8}"
        f"{'n_initial':<12}"
        f"{'n_raw':<10}"
        f"{'n_encoded':<12}"
        f"{'n_classes':<12}"
        f"{'f1_train':<10}"
    )

    for (
            place_name,
            n_initial,
            n_raw_features,
            n_encoded_features,
            n_classes,
            f1,
    ) in summary_rows:
        f1_str = f"{f1:.3f}" if f1 is not None else "--"

        print(
            f"{place_name:<8}"
            f"{n_initial:<12}"
            f"{n_raw_features:<10}"
            f"{n_encoded_features:<12}"
            f"{n_classes:<12}"
            f"{f1_str:<10}"
        )

    n_built = len(model)
    n_trivial = sum(1 for v in model.values() if v["n_classes"] < 2)
    print(f"\n{n_built}/{len(csv_files)} decision points have a normative model tree "
          f"({n_trivial} of those are single-class/trivial guards -- not useful drift-injection "
          f"candidates for Scenario 1/2, but still valid, correct parts of the normative model)")

    if args.save:
        with open(args.save, "wb") as f:
            pickle.dump(model, f)
        print(f"\nSaved normative model ({n_built} trees) to {args.save}")


if __name__ == "__main__":
    main()
