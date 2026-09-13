import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from repair.run_perturbation_repair import encode, extend_columns_for_regrow, NON_FEATURE_COLUMNS
from keep_remine_prune.tree import Condition, Operator
BASE_COLUMNS = ["num_feature", "cat_feature_x", "cat_feature_y"]
CAT_COLS = ["cat_feature"]


def _build_test_df():
    return pd.DataFrame({
        "case_id": [1, 2, 3, 4, 5],
        "branch": ["A", "A", "B", "B", "A"],
        "cat_feature": ["x", "y", "z", "w", "x"],
        "num_feature": [1.0, 2.0, 3.0, 4.0, 5.0],
    })


def test_extend_columns_for_regrow_preserves_order_and_appends_novel():
    df = _build_test_df()
    extended = extend_columns_for_regrow(df, CAT_COLS, BASE_COLUMNS)
    assert extended[:len(BASE_COLUMNS)] == BASE_COLUMNS
    assert extended[len(BASE_COLUMNS):] == ["cat_feature_w", "cat_feature_z"]
    print("test_extend_columns_for_regrow_preserves_order_and_appends_novel: PASS")
    print(f"  Verified: {extended}")


def test_extend_columns_for_regrow_handles_missing_cat_col():
    df = _build_test_df()
    extended = extend_columns_for_regrow(df, CAT_COLS + ["other_cat_not_in_df"], BASE_COLUMNS)
    assert extended[:len(BASE_COLUMNS)] == BASE_COLUMNS
    print("test_extend_columns_for_regrow_handles_missing_cat_col: PASS")


def test_extend_columns_for_regrow_no_novel_categories():
    df = _build_test_df()
    df["cat_feature"] = ["x", "y", "x", "y", "x"]
    extended = extend_columns_for_regrow(df, CAT_COLS, BASE_COLUMNS)
    assert extended == BASE_COLUMNS
    print("test_extend_columns_for_regrow_no_novel_categories: PASS")


def test_keep_side_unaffected_by_extension():
    df = _build_test_df()
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS and c != "branch"]

    extended_columns = extend_columns_for_regrow(df, CAT_COLS, BASE_COLUMNS)
    assert extended_columns != BASE_COLUMNS

    X_base = encode(df[feature_cols], CAT_COLS, BASE_COLUMNS)
    X_extended = encode(df[feature_cols], CAT_COLS, extended_columns)

    cond = Condition(attribute="cat_feature_x", attribute_pos=1, operator=Operator.LE, threshold=0.5)

    fires_base = [cond.fire(row) for row in X_base.to_numpy()]
    fires_extended = [cond.fire(row) for row in X_extended.to_numpy()]
    assert fires_base == fires_extended, (
        f"keep-side condition must fire identically regardless of the extension: "
        f"base={fires_base}, extended={fires_extended}"
    )
    print("test_keep_side_unaffected_by_extension: PASS")
    print(f"  Verified: condition fires identically on both encodings, {fires_base}")


if __name__ == "__main__":
    test_extend_columns_for_regrow_preserves_order_and_appends_novel()
    test_extend_columns_for_regrow_handles_missing_cat_col()
    test_extend_columns_for_regrow_no_novel_categories()
    test_keep_side_unaffected_by_extension()
    print("\nAll tests passed.")
