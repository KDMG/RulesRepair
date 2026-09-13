import argparse
from pathlib import Path

import pandas as pd

from analysis.core.compare_kr_cart_dominance import _holm_bonferroni
from analysis.core.validation import ValidationReport
from analysis.rq2.statistical.rq_table import DATASET_DISPLAY, unified_results_to_latex_by_dataset


HOLM_REQUIRED_COLUMNS = ["dataset", "view", "baseline", "rq12_not_applicable", "wilcoxon_p", "p_holm"]


CAPTION = (
    "RQ1+RQ2 -- SUPPLEMENTARY, TRIAL-LEVEL, POOLED ACROSS DECISION POINTS "
    "(no per-seed aggregation, no per-decision-point split). Holm-Bonferroni "
    "correction applied ACROSS the three pairwise comparisons (CART, C4.5, "
    "REPTree, each vs RulesRepair) for the same (dataset, view) cell, results "
    "still reported as three separate rows."
)


def load_and_concat(paths, report):
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        missing = [c for c in HOLM_REQUIRED_COLUMNS if c not in df.columns]
        report.check(
            f"{Path(p).name} has all required columns (identified from the actual CSV header, not assumed)",
            len(missing) == 0,
            f"missing: {missing}" if missing else f"all {len(HOLM_REQUIRED_COLUMNS)} required columns present",
        )
        if missing:
            raise SystemExit(f"{p}: missing required column(s) {missing} -- cannot proceed.")
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)

    present_datasets = sorted(combined["dataset"].unique())
    missing_datasets = sorted(set(DATASET_DISPLAY) - set(present_datasets))
    report.check(
        "every canonical dataset (analysis.rq_results_latex.DATASET_DISPLAY) is present",
        len(missing_datasets) == 0,
        f"missing: {missing_datasets}" if missing_datasets else f"all {len(DATASET_DISPLAY)} datasets present",
    )
    extra = sorted(set(present_datasets) - set(DATASET_DISPLAY))
    if extra:
        report.warn(f"dataset(s) present but not in DATASET_DISPLAY (kept, just not a 'known' name): {extra}")
    return combined


def apply_cross_baseline_holm(df, alpha, report):
    df = df.copy()
    df["p_holm_per_baseline_family1"] = df["p_holm"]
    new_p_holm = df["p_holm"].astype(float).copy()

    n_groups = 0
    n_families_lt3 = 0
    for (dataset, view), idx in df.groupby(["dataset", "view"]).groups.items():
        n_groups += 1
        sub = df.loc[idx]
        testable_mask = sub["wilcoxon_p"].notna()
        n_baselines_present = len(sub)
        n_testable = int(testable_mask.sum())
        if n_baselines_present != 3:
            n_families_lt3 += 1
            report.warn(
                f"({dataset}, {view}): {n_baselines_present} baseline row(s) present, expected 3 "
                f"(CART, J48, REPTree) -- Holm family size for this cell is whatever is actually testable, "
                f"not silently padded to 3."
            )
        if n_testable == 0:
            continue
        raw_ps = sub.loc[testable_mask, "wilcoxon_p"].tolist()
        adjusted = _holm_bonferroni(raw_ps)
        new_p_holm.loc[sub.index[testable_mask]] = adjusted

    df["p_holm"] = new_p_holm

    report.check(
        "Holm correction applied per (dataset, view) group, across whichever baselines were actually testable",
        True,
        f"{n_groups} (dataset, view) group(s) processed, {n_families_lt3} with a family size != 3",
    )

    testable = df["wilcoxon_p"].notna()
    violation = (df.loc[testable, "p_holm"] + 1e-12 < df.loc[testable, "p_holm_per_baseline_family1"])
    report.check(
        "cross-baseline Holm p_holm is never smaller than the original family-of-1 p_holm for any row",
        not violation.any(),
        "" if not violation.any() else f"{int(violation.sum())} row(s) violate this",
    )
    return df


def main_holm_table(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "paths", nargs="+",
        help="rq_unified_trial_level_by_dataset_<dataset>.csv paths (one or more, typically the six "
             "datasets), as written by analysis/run_multi_baseline_rq_table.py --trial-level --by-dataset.",
    )
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance threshold for the '*' marker.")
    parser.add_argument(
        "--out-dir", default=None,
        help="Directory to write outputs into (created if missing). Default: "
             "quantitative_evaluation/rq2/holm_table/.",
    )
    args = parser.parse_args(argv)

    if args.out_dir is None:
        args.out_dir = str(Path("quantitative_evaluation") / "rq2" / "holm_table")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = ValidationReport()

    print(f"Loading {len(args.paths)} per-dataset CSV(s)...")
    df = load_and_concat(args.paths, report)
    print(f"Loaded {len(df)} row(s) across {df['dataset'].nunique()} dataset(s) and {df['baseline'].nunique()} baseline(s).")

    corrected = apply_cross_baseline_holm(df, args.alpha, report)

    ok = report.print_report()

    csv_out = out_dir / "rq_unified_trial_level_by_dataset_holm_corrected.csv"
    corrected.to_csv(csv_out, index=False)
    print(f"\nWrote {csv_out} ({len(corrected)} row(s)).")

    tex = unified_results_to_latex_by_dataset(corrected, alpha=args.alpha, caption=CAPTION)
    tex_out = out_dir / "rq_unified_trial_level_by_dataset_holm_corrected.tex"
    tex_out.write_text(tex)
    print(f"Wrote {tex_out}.")

    print("\ngenerated latex table\n")
    print(tex)

    if not ok:
        raise SystemExit(1)
