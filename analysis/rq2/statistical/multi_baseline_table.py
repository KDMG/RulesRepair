import argparse
import tempfile
from pathlib import Path

import pandas as pd

from analysis.core.compare_rulesrepair_cart_dominance import DEFAULT_MAX_DEPTH, DEFAULT_NODES_MIN, DEFAULT_TOL
from analysis.rq2.statistical.rq_table import (
    build_unified_results_table, build_unified_results_table_trial_level,
    build_unified_results_table_trial_level_by_dataset,
    unified_results_to_latex, unified_results_to_latex_by_dataset,
)


def _filter_out_w_simp_values(paths, excluded_w_simps, tmp_dir):
    import re

    filtered_paths = []
    for p in paths:
        orig = Path(p)
        seed_match = re.search(r"seed_(\d+)", str(orig))
        if seed_match is None:
            raise ValueError(f"{p}: no 'seed_<N>' segment found -- cannot mirror path for filtering.")
        if not (orig.parent.name == "analysis" and orig.parent.parent.name in ("mutated", "baseline")):
            raise ValueError(
                f"{p}: expected .../<dp>/{{mutated,baseline}}/analysis/{orig.name} shape -- got {orig}."
            )
        dp_name = orig.parent.parent.parent.name
        scenario = orig.parent.parent.name

        df = pd.read_csv(p)
        mask = pd.Series(True, index=df.index)
        for excl in excluded_w_simps:
            mask &= (df["w_simp"] - excl).abs() > DEFAULT_TOL
        df = df[mask]

        out_path = Path(tmp_dir) / f"seed_{seed_match.group(1)}" / dp_name / scenario / "analysis" / orig.name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        filtered_paths.append(str(out_path))
    return filtered_paths


DEFAULT_MULTI_BASELINES = [
    {"label": "CART", "acc_col": "acc_cart_test", "nodes_col": "cart_total_nodes",
     "jaccard_col": "sim_old_cart_jaccard"},
    {"label": "J48", "acc_col": "acc_j48_test", "nodes_col": "j48_total_nodes",
     "jaccard_col": "sim_old_j48_jaccard"},
    {"label": "REPTree", "acc_col": "acc_reptree_test", "nodes_col": "reptree_total_nodes",
     "jaccard_col": "sim_old_reptree_jaccard"},
]


def main_multi_baseline_table(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "paths", nargs="+",
        help="pareto_per_trial.csv paths for ONE dataset (glob-expanded by your shell, one or more "
             "decision points x seeds -- same seed_<N> contract as the rest of this pipeline).",
    )
    parser.add_argument(
        "--dataset", required=True,
        help="Dataset label (e.g. sepsis, hospital_billing, road_traffic) -- used as the 'dataset' "
             "column value and in the output filenames.",
    )
    parser.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH)
    parser.add_argument("--nodes-min", type=int, default=DEFAULT_NODES_MIN)
    parser.add_argument("--tol", type=float, default=DEFAULT_TOL)
    parser.add_argument(
        "--alpha", type=float, default=0.05,
        help="Significance threshold for the RQ1/RQ2 'winner' column (applied to the Holm-adjusted "
             "p-value, same as igd_statistical_comparison.py).",
    )
    parser.add_argument(
        "--sd-zero-tol", type=float, default=DEFAULT_TOL,
        help="Threshold below which the paired differences' standard deviation is treated as "
             "numerically zero for Cohen's d_z's cohens_dz_status classification (default: DEFAULT_TOL, "
             "this codebase's existing 1e-9 'numerically zero' constant).",
    )
    parser.add_argument(
        "--by-mutation-type", action="store_true", default=True,
        help="Also compute one view per canonical mutation type, in addition to overall (default: on).",
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="Directory to write outputs into (created if missing). Default: "
             "quantitative_evaluation/rq2/multi_baseline_table/.",
    )
    parser.add_argument(
        "--trial-level", action="store_true",
        help="2026 addition, explicit user request. SUPPLEMENTARY ONLY -- pools every trial from every "
             "seed with NO per-seed aggregation, instead of this pipeline's primary per-seed-median "
             "methodology (analysis.igd_statistical_comparison.decision_point_igd_test_trial_level()). "
             "Writes rq_unified_trial_level_<dataset>.csv/.tex instead of rq_unified_<dataset>.csv/.tex "
             "so it never collides with the primary output. CAVEAT: trials within a seed are not "
             "independent (same T_old, same adapt/test split, same mutation-sampling random stream) -- "
             "the seed-level table (no --trial-level) remains the PRIMARY, statistically valid result.",
    )
    parser.add_argument(
        "--by-dataset", action="store_true",
        help="2026 addition, explicit user request (\"I'd like a single table with one row per "
             "dataset\"). SUPPLEMENTARY ONLY, requires --trial-level -- pools every decision point of "
             "--dataset into ONE row per baseline (not one row per decision point): every trial from every "
             "decision point x every seed is pooled into a single Wilcoxon/Hodges-Lehmann test per "
             "(view, baseline). Writes rq_unified_trial_level_by_dataset_<dataset>.csv/.tex instead of the "
             "usual rq_unified_trial_level_<dataset>.csv/.tex. CAVEAT (on top of --trial-level's own "
             "seed-independence caveat): different decision points are different repair problems (their "
             "own T_old, their own feature space) -- pooling their trials together treats them as "
             "exchangeable observations of the same effect, a real methodological simplification, not an "
             "oversight (see build_unified_results_table_trial_level_by_dataset()'s own docstring). The "
             "per-decision-point trial-level table (--trial-level alone) and the primary per-decision-point "
             "seed-level table (neither flag) remain available/authoritative.",
    )
    parser.add_argument(
        "--exclude-w-simp-grid-values", default=None,
        help="2026 addition, explicit user request. SUPPLEMENTARY ONLY -- comma-separated w_simp "
             "value(s) (e.g. '5.0' or '3.0,5.0') to drop from the input pareto_per_trial.csv rows "
             "before recomputing the Pareto front / IGD+ (see _filter_out_w_simp_values() docstring "
             "above). Writes rq_unified[_trial_level]_excl_w_simp<V>_<dataset>.csv/.tex -- an ADDITIONAL "
             "table alongside the primary one, never replacing it. The primary table (no exclusion, "
             "full w_simp/w_simi grid) stays authoritative -- explicit user decision not to cut the "
             "Pareto front in the primary RQ1/RQ2 result, since the repair problem is genuinely "
             "3-objective and removing grid points would misrepresent the trade-off it reports.",
    )
    args = parser.parse_args(argv)

    if args.by_dataset and not args.trial_level:
        raise SystemExit(
            "--by-dataset requires --trial-level (there is no seed-level, pooled-across-decision-points "
            "table -- see --by-dataset's own help text)."
        )

    if args.out_dir is None:
        args.out_dir = str(Path("quantitative_evaluation") / "rq2" / "multi_baseline_table")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    excluded_w_simps = None
    excl_tag = ""
    if args.exclude_w_simp_grid_values:
        excluded_w_simps = [float(v.strip()) for v in args.exclude_w_simp_grid_values.split(",")]
        excl_tag = "_excl_w_simp" + "_".join(f"{v:g}" for v in excluded_w_simps)

    baseline_names = ", ".join(b["label"] for b in DEFAULT_MULTI_BASELINES)
    print(f"Loaded {len(args.paths)} input CSV(s) for dataset={args.dataset}.")
    print(
        f"Computing against {len(DEFAULT_MULTI_BASELINES)} baselines: {baseline_names}."
        + ("  (Supplementary, TRIAL-LEVEL -- no per-seed aggregation)" if args.trial_level else "")
        + (f"  (Supplementary, EXCLUDING w_simp={excluded_w_simps} from the Pareto front)" if excluded_w_simps else "")
    )

    input_paths = args.paths
    tmp_dir_ctx = None
    if excluded_w_simps:
        tmp_dir_ctx = tempfile.TemporaryDirectory()
        input_paths = _filter_out_w_simp_values(args.paths, excluded_w_simps, tmp_dir_ctx.name)

    if args.by_dataset:
        build_fn = build_unified_results_table_trial_level_by_dataset
        csv_basename = "rq_unified_trial_level_by_dataset" + excl_tag
    else:
        build_fn = build_unified_results_table_trial_level if args.trial_level else build_unified_results_table
        csv_basename = ("rq_unified_trial_level" if args.trial_level else "rq_unified") + excl_tag
    caption_parts = []
    if args.trial_level:
        caption_parts.append(
            "SUPPLEMENTARY, TRIAL-LEVEL (no per-seed aggregation) -- trials within a seed are not "
            "independent (same T\\_old, same adapt/test split, same mutation-sampling random stream)."
        )
    if args.by_dataset:
        caption_parts.append(
            "POOLED ACROSS DECISION POINTS -- one row per baseline for the WHOLE dataset, every decision "
            "point's trials pooled into a single test per (view, baseline). Different decision points are "
            "different repair problems (their own T_old, their own feature space); pooling treats their "
            "trials as exchangeable observations of the same effect."
        )
    if excluded_w_simps:
        caption_parts.append(
            f"SUPPLEMENTARY -- Pareto front/IGD+ recomputed EXCLUDING w_simp={excluded_w_simps} from the "
            "grid. NOT the primary result: the primary table (full w_simp/w_simi grid) is authoritative -- "
            "the repair problem is genuinely 3-objective (accuracy, simplicity, similarity), so removing "
            "grid points would misrepresent the trade-off it reports. Shown here only to quantify how "
            "much the excluded value(s) affect the comparison."
        )
    caption = "RQ1+RQ2" + ((" -- " + " ".join(caption_parts)) if caption_parts else "")
    if caption_parts:
        caption += " See the full-grid table for the validated RQ1/RQ2 result."

    df = build_fn(
        {args.dataset: input_paths}, by_mutation_type=args.by_mutation_type,
        max_depth=args.max_depth, nodes_min=args.nodes_min, tol=args.tol, alpha=args.alpha,
        sd_zero_tol=args.sd_zero_tol, baselines=DEFAULT_MULTI_BASELINES,
    )
    csv_out = out_dir / f"{csv_basename}_{args.dataset}.csv"
    df.to_csv(csv_out, index=False)
    print(f"Wrote {csv_out} ({len(df)} row(s)).")

    latex_fn = unified_results_to_latex_by_dataset if args.by_dataset else unified_results_to_latex
    tex = latex_fn(df, caption=caption)
    tex_out = out_dir / f"{csv_basename}_{args.dataset}.tex"
    tex_out.write_text(tex)
    print(f"Wrote {tex_out}.")

    if tmp_dir_ctx is not None:
        tmp_dir_ctx.cleanup()
