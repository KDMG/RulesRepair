import argparse
import re
from pathlib import Path

import pandas as pd

from analysis.core.validation import ValidationReport
from analysis.rq1._common import DATASET_DISPLAY_NAMES, BASELINE_ALGORITHMS


DATASET_COLUMN_ORDER = [
    "sepsis", "production", "hospital_billing", "road_traffic",
    "prepaid_travel_costs", "international_declarations",
]


METRICS = [
    ("accuracy", "ΔAcc", r"$\Delta_{\mathrm{Acc}}$", 1.0, 2),
    ("similarity", "ΔSim", r"$\Delta_{\mathrm{Sim}}$", 1.0, 2),
    ("simplicity", "ΔSimp", r"$\Delta_{\mathrm{Simp}}$", 1.0, 2),
]


REQUIRED_GROUP_COLUMNS = ["dataset", "mine_algorithm"] + [
    f"{stat}_of_median_delta_{key}"
    for key, *_ in METRICS
    for stat in ("median", "q1", "q3")
]


_DATASET_FROM_FILENAME_RE = re.compile(r"rq2_group_summary_(.+)\.csv$")


def _validate_columns(df, required_columns, path, report, kind):
    missing = [c for c in required_columns if c not in df.columns]
    report.check(
        f"{kind} file {path.name} has all required columns (identified from the actual CSV header, "
        f"not assumed)",
        len(missing) == 0,
        f"missing: {missing}" if missing else f"all {len(required_columns)} required columns present",
    )
    if missing:
        raise SystemExit(f"{path}: missing required column(s) {missing} -- cannot proceed "
                          f"(refusing to guess column names).")


def load_group_summaries(rq2_dir, report):
    paths = sorted(Path(rq2_dir).glob("rq2_group_summary_*.csv"))
    frames = []
    for path in paths:
        df = pd.read_csv(path)
        _validate_columns(df, REQUIRED_GROUP_COLUMNS, path, report, "group_summary")
        present = df["dataset"].dropna().unique()
        m = _DATASET_FROM_FILENAME_RE.search(path.name)
        filename_dataset = m.group(1) if m else None
        if len(present) != 1:
            report.warn(f"{path.name}: expected exactly one distinct 'dataset' value, found {list(present)}")
        elif filename_dataset is not None and present[0] != filename_dataset:
            report.warn(f"{path.name}: filename suggests dataset='{filename_dataset}' but CSV content says "
                        f"'{present[0]}' -- using the CSV's own content, not the filename.")
        frames.append(df)
    if not frames:
        raise SystemExit(f"No rq2_group_summary_*.csv files found under {rq2_dir}")
    combined = pd.concat(frames, ignore_index=True)

    dup_mask = combined.duplicated(subset=["dataset", "mine_algorithm"], keep=False)
    report.check(
        "no duplicate (dataset, mine_algorithm) rows across the loaded group_summary files",
        not dup_mask.any(),
        "" if not dup_mask.any() else
        f"duplicate rows for: {sorted(combined.loc[dup_mask, ['dataset', 'mine_algorithm']].drop_duplicates().itertuples(index=False, name=None))}",
    )
    if dup_mask.any():
        raise SystemExit("Duplicate (dataset, mine_algorithm) rows found in the group_summary files -- "
                          "cannot pick a single value unambiguously; fix the input files.")
    return combined


def _format_cell(row, metric_key, scale, decimals):
    med = row[f"median_of_median_delta_{metric_key}"] * scale
    q1 = row[f"q1_of_median_delta_{metric_key}"] * scale
    q3 = row[f"q3_of_median_delta_{metric_key}"] * scale
    if pd.isna(med) or pd.isna(q1) or pd.isna(q3):
        return None
    return f"{med:.{decimals}f} [{q1:.{decimals}f}, {q3:.{decimals}f}]", med, q1, q3


def build_table(group_df, report):
    present_datasets = set(group_df["dataset"].dropna().unique())
    missing_datasets = [d for d in DATASET_COLUMN_ORDER if d not in present_datasets]
    report.check(
        "every expected dataset is present in the group_summary files",
        len(missing_datasets) == 0,
        f"missing: {missing_datasets}" if missing_datasets else f"all {len(DATASET_COLUMN_ORDER)} datasets present",
    )
    extra_datasets = sorted(present_datasets - set(DATASET_COLUMN_ORDER))
    if extra_datasets:
        report.warn(f"dataset(s) present in the input CSVs but not in DATASET_COLUMN_ORDER (ignored by "
                    f"this table): {extra_datasets}")

    expected_algos = [a["display"] for a in BASELINE_ALGORITHMS]
    n_cells_total = 0
    n_cells_missing = 0

    table = {}
    for algo in expected_algos:
        table[algo] = {}
        present_algo_datasets = set(group_df.loc[group_df["mine_algorithm"] == algo, "dataset"].unique())
        missing_for_algo = [d for d in DATASET_COLUMN_ORDER if d not in present_algo_datasets]
        report.check(
            f"algorithm '{algo}' has a group_summary row for every expected dataset",
            len(missing_for_algo) == 0,
            f"missing: {missing_for_algo}" if missing_for_algo else "present for all 6 datasets",
        )
        for metric_key, _, _, scale, decimals in METRICS:
            cell_map = {}
            for dataset in DATASET_COLUMN_ORDER:
                n_cells_total += 1
                rows = group_df[(group_df["dataset"] == dataset) & (group_df["mine_algorithm"] == algo)]
                if rows.empty:
                    cell_map[dataset] = None
                    n_cells_missing += 1
                    continue
                formatted = _format_cell(rows.iloc[0], metric_key, scale, decimals)
                cell_map[dataset] = formatted
                if formatted is None:
                    n_cells_missing += 1
                else:
                    text, med, q1, q3 = formatted
                    report.check(
                        f"Q1 <= median <= Q3 for {algo}/{metric_key}/{dataset}",
                        q1 <= med + 1e-9 and med <= q3 + 1e-9,
                        f"q1={q1}, median={med}, q3={q3}",
                    )
            table[algo][metric_key] = cell_map

    report.check(
        "no algorithm/metric/dataset combination silently dropped (all 3x3x6=54 cells accounted for, "
        "present or explicitly missing)",
        n_cells_total == len(expected_algos) * len(METRICS) * len(DATASET_COLUMN_ORDER),
        f"{n_cells_total} cell(s) evaluated, {n_cells_missing} rendered as '--' (no dominance trials for "
        f"that combination), {n_cells_total - n_cells_missing} numeric",
    )
    return table


def _cell_text(cell):
    if cell is None:
        return "--"
    return cell[0]


def print_console_preview(table):
    dataset_labels = [DATASET_DISPLAY_NAMES.get(d, d) for d in DATASET_COLUMN_ORDER]
    col_width = max([len(l) for l in dataset_labels] + [22]) + 2
    print("\nrq2 combined table (preview)\n")
    header = f"{'Mine':<8}{'Metric':<8}" + "".join(f"{l:<{col_width}}" for l in dataset_labels)
    print(header)
    for algo in [a["display"] for a in BASELINE_ALGORITHMS]:
        for metric_key, metric_label, _, _, _ in METRICS:
            cell_map = table[algo][metric_key]
            row_text = f"{algo:<8}{metric_label:<8}" + "".join(
                f"{_cell_text(cell_map[d]):<{col_width}}" for d in DATASET_COLUMN_ORDER
            )
            print(row_text)


def write_csv(table, out_path):
    rows = []
    for algo in [a["display"] for a in BASELINE_ALGORITHMS]:
        for metric_key, metric_label, _, _, _ in METRICS:
            cell_map = table[algo][metric_key]
            row = {"Mine": algo, "Metric": metric_label}
            for dataset in DATASET_COLUMN_ORDER:
                row[DATASET_DISPLAY_NAMES.get(dataset, dataset)] = _cell_text(cell_map[dataset])
            rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, index=False)


def build_latex_table(table):
    backslash = chr(92)
    row_end = backslash + backslash
    dataset_labels = [DATASET_DISPLAY_NAMES.get(d, d) for d in DATASET_COLUMN_ORDER]
    col_spec = "ll" + "c" * len(DATASET_COLUMN_ORDER)

    lines = []
    lines.append(backslash + "begin{table}[tb]")
    lines.append(backslash + "centering")
    lines.append(backslash + "resizebox{" + backslash + "textwidth}{!}{%")
    lines.append(backslash + "begin{tabular}{" + col_spec + "}")
    lines.append(backslash + "toprule")
    lines.append("Mine & Metric & " + " & ".join(dataset_labels) + " " + row_end)
    lines.append(backslash + "midrule")

    algo_displays = [a["display"] for a in BASELINE_ALGORITHMS]
    for ai, algo in enumerate(algo_displays):
        if ai > 0:
            lines.append(backslash + "midrule")
        first_algo_line = True
        for metric_key, _, metric_math, _, _ in METRICS:
            cell_map = table[algo][metric_key]
            cells = [_cell_text(cell_map[d]) for d in DATASET_COLUMN_ORDER]
            algo_cell = (
                backslash + "multirow{" + str(len(METRICS)) + "}{*}{" + algo + "}"
                if first_algo_line else ""
            )
            lines.append(algo_cell + " & " + metric_math + " & " + " & ".join(cells) + " " + row_end)
            first_algo_line = False

    lines.append(backslash + "bottomrule")
    lines.append(backslash + "end{tabular}")
    lines.append("}")
    lines.append(backslash + "vspace{2pt}")
    caption = ("Gaps in accuracy ($\Delta_{fit}$), simplicity ($\Delta_{simp}$), and similarity ($\Delta_{simi}$) for \textsc{RulesRepair} solutions dominated by \textsc{Mine}")
    lines.append(backslash + "caption{" + caption + "}")
    lines.append(backslash + "label{tab:rq2_dominance_advantage}")
    lines.append(backslash + "end{table}")
    return chr(10).join(lines)


def main_build_table(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--rq2-dir", default=str(Path("quantitative_evaluation") / "rq1" / "dominance_advantage"),
        help="Directory containing rq2_group_summary_*.csv (default: where main_dominance_advantage "
             "writes them by default, quantitative_evaluation/rq1/dominance_advantage/)",
    )
    parser.add_argument("--out-csv", default=None, help="Path for the combined CSV table "
                                                          "(default: <rq2-dir>/rq2_combined_table.csv)")
    parser.add_argument("--out-tex", default=None, help="Path for the combined LaTeX table "
                                                          "(default: <rq2-dir>/rq2_combined_table.tex)")
    args = parser.parse_args(argv)

    rq2_dir = Path(args.rq2_dir)
    out_csv = Path(args.out_csv) if args.out_csv else rq2_dir / "rq2_combined_table.csv"
    out_tex = Path(args.out_tex) if args.out_tex else rq2_dir / "rq2_combined_table.tex"

    report = ValidationReport()

    print(f"Loading group summaries (overall only) from {rq2_dir}/rq2_group_summary_*.csv...")
    group_df = load_group_summaries(rq2_dir, report)

    table = build_table(group_df, report)

    ok = report.print_report()
    print_console_preview(table)

    if not ok:
        raise SystemExit(1)

    write_csv(table, out_csv)
    print(f"\nSaved combined CSV table to {out_csv}")

    latex_text = build_latex_table(table)
    out_tex.write_text(latex_text + chr(10))
    print(f"Saved combined LaTeX table to {out_tex}")

    print("\ngenerated latex table\n")
    print(latex_text)
