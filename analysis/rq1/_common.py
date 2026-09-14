import re
from pathlib import Path
import glob as glob_module

import pandas as pd

from analysis.core.compare_kr_cart_dominance import DEFAULT_TOL, _dominates


_DATASET_RE = re.compile(r"experiments[/\\]([^/\\]+)[/\\]repair[/\\]")

_SEED_RE = re.compile(r"seed_([^/\\]+)[/\\]")


def _quartile_row(vals, label):
    vals = pd.Series(vals).dropna()
    n = len(vals)
    if n == 0:
        return {
            "group": label, "n_trials": 0,
            "median": float("nan"), "q1": float("nan"), "q3": float("nan"),
            "iqr": float("nan"), "mean": float("nan"),
        }
    q1, med, q3 = vals.quantile([0.25, 0.5, 0.75])
    return {
        "group": label, "n_trials": n,
        "median": float(med), "q1": float(q1), "q3": float(q3),
        "iqr": float(q3 - q1), "mean": float(vals.mean()),
    }


def _load_merge_csvs(csv_paths, required_columns, missing_column_hint, empty_message):
    frames = []
    for raw_path in csv_paths:
        p = Path(raw_path)
        if not p.exists():
            print(f"  {p}: file not found")
            continue
        df = pd.read_csv(p)
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            raise SystemExit(f"{p}: missing column(s) {missing} -- {missing_column_hint}")
        frames.append(df)
    if not frames:
        raise SystemExit(empty_message)
    return pd.concat(frames, ignore_index=True)


def infer_dataset(csv_path):
    m = _DATASET_RE.search(str(csv_path))
    if m:
        return m.group(1)
    return None


def infer_seed(csv_path):
    m = _SEED_RE.search(str(csv_path))
    return m.group(1) if m else None


def _expand_and_label_paths(csv_paths, quiet=False):
    expanded_paths = []
    for raw_path in csv_paths:
        matches = sorted(glob_module.glob(str(raw_path)))
        if matches:
            expanded_paths.extend(matches)
        else:
            expanded_paths.append(raw_path)

    n_unmatched = 0
    for raw_path in expanded_paths:
        path = Path(raw_path)
        if not path.exists():
            if not quiet:
                print(f"  {path}: file not found")
            continue
        dataset = infer_dataset(path)
        seed = infer_seed(path)
        try:
            decision_point = path.parents[2].name
        except IndexError:
            decision_point = path.parent.name
        if dataset is None:
            n_unmatched += 1
            if not quiet:
                print(f"  Warning: could not infer dataset from path, leaving as 'unknown': {path}")
            dataset = "unknown"
        yield path, dataset, seed, decision_point

    if n_unmatched and not quiet:
        print(f"\n{n_unmatched} file(s) had an unexpected path, kept under dataset='unknown'.\n")


OPERATOR_ORDER = [
    "prune", "change_label", "branch_swap",
    "change_threshold", "change_feature",
    "regrow_leaf", "regrow_internal",
]


OPERATOR_DISPLAY_NAMES = {
    "prune": "Prune",
    "change_label": "Change label",
    "branch_swap": "Branch swap",
    "change_threshold": "Change threshold",
    "change_feature": "Change feature",
    "regrow_leaf": "Regrow leaf",
    "regrow_internal": "Regrow internal",
}


DATASET_DISPLAY_NAMES = {
    "sepsis": "Sepsis",
    "road_traffic": "Road Traffic Fine",
    "production": "Production",
    "prepaid_travel_costs": "Prepaid Travel Costs",
    "hospital_billing": "Hospital Billing",
    "international_declarations": "International Declarations",
}


def _dataset_display(name):
    return DATASET_DISPLAY_NAMES.get(name, str(name).replace("_", " ").title())


BASELINE_ALGORITHMS = [
    {"key": "cart", "display": "CART",
     "acc_col": "acc_cart_test", "nodes_col": "cart_total_nodes", "jaccard_col": "sim_old_cart_jaccard"},
    {"key": "c45", "display": "C4.5",
     "acc_col": "acc_j48_test", "nodes_col": "j48_total_nodes", "jaccard_col": "sim_old_j48_jaccard"},
    {"key": "reptree", "display": "REPTree",
     "acc_col": "acc_reptree_test", "nodes_col": "reptree_total_nodes", "jaccard_col": "sim_old_reptree_jaccard"},
]


DOM_COV_RAW_COLUMNS = [
    "dataset", "seed", "decision_point", "trial_id", "operator", "algorithm",
    "front_size", "n_dominated", "d_i",
]


def load_dom_cov_raw(csv_paths):
    return _load_merge_csvs(
        csv_paths, DOM_COV_RAW_COLUMNS,
        missing_column_hint="is this really a --dom-cov-raw-out file from a previous run of this script?",
        empty_message="No usable --dom-cov-raw-out CSV files found among the given paths.",
    )


def collect_multi_algorithm_dominance(csv_paths, algorithms=BASELINE_ALGORITHMS, tol=DEFAULT_TOL):
    rows = []
    warned = set()
    for path, dataset, seed, decision_point in _expand_and_label_paths(csv_paths, quiet=True):
        df = pd.read_csv(path)
        base_required = ["trial_id", "is_pareto", "acc_new_test", "total_nodes", "sim_old_new_jaccard"]
        if any(c not in df.columns for c in base_required):
            continue
        pareto_df = df[df["is_pareto"]]
        for trial_id, group in pareto_df.groupby("trial_id"):
            front = list(zip(
                group["acc_new_test"].astype(float),
                group["total_nodes"].astype(float),
                group["sim_old_new_jaccard"].astype(float),
            ))
            front_size = len(front)
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
                n_dominated = sum(
                    1 for (acc, nodes, jaccard) in front
                    if _dominates(a_acc, a_nodes, a_jaccard, acc, nodes, jaccard, tol)
                )
                rows.append({
                    "dataset": dataset, "seed": seed, "decision_point": decision_point,
                    "trial_id": trial_id, "operator": operator, "algorithm": algo["display"],
                    "front_size": front_size, "n_dominated": n_dominated,
                    "d_i": (n_dominated / front_size) if front_size else float("nan"),
                })
    return pd.DataFrame(rows)
