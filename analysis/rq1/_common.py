import re
from pathlib import Path
import glob as glob_module

import numpy as np
import pandas as pd

from analysis.core.compare_rulesrepair_cart_dominance import DEFAULT_TOL, _dominates
from keep_remine_prune.param_grid import GROW_CONFIGS


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


DATASET_COLUMN_ORDER = [
    "sepsis", "production", "hospital_billing", "road_traffic",
    "prepaid_travel_costs", "international_declarations",
]


BASELINE_ALGORITHMS = [
    {"key": "cart", "display": "CART",
     "mining_algorithm": "cart",
     "acc_col": "acc_cart_test", "nodes_col": "cart_total_nodes", "jaccard_col": "sim_old_cart_jaccard"},
    {"key": "c45", "display": "C4.5",
     "mining_algorithm": "j48",
     "acc_col": "acc_j48_test", "nodes_col": "j48_total_nodes", "jaccard_col": "sim_old_j48_jaccard"},
    {"key": "reptree", "display": "REPTree",
     "mining_algorithm": "reptree",
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


SWEEP_OBJECTIVES = ["acc_test", "total_nodes", "sim_old_jaccard"]

RULESREPAIR_OBJECTIVES = ["acc_new_test", "total_nodes", "sim_old_new_jaccard"]


def _pareto_points(points, tol=DEFAULT_TOL):
    unique = np.unique(np.round(np.asarray(points, dtype=float), 9), axis=0)
    keep = []
    for i, p in enumerate(unique):
        dominated = False
        for j, q in enumerate(unique):
            if i != j and _dominates(q[0], q[1], q[2], p[0], p[1], p[2], tol):
                dominated = True
                break
        if not dominated:
            keep.append(p)
    return np.array(keep)


def _expected_config_names(mining_algorithm):
    return [c["name"] for c in GROW_CONFIGS if c["algorithm"] == mining_algorithm]


def iter_front_pairs(csv_paths, algorithms=BASELINE_ALGORITHMS, tol=DEFAULT_TOL):
    warned = set()
    for path, dataset, seed, _ in _expand_and_label_paths(csv_paths, quiet=True):
        decision_point = path.parents[1].name
        sweep = pd.read_csv(path)
        missing = [c for c in ["trial_id", "algorithm"] + SWEEP_OBJECTIVES if c not in sweep.columns]
        if missing:
            raise SystemExit(f"{path}: missing column(s) {missing} -- expected a baseline_sweep.csv file.")
        configs_dir = path.parent / "configs"

        for algo in algorithms:
            mining_algorithm = algo["mining_algorithm"]
            config_names = _expected_config_names(mining_algorithm)
            absent = [n for n in config_names if not (configs_dir / n / "results.csv").exists()]
            if absent:
                if (str(path), mining_algorithm) not in warned:
                    print(f"  Warning: {path.parent}: missing RulesRepair results for {absent}, "
                          f"skipping {algo['display']}.")
                    warned.add((str(path), mining_algorithm))
                continue
            results = pd.concat(
                [pd.read_csv(configs_dir / name / "results.csv") for name in config_names], ignore_index=True
            )
            missing = [c for c in ["trial_id"] + RULESREPAIR_OBJECTIVES if c not in results.columns]
            if missing:
                raise SystemExit(f"{configs_dir}: missing column(s) {missing} in results.csv.")

            mine_all = sweep[sweep["algorithm"] == mining_algorithm]
            operators = results.groupby("trial_id")["operator"].first() if "operator" in results.columns else None
            for trial_id, rr_group in results.groupby("trial_id"):
                mine_group = mine_all[mine_all["trial_id"] == trial_id]
                if mine_group.empty:
                    continue
                yield {
                    "dataset": dataset, "seed": seed, "decision_point": decision_point,
                    "trial_id": trial_id,
                    "operator": operators[trial_id] if operators is not None else None,
                    "algorithm": algo["display"],
                    "rr_front": _pareto_points(rr_group[RULESREPAIR_OBJECTIVES].to_numpy(), tol),
                    "mine_front": _pareto_points(mine_group[SWEEP_OBJECTIVES].to_numpy(), tol),
                }


def collect_multi_algorithm_dominance(csv_paths, algorithms=BASELINE_ALGORITHMS, tol=DEFAULT_TOL):
    rows = []
    for pair in iter_front_pairs(csv_paths, algorithms, tol):
        rr_front, mine_front = pair["rr_front"], pair["mine_front"]
        n_dominated = sum(
            1 for r in rr_front
            if any(_dominates(m[0], m[1], m[2], r[0], r[1], r[2], tol) for m in mine_front)
        )
        rows.append({
            "dataset": pair["dataset"], "seed": pair["seed"], "decision_point": pair["decision_point"],
            "trial_id": pair["trial_id"], "operator": pair["operator"], "algorithm": pair["algorithm"],
            "front_size": len(rr_front), "mine_front_size": len(mine_front),
            "n_dominated": n_dominated, "d_i": n_dominated / len(rr_front),
        })
    return pd.DataFrame(rows)
