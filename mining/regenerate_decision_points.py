import argparse
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import pm4py
from pm4py.algo.conformance.alignments.petri_net import algorithm as align_alg

from mining.extract_decision_points import fix_final_marking, encode_var, NON_DATA_KEYS
from mining.mine_decisions import find_decision_points

NON_FEATURE_COLUMNS = {"case_id", "timestamp", "branch_index", "branch_label"}


def sort_log_chronologically(log):
    def first_ts(tr):
        if len(tr) == 0:
            return None
        return tr[0].get("time:timestamp")

    with_ts = [tr for tr in log if first_ts(tr) is not None]
    without_ts = [tr for tr in log if first_ts(tr) is None]
    with_ts.sort(key=first_ts)
    return with_ts + without_ts


def build_instances_with_metadata(net, im, fm, log, trans_to_dp, min_fitness=0.944):
    name_to_label = {t.name: (t.label if t.label is not None else f"skip ({t.name})") for t in net.transitions}

    rows = defaultdict(list)
    total = len(log)
    progress_every = max(1, total // 100)
    start_time = time.perf_counter()

    alignment_cache = {}
    n_variant_hits = 0

    for i, tr in enumerate(log, start=1):
        case_id = tr.attributes.get("concept:name")
        variant_key = tuple(ev.get("concept:name") for ev in tr)
        res = alignment_cache.get(variant_key)
        if res is not None:
            n_variant_hits += 1
        else:
            res = align_alg.apply_trace(tr, net, im, fm, parameters={"ret_tuple_as_trans_desc": True})
            alignment_cache[variant_key] = res
        if res["fitness"] < min_fitness:
            continue
        state = {}
        last_timestamp = tr[0].get("time:timestamp") if len(tr) > 0 else None
        log_idx = 0
        for (log_trans, model_trans), (log_act, model_act) in res["alignment"]:
            if model_trans in trans_to_dp and model_trans != ">>":
                for place_name in trans_to_dp[model_trans]:
                    snapshot = dict(state)
                    snapshot["branch"] = name_to_label.get(model_trans, model_trans)
                    snapshot["case_id"] = case_id
                    snapshot["timestamp"] = last_timestamp
                    rows[place_name].append(snapshot)
            if log_act != ">>":
                if log_idx < len(tr):
                    ev = tr[log_idx]
                    last_timestamp = ev.get("time:timestamp", last_timestamp)
                    for k, v in ev.items():
                        if k not in NON_DATA_KEYS:
                            state[encode_var(k)] = v
                log_idx += 1

        if i % progress_every == 0 or i == total:
            elapsed = time.perf_counter() - start_time
            rate = i / elapsed if elapsed > 0 else 0
            eta_str = f"~{(total - i) / rate:.0f}s remaining" if rate > 0 else "ETA unknown"
            print(f"  aligned {i}/{total} traces ({100 * i / total:.1f}%), "
                  f"{elapsed:.0f}s elapsed, {eta_str}, "
                  f"{len(alignment_cache)} distinct variant(s) so far, {n_variant_hits} cache hit(s)", flush=True)

    print(f"variant cache: {len(alignment_cache)} distinct variant(s) out of {total} trace(s), "
          f"{n_variant_hits} A* alignment(s) skipped via cache reuse "
          f"({100 * n_variant_hits / total:.1f}% of traces)", flush=True)
    return rows


def distinct_dp_places(trans_to_dp):
    places = set()
    for place_names in trans_to_dp.values():
        places.update(place_names)
    return places


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pnml", default="datasets/sepsis_cut/pn.pnml")
    parser.add_argument("--xes", default="datasets/sepsis/sepsis.xes")
    parser.add_argument("--min-fitness", type=float, default=0)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    net, im, fm = pm4py.read_pnml(args.pnml)
    fm = fix_final_marking(net, fm)
    trans_to_dp = find_decision_points(net)
    all_places = distinct_dp_places(trans_to_dp)
    print(f"Petri net: {len(all_places)} decision points, {len(trans_to_dp)} branching transitions")

    log = pm4py.read_xes(args.xes, return_legacy_log_object=True)
    print(f"log: {len(log)} traces (unsorted)", flush=True)
    log = sort_log_chronologically(log)
    print("sorted traces chronologically by first-event timestamp", flush=True)

    print(f"Aligning {len(log)} traces against the Petri net", flush=True)
    rows = build_instances_with_metadata(net, im, fm, log, trans_to_dp, min_fitness=args.min_fitness)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for place_name, recs in sorted(rows.items()):
        df = pd.DataFrame(recs)
        meta_cols = [c for c in ("case_id", "timestamp", "branch") if c in df.columns]
        feature_cols = [c for c in df.columns if c not in meta_cols]
        df = df[feature_cols + meta_cols]

        n_features = len(feature_cols)
        branch_dist = Counter(r["branch"] for r in recs)
        usable = n_features > 0 and len(branch_dist) >= 2
        path = out_dir / f"dp_{place_name}.csv"
        df.to_csv(path, index=False)
        summary.append((place_name, len(recs), n_features, dict(branch_dist), usable))

    print(f"\nsummary: {len(summary)}/{len(all_places)} decision-point place(s) got at least 1 row")
    print(f"{'place':<10}{'rows':<8}{'n_features':<12}{'branches':<10}{'usable':<8}")
    for place_name, n_rows, n_features, branch_dist, usable in summary:
        print(f"{place_name:<10}{n_rows:<8}{n_features:<12}{len(branch_dist):<10}{str(usable):<8}")

    written_places = {p for p, _, _, _, _ in summary}
    missing_places = sorted(all_places - written_places)
    if missing_places:
        print(f"\n{len(missing_places)} decision points had zero rows, no CSV written: {missing_places}")

    n_usable = sum(1 for *_, usable in summary if usable)
    print(f"\n{n_usable}/{len(summary)} decision points are usable (at least 1 feature, at least 2 branches)")

    row_counts = [n_rows for _, n_rows, _, _, _ in summary]
    if row_counts:
        mean_rows = statistics.mean(row_counts)
        std_rows = statistics.pstdev(row_counts)
        print(f"\nDecision points: {len(summary)}, mean rows: {mean_rows:.1f}, std: {std_rows:.1f}")


if __name__ == "__main__":
    main()
