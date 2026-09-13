import argparse
import re
import xml.etree.ElementTree as ET
from collections import defaultdict, namedtuple
from pathlib import Path

import pandas as pd
import pm4py
from pm4py.objects.petri_net.obj import Marking
from pm4py.algo.conformance.alignments.petri_net import algorithm as align_alg


def declared_variables(pnml_path):
    tree = ET.parse(pnml_path)
    root = tree.getroot()
    names = set()
    for var in root.iter():
        if var.tag.split("}")[-1] == "variable":
            for child in var:
                if child.tag.split("}")[-1] == "name" and child.text:
                    names.add(child.text.strip())
    return names

NON_DATA_KEYS = {"concept:name", "lifecycle:transition", "time:timestamp"}

DecisionPoint = namedtuple("DecisionPoint", ["place_name", "variables", "branches"])

def encode_var(name):
    return name.replace(":", "$3A")


def fix_final_marking(net, fm):
    if len(fm) > 0:
        return fm
    out_deg = defaultdict(int)
    for a in net.arcs:
        if a.source in net.places:
            out_deg[a.source] += 1
    sinks = [p for p in net.places if out_deg[p] == 0]
    new_fm = Marking()
    for p in sinks:
        new_fm[p] = 1
    return new_fm


def find_decision_points(net, valid_variable_names):
    out_arcs = defaultdict(list)
    for a in net.arcs:
        if a.source in net.places:
            out_arcs[a.source].append(a.target)

    decision_points = {}
    trans_to_dp = defaultdict(list)

    for place, targets in out_arcs.items():
        if len(targets) < 2:
            continue
        targets_sorted = sorted(targets, key=lambda t: t.name)
        guards = [t.properties.get("guard") for t in targets_sorted]
        variables = set()
        for g in guards:
            if not g:
                continue
            for m in re.finditer(r"[A-Za-z_][A-Za-z0-9_$]*", g):
                variables.add(m.group(0))
        variables = variables & valid_variable_names

        branches = {}
        for i, t in enumerate(targets_sorted):
            branches[t.name] = {
                "index": i,
                "label": t.label,
                "guard": t.properties.get("guard"),
                "tag": t.properties.get("trans_name_tag"),
            }
            trans_to_dp[t.name].append((place.name, i))

        decision_points[place.name] = DecisionPoint(place.name, variables, branches)

    return decision_points, trans_to_dp


def process_log(net, im, fm, log, decision_points, trans_to_dp, max_traces=None):
    name_to_label = {t.name: (t.label if t.label is not None else f"skip ({t.name})") for t in net.transitions}

    rows = defaultdict(list)
    n_traces = len(log) if max_traces is None else min(max_traces, len(log))

    for i in range(n_traces):
        tr = log[i]
        try:
            res = align_alg.apply_trace(
                tr, net, im, fm,
                parameters={"ret_tuple_as_trans_desc": True},
            )
        except Exception as e:
            print(f"  [warn] alignment failed for trace {i}: {e}")
            continue

        case_id = tr.attributes.get("concept:name", str(i))
        case_timestamp = tr[0].get("time:timestamp") if len(tr) > 0 else None

        state = {}
        log_idx = 0
        for (log_trans, model_trans), (log_act, model_act) in res["alignment"]:
            is_decision_move = model_trans in trans_to_dp and model_trans != ">>"

            if is_decision_move:
                for place_name, branch_idx in trans_to_dp[model_trans]:
                    dp = decision_points[place_name]
                    snapshot = dict(state)
                    branch_info = dp.branches[model_trans]
                    snapshot["branch"] = name_to_label.get(model_trans, model_trans)
                    snapshot["branch_index"] = branch_idx
                    snapshot["branch_label"] = branch_info["label"]
                    snapshot["case_id"] = case_id
                    snapshot["timestamp"] = case_timestamp
                    rows[place_name].append(snapshot)

            if log_act != ">>":
                if log_idx < len(tr):
                    ev = tr[log_idx]
                    for k, v in ev.items():
                        if k not in NON_DATA_KEYS:
                            state[encode_var(k)] = v
                log_idx += 1

    return rows


def main():
    parser = argparse.ArgumentParser(description="Extract decision-point tables from a log + Data Petri Net")
    parser.add_argument("--pnml", required=True)
    parser.add_argument("--xes", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-traces", type=int, default=None, help="Limit the number of traces (debug/speed)")
    args = parser.parse_args()

    net, im, fm = pm4py.read_pnml(args.pnml)
    fm = fix_final_marking(net, fm)

    valid_vars = declared_variables(args.pnml)
    decision_points, trans_to_dp = find_decision_points(net, valid_vars)
    print(f"found decision points: {len(decision_points)}")
    for place_name, dp in decision_points.items():
        print(f"{place_name}: {len(dp.branches)} branches, variables={sorted(dp.variables)}")

    log = pm4py.read_xes(args.xes, return_legacy_log_object=True)

    rows = process_log(net, im, fm, log, decision_points, trans_to_dp, max_traces=args.max_traces)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for place_name, recs in rows.items():
        df = pd.DataFrame(recs)
        path = out_dir / f"dp_{place_name}.csv"
        df.to_csv(path, index=False)
        print(f"{path}: {len(df)} rows, columns={list(df.columns)}")

if __name__ == "__main__":
    main()
