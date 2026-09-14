import argparse
import pickle
from collections import defaultdict

import pm4py
from pm4py.algo.conformance.alignments.petri_net import algorithm as align_alg

from mining.extract_decision_points import fix_final_marking, encode_var, NON_DATA_KEYS
# build_tree/prune_tree/print_tree/format_cond used to be defined in this
# file, but they don't need pm4py/extract_decision_points at all -- only
# find_decision_points/build_instances/mine_decisions() below (the actual
# process-mining/alignment logic) do. They now live in cart_training.py (no
# pm4py dependency) and are re-exported here so mine_decisions() (below) and
# any existing `from mine_decisions import build_tree` caller keep working
# unchanged. Callers that only need build_tree() -- the repair pipeline
# scripts -- import directly from cart_training.py instead, so they no
# longer need pm4py/extract_decision_points installed at all.
from mining.cart_training import build_tree, prune_tree, print_tree, format_cond  # noqa: F401


def find_decision_points(net):
    out_arcs = defaultdict(list)
    for a in net.arcs:
        if a.source in net.places:
            out_arcs[a.source].append(a.target)
    trans_to_dp = defaultdict(list)
    for place, targets in out_arcs.items():
        if len(targets) < 2:
            continue
        for t in targets:
            trans_to_dp[t.name].append(place.name)
    return trans_to_dp


def build_instances(net, im, fm, log, trans_to_dp, min_fitness=0.944):
    rows = defaultdict(list)
    for tr in log:
        res = align_alg.apply_trace(tr, net, im, fm, parameters={"ret_tuple_as_trans_desc": True})
        if res["fitness"] < min_fitness:
            continue
        state = {}
        log_idx = 0
        for (log_trans, model_trans), (log_act, model_act) in res["alignment"]:
            if model_trans in trans_to_dp and model_trans != ">>":
                for place_name in trans_to_dp[model_trans]:
                    snapshot = dict(state)
                    snapshot["branch"] = model_trans
                    rows[place_name].append(snapshot)
            if log_act != ">>":
                if log_idx < len(tr):
                    for k, v in tr[log_idx].items():
                        if k not in NON_DATA_KEYS:
                            state[encode_var(k)] = v
                log_idx += 1
    return rows


def mine_decisions(pnml_path, xes_path, max_depth=4, min_samples_leaf=0.02, min_fitness=0.944):
    net, im, fm = pm4py.read_pnml(pnml_path)
    fm = fix_final_marking(net, fm)
    trans_to_dp = find_decision_points(net)
    log = pm4py.read_xes(xes_path, return_legacy_log_object=True)
    rows = build_instances(net, im, fm, log, trans_to_dp, min_fitness)
    return {place_name: build_tree(records, max_depth, min_samples_leaf) for place_name, records in rows.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pnml", required=True)
    parser.add_argument("--xes", required=True)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--min-samples-leaf", type=float, default=0.02)
    parser.add_argument("--min-fitness", type=float, default=0.944)
    parser.add_argument("--save", default=None, help="Save the mined trees to this file (pickle)")
    args = parser.parse_args()

    trees = mine_decisions(args.pnml, args.xes, args.max_depth, args.min_samples_leaf, args.min_fitness)
    for place_name, (tree, f1, onehot_map, columns) in trees.items():
        print(place_name)
        print_tree(tree, onehot_map, 1)

    if args.save:
        with open(args.save, "wb") as f:
            pickle.dump(trees, f)
        print(f"\nSaved mined trees to {args.save}")


if __name__ == "__main__":
    main()
