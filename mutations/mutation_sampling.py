import copy
import math
import random

import numpy as np

from mutations.tree_mutations import (
    get_internal_nodes, get_leaf_nodes, get_all_nodes, find_node_by_id,
    prune, branch_swap, regrow,
    apply_change_label, apply_change_threshold, apply_change_feature,
)

DEFAULT_P = 0.20
DEFAULT_DEGRADATION_THRESHOLD = 0.20
DEFAULT_MAX_REGROW_ATTEMPTS = 10
DEFAULT_REGROW_MAX_DEPTH = 3
DEFAULT_REGROW_SPLIT_PROB = 0.7
DEFAULT_MAX_ALTERNATIVES_PER_NODE = 100

OPERATOR_NODE_KIND = {
    "prune": "internal",
    "change_label": "leaf",
    "branch_swap": "internal",
    "change_threshold": "internal",
    "change_feature": "internal",
    "regrow_leaf": "leaf",
    "regrow_internal": "internal",
}


def _node_pool(root, kind):
    if kind == "leaf":
        return get_leaf_nodes(root)
    if kind == "internal":
        return get_internal_nodes(root, exclude_root=False)
    if kind == "internal_no_root":
        return get_internal_nodes(root, exclude_root=True)
    if kind == "any":
        return get_all_nodes(root)
    raise ValueError(f"unknown node kind: {kind}")


def _accuracy(root, X_np, y):
    preds = [root.predict(row) for row in X_np]
    return float(np.mean([p == t for p, t in zip(preds, y)]))


def _alternatives_for_node(operator, node, classes, thresholds_pool, columns,
                            rng, max_regrow_attempts, regrow_max_depth, regrow_split_prob,
                            max_alternatives=None):
    if operator == "prune":
        alts = [c for c in classes if c != node.label] or list(classes)
        rng.shuffle(alts)
        for label in alts:
            def apply_fn(n, _label=label):
                # force the specific label deterministically instead of
                # letting prune() re-randomize it
                prune(n, classes, rng=random)
                n.label = _label
                n.value = [1 if c == _label else 0 for c in classes]
            yield f"label={label}", apply_fn

    elif operator == "change_label":
        alts = [c for c in classes if c != node.label]
        rng.shuffle(alts)
        for label in alts:
            def apply_fn(n, _label=label):
                apply_change_label(n, _label, classes)
            yield f"label={label}", apply_fn

    elif operator == "branch_swap":
        def apply_fn(n):
            branch_swap(n)
        yield "swap", apply_fn

    elif operator == "change_threshold":
        pos = node.conditions[0].attribute_pos
        current = node.conditions[0].threshold
        alts = [t for t in thresholds_pool[pos] if t != current]
        rng.shuffle(alts)
        if max_alternatives is not None:
            alts = alts[:max_alternatives]
        for thr in alts:
            def apply_fn(n, _thr=thr):
                apply_change_threshold(n, _thr)
            yield f"threshold={thr}", apply_fn

    elif operator == "change_feature":
        current_pos = node.conditions[0].attribute_pos
        combos = [
            (pos, thr)
            for pos in range(len(columns)) if pos != current_pos
            for thr in thresholds_pool[pos]
        ]
        rng.shuffle(combos)
        if max_alternatives is not None:
            combos = combos[:max_alternatives]
        for pos, thr in combos:
            def apply_fn(n, _pos=pos, _thr=thr):
                apply_change_feature(n, _pos, _thr, columns)
            yield f"feature={columns[pos]},threshold={thr}", apply_fn

    elif operator in ("regrow_leaf", "regrow_internal"):
        for attempt in range(max_regrow_attempts):
            seed_for_attempt = rng.randrange(2**31)
            def apply_fn(n, _root_ref=node, _seed=seed_for_attempt):
                local_rng = random.Random(_seed)
                # `root` argument to regrow() is only used to compute a
                # collision-free starting node_id -- pass the CLONED
                # tree's own root (n may not be the root, so walk up).
                cloned_root = n
                while cloned_root.parent is not None:
                    cloned_root = cloned_root.parent
                regrow(n, cloned_root, thresholds_pool, columns, classes,
                       max_depth=regrow_max_depth, split_prob=regrow_split_prob,
                       rng=local_rng)
            yield f"regrow_attempt={attempt}", apply_fn

    else:
        raise ValueError(f"unknown operator: {operator}")


def generate_mutants(root, X_train, columns, classes, X_eval, y_eval,
                      p=DEFAULT_P, degradation_threshold=DEFAULT_DEGRADATION_THRESHOLD,
                      seed=0, n_thresholds=10,
                      max_regrow_attempts=DEFAULT_MAX_REGROW_ATTEMPTS,
                      regrow_max_depth=DEFAULT_REGROW_MAX_DEPTH,
                      regrow_split_prob=DEFAULT_REGROW_SPLIT_PROB,
                      max_alternatives_per_node=DEFAULT_MAX_ALTERNATIVES_PER_NODE):
    from mutations.tree_mutations import prepare_thresholds_pool
    # Converted once here instead of inside _accuracy(): X_eval never changes across
    # the many candidate evaluations below, only the tree does, and re-running
    # X_eval.to_numpy() on every candidate was both slow and, since X_eval mixes
    # bool/int/float columns, produced a bloated dtype=object array (not float64)
    # -- a major cost on decision points with many one-hot columns.
    X_eval_np = np.asarray(X_eval, dtype=np.float64)
    acc_orig = _accuracy(root, X_eval_np, y_eval)
    if len(set(classes)) <= 1:
        only_class = classes[0] if classes else None
        print(
            f"Skipping mutant generation: only one class ({only_class!r}) present, "
            f"T_old's accuracy is already {acc_orig:.3f}. Returning zero mutants.",
            flush=True,
        )
        results = {op: [] for op in OPERATOR_NODE_KIND}
        summary = {
            op: {
                "n_pool": 0, "target_count": 0, "n_accepted": 0,
                "n_nodes_tried": 0, "n_nodes_exhausted_without_success": 0,
            }
            for op in OPERATOR_NODE_KIND
        }
        return results, summary

    thresholds_pool = prepare_thresholds_pool(X_train, columns, n_thresholds=n_thresholds)

    master_rng = random.Random(seed)
    results = {}
    summary = {}

    for operator, kind in OPERATOR_NODE_KIND.items():
        pool = _node_pool(root, kind)
        n_pool = len(pool)
        target_count = math.ceil(p * n_pool) if n_pool > 0 else 0
        shuffled_ids = [n.node_id for n in pool]
        master_rng.shuffle(shuffled_ids)

        accepted = []
        n_nodes_tried = 0
        n_nodes_exhausted = 0

        for node_id in shuffled_ids:
            if len(accepted) >= target_count:
                break
            n_nodes_tried += 1
            node = find_node_by_id(root, node_id)
            alt_rng = random.Random(master_rng.randrange(2**31))
            n_alt_tried = 0
            found = False
            for description, apply_fn in _alternatives_for_node(
                operator, node, classes, thresholds_pool, columns, alt_rng,
                max_regrow_attempts, regrow_max_depth, regrow_split_prob,
                max_alternatives=max_alternatives_per_node,
            ):
                n_alt_tried += 1
                candidate_root = copy.deepcopy(root)
                candidate_node = find_node_by_id(candidate_root, node_id)
                apply_fn(candidate_node)
                acc_after = _accuracy(candidate_root, X_eval_np, y_eval)
                degradation = acc_orig - acc_after
                if degradation >= degradation_threshold:
                    accepted.append({
                        "node_id": node_id,
                        "description": description,
                        "acc_before": acc_orig,
                        "acc_after": acc_after,
                        "degradation": degradation,
                        "n_alternatives_tried": n_alt_tried,
                        "tree": candidate_root,
                    })
                    found = True
                    break
            if not found:
                n_nodes_exhausted += 1

        results[operator] = accepted
        summary[operator] = {
            "n_pool": n_pool,
            "target_count": target_count,
            "n_accepted": len(accepted),
            "n_nodes_tried": n_nodes_tried,
            "n_nodes_exhausted_without_success": n_nodes_exhausted,
        }

    return results, summary


def print_mutation_summary(summary):
    for operator, s in summary.items():
        print(operator)
        print(f"  pool={s['n_pool']}  target={s['target_count']} "
              f"(p={DEFAULT_P:.0%})  accepted={s['n_accepted']}  "
              f"nodes_tried={s['n_nodes_tried']}  "
              f"nodes_exhausted_without_success={s['n_nodes_exhausted_without_success']}")
        if s['n_accepted'] < s['target_count']:
            print(f"  Warning: only {s['n_accepted']}/{s['target_count']} target mutants reached, "
                  f"pool exhausted before hitting the degradation threshold enough times.")
        print()
