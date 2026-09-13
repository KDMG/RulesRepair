import random as _random

import numpy as np

from keep_remine_prune.tree import DecisionNode, Condition, Operator

_COMPLEMENT_OPERATOR = {
    Operator.LE: Operator.GT,
    Operator.GT: Operator.LE,
    Operator.LT: Operator.GE,
    Operator.GE: Operator.LT,
    Operator.EQ: Operator.NE,
    Operator.NE: Operator.EQ,
}


def get_all_nodes(root):
    nodes = []
    root.walk(lambda n: nodes.append(n))
    return nodes


def get_internal_nodes(root, exclude_root=False):
    return [n for n in get_all_nodes(root)
            if not n.is_leaf() and not (exclude_root and n.is_root())]


def get_leaf_nodes(root):
    return [n for n in get_all_nodes(root) if n.is_leaf()]


def max_node_id(root):
    return max(n.node_id for n in get_all_nodes(root))


def find_node_by_id(root, node_id):
    for n in get_all_nodes(root):
        if n.node_id == node_id:
            return n
    return None


def root_to_leaf_paths(root):
    paths = []
    for leaf in get_leaf_nodes(root):
        ids = []
        node = leaf
        while node is not None:
            ids.append(node.node_id)
            node = node.parent
        paths.append((leaf.node_id, frozenset(ids)))
    return paths


def prepare_thresholds_pool(X, columns, n_thresholds=10):
    X = np.asarray(X, dtype=np.float64)
    if X.shape[1] != len(columns):
        raise ValueError(
            f"X has {X.shape[1]} columns but `columns` has {len(columns)} entries"
        )
    n_samples = X.shape[0]
    pool = {}
    for pos in range(len(columns)):
        col = np.sort(X[:, pos])
        candidates = []
        for j in range(n_thresholds):
            idx = int(n_samples / (n_thresholds + 1) * (j + 1))
            idx = min(idx, n_samples - 1)
            candidates.append(float(col[idx]))
        pool[pos] = candidates
    return pool


def prune(node, classes, rng=None):
    rng = rng or _random
    if node.is_leaf():
        raise ValueError(f"node {node.node_id} is already a leaf")
    node.children = []
    node.conditions = []
    node.label = rng.choice(classes)
    node.value = [1 if c == node.label else 0 for c in classes]
    node.impurity = 0.0
    return node


def apply_change_label(node, label, classes):
    """Deterministic core of change_label(): set `node` (a leaf) to `label`.
    Split out (2026) so both change_label() below (single random pick) and
    mutations/mutation_sampling.py's enumerate-every-alternative sampler
    apply the exact same logic instead of each having their own copy --
    explicit user request ("everything must reference the perturbation
    folder")."""
    node.label = label
    node.value = [1 if c == label else 0 for c in classes]
    return node


def change_label(node, classes, rng=None):
    rng = rng or _random
    if not node.is_leaf():
        raise ValueError(f"node {node.node_id} is not a leaf")
    if len(classes) <= 1:
        return node
    candidates = [c for c in classes if c != node.label]
    label = rng.choice(candidates)
    return apply_change_label(node, label, classes)

def branch_swap(node):
    if node.is_leaf():
        raise ValueError(f"node {node.node_id} is a leaf, nothing to swap")
    for cond in node.conditions:
        cond.operator = _COMPLEMENT_OPERATOR[cond.operator]
    return node


def apply_change_threshold(node, threshold):
    """Deterministic core of change_threshold() -- see apply_change_label()'s
    docstring for why this is split out."""
    for cond in node.conditions:
        cond.threshold = threshold
    return node


def change_threshold(node, thresholds_pool, rng=None):
    rng = rng or _random
    if node.is_leaf():
        raise ValueError(f"node {node.node_id} is a leaf, not a decision node")
    attribute_pos = node.conditions[0].attribute_pos
    new_threshold = rng.choice(thresholds_pool[attribute_pos])
    return apply_change_threshold(node, new_threshold)


def apply_change_feature(node, pos, threshold, columns):
    """Deterministic core of change_feature() -- see apply_change_label()'s
    docstring for why this is split out."""
    new_attribute = columns[pos]
    for cond in node.conditions:
        cond.attribute_pos = pos
        cond.attribute = new_attribute
        cond.threshold = threshold
    return node


def change_feature(node, thresholds_pool, columns, rng=None):
    rng = rng or _random
    if node.is_leaf():
        raise ValueError(f"node {node.node_id} is a leaf, not a decision node")
    n_features = len(columns)
    if n_features <= 1:
        raise ValueError("need at least 2 features to change the split feature")
    current_pos = node.conditions[0].attribute_pos
    candidates_pos = [p for p in range(n_features) if p != current_pos]
    new_pos = rng.choice(candidates_pos)
    new_threshold = rng.choice(thresholds_pool[new_pos])
    return apply_change_feature(node, new_pos, new_threshold, columns)


def _generate_random_subtree(start_id, thresholds_pool, columns, classes,
                              max_depth, split_prob, rng):
    counter = [start_id]
    n_features = len(columns)

    def build(depth_left, force_split):
        my_id = counter[0]
        counter[0] += 1
        make_leaf = (depth_left == 0) or (
            (not force_split) and rng.random() > split_prob
        )
        if make_leaf:
            label = rng.choice(classes)
            value = [1 if c == label else 0 for c in classes]
            return DecisionNode(label, my_id, value, 0.0)
        pos = rng.randrange(n_features)
        threshold = rng.choice(thresholds_pool[pos])
        attribute = columns[pos]
        parent_node = DecisionNode(rng.choice(classes), my_id,
                                    [0] * len(classes), 0.0)
        left = build(depth_left - 1, force_split=False)
        right = build(depth_left - 1, force_split=False)
        parent_node.add_child(Condition(attribute, pos, Operator.LE, threshold), left)
        parent_node.add_child(Condition(attribute, pos, Operator.GT, threshold), right)
        return parent_node

    subtree_root = build(max_depth, force_split=True)
    return subtree_root, counter[0]


def regrow(node, root, thresholds_pool, columns, classes,
           max_depth=3, split_prob=0.7, rng=None):
    rng = rng or _random
    if max_depth < 1:
        raise ValueError("max_depth must be >= 1 to guarantee at least one split")
    start_id = max_node_id(root) + 1
    subtree_root, _ = _generate_random_subtree(
        start_id, thresholds_pool, columns, classes, max_depth, split_prob, rng
    )
    node.label = subtree_root.label
    node.value = subtree_root.value
    node.impurity = subtree_root.impurity
    node.children = subtree_root.children
    node.conditions = subtree_root.conditions
    for child in node.children:
        child.parent = node
    return node
