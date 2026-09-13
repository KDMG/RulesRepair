from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from keep_remine_prune.tree import Condition, DecisionNode, Operator, gini_impurity


@dataclass
class RawSplit:
    attribute: Optional[str] = None
    children: List[Tuple[Operator, float, "RawSplit"]] = field(default_factory=list)

    @property
    def is_leaf(self):
        return self.attribute is None


def raw_to_decision_node(raw_root, X, y, columns, max_depth, classes=None):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    if classes is None:
        classes = sorted(set(y.tolist()))
    node_counter = [0]

    def build(raw_node, indices, depth):
        node_counter[0] += 1
        node_id = node_counter[0]
        y_here = y[indices]
        counts = [int(np.sum(y_here == c)) for c in classes]
        if len(indices) == 0:
            label = None
        else:
            label = classes[int(np.argmax(counts))]
        impurity = gini_impurity(y_here) if len(y_here) > 0 else 0.0
        node = DecisionNode(label, node_id, counts, impurity)

        stop = (
            raw_node.is_leaf
            or (max_depth is not None and depth >= max_depth)
            or len(indices) == 0
            or len(set(y_here.tolist())) < 2
        )
        if stop:
            return node

        attr_pos = columns.index(raw_node.attribute)
        for operator, threshold, child_raw in raw_node.children:
            mask = operator.op(X[indices, attr_pos], threshold)
            child_indices = indices[mask]
            cond = Condition(raw_node.attribute, attr_pos, operator, threshold)
            child_node = build(child_raw, child_indices, depth + 1)
            node.add_child(cond, child_node)
        return node

    return build(raw_root, np.arange(len(y)), 0)
