import numpy as np
from dataclasses import dataclass
from typing import Callable
from .tree import DecisionNode, TreeMetadata


# utility function
def _indent(s):
    return '\n'.join("    " + line for line in s.split('\n'))

@dataclass
class CostMetadata(TreeMetadata):
    w_simp: float
    w_simi: float
    grow_func: Callable[[np.ndarray, np.ndarray, int, TreeMetadata], DecisionNode]
    nodes_max: float = 1
    n_train: float = 1

@dataclass
class CostNode(DecisionNode):
    is_new: bool = True
    total_benefit: float = 0.0

    def recost(self, metadata):
        self.total_benefit = -metadata.w_simp / metadata.nodes_max

        if self.is_new:
            self.total_benefit -= metadata.w_simi / metadata.nodes_max

        if self.is_leaf():
            if self.label not in metadata.classes:
                correct = 0
            else:
                correct = self.value[self.label_index(metadata.classes)]

            incorrect = sum(self.value) - correct
            self.total_benefit -= incorrect / metadata.n_train

        for child in self.children:
            self.total_benefit += child.total_benefit

    def matches_row(self, x):
        node = self

        while True:
            cond = node.find_to_condition()
            if cond is None:
                # reached the root node and all conditions in parent chain held
                return True
            if not cond.fire(x):
                # condition in parent chain did not hold
                return False
            node = node.parent

    def matches(self, X):
        conditions = []
        node = self
        while True:
            cond = node.find_to_condition()
            if cond is None:
                break
            conditions.append(cond)
            node = node.parent

        mask = np.ones(len(X), dtype=bool)
        for cond in conditions:
            mask &= cond.operator.op(X[:, cond.attribute_pos], cond.threshold)
        return mask

    def pretty_print(self):
        # TODO: reimplement
        if self.is_leaf():
            return (
                f"{'NEW' if self.is_new else 'OLD'} LEAF {self.find_to_condition()} (Benefit: {self.total_benefit}, Label: {self.label}, Values: {self.value})"
            )

        s = f"{'NEW' if self.is_new else 'OLD'} NODE {'ROOT' if self.is_root() else self.find_to_condition()} (Benefit: {self.total_benefit})\n"
        s += '\n'.join(_indent(child.pretty_print()) for child in self.children)
        return s

    @property
    def depth(self):
        # root node is 0 depth
        depth = 0
        p = self.parent
        while p is not None:
            p = p.parent
            depth += 1
        return depth


def to_cost_tree(node):
    new_node = CostNode(
        node.label,
        node.node_id,
        node.value,
        node.impurity
    )

    for child, condition in zip(node.children, node.conditions):
        new_child = to_cost_tree(child)
        new_node.add_child(condition, new_child)
    return new_node
