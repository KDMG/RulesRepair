import numpy as np

from dataclasses import dataclass, field

from functools import partial
from enum import Enum
from collections import deque

from typing import Tuple, List, Counter
from operator import eq, lt, le, ge, ne, gt

import graphviz

class Operator(str, Enum):
    EQ = "=="
    LT = "<"
    GT = ">"
    LE = "<="
    GE = ">="
    NE = "!="

    @property
    def op(self):
        return {self.EQ: eq,
                self.LT: lt,
                self.LE: le,
                self.GE: ge,
                self.NE: ne,
                self.GT: gt}[self]

@dataclass
class Condition:
    attribute: str
    attribute_pos: int
    operator: Operator
    threshold: float

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"{self.attribute} {self.operator} {self.threshold}"

    def fire(self, x):
        return self.operator.op(x[self.attribute_pos], self.threshold)

@dataclass
class Split:
    score: float
    attribute_pos: int
    ids: Tuple[float]
    operations: List[Tuple[Operator, float]]

@dataclass
class TreeMetadata:
    classes: List[str]
    column_names: List[str]

@dataclass
class DecisionNode:
    label: str
    node_id: int
    value: List[int]
    impurity: float = 0.0
    parent: 'DecisionNode' = None
    children: List['DecisionNode'] = field(default_factory=list)
    conditions: List[Condition] = field(default_factory=list)

    def label_index(self, classes):
        return classes.index(self.label)

    def walk(self, callback):
        callback(self)
        for n in self.children:
            n.walk(callback)

    def add_child(self, condition, node):
        assert isinstance(condition, Condition)
        assert isinstance(node, DecisionNode)
        self.conditions.append(condition)
        node.parent = self
        self.children.append(node)

    def predict(self, x):
        stack = deque()
        stack.append(self)
        node = self

        while not node.is_leaf():
            node = stack.pop()

            matched = False
            for i, cond in enumerate(node.conditions):
                if cond.fire(x):
                    stack.append(node.children[i])
                    matched = True
                    break  # binary splits: exactly one branch should ever fire

            if not matched:
                return node.label

        return node.label

    def is_leaf(self):
        return not self.children

    def is_root(self):
        return not self.parent

    def plot(self):
        dot = graphviz.Digraph('tree', comment='Decision Tree')

        def update_dot(dot, node):
            dot.node(f"{node.node_id}", f"Node_{node.node_id}\nImpurity: {node.impurity:0.3f}\nLabel: {node.label}\nValue: {node.value}\nSamples: {sum(node.value)}")
            if node.parent:
                cond = node.find_to_condition()
                dot.edge(f"{node.parent.node_id}" , f"{node.node_id}", str(cond))

        update_dot_partial = partial(update_dot, dot)
        self.walk(update_dot_partial)
        return dot

    def find_to_condition(self):
        if self.is_root():
            return None
        else:
            index = -1
            for i, node in enumerate(self.parent.children):
                if node.node_id == self.node_id:
                    index = i
            if index < 0:
                raise ValueError("Incorrect tree")
            return self.parent.conditions[index]

    def __str__(self):
        return f"Node_{self.node_id}"

def stopping_criteria(tree_depth, **kwargs):
    max_depth = kwargs.pop('max_depth', -1)
    if tree_depth >= max_depth:
        return True
    return False


# 2026: removed entropy_impurity (alternative impurity criterion, never
# selected anywhere -- every caller hardcodes measure=gini_impurity).
def gini_impurity(y):
    counts = Counter(y)
    total = sum(counts.values())
    return round(1 - sum(map(lambda x: (x / total) ** 2, counts.values())), 3)


def evaluation_measure(groups: Tuple, measure):
    N = sum(map(len,groups))
    return sum(map(lambda x: len(x) / N * measure(x), groups))


def calculate_current_depth(current_node):
    depth_counter = 0
    parent_node = current_node.parent
    while parent_node:
        depth_counter += 1
        parent_node = parent_node.parent
    return depth_counter

def count_values(array, values):
    return [np.count_nonzero(array == i) for i in sorted(values)]

def find_best_split(X, y, **kwargs):
    split = Split(1,-1,(),())
    measure = kwargs.pop("measure", gini_impurity)

    for i in range(0, X.shape[1]):
        x_s = X[:, i]

        for threshold in np.unique(x_s):

            ids = (x_s <= threshold, x_s > threshold)
            operations = [(Operator.LE, threshold), (Operator.GT, threshold)]

            y_values = [y[i] for i in ids]
            score = evaluation_measure(y_values, measure)

            if score < split.score:
                split = Split(score, i, ids, operations)

    return split

def grow_tree(X, y, **kwargs):
    column_name = X.columns
    X = X.to_numpy()
    node_counter = 1
    stack = deque()

    if len(y) == 0:
        return DecisionNode(None, node_counter, [], 0.0)

    # Set up decision tree
    classes = np.unique(y)
    counts = count_values(y, classes)
    majority_class = classes[np.argmax(count_values(y, classes))]
    tree = DecisionNode(majority_class, node_counter, counts, gini_impurity(y))
    stack.append((tree, X, y))

    while len(stack) != 0:
        current_node, parent_X, parent_y = stack.pop()

        # Stop once reached max depth branching
        current_depth = calculate_current_depth(current_node)
        if stopping_criteria(current_depth, **kwargs):
            continue

        # Stop branching if node contains a single class
        values = np.unique(parent_y)
        if len(values) < 2:
            continue

        # Determine best attribute and split
        split = find_best_split(parent_X, parent_y, **kwargs)

        # Update tree with new split
        for cond, split_ids in zip(split.operations, split.ids):
            new_y = parent_y[split_ids]
            new_X = parent_X[split_ids]

            # Ensure new node is processed later
            counts = count_values(new_y, classes)
            if len(counts) == 0:
                continue

            condition = Condition(column_name[split.attribute_pos],
                                  split.attribute_pos,
                                  cond[0],
                                  cond[1])

            label = classes[np.argmax(counts)]
            node_counter += 1

            score = gini_impurity(new_y)

            new_node = DecisionNode(label, node_counter, counts, score)
            current_node.add_child(condition, new_node)

            stack.append((new_node, new_X, new_y))

    return tree
