import numpy as np
import pandas as pd
from collections import Counter
import sklearn
import sklearn.tree
import uuid
from .conversion import sklearn_to_tree
from .cost_tree import CostNode, CostMetadata, to_cost_tree
from .tree import DecisionNode, count_values, gini_impurity
from .tree import grow_tree as our_grow_tree


OLD = False # Old node (is_new = False)
NEW = True  # New node (is_new = True)


def _uid():
    return uuid.uuid4().int


def relabel(node, id_counter=1):
    node.node_id = id_counter
    id_counter += 1

    for child in node.children:
        id_counter = relabel(child, id_counter)

    return id_counter


def sklearn_grow_func(X, y, max_depth, metadata):
    clf = sklearn.tree.DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=0.02, random_state=0)
    clf = clf.fit(X, y)
    return sklearn_to_tree(clf, metadata.column_names)


def tree_grow_func(X, y, max_depth, metadata):
    # Our implementation of tree growing
    return our_grow_tree(pd.DataFrame(X, columns=metadata.column_names), y, max_depth=max_depth)


def grow_tree(X, y, old_tree=None, max_depth=4, w_simp=1, w_simi=0.25, grow_func=tree_grow_func, cache=None,
              forced_keep_ids=None, **kwargs):
    column_names = X.columns
    if cache is not None: # for speeding up
        bucket = cache.setdefault('X_numpy', {})
        key = id(X)
        X_np = bucket.get(key)
        if X_np is None:
            X_np = X.to_numpy()
            bucket[key] = X_np
        X = X_np
    else:
        X = X.to_numpy()

    classes = list(np.unique(y))

    nodes_max = 2 ** (max_depth + 1) - 1
    n_train = max(len(y), 1)

    metadata = CostMetadata(
        classes, column_names, w_simp, w_simi, grow_func,
        nodes_max=nodes_max, n_train=n_train,
    )

    if old_tree is None:
        old_tree = CostNode(best_pred(y, classes, metadata), 0, [])

    if not isinstance(old_tree, CostNode):
        assert isinstance(old_tree, DecisionNode)
        if cache is not None:
            bucket = cache.setdefault('to_cost_tree', {})
            key = id(old_tree)
            converted = bucket.get(key)
            if converted is None:
                converted = to_cost_tree(old_tree)
                bucket[key] = converted
            old_tree = converted
        else:
            old_tree = to_cost_tree(old_tree)

    flag_old(old_tree)
    recost(X, y, old_tree, metadata, cache=cache)

    tree = reduce(X, y, old_tree, max_depth, metadata, cache=cache, forced_keep_ids=forced_keep_ids)
    relabel(tree)
    return tree


def filter_data(X, y, node=None, cache=None):
    if node is None:
        return X, y

    if cache is not None:
        bucket = cache.setdefault('filter_data', {})
        key = id(node)
        cached = bucket.get(key)
        if cached is not None:
            return cached

    matches = node.matches(X)
    result = (X[matches], y[matches])

    if cache is not None:
        bucket[key] = result

    return result


def _majority_pred(y, classes):
    data = Counter(classes + list(y))
    return data.most_common(1)[0][0]


def best_pred(y, classes, metadata=None):
    return _majority_pred(y, classes)


def regrow(X, y, node, max_depth, metadata, cache=None):
    # TODO: Option to switch between sklearn and our own implementation
    if cache is not None:
        bucket = cache.setdefault('regrow', {})
        key = (id(node), max_depth, id(metadata.grow_func))
        cached = bucket.get(key)
        if cached is not None:
            return cached

    _X, _y = filter_data(X, y, node, cache=cache)
    max_depth = max_depth - node.depth

    if max_depth <= 0 or len(_y) == 0:
        tree = DecisionNode(None, 0, [])
    else:
        # sklearn splits are non-deterministic unless we set random_state
        tree = metadata.grow_func(_X, _y, max_depth, metadata)

    tree = to_cost_tree(tree)

    if cache is not None:
        bucket[key] = tree

    return tree


def flag_old(node):
    node.is_new = OLD

    for child in node.children:
        flag_old(child)


def reeval(X, y, node, metadata, fix_label=False, cache=None):
    _, _y = filter_data(X, y, node, cache=cache)
    if fix_label:
        node.label = best_pred(_y, metadata.classes, metadata)
        print(f"fixing label {node}")

    node.value = count_values(_y, metadata.classes)
    node.impurity = gini_impurity(_y)


def recost(X, y, node, metadata, fix_label=False, cache=None):
    reeval(X, y, node, metadata, fix_label, cache=cache)

    # visit children first so information can propogate bottom up
    for child in node.children:
        recost(X, y, child, metadata, fix_label, cache=cache)

    node.recost(metadata)


def prune(X, y, node, metadata, cache=None):
    # x, y must be filtered to the current node
    pred = best_pred(y, metadata.classes, metadata)
    value = count_values(y, metadata.classes)
    impurity = gini_impurity(y)

    prune_tree = CostNode(pred, _uid(), value, impurity)
    prune_tree.recost(metadata)

    if node.is_leaf():
        # keeping existing leaf node is never better than prune_tree.
        keep_tree = prune_tree
    else:
        keep_tree = CostNode(pred, _uid(), value, impurity)

        for child, condition in zip(node.children, node.conditions):
            _X, _y = filter_data(X, y, child, cache=cache)
            new_child = prune(_X, _y, child, metadata, cache=cache)
            keep_tree.add_child(condition, new_child)
        if keep_tree.children and all(c.is_leaf() for c in keep_tree.children) \
                and len({c.label for c in keep_tree.children}) == 1:
            collapsed_value = [sum(v) for v in zip(*(c.value for c in keep_tree.children))]
            keep_tree = CostNode(keep_tree.children[0].label, _uid(), collapsed_value, impurity)

        keep_tree.recost(metadata)

    if keep_tree.total_benefit >= prune_tree.total_benefit:
        return keep_tree
    return prune_tree


def reduce(X, y, node, max_depth, metadata, cache=None, forced_keep_ids=None):
    if node.is_leaf():
        keep_tree = CostNode(
            node.label, node.node_id, node.value, node.impurity,
            None, [], [], OLD, node.total_benefit
        )
    else:
        new_children = [
            reduce(X, y, child, max_depth, metadata, cache=cache, forced_keep_ids=forced_keep_ids)
            for child in node.children
        ]
        # We do not pay changed node penality if retaining structure of old tree.
        keep_tree = CostNode(
            node.label, node.node_id, node.value, node.impurity,
            None, [], [], OLD, node.total_benefit
        )
        for child, condition in zip(new_children, node.conditions):
            keep_tree.add_child(condition, child)
        keep_tree.recost(metadata)

    if forced_keep_ids is not None and node.node_id in forced_keep_ids:
        # comparison that merely happens to favour keep.
        return keep_tree

    regrow_tree = regrow(X, y, node, max_depth, metadata, cache=cache)
    _X, _y = filter_data(X, y, node, cache=cache) # need to ensure that we filter data to just the node we are pruning
    regrow_tree = prune(_X, _y, regrow_tree, metadata, cache=cache)

    if keep_tree.total_benefit >= regrow_tree.total_benefit:
        return keep_tree
    return regrow_tree
