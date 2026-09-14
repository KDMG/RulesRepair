import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score

from keep_remine_prune.conversion import sklearn_to_tree
from keep_remine_prune.tree import Operator


def prune_tree(node):
    if node.is_leaf():
        return node
    node.children = [prune_tree(c) for c in node.children]
    if all(c.is_leaf() for c in node.children) and len({c.label for c in node.children}) == 1:
        value = [sum(v) for v in zip(*(c.value for c in node.children))]
        return type(node)(node.children[0].label, node.node_id, value, node.impurity, node.parent)
    return node


def encode_records(records):
    if not records:
        return None, None, {}, []
    df = pd.DataFrame(records)
    y = df["branch"]
    X = df.drop(columns=["branch"])
    if X.shape[1] == 0:
        return None, None, {}, []
    cat_cols = [c for c in X.columns if X[c].dtype == object or str(X[c].dtype) == "bool"]
    for col in X.columns:
        if col not in cat_cols:
            X[col] = X[col].fillna(X[col].median()).fillna(0)
    onehot_map = {col: (c, col[len(c) + 1:]) for c in cat_cols for col in pd.get_dummies(X[c], prefix=c).columns}
    X = pd.get_dummies(X, columns=cat_cols)
    return X, y, onehot_map, list(X.columns)


def build_tree(records, max_depth=4, min_samples_leaf=0.02, criterion="gini"):
    X, y, onehot_map, columns = encode_records(records)
    if X is None:
        return None, None, {}, []
    clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, criterion=criterion, random_state=0)
    clf.fit(X, y)
    f1 = f1_score(y, clf.predict(X), average="macro", zero_division=0)
    root = prune_tree(sklearn_to_tree(clf, columns))
    return root, f1, onehot_map, columns


def format_cond(cond, onehot_map):
    if cond.attribute in onehot_map:
        col, val = onehot_map[cond.attribute]
        return f"{col} == {val}" if cond.operator == Operator.GT else f"{col} != {val}"
    return str(cond)


def print_tree(node, onehot_map, depth=0):
    if node is None:
        print("  " * depth + "None")
        return
    if node.is_leaf():
        print("  " * depth + f"-> {node.label}")
        return
    for child, cond in zip(node.children, node.conditions):
        print("  " * depth + format_cond(cond, onehot_map))
        print_tree(child, onehot_map, depth + 1)
