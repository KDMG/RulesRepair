import re
import subprocess

import numpy as np

from keep_remine_prune.tree import Operator
from mining.binary_tree_from_splits import RawSplit, raw_to_decision_node

_jvm_started = False
_jvm_unavailable_reason = None

_JAVA_VERSION_RE = re.compile(r'version "(\d+)(?:\.(\d+))?')


def _detect_java_major_version():
    try:
        result = subprocess.run(["java", "-version"], capture_output=True, text=True, timeout=10)
    except Exception:
        return None
    m = _JAVA_VERSION_RE.search((result.stdout or "") + (result.stderr or ""))
    if not m:
        return None
    major = int(m.group(1))
    if major == 1 and m.group(2):
        return int(m.group(2))
    return major


def ensure_jvm():
    global _jvm_started, _jvm_unavailable_reason
    if _jvm_started:
        return
    if _jvm_unavailable_reason is not None:
        raise RuntimeError(_jvm_unavailable_reason)

    java_major = _detect_java_major_version()
    if java_major is not None and java_major < 9:
        _jvm_unavailable_reason = (
            f"Local Java is version {java_major}, but python-weka-wrapper3 requires Java 9 or "
            f"later -- refusing to start the JVM with it (older JVMs have been seen to crash the "
            f"whole process natively on this kind of mismatch, instead of failing cleanly). "
            f"Install a newer JDK (e.g. OpenJDK 17) and make sure it's the one on PATH."
        )
        raise RuntimeError(_jvm_unavailable_reason)

    try:
        import weka.core.jvm as jvm
        jvm.start()
    except Exception as ex:
        _jvm_unavailable_reason = (
            f"Weka JVM unavailable ({type(ex).__name__}: {ex}) -- is the "
            f"`python-weka-wrapper3` package installed (see pyproject.toml) "
            f"and a JVM/JDK (OpenJDK 8+) on PATH in this environment?"
        )
        raise RuntimeError(_jvm_unavailable_reason) from ex
    _jvm_started = True


def _to_weka_instances(X, y, columns):
    from weka.core.dataset import create_instances_from_matrices
    from weka.filters import Filter

    data = create_instances_from_matrices(
        np.asarray(X, dtype=float), np.asarray(y, dtype=object),
        name="baseline", cols_x=list(columns), col_y="class",
    )
    data.class_is_last()
    to_nominal = Filter(classname="weka.filters.unsupervised.attribute.StringToNominal", options=["-R", "last"])
    to_nominal.inputformat(data)
    data = to_nominal.filter(data)
    data.class_is_last()
    return data


_NODE_RE = re.compile(r'^(N[\w]+)\s*\[label="([^"]*)"(.*)\]\s*$')
_EDGE_RE = re.compile(r'^(N[\w]+)->(N[\w]+)\s*\[label="\s*(<=|<|>=|>)\s*([^"]+)"\]\s*$')
_REPTREE_PREFIX_RE = re.compile(r'^\d+:\s*(.*)$')

_OP_MAP = {"<=": Operator.LE, "<": Operator.LT, ">=": Operator.GE, ">": Operator.GT}


def parse_weka_dot(dot_text):
    is_leaf = {}
    attribute = {}
    children_by_parent = {}
    all_ids = set()
    child_ids = set()

    for line in dot_text.splitlines():
        line = line.strip()
        m = _EDGE_RE.match(line)
        if m:
            parent_id, child_id, op_text, threshold_text = m.groups()
            children_by_parent.setdefault(parent_id, []).append(
                (_OP_MAP[op_text], float(threshold_text), child_id)
            )
            all_ids.add(parent_id)
            all_ids.add(child_id)
            child_ids.add(child_id)
            continue
        m = _NODE_RE.match(line)
        if m:
            node_id, label, extra = m.groups()
            all_ids.add(node_id)
            leaf = "shape=box" in extra
            is_leaf[node_id] = leaf
            if not leaf:
                prefix_m = _REPTREE_PREFIX_RE.match(label)
                attribute[node_id] = prefix_m.group(1) if prefix_m else label

    roots = all_ids - child_ids
    if len(roots) != 1:
        raise ValueError(
            f"parse_weka_dot: expected exactly 1 root node, found {len(roots)} ({roots}) -- "
            f"Weka's .graph format may have changed. Raw text:\n{dot_text}"
        )
    root_id = next(iter(roots))

    def build(node_id):
        if is_leaf.get(node_id, True):
            return RawSplit()
        raw = RawSplit(attribute=attribute[node_id])
        for op, threshold, child_id in children_by_parent.get(node_id, []):
            raw.children.append((op, threshold, build(child_id)))
        return raw

    return build(root_id)


def build_j48_trees(X_adapt, y_adapt, columns, max_depth):
    try:
        ensure_jvm()
        from weka.classifiers import Classifier
        data = _to_weka_instances(X_adapt, y_adapt, columns)
        clf = Classifier(classname="weka.classifiers.trees.J48", options=["-B"])
        clf.build_classifier(data)
        raw_root = parse_weka_dot(clf.graph)
    except Exception as ex:
        print(f"Warning: Weka J48 failed to fit this trial's D_adapt ({ex}), "
              f"reporting None for every J48/J48-unbounded column on this row.")
        return None, None
    bounded = raw_to_decision_node(raw_root, X_adapt, y_adapt, columns, max_depth)
    unbounded = raw_to_decision_node(raw_root, X_adapt, y_adapt, columns, max_depth=None)
    return bounded, unbounded


def build_reptree_tree(X_adapt, y_adapt, columns, max_depth):
    try:
        ensure_jvm()
        from weka.classifiers import Classifier
        data = _to_weka_instances(X_adapt, y_adapt, columns)
        clf = Classifier(classname="weka.classifiers.trees.REPTree", options=["-L", str(max_depth)])
        clf.build_classifier(data)
        raw_root = parse_weka_dot(clf.graph)
    except Exception as ex:
        print(f"Warning: Weka REPTree failed to fit this trial's D_adapt ({ex}), "
              f"reporting None for every REPTree column on this row.")
        return None
    return raw_to_decision_node(raw_root, X_adapt, y_adapt, columns, max_depth)
