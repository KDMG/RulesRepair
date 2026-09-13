from keep_remine_prune.tree import DecisionNode  # noqa: F401  (for type/documentation reference only)


def _conditions_match(c1, c2, float_tol=1e-9):
    return (
        c1.attribute == c2.attribute
        and c1.operator == c2.operator
        and abs(c1.threshold - c2.threshold) <= float_tol
    )


def _mark_all_reaudit(node):
    node.needs_reaudit = True
    for child in node.children:
        _mark_all_reaudit(child)


def mark_reaudit_nodes(old_node, new_node, float_tol=1e-9):
    old_leaf = old_node.is_leaf()
    new_leaf = new_node.is_leaf()

    if old_leaf != new_leaf:
        _mark_all_reaudit(new_node)
        return

    if old_leaf and new_leaf:
        new_node.needs_reaudit = (old_node.label != new_node.label)
        return

    same_shape = len(old_node.conditions) == len(new_node.conditions)
    same_conditions = same_shape and all(
        _conditions_match(oc, nc, float_tol) for oc, nc in zip(old_node.conditions, new_node.conditions)
    )
    if not same_conditions:
        _mark_all_reaudit(new_node)
        return

    new_node.needs_reaudit = False
    for old_child, new_child in zip(old_node.children, new_node.children):
        mark_reaudit_nodes(old_child, new_child, float_tol)


def collect_reaudit_nodes(new_node):
    out = []

    def walk(node, path_str):
        if getattr(node, "needs_reaudit", False):
            out.append({
                "node_id": node.node_id,
                "path": path_str or "(root)",
                "is_leaf": node.is_leaf(),
                "label": node.label if node.is_leaf() else None,
            })
        for child in node.children:
            cond = child.find_to_condition()
            child_path = f"{path_str} AND {cond}" if path_str else str(cond)
            walk(child, child_path)

    walk(new_node, "")
    return out


def reaudit_summary(new_node):
    n_total = 0
    n_reaudit = 0

    def walk(node):
        nonlocal n_total, n_reaudit
        n_total += 1
        if getattr(node, "needs_reaudit", False):
            n_reaudit += 1
        for child in node.children:
            walk(child)

    walk(new_node)
    pct = round(100 * n_reaudit / n_total, 1) if n_total else 0.0
    return n_total, n_reaudit, pct
