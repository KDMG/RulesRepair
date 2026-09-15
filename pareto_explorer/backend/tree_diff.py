from ._paths import REPO_ROOT
from mutations.tree_mutations import get_leaf_nodes, root_to_leaf_paths

from .rendering import GuardTree, _condition_text


def _mark_subtree(node, changed_ids):
    if node is None:
        return
    changed_ids.add(node.node_id)
    for child in node.children:
        _mark_subtree(child, changed_ids)



def find_changed_nodes(old_tree, new_tree):
    changed = set()

    def walk(old_node, new_node):
        if old_node is None or new_node is None:
            _mark_subtree(new_node, changed)
            return
        if old_node.is_leaf() != new_node.is_leaf():
            _mark_subtree(new_node, changed)
            return
        if old_node.is_leaf():
            if old_node.label != new_node.label:
                changed.add(new_node.node_id)
            return
        old_cond = old_node.conditions[0] if old_node.conditions else None
        new_cond = new_node.conditions[0] if new_node.conditions else None
        if _condition_text(old_cond) != _condition_text(new_cond):
            _mark_subtree(new_node, changed)
            return
        n = max(len(old_node.children), len(new_node.children))
        for i in range(n):
            old_child = old_node.children[i] if i < len(old_node.children) else None
            new_child = new_node.children[i] if i < len(new_node.children) else None
            walk(old_child, new_child)

    walk(old_tree, new_tree)
    return changed



def unchanged_id_map(old_tree, new_tree):
    mapping = {}

    def walk(old_node, new_node):
        if old_node is None or new_node is None:
            return
        if old_node.is_leaf() != new_node.is_leaf():
            return
        if old_node.is_leaf():
            if old_node.label == new_node.label:
                mapping[new_node.node_id] = old_node.node_id
            return
        old_cond = old_node.conditions[0] if old_node.conditions else None
        new_cond = new_node.conditions[0] if new_node.conditions else None
        if _condition_text(old_cond) != _condition_text(new_cond):
            return
        mapping[new_node.node_id] = old_node.node_id
        n = max(len(old_node.children), len(new_node.children))
        for i in range(n):
            old_child = old_node.children[i] if i < len(old_node.children) else None
            new_child = new_node.children[i] if i < len(new_node.children) else None
            walk(old_child, new_child)

    walk(old_tree, new_tree)
    return mapping



def _max_node_id(tree):
    return max([tree.node_id] + [_max_node_id(c) for c in tree.children])



def display_id_map_for_tree(tree_old, target_tree, id_offset=0):
    mapping = dict(unchanged_id_map(tree_old, target_tree))
    next_id = _max_node_id(tree_old) + 1 + id_offset

    def walk(node):
        nonlocal next_id
        if node.node_id not in mapping:
            mapping[node.node_id] = next_id
            next_id += 1
        for child in node.children:
            walk(child)

    walk(target_tree)
    return mapping



def leaf_choices(tree):
    if isinstance(tree, GuardTree):
        choices = []
        for trans_name, info in sorted(tree.branches.items(), key=lambda kv: kv[1]["index"]):
            choices.append({
                "node_id": info["index"],
                "label": info.get("label") or info.get("tag") or trans_name,
                "path_str": info.get("guard") or "else",
            })
        return choices

    choices = []
    for leaf in get_leaf_nodes(tree):
        conds = []
        node = leaf
        while node is not None and not node.is_root():
            cond = node.find_to_condition()
            if cond is not None:
                conds.append(str(cond))
            node = node.parent
        conds.reverse()
        choices.append({
            "node_id": leaf.node_id,
            "label": leaf.label,
            "path_str": " AND ".join(conds) if conds else "(single leaf)",
        })
    return choices



def forced_ids_for_leaves(tree, leaf_node_ids):
    if not leaf_node_ids:
        return None
    paths = dict(root_to_leaf_paths(tree))
    union = set()
    for leaf_id in leaf_node_ids:
        union |= set(paths[leaf_id])
    return frozenset(union)



def _branch_path_to_node_id(root, target_id):
    if root.node_id == target_id:
        return []
    for i, child in enumerate(root.children):
        sub = _branch_path_to_node_id(child, target_id)
        if sub is not None:
            return [i] + sub
    return None



def mandatory_ids_for_tree(tree_old, target_tree, mandatory_leaf_ids):
    leaf_ids = set()
    path_ids = set()
    if not mandatory_leaf_ids or tree_old is None or target_tree is None:
        return leaf_ids, path_ids
    for original_leaf_id in mandatory_leaf_ids:
        branch_path = _branch_path_to_node_id(tree_old, original_leaf_id)
        if branch_path is None:
            continue
        node = target_tree
        path_ids.add(node.node_id)
        resolved = True
        for i in branch_path:
            if i >= len(node.children):
                resolved = False
                break
            node = node.children[i]
            path_ids.add(node.node_id)
        if resolved:
            leaf_ids.add(node.node_id)
    return leaf_ids, path_ids

