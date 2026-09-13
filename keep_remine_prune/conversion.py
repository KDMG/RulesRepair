import numpy as np

from .tree import DecisionNode, Condition, Operator


def sklearn_to_tree(sklearn_tree, column_names):
    n_nodes = sklearn_tree.tree_.node_count
    children_left = sklearn_tree.tree_.children_left
    children_right = sklearn_tree.tree_.children_right
    feature = sklearn_tree.tree_.feature
    threshold = sklearn_tree.tree_.threshold

    class_values = sklearn_tree.tree_.value  # shape (n_nodes, 1, n_classes)
    n_node_samples = sklearn_tree.tree_.n_node_samples
    impurities = sklearn_tree.tree_.impurity
    classes = sklearn_tree.classes_

    def node_label_and_value(node_id):
        counts = class_values[node_id][0]
        label = classes[int(np.argmax(counts))]
        total = counts.sum()
        if total > 0:
            counts = counts / total * n_node_samples[node_id]
        value = [int(round(c)) for c in counts]
        return label, value

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, 0)]  # start with the root node id (0) and its depth (0)

    node_map = {} # node_id -> node

    # pass 1 (construct nodes)
    while len(stack) > 0:
        # `pop` ensures each node is only visited once
        node_id, depth = stack.pop()
        node_depth[node_id] = depth

        is_split_node = children_left[node_id] != children_right[node_id]
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))

            # Create Node
            label, value = node_label_and_value(node_id)
            new_node = DecisionNode(label, node_id, value, float(impurities[node_id]))
            node_map[node_id] = new_node
        else:
            is_leaves[node_id] = True

            # Create Leaf Node
            label, value = node_label_and_value(node_id)
            new_node = DecisionNode(label, node_id, value, float(impurities[node_id]))
            node_map[node_id] = new_node

    # pass 2 (link nodes)
    stack = [(0, 0)]  # start with the root node id (0) and its depth (0)

    while len(stack) > 0:
        node_id, depth = stack.pop()

        is_split_node = children_left[node_id] != children_right[node_id]
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))

            # Link children
            node = node_map[node_id]
            left = node_map[children_left[node_id]]
            right = node_map[children_right[node_id]]

            attribute_pos = feature[node_id]
            attribute = column_names[attribute_pos]
            thresh = threshold[node_id]

            left_cond = Condition(
                attribute,
                attribute_pos,
                Operator.LE,
                thresh
            )

            right_cond = Condition(
                attribute,
                attribute_pos,
                Operator.GT,
                thresh
            )

            node.add_child(left_cond, left)
            node.add_child(right_cond, right)

    return node_map[0] # root node
