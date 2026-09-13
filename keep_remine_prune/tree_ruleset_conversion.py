from .similar_tree import Rule, Ruleset

def tuple_tree_conversion(tree):
    rules = []

    def walk(node, path_conditions):
        if node.is_leaf():
            rules.append(Rule(node.label, list(path_conditions)))
            return
        for condition, child in zip(node.conditions, node.children):
            walk(child, path_conditions + [condition])

    walk(tree, [])
    return Ruleset(rules)
