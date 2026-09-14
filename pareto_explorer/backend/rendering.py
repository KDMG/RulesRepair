import graphviz

DEFAULT_DPI = 130
DECISION_POINT_FILLCOLOR = "#ffcc66"


class GuardTree:
    def __init__(self, place_name, branches):
        self.place_name = place_name
        self.branches = branches  # dict: trans_name -> {index, label, guard, tag}

    def plot(self, show_ids=True, dark=False):
        text_color = "#e8e8e8" if dark else "black"
        dot = graphviz.Digraph("guard_tree")
        dot.graph_attr["bgcolor"] = "transparent"
        dot.attr("node", fontcolor=text_color, color=text_color)
        dot.attr("edge", color=text_color, fontcolor=text_color)
        dot.node("root", _esc(self.place_name))
        leaf_id_map = {}
        for trans_name, info in sorted(self.branches.items(), key=lambda kv: kv[1]["index"]):
            dot_name = f"b{info['index']}"
            label = _esc(info.get("label") or info.get("tag") or trans_name)
            leaf_label = f"l<SUB>{info['index']}</SUB>: {label}" if show_ids else label
            dot.node(dot_name, f"<{leaf_label}>")
            dot.edge("root", dot_name, _esc(info["guard"]) if info.get("guard") else "False")
            leaf_id_map[dot_name] = info["index"]
        return dot, leaf_id_map


def _esc(text):
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _condition_text(cond):
    if cond is None:
        return "?"
    threshold = cond.threshold
    threshold_str = f"{threshold:g}" if isinstance(threshold, float) else str(threshold)
    return _esc(f"{cond.attribute} {cond.operator} {threshold_str}")



def _build_decision_dot(tree, show_ids=True, dark=False, highlight_change_ids=None,
                         highlight_mandatory_ids=None, highlight_path_ids=None,
                         display_id_map=None):
    text_color = "#e8e8e8" if dark else "black"
    dot = graphviz.Digraph("tree")
    dot.graph_attr["bgcolor"] = "transparent"
    dot.attr("node", fontcolor=text_color, color=text_color, fontsize="20")
    dot.attr("edge", color=text_color, fontcolor=text_color, fontsize="16")
    leaf_id_map = {}
    compare_to_normative = highlight_change_ids is not None
    highlight_change_ids = highlight_change_ids or set()
    highlight_mandatory_ids = highlight_mandatory_ids or set()
    highlight_path_ids = highlight_path_ids or set()
    display_id_map = display_id_map or {}

    def _color_for(node):
        if node.node_id in highlight_mandatory_ids or node.node_id in highlight_path_ids:
            return "#ff3b30"
        if compare_to_normative:
            different = node.node_id in highlight_change_ids
            return "#9c27b0" if different else "#ff9800"
        return None

    def add(node):
        node_id = str(node.node_id)
        display_id = display_id_map.get(node.node_id, node.node_id)
        if node.is_leaf():
            content = _esc(node.label)
            label = f"l<SUB>{display_id}</SUB>: {content}" if show_ids else content
            leaf_id_map[node_id] = node.node_id
        else:
            cond = node.conditions[0] if node.conditions else None
            content = _condition_text(cond)
            label = f"n<SUB>{display_id}</SUB>: {content}" if show_ids else content
        color = _color_for(node)
        node_kwargs = {"color": color, "penwidth": "3"} if color else {}
        dot.node(node_id, f"<{label}>", **node_kwargs)
        for i, child in enumerate(node.children):
            edge_label = "true" if i == 0 else ("false" if i == 1 else str(i))
            add(child)
            child_color = _color_for(child)
            edge_kwargs = {"color": child_color, "penwidth": "3"} if child_color else {}
            dot.edge(node_id, str(child.node_id), edge_label, **edge_kwargs)

    add(tree)
    return dot, leaf_id_map



def _node_boxes_from_plain(dot, dpi=DEFAULT_DPI):
    raw = dot.pipe(format="plain").decode("utf-8", errors="replace")
    height_in = None
    node_lines = []
    for line in raw.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "graph":
            height_in = float(parts[3])
        elif parts[0] == "node":
            node_lines.append(parts)
    if height_in is None:
        return {}
    height_px = height_in * dpi
    boxes = {}
    for parts in node_lines:
        name = parts[1].strip('"')
        x_in, y_in, w_in, h_in = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
        cx, cy = x_in * dpi, height_px - y_in * dpi
        w, h = w_in * dpi, h_in * dpi
        boxes[name] = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
    return boxes



def render_tree_png(tree, out_png_stem, show_ids=True, dpi=DEFAULT_DPI, dark=False,
                     highlight_change_ids=None, highlight_mandatory_ids=None,
                     highlight_path_ids=None, display_id_map=None):
    if isinstance(tree, GuardTree):
        dot, leaf_id_map = tree.plot(show_ids=show_ids, dark=dark)
    else:
        dot, leaf_id_map = _build_decision_dot(
            tree, show_ids=show_ids, dark=dark,
            highlight_change_ids=highlight_change_ids, highlight_mandatory_ids=highlight_mandatory_ids,
            highlight_path_ids=highlight_path_ids, display_id_map=display_id_map,
        )
    dot.graph_attr["pad"] = "0"
    dot.graph_attr["dpi"] = str(dpi)
    dot.format = "png"
    png_path = dot.render(filename=str(out_png_stem), cleanup=True)
    node_boxes = _node_boxes_from_plain(dot, dpi=dpi)
    leaf_boxes = {leaf_id: node_boxes[dot_name] for dot_name, leaf_id in leaf_id_map.items() if dot_name in node_boxes}
    return png_path, leaf_boxes



def _render_petri_net_dot(net, im, fm, decision_point_names=None, dark=False):
    decision_point_names = decision_point_names or set()
    text_color = "#e8e8e8" if dark else "black"
    dot = graphviz.Digraph("petri_net")
    dot.graph_attr["bgcolor"] = "transparent"
    dot.attr(rankdir="LR", fontsize="12")
    dot.attr("node", fontsize="12", fontcolor=text_color, color=text_color)
    dot.attr("edge", color=text_color)

    place_id_map = {}
    for p in net.places:
        if p.name in decision_point_names:
            # black text reads fine on this orange on both themes.
            style, fillcolor, fontcolor = "filled", DECISION_POINT_FILLCOLOR, "black"
        elif p in im or p in fm:
            style, fillcolor, fontcolor = "filled", "lightgray", "black"
        else:
            style, fillcolor, fontcolor = "", "none", text_color
        dot_name = str(id(p))
        dot.node(dot_name, _esc(p.name), shape="ellipse", style=style, fillcolor=fillcolor, fontcolor=fontcolor)
        place_id_map[dot_name] = p.name

    for t in net.transitions:
        if t.label is not None:
            dot.node(str(id(t)), _esc(t.label), shape="box", style="", fillcolor="none", fontcolor=text_color)
        else:
            dot.node(str(id(t)), _esc(t.name), shape="box", style="filled", fillcolor="black", fontcolor="white")

    for a in net.arcs:
        dot.edge(str(id(a.source)), str(id(a.target)))

    return dot, place_id_map

