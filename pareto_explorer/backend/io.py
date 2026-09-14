import pickle
from pathlib import Path

import pm4py

from ._paths import REPO_ROOT
from mining.extract_decision_points import find_decision_points, declared_variables, fix_final_marking, process_log

from .rendering import DEFAULT_DPI, GuardTree, _node_boxes_from_plain, _render_petri_net_dot


def list_known_pnml_files():
    return sorted(REPO_ROOT.glob("datasets/*_cut/pn_normative.pnml"))



def open_pnml(pnml_path, out_png, dpi=DEFAULT_DPI, dark=False):
    net, im, fm = pm4py.read_pnml(str(pnml_path))
    fm = fix_final_marking(net, fm)

    valid_vars = declared_variables(pnml_path)
    decision_points, trans_to_dp = find_decision_points(net, valid_vars)

    dot, place_id_map = _render_petri_net_dot(net, im, fm, decision_point_names=set(decision_points), dark=dark)
    dot.graph_attr["pad"] = "0"
    dot.graph_attr["dpi"] = str(dpi)
    dot.format = "png"
    dot.render(filename=str(Path(out_png).with_suffix("")), cleanup=True)
    node_boxes = _node_boxes_from_plain(dot, dpi=dpi)
    place_boxes = {name: node_boxes[dot_name] for dot_name, name in place_id_map.items() if dot_name in node_boxes}
    dp_boxes = {name: box for name, box in place_boxes.items() if name in decision_points}

    return {
        "net": net, "im": im, "fm": fm,
        "decision_points": decision_points, "trans_to_dp": trans_to_dp,
        "dp_boxes": dp_boxes,
    }



class _NumpyCompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError):
            if module.startswith("numpy._core"):
                alt_module = module.replace("numpy._core", "numpy.core", 1)
            elif module.startswith("numpy.core"):
                alt_module = module.replace("numpy.core", "numpy._core", 1)
            else:
                raise
            return super().find_class(alt_module, name)



def load_model(pkl_path):
    with open(pkl_path, "rb") as f:
        return _NumpyCompatUnpickler(f).load()



def extract_observations(net_data, xes_path, max_traces=None):
    log = pm4py.read_xes(str(xes_path), return_legacy_log_object=True)
    return process_log(
        net_data["net"], net_data["im"], net_data["fm"], log,
        net_data["decision_points"], net_data["trans_to_dp"], max_traces=max_traces,
    )



def get_dp_tree(dp_name, dp_info, model):
    if model is not None:
        entry = model.get(dp_name)
        if entry is not None and entry.get("tree") is not None:
            return entry["tree"], "trained"
    return GuardTree(dp_info.place_name, dp_info.branches), "guard"

