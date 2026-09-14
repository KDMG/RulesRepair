import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pm4py

from mining.extract_decision_points import find_decision_points, declared_variables, fix_final_marking

# pareto_explorer is a local GUI tool, not part of the public repo (see
# .gitignore) -- import it lazily inside render_preview() so this module
# (needed by the core mining pipeline) still works without it. Preview
# rendering is then simply skipped where it's not available.
DEFAULT_DPI = 130


def build_pn_normative(xes_path, out_pnml, noise_threshold=0.2, multi_processing=False):
    log = pm4py.read_xes(str(xes_path))
    net, im, fm = pm4py.discover_petri_net_inductive(
        log, noise_threshold=noise_threshold, multi_processing=multi_processing,
    )
    fm = fix_final_marking(net, fm)
    pm4py.write_pnml(net, im, fm, str(out_pnml))
    return net, im, fm


def render_preview(net, im, fm, out_png, pnml_path=None, dpi=DEFAULT_DPI):
    try:
        from pareto_explorer.backend import _render_petri_net_dot
    except ImportError:
        print("Skipping preview: pareto_explorer (local GUI tool, not part of the public repo) is not installed.")
        return None
    decision_point_names = set()
    if pnml_path is not None:
        valid_vars = declared_variables(pnml_path)
        decision_points, _ = find_decision_points(net, valid_vars)
        decision_point_names = set(decision_points)
    dot, _ = _render_petri_net_dot(net, im, fm, decision_point_names=decision_point_names)
    dot.graph_attr["pad"] = "0"
    dot.graph_attr["dpi"] = str(dpi)
    dot.format = "png"
    return dot.render(filename=str(Path(out_png).with_suffix("")), cleanup=True)


def main():
    parser = argparse.ArgumentParser(
        description="Mine pn_normative.pnml directly via pm4py's Inductive Miner infrequent -- no external tool needed."
    )
    parser.add_argument("--xes", required=True, help="<dataset>_normative.xes (the 'old' split, from mining/split_log.py)")
    parser.add_argument("--out", required=True, help="Output .pnml path (e.g. datasets/<dataset>_cut/pn_normative.pnml)")
    parser.add_argument("--noise", type=float, default=0.2, help="Noise threshold for Inductive Miner infrequent (default: 0.2)")
    parser.add_argument("--preview", default=None, help="PNG preview path (default: --out with .png instead of .pnml)")
    parser.add_argument("--no-preview", action="store_true", help="Skip rendering the PNG preview")
    args = parser.parse_args()

    net, im, fm = build_pn_normative(args.xes, args.out, noise_threshold=args.noise)
    print(f"Mined: {len(net.places)} place(s), {len(net.transitions)} transition(s) -> {args.out}")

    if not args.no_preview:
        preview_path = args.preview or str(Path(args.out).with_suffix(".png"))
        if render_preview(net, im, fm, preview_path, pnml_path=args.out) is not None:
            print(f"Preview -> {preview_path}")


if __name__ == "__main__":
    main()
