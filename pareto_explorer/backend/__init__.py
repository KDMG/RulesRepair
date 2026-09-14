"""
pareto_explorer/backend/ -- backend for the interactive Pareto-front
explorer GUI (pareto_explorer/app.py).

Split (2026 reorganization, no behavior change) into one module per
concern:
  _paths.py      -- REPO_ROOT + sys.path setup, shared by every submodule
                     below that imports a repo-level package.
  rendering.py   -- GuardTree + every graphviz dot/PNG builder.
  tree_diff.py   -- T_old-vs-repaired-tree structural comparisons, leaf/
                     mandatory-constraint helpers.
  io.py          -- the 3 explicit inputs: Petri net (.pnml), model (.pkl),
                     log (.xes).
  live_repair.py -- the live alpha/beta repair sweep + on-demand CART/J48/
                     REPTree baselines.

This __init__ re-exports every name app.py (and mining/build_pn_normative_
pm4py.py's optional preview, which does `from pareto_explorer.backend
import _render_petri_net_dot`) already reaches through the flat `backend.X`
attribute access -- so every existing caller keeps working unchanged; only
this file's own internals moved.

Reuses this repo's own validated primitives -- log-to-decision-point
extraction, Keep-Remine-Prune's own grow_tree(), Jaccard rule-set
similarity, the CART "remining" baseline, and the same non-dominance
definition already used throughout analysis/ -- rather than reimplementing
any of them. No new science: only a different (interactive, single-
decision-point, live) orchestration of it.
"""
from ._paths import REPO_ROOT

from .rendering import (
    DEFAULT_DPI, DECISION_POINT_FILLCOLOR,
    GuardTree, render_tree_png,
    _esc, _condition_text, _build_decision_dot, _node_boxes_from_plain, _render_petri_net_dot,
)
from .tree_diff import (
    find_changed_nodes, unchanged_id_map, display_id_map_for_tree, mandatory_ids_for_tree,
    leaf_choices, forced_ids_for_leaves,
)
from .io import (
    list_known_pnml_files, open_pnml, load_model, extract_observations, get_dp_tree,
    _NumpyCompatUnpickler,
)
from .live_repair import (
    DEFAULT_MAX_DEPTH, DEFAULT_ALPHAS, DEFAULT_BETAS,
    MissingDataError, BASELINE_LABELS, BASELINE_BUILDERS,
    compute_baseline_point, compute_baseline_point_for_tree,
    prepare_normative_base, split_log_train_test, compute_live_pareto_front,
    _find_offline_shared_train, _find_offline_shared_test,
)

__all__ = [
    "REPO_ROOT",
    "DEFAULT_DPI", "DECISION_POINT_FILLCOLOR", "GuardTree", "render_tree_png",
    "find_changed_nodes", "unchanged_id_map", "display_id_map_for_tree", "mandatory_ids_for_tree",
    "leaf_choices", "forced_ids_for_leaves",
    "list_known_pnml_files", "open_pnml", "load_model", "extract_observations", "get_dp_tree",
    "DEFAULT_MAX_DEPTH", "DEFAULT_ALPHAS", "DEFAULT_BETAS",
    "MissingDataError", "BASELINE_LABELS", "BASELINE_BUILDERS",
    "compute_baseline_point", "compute_baseline_point_for_tree",
    "prepare_normative_base", "split_log_train_test", "compute_live_pareto_front",
]
