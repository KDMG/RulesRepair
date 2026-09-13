import numpy as np

from keep_remine_prune.tree import Operator
from mining.binary_tree_from_splits import RawSplit, raw_to_decision_node

_r_ready = False
_extract_edges_r = None

_EXTRACT_EDGES_R_SOURCE = """
.extract_edges_for_python <- function(fit) {
  ids <- partykit::nodeids(fit)
  var_names <- names(fit$data)
  rows <- vector("list", length(ids))
  for (i in seq_along(ids)) {
    id <- ids[i]
    node <- partykit::nodeapply(fit, ids = id, FUN = function(n) n)[[1]]
    if (partykit::is.terminal(node)) {
      rows[[i]] <- data.frame(id=id, is_leaf=TRUE, attribute=NA_character_,
                               threshold=NA_real_, left_id=NA_integer_, right_id=NA_integer_,
                               stringsAsFactors=FALSE)
    } else {
      sp <- partykit::split_node(node)
      vid <- partykit::varid_split(sp)
      brk <- partykit::breaks_split(sp)[1]
      kids <- partykit::kids_node(node)
      left_id <- partykit::id_node(kids[[1]])
      right_id <- partykit::id_node(kids[[2]])
      rows[[i]] <- data.frame(id=id, is_leaf=FALSE, attribute=var_names[vid],
                               threshold=brk, left_id=left_id, right_id=right_id,
                               stringsAsFactors=FALSE)
    }
  }
  do.call(rbind, rows)
}
"""


def _ensure_r():
    global _r_ready, _extract_edges_r
    if _r_ready:
        return
    import rpy2.robjects as robjects
    from rpy2.robjects.packages import importr
    importr("partykit")
    robjects.r(_EXTRACT_EDGES_R_SOURCE)
    _extract_edges_r = robjects.globalenv[".extract_edges_for_python"]
    _r_ready = True


def _fit_ctree_edges(X, y, columns, max_depth):
    import pandas as pd
    import rpy2.robjects as robjects
    from rpy2.robjects import pandas2ri, default_converter
    from rpy2.robjects.conversion import localconverter

    _ensure_r()
    df = pd.DataFrame(np.asarray(X, dtype=float), columns=list(columns))
    df["class"] = pd.Categorical(np.asarray(y, dtype=object))

    with localconverter(default_converter + pandas2ri.converter):
        r_df = robjects.conversion.get_conversion().py2rpy(df)
    robjects.globalenv["._ctree_data"] = r_df
    robjects.r(
        "._ctree_fit <- partykit::ctree(class ~ ., data = ._ctree_data, "
        f"control = partykit::ctree_control(maxdepth = {int(max_depth)}))"
    )
    fit = robjects.globalenv["._ctree_fit"]
    r_edges = _extract_edges_r(fit)
    with localconverter(default_converter + pandas2ri.converter):
        edges = robjects.conversion.get_conversion().rpy2py(r_edges)
    return edges


def _edges_to_raw_split(edges):
    edges = edges.set_index("id")
    child_ids = set()
    for _, row in edges.iterrows():
        if not bool(row["is_leaf"]):
            child_ids.add(int(row["left_id"]))
            child_ids.add(int(row["right_id"]))
    all_ids = set(int(i) for i in edges.index)
    roots = all_ids - child_ids
    if len(roots) != 1:
        raise ValueError(f"_edges_to_raw_split: expected exactly 1 root node, found {roots}")
    root_id = next(iter(roots))

    def build(node_id):
        row = edges.loc[node_id]
        if bool(row["is_leaf"]):
            return RawSplit()
        raw = RawSplit(attribute=str(row["attribute"]))
        threshold = float(row["threshold"])
        left_id = int(row["left_id"])
        right_id = int(row["right_id"])
        raw.children.append((Operator.LE, threshold, build(left_id)))
        raw.children.append((Operator.GT, threshold, build(right_id)))
        return raw

    return build(root_id)


def build_ctree_tree(X_adapt, y_adapt, columns, max_depth):
    try:
        edges = _fit_ctree_edges(X_adapt, y_adapt, columns, max_depth)
        raw_root = _edges_to_raw_split(edges)
    except Exception as ex:
        print(f"Warning: R/ctree failed to fit this trial's D_adapt ({ex}), "
              f"reporting None for every CTree column on this row.")
        return None
    return raw_to_decision_node(raw_root, X_adapt, y_adapt, columns, max_depth)
