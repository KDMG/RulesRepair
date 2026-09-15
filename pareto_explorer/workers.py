from PySide6.QtCore import QObject, Signal

import backend

class ExtractWorker(QObject):
    finished = Signal(str, object)
    failed = Signal(str)

    def __init__(self, net_data, xes_path):
        super().__init__()
        self.net_data = net_data
        self.xes_path = xes_path

    def run(self):
        try:
            observations = backend.extract_observations(self.net_data, self.xes_path)
            self.finished.emit(self.xes_path, observations)
        except Exception as exc:
            self.failed.emit(str(exc))


class ComputeWorker(QObject):
    progress = Signal(int, int)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, dp_name, model, mandatory_leaf_ids, base_tree, df_adapt, df_test=None):
        super().__init__()
        self.dp_name = dp_name
        self.model = model
        self.mandatory_leaf_ids = mandatory_leaf_ids
        self.base_tree = base_tree
        self.df_adapt = df_adapt
        self.df_test = df_test

    def run(self):
        try:
            result = backend.compute_live_pareto_front(
                self.dp_name, self.model, self.mandatory_leaf_ids,
                self.base_tree, self.df_adapt, df_test_raw=self.df_test,
                progress_cb=lambda done, total: self.progress.emit(done, total),
            )
            self.finished.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class BaselineWorker(QObject):
    finished = Signal(str, str, object)
    failed = Signal(str, str)

    def __init__(self, prefix, dp_name, tree_old, df_adapt, df_test=None):
        super().__init__()
        self.prefix = prefix
        self.dp_name = dp_name
        self.tree_old = tree_old
        self.df_adapt = df_adapt
        self.df_test = df_test

    def run(self):
        try:
            point = backend.compute_baseline_point_for_tree_isolated(self.prefix, self.tree_old, self.df_adapt, df_test_raw=self.df_test)
            self.finished.emit(self.prefix, self.dp_name, point)
        except Exception as exc:
            self.failed.emit(self.dp_name, str(exc))
