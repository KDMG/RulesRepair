import pandas as pd

from PySide6.QtCore import Qt, QThread
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QListWidget,
    QListWidgetItem, QAbstractItemView, QScrollArea, QProgressBar, QMessageBox,
)

import backend
from common import TMP_DIR, _dp_display_html
from widgets import ClickableImageLabel
from workers import ComputeWorker

class RuleSelectionScreen(QWidget):
    def __init__(self, on_back, on_computed, on_theme_toggle=None):
        super().__init__()
        self.on_back = on_back
        self.on_computed = on_computed
        self.on_theme_toggle = on_theme_toggle
        self.dp_name = None
        self.dp_info = None
        self.model = None
        self.tree = None
        self.thread = None
        self.worker = None
        self._df_adapt = None
        self._df_test = None
        self._base_ready = False
        self._worker_mandatory_ids = []
        self.dataset_name = ""
        self.dark = False

        layout = QHBoxLayout(self)

        left = QVBoxLayout()
        top_row = QHBoxLayout()
        back_btn = QPushButton("Back")
        back_btn.clicked.connect(self.on_back)
        top_row.addWidget(back_btn)
        top_row.addStretch(1)
        self.theme_btn = QPushButton("Dark mode")
        if self.on_theme_toggle is not None:
            self.theme_btn.clicked.connect(self.on_theme_toggle)
        top_row.addWidget(self.theme_btn)
        left.addLayout(top_row)

        self.title_label = QLabel("<b>Decision point</b>")
        left.addWidget(self.title_label)

        self.source_label = QLabel("")
        self.source_label.setWordWrap(True)
        left.addWidget(self.source_label)

        left.addWidget(QLabel("Rules -- select one or more as mandatory:"))
        self.leaf_list = QListWidget()
        self.leaf_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.leaf_list.itemSelectionChanged.connect(self._render_tree)
        left.addWidget(self.leaf_list)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        left.addWidget(self.progress)

        self.compute_btn = QPushButton("Compute front ->")
        self.compute_btn.clicked.connect(self._start_compute)
        left.addWidget(self.compute_btn)

        left_widget = QWidget()
        left_widget.setLayout(left)
        left_widget.setMaximumWidth(360)

        right = QVBoxLayout()
        right.addWidget(QLabel("<b>Tree</b>  (click a leaf to toggle it as mandatory)"))
        self.tree_scroll = QScrollArea()
        self.tree_label = ClickableImageLabel(on_click=self._toggle_leaf)
        self.tree_scroll.setWidget(self.tree_label)
        self.tree_scroll.setWidgetResizable(True)
        right.addWidget(self.tree_scroll)
        right_widget = QWidget()
        right_widget.setLayout(right)

        layout.addWidget(left_widget)
        layout.addWidget(right_widget, stretch=1)

    def _toggle_leaf(self, leaf_id):
        for i in range(self.leaf_list.count()):
            item = self.leaf_list.item(i)
            if item.data(Qt.UserRole) == leaf_id:
                item.setSelected(not item.isSelected())
                return

    def open_dp(self, dp_name, dp_info, model, observations, dataset_name=""):
        self.dp_name = dp_name
        self.dp_info = dp_info
        self.model = model
        self.dataset_name = dataset_name
        self.leaf_list.clear()
        prefix = f"{dataset_name}, " if dataset_name else ""
        self.title_label.setText(f"<b>{prefix}decision point {_dp_display_html(dp_name)}</b>")

        tree, source = backend.get_dp_tree(dp_name, dp_info, model)

        try:
            base_tree, df_adapt = backend.prepare_normative_base(dp_name, model, observations)
        except backend.MissingDataError as exc:
            self.tree = tree
            self._df_adapt = None
            self._df_test = None
            self._base_ready = False
            source_text = "trained tree (T_old)" if source == "trained" else "guard-only tree (from the Petri net)"
            self.source_label.setText(f"Showing {source_text}. Not computable here: {exc}")
            self.compute_btn.setEnabled(False)
            self.compute_btn.setToolTip(str(exc))
        else:
            self.tree = base_tree
            self._df_adapt = df_adapt
            self._base_ready = True
            test_path = backend._find_offline_shared_test(dp_name)
            self._df_test = pd.read_csv(test_path) if test_path is not None else None
            test_note = f", {len(self._df_test)} held-out test row(s) found" if self._df_test is not None else ""
            self.source_label.setText(
                f"Trained tree (T_old) -- {len(df_adapt)} observation(s) used as D_adapt{test_note}. Ready to compute."
            )
            self.compute_btn.setEnabled(True)
            self.compute_btn.setToolTip("")

        self._render_tree()

        for choice in backend.leaf_choices(self.tree):
            item = QListWidgetItem(f"Leaf {choice['node_id']}: {choice['label']}")
            item.setData(Qt.UserRole, choice["node_id"])
            item.setToolTip(choice["path_str"])
            self.leaf_list.addItem(item)

    def set_dark(self, dark):
        self.dark = dark
        self._render_tree()

    def _render_tree(self):
        if self.tree is None:
            return
        selected_ids = [item.data(Qt.UserRole) for item in self.leaf_list.selectedItems()]
        path_ids = None
        if selected_ids and not isinstance(self.tree, backend.GuardTree):
            try:
                path_ids = backend.forced_ids_for_leaves(self.tree, selected_ids)
            except Exception:  # noqa: BLE001 -- best-effort highlight, never blocks the render itself
                path_ids = None
        png_stem = TMP_DIR / f"tree_{self.dp_name}"
        try:
            png_path, leaf_boxes = backend.render_tree_png(
                self.tree, png_stem, dark=self.dark, highlight_path_ids=path_ids,
            )
            self.tree_label.set_image(QPixmap(str(png_path)), boxes=leaf_boxes)
        except Exception as exc:  # noqa: BLE001
            self.tree_label.clear_image(f"Could not draw tree:\n{exc}")

    def _start_compute(self):
        if not self._base_ready:
            QMessageBox.warning(self, "Compute", "No usable data for this decision point -- load a model and a log that cover it.")
            return
        selected_ids = [item.data(Qt.UserRole) for item in self.leaf_list.selectedItems()]
        self._worker_mandatory_ids = selected_ids  # stashed for _on_finished(), see its own comment

        self.compute_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setValue(0)

        self.thread = QThread()
        self.worker = ComputeWorker(self.dp_name, self.model, selected_ids, self.tree, self._df_adapt, df_test=self._df_test)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.start()

    def _on_progress(self, done, total):
        self.progress.setMaximum(total)
        self.progress.setValue(done)

    def _on_finished(self, result):
        df, cart_point, normative_point = result
        self.compute_btn.setEnabled(True)
        self.progress.setVisible(False)
        mandatory_leaf_ids = set(self._worker_mandatory_ids)
        self.on_computed(self.dp_name, self.tree, df, cart_point, normative_point, mandatory_leaf_ids, self._df_adapt, self.dataset_name, self._df_test)

    def _on_failed(self, message):
        self.compute_btn.setEnabled(True)
        self.progress.setVisible(False)
        QMessageBox.critical(self, "Compute failed", message)
