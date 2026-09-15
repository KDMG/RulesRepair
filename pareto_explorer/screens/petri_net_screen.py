from pathlib import Path

from PySide6.QtCore import QThread
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QListWidget, QListWidgetItem, QScrollArea, QProgressBar, QMessageBox,
    QGroupBox, QFormLayout, QFileDialog,
)

import backend
from common import TMP_DIR
from widgets import ClickableImageLabel
from workers import ExtractWorker

class PetriNetScreen(QWidget):
    def __init__(self, on_open_dp, on_theme_toggle=None):
        super().__init__()
        self.on_open_dp = on_open_dp
        self.on_theme_toggle = on_theme_toggle
        self.net_data = None
        self.model = None
        self.observations = None
        self.extract_thread = None
        self.extract_worker = None
        self._extracting = False
        self.dark = False
        self._current_pnml_path = None
        self._current_log_path = None

        layout = QHBoxLayout(self)

        left = QVBoxLayout()
        title_row = QHBoxLayout()
        title_row.addWidget(QLabel("<b>Input</b>"))
        title_row.addStretch(1)
        self.theme_btn = QPushButton("Dark mode")
        if self.on_theme_toggle is not None:
            self.theme_btn.clicked.connect(self.on_theme_toggle)
        title_row.addWidget(self.theme_btn)
        left.addLayout(title_row)

        data_pn_box = QGroupBox("1. Data Petri net")
        data_pn_layout = QVBoxLayout()

        data_pn_layout.addWidget(QLabel("<b>Petri net</b>"))
        form = QFormLayout()
        self.net_combo = QComboBox()
        for path in backend.list_known_pnml_files():
            self.net_combo.addItem(path.parent.name, str(path))
        form.addRow("Known:", self.net_combo)
        data_pn_layout.addLayout(form)
        row = QHBoxLayout()
        load_btn = QPushButton("Load")
        load_btn.clicked.connect(self._load_selected_net)
        row.addWidget(load_btn)
        open_btn = QPushButton("Open...")
        open_btn.clicked.connect(self._open_net_file)
        row.addWidget(open_btn)
        data_pn_layout.addLayout(row)
        self.net_status = QLabel("Not loaded.")
        self.net_status.setWordWrap(True)
        data_pn_layout.addWidget(self.net_status)

        data_pn_layout.addWidget(QLabel("<b>Normative decision trees</b>"))
        model_open_btn = QPushButton("Open...")
        model_open_btn.clicked.connect(self._open_model_file)
        data_pn_layout.addWidget(model_open_btn)
        self.model_status = QLabel("Not loaded.")
        self.model_status.setWordWrap(True)
        data_pn_layout.addWidget(self.model_status)

        data_pn_box.setLayout(data_pn_layout)
        left.addWidget(data_pn_box)

        log_box = QGroupBox("2. Event log (.xes)")
        log_layout = QVBoxLayout()
        log_open_btn = QPushButton("Open...")
        log_open_btn.clicked.connect(self._open_log_file)
        log_layout.addWidget(log_open_btn)
        self.log_progress = QProgressBar()
        self.log_progress.setMinimum(0)
        self.log_progress.setMaximum(0)  # indeterminate
        self.log_progress.setVisible(False)
        log_layout.addWidget(self.log_progress)
        self.log_status = QLabel("Not loaded.")
        self.log_status.setWordWrap(True)
        log_layout.addWidget(self.log_status)
        log_box.setLayout(log_layout)
        left.addWidget(log_box)

        left.addWidget(QLabel("Decision points:"))
        self.dp_list = QListWidget()
        self.dp_list.itemDoubleClicked.connect(self._open_selected)
        left.addWidget(self.dp_list, stretch=1)

        open_dp_btn = QPushButton("Open")
        open_dp_btn.clicked.connect(self._open_selected)
        left.addWidget(open_dp_btn)

        left_widget = QWidget()
        left_widget.setLayout(left)
        left_widget.setMaximumWidth(320)

        right = QVBoxLayout()
        right.addWidget(QLabel("<b>Data Petri net</b>  (click a decision point to open it)"))
        self.net_scroll = QScrollArea()
        self.net_label = ClickableImageLabel(on_click=self._open_dp_by_name)
        self.net_label.setText("No net loaded.")
        self.net_scroll.setWidget(self.net_label)
        self.net_scroll.setWidgetResizable(True)
        right.addWidget(self.net_scroll)
        right_widget = QWidget()
        right_widget.setLayout(right)

        layout.addWidget(left_widget)
        layout.addWidget(right_widget, stretch=1)

    def _load_selected_net(self):
        path = self.net_combo.currentData()
        if path:
            self._load_pnml(path)

    def _open_net_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Petri net", str(backend.REPO_ROOT), "Petri net (*.pnml)")
        if path:
            self._load_pnml(path)

    def _load_pnml(self, path, reset_log=True):
        self._current_pnml_path = path
        self.dp_list.clear()
        if reset_log:
            self.observations = None
            self.log_status.setText("Not loaded.")
        try:
            png_path = TMP_DIR / "net.png"
            self.net_data = backend.open_pnml(path, png_path, dpi=70, dark=self.dark)
            self.net_label.set_image(QPixmap(str(png_path)), boxes=self.net_data["dp_boxes"])
        except Exception as exc:  # noqa: BLE001
            self.net_data = None
            self.net_status.setText(f"Could not load: {exc}")
            self.net_label.clear_image(f"Could not load net:\n{exc}")
            return

        self.net_status.setText(f"Loaded: {Path(path).name}  ({len(self.net_data['decision_points'])} decision point(s))")
        for dp_name in sorted(self.net_data["decision_points"]):
            self.dp_list.addItem(QListWidgetItem(dp_name))

    def set_dark(self, dark):
        self.dark = dark
        if self._current_pnml_path is not None:
            self._load_pnml(self._current_pnml_path, reset_log=False)

    def _open_model_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open model", str(backend.REPO_ROOT), "Model (*.pkl)")
        if not path:
            return
        try:
            self.model = backend.load_model(path)
        except Exception as exc:  # noqa: BLE001
            self.model = None
            self.model_status.setText(f"Could not load: {exc}")
            return
        n_trees = sum(1 for v in self.model.values() if v.get("tree") is not None)
        self.model_status.setText(f"Loaded: {Path(path).name}  ({n_trees} tree(s))")

    def _open_log_file(self):
        if self.net_data is None:
            QMessageBox.information(self, "Log", "Open a Petri net first.")
            return
        path, _ = QFileDialog.getOpenFileName(self, "Open log", str(backend.REPO_ROOT), "Event log (*.xes)")
        if not path:
            return

        self.log_progress.setVisible(True)
        self.log_status.setText("Extracting observations instances (alignment-based)...")
        self._extracting = True

        self.extract_thread = QThread()
        self.extract_worker = ExtractWorker(self.net_data, path)
        self.extract_worker.moveToThread(self.extract_thread)
        self.extract_thread.started.connect(self.extract_worker.run)
        self.extract_worker.finished.connect(self._on_log_finished)
        self.extract_worker.failed.connect(self._on_log_failed)
        self.extract_worker.finished.connect(self.extract_thread.quit)
        self.extract_worker.failed.connect(self.extract_thread.quit)
        self.extract_worker.finished.connect(self.extract_worker.deleteLater)
        self.extract_worker.failed.connect(self.extract_worker.deleteLater)
        self.extract_thread.finished.connect(self.extract_thread.deleteLater)
        self.extract_thread.start()

    def _on_log_finished(self, path, observations):
        self._extracting = False
        self.observations = observations
        self._current_log_path = path
        self.log_progress.setVisible(False)
        n_dp_covered = sum(1 for v in observations.values() if v)
        total_rows = sum(len(v) for v in observations.values())
        self.log_status.setText(
            f"Loaded: {Path(path).name}  ({n_dp_covered} decision point(s) covered, {total_rows} observation(s))"
        )

    def _on_log_failed(self, message):
        self._extracting = False
        self.observations = None
        self.log_progress.setVisible(False)
        self.log_status.setText(f"Could not extract: {message}")
        QMessageBox.critical(self, "Log extraction failed", message)

    def _open_selected(self):
        if self.net_data is None:
            QMessageBox.information(self, "Decision point", "Open a Petri net first.")
            return
        if self._extracting:
            QMessageBox.information(self, "Decision point", "Still extracting observations from the log.")
            return
        item = self.dp_list.currentItem()
        if item is None:
            QMessageBox.information(self, "Decision point", "Select a decision point first.")
            return
        self._open_dp_by_name(item.text())

    def _dataset_name(self):
        path = self._current_log_path or self._current_pnml_path
        if not path:
            return ""
        humanized = Path(path).stem.replace("_", " ").replace("-", " ").strip()
        first_word = humanized.split(" ", 1)[0] if humanized else ""
        return first_word.capitalize()

    def _open_dp_by_name(self, dp_name):
        if self.net_data is None or dp_name not in self.net_data["decision_points"]:
            return
        if self._extracting:
            QMessageBox.information(self, "Decision point", "Still extracting observations from the log.")
            return
        if self.model is None:
            QMessageBox.information(
                self, "Model",
                "Load a model (.pkl) first.",
            )
            return
        dp_info = self.net_data["decision_points"][dp_name]
        dataset_name = self._dataset_name()
        self.on_open_dp(dp_name, dp_info, self.model, self.observations, dataset_name)

