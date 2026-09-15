import numpy as np

from PySide6.QtCore import Qt, QThread, QPoint
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QSlider, QScrollArea, QSplitter, QGroupBox, QFormLayout, QToolTip,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

import backend
from common import TMP_DIR, _dp_display_html
from widgets import FitImageLabel
from workers import BaselineWorker


class FrontExplorerScreen(QWidget):
    BASELINE_PREFIXES = ["cart", "j48", "reptree"]

    def __init__(self, on_back, on_theme_toggle=None):
        super().__init__()
        self.on_back = on_back
        self.on_theme_toggle = on_theme_toggle
        self.dp_name = None
        self.tree_old = None
        self.df = None
        self.df_adapt = None
        self.df_test = None
        self.normative_point = None
        self.dataset_name = ""
        self.mandatory_leaf_ids = set()
        self.baseline_points = {}
        self.baseline_prefix = "cart"
        self._filtered_df = None
        self._hover_marker_a = None
        self._hover_marker_b = None
        self._selected_marker_a = None
        self._selected_marker_b = None
        self._last_shown_row = None
        self._baseline_threads = []
        self.dark = False

        layout = QVBoxLayout(self)

        top = QHBoxLayout()
        back_btn = QPushButton("Back")
        back_btn.clicked.connect(self.on_back)
        top.addWidget(back_btn)
        self.title_label = QLabel("<b>Decision point</b>")
        self.title_label.setAlignment(Qt.AlignCenter)
        top.addStretch(1)
        top.addWidget(self.title_label)
        top.addStretch(1)
        self.theme_btn = QPushButton("Dark mode")
        if self.on_theme_toggle is not None:
            self.theme_btn.clicked.connect(self.on_theme_toggle)
        top.addWidget(self.theme_btn)
        layout.addLayout(top)

        top_splitter = QSplitter(Qt.Horizontal)

        left = QWidget()
        left_layout = QVBoxLayout(left)

        filters_box = QGroupBox("Preferences")
        filters_layout = QFormLayout()

        self.min_acc_slider, self.min_acc_label = self._make_slider()
        filters_layout.addRow("Accuracy min:", self._with_label(self.min_acc_slider, self.min_acc_label))

        self.min_simpl_slider, self.min_simpl_label = self._make_slider()
        filters_layout.addRow("Simplicity min:", self._with_label(self.min_simpl_slider, self.min_simpl_label))

        self.min_jac_slider, self.min_jac_label = self._make_slider()
        filters_layout.addRow("Similarity min:", self._with_label(self.min_jac_slider, self.min_jac_label))

        filters_box.setLayout(filters_layout)
        left_layout.addWidget(filters_box)

        for slider in (self.min_acc_slider, self.min_simpl_slider, self.min_jac_slider):
            slider.valueChanged.connect(self._refresh)

        baseline_box = QGroupBox("Rediscovering")
        baseline_layout = QHBoxLayout()
        baseline_layout.addWidget(QLabel("Compare against:"))
        self.baseline_combo = QComboBox()
        for prefix in self.BASELINE_PREFIXES:
            self.baseline_combo.addItem(backend.BASELINE_LABELS[prefix], prefix)
        self.baseline_combo.currentIndexChanged.connect(self._on_baseline_selected)
        baseline_layout.addWidget(self.baseline_combo)
        self.baseline_status = QLabel("")
        self.baseline_status.setWordWrap(True)
        baseline_layout.addWidget(self.baseline_status, stretch=1)
        baseline_box.setLayout(baseline_layout)
        left_layout.addWidget(baseline_box)

        legend_box = QGroupBox("Legend")
        legend_layout = QVBoxLayout()
        for color, text_ in (
            ("#ff9800", "same as Normative"),
            ("#9c27b0", "different from Normative"),
            ("#ff3b30", "mandatory policy"),
        ):
            row = QLabel(f'<span style="color:{color}">&#9679;</span> {text_}')
            legend_layout.addWidget(row)
        legend_box.setLayout(legend_layout)
        left_layout.addWidget(legend_box)
        left_layout.addStretch(1)
        left.setMaximumWidth(360)

        right = QWidget()
        right_layout = QVBoxLayout(right)

        self.figure = Figure(figsize=(6, 7))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.figure.patch.set_alpha(0.0)
        self.canvas.setStyleSheet("background-color: transparent;")
        self.canvas.setAttribute(Qt.WA_TranslucentBackground)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.ax_a = self.figure.add_subplot(211)
        self.ax_b = self.figure.add_subplot(212, sharex=self.ax_a)
        self.canvas.mpl_connect("pick_event", self._on_pick)
        self.canvas.mpl_connect("motion_notify_event", self._on_hover)
        self._toolbar_layout = right_layout
        right_layout.addWidget(self.toolbar)
        right_layout.addWidget(self.canvas, stretch=1)

        top_splitter.addWidget(left)
        top_splitter.addWidget(right)
        top_splitter.setStretchFactor(0, 1)
        top_splitter.setStretchFactor(1, 3)
        layout.addWidget(top_splitter, stretch=1)

        trees_row = QHBoxLayout()
        self.normative_panel, self.normative_label, self.normative_metrics_label = self._make_tree_panel("Normative")
        self.repaired_panel, self.repaired_label, self.repaired_metrics_label = self._make_tree_panel("Repaired")
        self.rediscovered_panel, self.rediscovered_label, self.rediscovered_metrics_label = self._make_tree_panel("Rediscovered")
        trees_row.addWidget(self.normative_panel)
        trees_row.addWidget(self.repaired_panel)
        trees_row.addWidget(self.rediscovered_panel)
        layout.addLayout(trees_row, stretch=1)

    def _make_tree_panel(self, title):
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(2, 2, 2, 2)
        label_title = QLabel(f'<span style="font-size:18pt;"><b>{title}</b></span>')
        label_title.setWordWrap(True)
        v.addWidget(label_title)
        metrics_label = QLabel(self._metrics_text(None, None, None))
        metrics_label.setWordWrap(True)
        v.addWidget(metrics_label)
        scroll = QScrollArea()
        label = FitImageLabel()
        label.setText("--")
        scroll.setWidget(label)
        scroll.setWidgetResizable(True)
        v.addWidget(scroll, stretch=1)
        return w, label, metrics_label

    @staticmethod
    def _metrics_text(acc, simpl, jac):
        def fmt(v):
            return f"{v:.3f}" if v is not None else "--"
        return (
            '<span style="font-size:20pt;">'
            f"Accuracy: {fmt(acc)}&nbsp;&nbsp;&nbsp;Simplicity: {fmt(simpl)}&nbsp;&nbsp;&nbsp;Similarity: {fmt(jac)}"
            "</span>"
        )

    @staticmethod
    def _make_slider():
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(100)
        slider.setValue(0)
        slider.setTracking(True)
        label = QLabel("0.00")
        return slider, label

    @staticmethod
    def _with_label(slider, label):
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(slider)
        h.addWidget(label)
        return w

    def open_front(self, dp_name, tree_old, df, cart_point, normative_point, mandatory_leaf_ids, df_adapt, dataset_name="", df_test=None):
        self.dp_name = dp_name
        self.tree_old = tree_old
        self.df = df
        self.df_adapt = df_adapt
        self.df_test = df_test
        self.normative_point = normative_point
        self.dataset_name = dataset_name
        self.mandatory_leaf_ids = set(mandatory_leaf_ids or [])
        self.baseline_points = {"cart": cart_point}
        self.baseline_prefix = "cart"
        self.baseline_combo.blockSignals(True)
        self.baseline_combo.setCurrentIndex(0)
        self.baseline_combo.blockSignals(False)
        self.baseline_combo.setEnabled(True)
        self.baseline_status.setText("")
        prefix = f"{dataset_name}, " if dataset_name else ""
        self.title_label.setText(f'<span style="font-size:20pt;"><b>{prefix}decision point {_dp_display_html(dp_name)}</b></span>')
        for slider in (self.min_acc_slider, self.min_simpl_slider, self.min_jac_slider):
            slider.setValue(0)
        self._last_shown_row = None
        self.repaired_label.clear_pixmap("--")
        self.repaired_metrics_label.setText(self._metrics_text(None, None, None))
        self._show_normative_tree()
        self._refresh()
        self._update_baseline_display()

    def _refresh(self):
        if self.df is None:
            return
        min_acc = self.min_acc_slider.value() / 100.0
        min_simpl = self.min_simpl_slider.value() / 100.0
        min_jac = self.min_jac_slider.value() / 100.0
        self.min_acc_label.setText(f"{min_acc:.2f}")
        self.min_simpl_label.setText(f"{min_simpl:.2f}")
        self.min_jac_label.setText(f"{min_jac:.2f}")

        df = self.df
        mask = (
            (df["accuracy_display"] >= min_acc) & (df["simplicity"] >= min_simpl)
            & (df["jaccard"] >= min_jac) & df["is_pareto"]
        )
        self._filtered_df = df[mask].reset_index(drop=True)

        self._redraw_scatter()

    def _redraw_scatter(self):
        df = self._filtered_df
        if df is None:
            return
        baseline = self.baseline_points.get(self.baseline_prefix)
        baseline_label = f"{backend.BASELINE_LABELS[self.baseline_prefix]} (rediscovering)"
        if baseline is not None:
            min_acc = self.min_acc_slider.value() / 100.0
            min_simpl = self.min_simpl_slider.value() / 100.0
            min_jac = self.min_jac_slider.value() / 100.0
            if (
                baseline["accuracy_display"] < min_acc
                or baseline["simplicity"] < min_simpl
                or baseline["jaccard"] < min_jac
            ):
                baseline = None

        self.ax_a.clear()
        self.ax_b.clear()
        self.ax_a.patch.set_alpha(0.0)
        self.ax_b.patch.set_alpha(0.0)
        if len(df):
            self.ax_a.scatter(
                df["simplicity"], df["jaccard"], c="#1f77b4",
                picker=True, s=40, label="RulesRepair front points", zorder=4,
            )
            self.ax_b.scatter(
                df["simplicity"], df["accuracy_display"], c="#1f77b4",
                picker=True, s=40, label="RulesRepair front points", zorder=4,
            )

        if baseline is not None:
            self.ax_a.scatter(
                [baseline["simplicity"]], [baseline["jaccard"]],
                marker="*", s=220, c="#1565c0", edgecolors="black", label=baseline_label, zorder=5,
            )
            self.ax_b.scatter(
                [baseline["simplicity"]], [baseline["accuracy_display"]],
                marker="*", s=220, c="#1565c0", edgecolors="black", label=baseline_label, zorder=5,
            )

        if len(df) or baseline is not None:
            self.ax_a.legend(loc="upper left", fontsize=8)
            self.ax_b.legend(loc="upper left", fontsize=8)

        self.ax_a.tick_params(labelbottom=False)
        self.ax_a.set_ylabel("Similarity")
        self.ax_a.set_xlim(-0.02, 1.02)
        self.ax_a.set_ylim(-0.02, 1.02)

        self.ax_b.set_xlabel("Simplicity")
        self.ax_b.set_ylabel("Accuracy")
        self.ax_b.set_xlim(-0.02, 1.02)
        self.ax_b.set_ylim(-0.02, 1.02)

        self._hover_marker_a = self.ax_a.scatter(
            [], [], s=180, facecolors="none", edgecolors="#ff3b30", linewidths=2.5, zorder=6,
        )
        self._hover_marker_b = self.ax_b.scatter(
            [], [], s=180, facecolors="none", edgecolors="#ff3b30", linewidths=2.5, zorder=6,
        )

        self._selected_marker_a = self.ax_a.scatter(
            [], [], s=220, facecolors="none", edgecolors="#43a047", linewidths=2.5, zorder=7,
        )
        self._selected_marker_b = self.ax_b.scatter(
            [], [], s=220, facecolors="none", edgecolors="#43a047", linewidths=2.5, zorder=7,
        )
        self._update_selected_marker()

        self._apply_axes_theme()
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _apply_axes_theme(self):
        text_color = "#e8e8e8" if self.dark else "black"
        for ax in (self.ax_a, self.ax_b):
            ax.xaxis.label.set_color(text_color)
            ax.yaxis.label.set_color(text_color)
            ax.tick_params(colors=text_color)
            for spine in ax.spines.values():
                spine.set_color(text_color)
            legend = ax.get_legend()
            if legend is not None:
                legend.get_frame().set_alpha(0.0)
                for text in legend.get_texts():
                    text.set_color(text_color)

    def _update_selected_marker(self):
        if self._selected_marker_a is None or self._selected_marker_b is None:
            return
        row = self._last_shown_row
        min_acc = self.min_acc_slider.value() / 100.0
        min_simpl = self.min_simpl_slider.value() / 100.0
        min_jac = self.min_jac_slider.value() / 100.0
        visible = (
            row is not None
            and row["accuracy_display"] >= min_acc
            and row["simplicity"] >= min_simpl
            and row["jaccard"] >= min_jac
        )
        if visible:
            self._selected_marker_a.set_offsets([[row["simplicity"], row["jaccard"]]])
            self._selected_marker_b.set_offsets([[row["simplicity"], row["accuracy_display"]]])
        else:
            self._selected_marker_a.set_offsets(np.empty((0, 2)))
            self._selected_marker_b.set_offsets(np.empty((0, 2)))

    def _on_pick(self, event):
        if not len(event.ind):
            return
        idx = event.ind[0]
        self._show_tree(self._filtered_df.iloc[idx])

    def _on_hover(self, event):
        df = self._filtered_df
        if event.inaxes not in (self.ax_a, self.ax_b) or df is None or not len(df):
            self._clear_hover()
            return
        if event.inaxes is self.ax_a:
            xs, ys = df["simplicity"].to_numpy(), df["jaccard"].to_numpy()
        else:
            xs, ys = df["simplicity"].to_numpy(), df["accuracy_display"].to_numpy()

        disp = event.inaxes.transData.transform(list(zip(xs, ys)))
        dists = ((disp[:, 0] - event.x) ** 2 + (disp[:, 1] - event.y) ** 2) ** 0.5
        idx = int(dists.argmin())
        if dists[idx] > 10:  # pixels
            self._clear_hover()
            return

        row = df.iloc[idx]
        if self._hover_marker_a is not None:
            self._hover_marker_a.set_offsets([[row["simplicity"], row["jaccard"]]])
        if self._hover_marker_b is not None:
            self._hover_marker_b.set_offsets([[row["simplicity"], row["accuracy_display"]]])
        self.canvas.draw_idle()

        text = (
            f"Accuracy: {row['accuracy_display']:.3f}\n"
            f"Simplicity: {row['simplicity']:.3f}\n"
            f"Similarity: {row['jaccard']:.3f}\n"
            f"Pareto: {'yes' if row['is_pareto'] else 'no'}"
        )
        local_pos = QPoint(int(event.x), int(self.canvas.height() - event.y))
        QToolTip.showText(self.canvas.mapToGlobal(local_pos), text, self.canvas)

    def _clear_hover(self):
        QToolTip.hideText()
        changed = False
        for marker in (self._hover_marker_a, self._hover_marker_b):
            if marker is not None and len(marker.get_offsets()):
                marker.set_offsets(np.empty((0, 2)))
                changed = True
        if changed:
            self.canvas.draw_idle()

    def _show_normative_tree(self):
        if self.tree_old is None:
            self.normative_label.clear_pixmap("--")
            self.normative_metrics_label.setText(self._metrics_text(None, None, None))
            return
        leaf_ids, path_ids = backend.mandatory_ids_for_tree(self.tree_old, self.tree_old, self.mandatory_leaf_ids)
        same_as_self = backend.find_changed_nodes(self.tree_old, self.tree_old)
        png_stem = TMP_DIR / f"normative_{self.dp_name}"
        try:
            png_path, _ = backend.render_tree_png(
                self.tree_old, png_stem, dark=self.dark,
                highlight_change_ids=same_as_self,
                highlight_mandatory_ids=leaf_ids, highlight_path_ids=path_ids,
            )
            self.normative_label.set_full_pixmap(QPixmap(str(png_path)))
        except Exception as exc:  # noqa: BLE001
            self.normative_label.clear_pixmap(f"Could not draw tree:\n{exc}")
        if self.normative_point is not None:
            self.normative_metrics_label.setText(self._metrics_text(
                self.normative_point["accuracy_display"], self.normative_point["simplicity"], self.normative_point["jaccard"],
            ))
        else:
            self.normative_metrics_label.setText(self._metrics_text(None, None, 1.0))

    def _show_tree(self, row):
        self._last_shown_row = row
        changed_ids = backend.find_changed_nodes(self.tree_old, row["tree"]) if self.tree_old is not None else set()
        leaf_ids, path_ids = backend.mandatory_ids_for_tree(self.tree_old, row["tree"], self.mandatory_leaf_ids)
        display_id_map = backend.display_id_map_for_tree(self.tree_old, row["tree"], id_offset=0) if self.tree_old is not None else {}
        png_stem = TMP_DIR / f"selected_{self.dp_name}_{row.name}"
        try:
            png_path, _ = backend.render_tree_png(
                row["tree"], png_stem, dark=self.dark,
                highlight_change_ids=changed_ids, highlight_mandatory_ids=leaf_ids,
                highlight_path_ids=path_ids, display_id_map=display_id_map,
            )
            self.repaired_label.set_full_pixmap(QPixmap(str(png_path)))
        except Exception as exc:  # noqa: BLE001
            self.repaired_label.clear_pixmap(f"Could not draw tree:\n{exc}")
        self.repaired_metrics_label.setText(
            self._metrics_text(row["accuracy_display"], row["simplicity"], row["jaccard"])
        )
        self._update_selected_marker()
        self.canvas.draw_idle()

    def _on_baseline_selected(self):
        prefix = self.baseline_combo.currentData()
        if prefix is None:
            return
        self.baseline_prefix = prefix
        self._update_baseline_display()

    def _update_baseline_display(self):
        prefix = self.baseline_prefix
        if prefix in self.baseline_points:
            self._on_baseline_ready(self.baseline_points[prefix])
            return
        if self.df_adapt is None or self.tree_old is None:
            self.baseline_status.setText("No data to compute this baseline.")
            self.rediscovered_label.clear_pixmap("--")
            self.rediscovered_metrics_label.setText(self._metrics_text(None, None, None))
            return

        self.baseline_status.setText(f"")
        self.baseline_combo.setEnabled(False)
        self.rediscovered_label.clear_pixmap("Fitting...")

        thread = QThread()
        worker = BaselineWorker(prefix, self.dp_name, self.tree_old, self.df_adapt, df_test=self.df_test)
        worker.moveToThread(thread)
        entry = (thread, worker)
        self._baseline_threads.append(entry)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_baseline_computed)
        worker.failed.connect(self._on_baseline_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(lambda e=entry: self._baseline_threads.remove(e) if e in self._baseline_threads else None)
        thread.start()

    def _on_baseline_computed(self, prefix, dp_name, point):
        if dp_name != self.dp_name:
            return
        self.baseline_points[prefix] = point
        self.baseline_combo.setEnabled(True)
        if prefix == self.baseline_prefix:  # user might have switched again while this was fitting
            self._on_baseline_ready(point)

    def _on_baseline_failed(self, dp_name, message):
        if dp_name != self.dp_name:
            return
        self.baseline_combo.setEnabled(True)
        self.baseline_status.setText(f"Could not fit: {message}")
        self.rediscovered_label.clear_pixmap("--")
        self.rediscovered_metrics_label.setText(self._metrics_text(None, None, None))

    def _on_baseline_ready(self, point):
        self.baseline_status.setText("")
        try:
            if point is None:
                self.rediscovered_label.clear_pixmap("--")
                self.rediscovered_metrics_label.setText(self._metrics_text(None, None, None))
            else:
                try:
                    changed_ids = (
                        backend.find_changed_nodes(self.tree_old, point["tree"]) if self.tree_old is not None else None
                    )
                    display_id_map = (
                        backend.display_id_map_for_tree(self.tree_old, point["tree"], id_offset=1000)
                        if self.tree_old is not None else {}
                    )
                except Exception as exc:  # noqa: BLE001
                    changed_ids = None
                    display_id_map = {}
                    print(f"[baseline] could not compare {self.baseline_prefix} tree to the normative tree: {exc!r}", flush=True)
                png_stem = TMP_DIR / f"baseline_{self.baseline_prefix}_{self.dp_name}"
                try:
                    png_path, _ = backend.render_tree_png(
                        point["tree"], png_stem, dark=self.dark, highlight_change_ids=changed_ids,
                        display_id_map=display_id_map,
                    )
                    self.rediscovered_label.set_full_pixmap(QPixmap(str(png_path)))
                except Exception as exc:  # noqa: BLE001
                    self.rediscovered_label.clear_pixmap(f"Could not draw tree:\n{exc}")
                self.rediscovered_metrics_label.setText(
                    self._metrics_text(point["accuracy_display"], point["simplicity"], point["jaccard"])
                )
            self._redraw_scatter()
        except Exception as exc:
            print(f"failed for {self.baseline_prefix}: {exc!r}", flush=True)
            self.baseline_status.setText(f"Could not display this baseline: {exc}")

    def set_dark(self, dark):
        self.dark = dark
        self._rebuild_toolbar()
        if self.df is not None:
            self._apply_axes_theme()
            self.canvas.draw_idle()
        self._show_normative_tree()
        if self._last_shown_row is not None:
            self._show_tree(self._last_shown_row)
        point = self.baseline_points.get(self.baseline_prefix)
        if point is not None:
            self._on_baseline_ready(point)

    def _rebuild_toolbar(self):
        old = self.toolbar
        idx = self._toolbar_layout.indexOf(old)
        self._toolbar_layout.removeWidget(old)
        old.setParent(None)
        old.deleteLater()
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self._toolbar_layout.insertWidget(idx, self.toolbar)

