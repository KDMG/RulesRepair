from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QStackedWidget, QVBoxLayout

from screens.petri_net_screen import PetriNetScreen
from screens.rule_selection_screen import RuleSelectionScreen
from screens.front_explorer_screen import FrontExplorerScreen
from theme import _light_palette, _dark_palette


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pareto Front Explorer")
        self.resize(1400, 900)
        self.dark_mode = False

        central = QWidget()
        central.setAutoFillBackground(True)
        central_layout = QVBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)
        self.setCentralWidget(central)

        self.stack = QStackedWidget()
        self.stack.setAutoFillBackground(True)
        central_layout.addWidget(self.stack, stretch=1)

        self.petri_screen = PetriNetScreen(on_open_dp=self._open_dp, on_theme_toggle=self.toggle_theme)
        self.rule_screen = RuleSelectionScreen(
            on_back=self._back_to_petri, on_computed=self._front_computed, on_theme_toggle=self.toggle_theme,
        )
        self.front_screen = FrontExplorerScreen(on_back=self._back_to_rules, on_theme_toggle=self.toggle_theme)
        for screen in (self.petri_screen, self.rule_screen, self.front_screen):
            screen.setAutoFillBackground(True)

        self.stack.addWidget(self.petri_screen)
        self.stack.addWidget(self.rule_screen)
        self.stack.addWidget(self.front_screen)

    def toggle_theme(self):
        self.dark_mode = not self.dark_mode
        QApplication.instance().setPalette(_dark_palette() if self.dark_mode else _light_palette())
        theme_btn_text = "Light mode" if self.dark_mode else "Dark mode"
        self.petri_screen.theme_btn.setText(theme_btn_text)
        self.rule_screen.theme_btn.setText(theme_btn_text)
        self.front_screen.theme_btn.setText(theme_btn_text)
        self.petri_screen.set_dark(self.dark_mode)
        self.rule_screen.set_dark(self.dark_mode)
        self.front_screen.set_dark(self.dark_mode)

    def _open_dp(self, dp_name, dp_info, model, observations, dataset_name):
        self.rule_screen.open_dp(dp_name, dp_info, model, observations, dataset_name)
        self.stack.setCurrentWidget(self.rule_screen)

    def _back_to_petri(self):
        self.stack.setCurrentWidget(self.petri_screen)

    def _back_to_rules(self):
        self.stack.setCurrentWidget(self.rule_screen)

    def _front_computed(self, dp_name, tree_old, df, cart_point, normative_point, mandatory_leaf_ids, df_adapt, dataset_name, df_test=None):
        self.front_screen.open_front(dp_name, tree_old, df, cart_point, normative_point, mandatory_leaf_ids, df_adapt, dataset_name, df_test=df_test)
        self.stack.setCurrentWidget(self.front_screen)
