"""Settings dialog for the TreeQSM action.

CloudCompare's PythonRuntime ships its own Qt binding (PyQt5 on current
builds); importing a second Qt into CloudCompare's process can crash it, so
that binding is preferred and PySide6 is only a fallback (the combination
3DFin ships with).
"""

try:
    from PyQt5.QtCore import QSettings
    from PyQt5.QtWidgets import (
        QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
        QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
        QLabel, QLineEdit, QMessageBox, QProgressDialog, QPushButton,
        QSpinBox, QVBoxLayout, QWidget,
    )
except ImportError:
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import (
        QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
        QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
        QLabel, QLineEdit, QMessageBox, QProgressDialog, QPushButton,
        QSpinBox, QVBoxLayout, QWidget,
    )

from pathlib import Path

WARNING = (
    "TreeQSM uses randomised cover sets: every run gives a slightly "
    "different model.\nProgress appears in the console; the first run also "
    "compiles the numba kernels."
)


class _JobProgressDialog(QProgressDialog):
    """Qt emits canceled() on any close of the dialog, screen lock included.
    Closes are ignored until allow_close() so only the button cancels."""

    def __init__(self, *args):
        super().__init__(*args)
        self._closable = False

    def allow_close(self):
        self._closable = True

    def closeEvent(self, event):
        if self._closable:
            event.accept()
        else:
            event.ignore()

    def keyPressEvent(self, event):
        event.ignore()


def make_progress_dialog(title):
    """Non-modal indeterminate progress dialog with a working Cancel."""
    dlg = _JobProgressDialog(title, "Cancel", 0, 0)
    dlg.setWindowTitle("PyTLidar")
    dlg.setMinimumDuration(0)
    dlg.show()
    return dlg


def parse_diams(text):
    """Comma-separated PatchDiam values as a list of floats, each in (0, 1].

    Raises ValueError on anything else, including an empty string.
    """
    values = [float(v) for v in text.split(",") if v.strip()]
    if not values or any(not 0.0 < v <= 1.0 for v in values):
        raise ValueError(f"PatchDiam values must be in (0, 1]: {text!r}")
    return values


def _scalar_field_names(source):
    return [source.getScalarFieldName(i)
            for i in range(source.getNumberOfScalarFields())]


class _SettingsDialog(QDialog):
    def __init__(self, clouds, metrics):
        super().__init__()
        self.n_clouds = len(clouds)
        title = (clouds[0].getName() if self.n_clouds == 1
                 else f"{self.n_clouds} clouds")
        self.setWindowTitle(f"PyTLidar (TreeQSM): {title}")
        self.settings = None

        layout = QVBoxLayout(self)

        patch = QGroupBox("PatchDiam")
        patch_layout = QVBoxLayout(patch)
        self.custom_check = QCheckBox("Use custom values")
        self.custom_check.setToolTip(
            "Off: generate candidate PatchDiam values automatically. "
            "On: give the values to test, comma separated.")
        patch_layout.addWidget(self.custom_check)

        self.auto_box = QWidget()
        auto_form = QFormLayout(self.auto_box)
        self.n_pd1 = QSpinBox()
        self.n_pd2min = QSpinBox()
        self.n_pd2max = QSpinBox()
        for spin in (self.n_pd1, self.n_pd2min, self.n_pd2max):
            spin.setRange(1, 3)
            spin.setValue(1)
        auto_form.addRow("PatchDiam1 candidates", self.n_pd1)
        auto_form.addRow("PatchDiam2Min candidates", self.n_pd2min)
        auto_form.addRow("PatchDiam2Max candidates", self.n_pd2max)
        patch_layout.addWidget(self.auto_box)

        self.custom_box = QWidget()
        custom_form = QFormLayout(self.custom_box)
        self.pd1 = QLineEdit()
        self.pd2min = QLineEdit()
        self.pd2max = QLineEdit()
        for edit, hint in ((self.pd1, "e.g. 0.1"), (self.pd2min, "e.g. 0.02"),
                           (self.pd2max, "e.g. 0.06")):
            edit.setPlaceholderText(hint)
        custom_form.addRow("PatchDiam1", self.pd1)
        custom_form.addRow("PatchDiam2Min", self.pd2min)
        custom_form.addRow("PatchDiam2Max", self.pd2max)
        self.custom_box.hide()
        patch_layout.addWidget(self.custom_box)

        self.count_label = QLabel("")
        patch_layout.addWidget(self.count_label)

        self.custom_check.toggled.connect(self.auto_box.setHidden)
        self.custom_check.toggled.connect(self.custom_box.setVisible)
        self.custom_check.toggled.connect(self._update_count)
        for spin in (self.n_pd1, self.n_pd2min, self.n_pd2max):
            spin.valueChanged.connect(self._update_count)
        for edit in (self.pd1, self.pd2min, self.pd2max):
            edit.textChanged.connect(self._update_count)
        self._update_count()
        layout.addWidget(patch)

        # Union across the selection; clouds without the chosen field are
        # not filtered.
        self.sf_names = []
        for cloud in clouds:
            for name in _scalar_field_names(cloud):
                if name not in self.sf_names:
                    self.sf_names.append(name)
        self.sf_check = None
        if self.sf_names:
            sf_box = QGroupBox("Point filter")
            sf_form = QFormLayout(sf_box)
            self.sf_check = QCheckBox("Filter by scalar field")
            self.sf_combo = QComboBox()
            self.sf_combo.addItems(self.sf_names)
            for i, name in enumerate(self.sf_names):
                if "intensity" in name.lower():
                    self.sf_combo.setCurrentIndex(i)
                    break
            self.sf_threshold = QDoubleSpinBox()
            self.sf_threshold.setRange(-1e9, 1e9)
            self.sf_threshold.setDecimals(3)
            self.sf_combo.setEnabled(False)
            self.sf_threshold.setEnabled(False)
            self.sf_check.toggled.connect(self.sf_combo.setEnabled)
            self.sf_check.toggled.connect(self.sf_threshold.setEnabled)
            sf_form.addRow(self.sf_check)
            sf_form.addRow("Field", self.sf_combo)
            sf_form.addRow("Keep values at least", self.sf_threshold)
            layout.addWidget(sf_box)

        metric_form = QFormLayout()
        self.metric_combo = QComboBox()
        self.metric_combo.addItems(list(metrics))
        if "all_mean_dis" in metrics:
            self.metric_combo.setCurrentText("all_mean_dis")
        self.metric_combo.setToolTip(
            "Picks the best model when several PatchDiam combinations run. "
            "Ignored for a single combination.")
        metric_form.addRow("Optimal model metric", self.metric_combo)
        self.tria_check = QCheckBox("Triangulate the stem")
        self.tria_check.setToolTip(
            "Model the bottom of the trunk as a triangle mesh, better volume for "
            "large or buttressed stems. Adds a stem mesh to the output.")
        metric_form.addRow(self.tria_check)
        self.runs_spin = QSpinBox()
        self.runs_spin.setRange(1, 100)
        self.runs_spin.setValue(1)
        self.runs_spin.setToolTip(
            "TreeQSM is stochastic, so repeated runs give different models. "
            "Each run is added to the scene as its own group; keep the tree "
            "you like and delete the rest.")
        metric_form.addRow("Repeat runs", self.runs_spin)
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(1, 8)
        self.workers_spin.setValue(1)
        self.workers_spin.setToolTip(
            "How many runs fit at the same time. Each parallel run uses its "
            "own memory, so keep this low for very large clouds.")
        metric_form.addRow("Parallel workers", self.workers_spin)
        self.runs_spin.valueChanged.connect(self._update_count)
        self.save_check = QCheckBox("Save model files")
        self.save_check.setChecked(True)
        self.save_check.setToolTip(
            "Writes the QSM archive (npz), the cylinder, branch and treedata "
            "tables and a run_info.json for each run.")
        metric_form.addRow(self.save_check)
        default_dir = QSettings("PyTLidar", "pytlidar-cc").value(
            "results_dir", str(Path.home() / "Documents" / "PyTLidar" / "results"))
        self.dir_edit = QLineEdit(str(default_dir))
        browse = QPushButton("Browse")
        browse.clicked.connect(self._pick_dir)
        dir_row = QWidget()
        dir_lay = QHBoxLayout(dir_row)
        dir_lay.setContentsMargins(0, 0, 0, 0)
        dir_lay.addWidget(self.dir_edit)
        dir_lay.addWidget(browse)
        self.save_check.toggled.connect(dir_row.setEnabled)
        metric_form.addRow("Results folder", dir_row)
        layout.addLayout(metric_form)

        warning = QLabel(WARNING)
        warning.setWordWrap(True)
        layout.addWidget(warning)

        self.error_label = QLabel("")
        self.error_label.setStyleSheet("color: red")
        self.error_label.setWordWrap(True)
        layout.addWidget(self.error_label)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _update_count(self):
        if self.custom_check.isChecked():
            try:
                n = (len(parse_diams(self.pd1.text()))
                     * len(parse_diams(self.pd2min.text()))
                     * len(parse_diams(self.pd2max.text())))
            except ValueError:
                self.count_label.setText("")
                return
        else:
            n = (self.n_pd1.value() * self.n_pd2min.value()
                 * self.n_pd2max.value())
        runs = self.runs_spin.value() if hasattr(self, "runs_spin") else 1
        total = n * runs * self.n_clouds
        if total == 1:
            self.count_label.setText("1 model will be fitted.")
        else:
            parts = [f"{n} PatchDiam combination(s)"]
            if runs > 1:
                parts.append(f"{runs} runs")
            if self.n_clouds > 1:
                parts.append(f"{self.n_clouds} clouds")
            self.count_label.setText(
                f"{total} models will be fitted ({' x '.join(parts)}); "
                "the runtime multiplies accordingly.")

    def _pick_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Results folder",
                                             self.dir_edit.text())
        if d:
            self.dir_edit.setText(d)

    def _validate_and_accept(self):
        if self.runs_spin.value() > 20:
            reply = QMessageBox.question(
                self, "PyTLidar",
                f"{self.runs_spin.value()} repeat runs, times the PatchDiam "
                "combinations, is a lot of fitting. Are you sure?")
            if reply != QMessageBox.StandardButton.Yes:
                return
        custom = None
        if self.custom_check.isChecked():
            try:
                custom = (parse_diams(self.pd1.text()),
                          parse_diams(self.pd2min.text()),
                          parse_diams(self.pd2max.text()))
            except ValueError as exc:
                self.error_label.setText(str(exc))
                return
        intensity_sf = None
        threshold = 0.0
        if self.sf_check is not None and self.sf_check.isChecked():
            intensity_sf = self.sf_combo.currentText()
            threshold = float(self.sf_threshold.value())
        QSettings("PyTLidar", "pytlidar-cc").setValue(
            "results_dir", self.dir_edit.text())
        self.settings = {
            "custom": custom,
            "n_patchdiam": (self.n_pd1.value(), self.n_pd2min.value(),
                            self.n_pd2max.value()),
            "metric": self.metric_combo.currentText(),
            "tria": self.tria_check.isChecked(),
            "runs": self.runs_spin.value(),
            "workers": self.workers_spin.value(),
            "save_files": self.save_check.isChecked(),
            "results_dir": self.dir_edit.text(),
            "intensity_sf": intensity_sf,
            "intensity_threshold": threshold,
        }
        self.accept()


def show_settings_dialog(clouds, metrics):
    """Modal settings dialog for one job over the selected clouds. Returns
    the settings dict, or None when the user cancels. CloudCompare's Qt event
    loop is already running, so QDialog.exec() is enough; no QApplication or
    QEventLoop needed."""
    dialog = _SettingsDialog(clouds, metrics)
    dialog.exec()
    return dialog.settings
