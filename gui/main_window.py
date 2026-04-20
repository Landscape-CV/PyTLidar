"""
EcomodelMainWindow – PySide6 main window for the Ecomodel pipeline.

Layout
──────
  Left panel  (scrollable QScrollArea)
      Parameter inputs grouped by pipeline stage:
        Input / Output
        Tile Subdivision
        Ground Filtering (CSF)
        Terrain Model (DEM)
        Optional Cleanup / Denoise
        Tree Segmentation
        QSM
        Cover Sets        (stored in config; not yet wired into ecomodel.py)
        RGI Leaf/Wood     (stored in config; not yet wired into ecomodel.py)
        Checkpoint

  Right panel
        Run / Stop buttons
        Progress bar
        Step label
        Log text area (read-only, monospaced)
        Clear Log button

Threading model:
    QThread + EcomodelWorker(QObject).moveToThread(thread)
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QThread, Qt
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from gui.config import EcomodelConfig
from gui.query_widgets import QueryPage
from gui.results_widgets import ResultsPage
from gui.worker import EcomodelWorker


class EcomodelMainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Ecomodel")
        self.resize(1150, 820)

        self._thread: QThread | None = None
        self._worker: EcomodelWorker | None = None
        self._defaults = EcomodelConfig()

        # ── Stacked widget: page 0 = pipeline, page 1 = results, page 2 = query
        self._stack = QStackedWidget()
        self.setCentralWidget(self._stack)

        # Page 0 — pipeline
        pipeline_page = QWidget()
        pipeline_root = QHBoxLayout(pipeline_page)
        splitter = QSplitter(Qt.Horizontal)
        pipeline_root.addWidget(splitter)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        param_widget = QWidget()
        scroll.setWidget(param_widget)
        splitter.addWidget(scroll)

        right = QWidget()
        splitter.addWidget(right)
        splitter.setSizes([560, 590])

        self._build_param_panel(param_widget)
        self._build_right_panel(right)

        self._stack.addWidget(pipeline_page)          # index 0

        # Initialise status bar before connecting signals that write to it.
        self.setStatusBar(QStatusBar())

        # Page 1 — results viewer
        self._results_page = ResultsPage(
            results_folder=self._defaults.results_folder,
        )
        self._results_page.back_requested.connect(
            lambda: self._stack.setCurrentIndex(0)
        )
        self._results_page.status_message.connect(
            lambda msg: self.statusBar().showMessage(msg)
        )
        self._results_page.query_requested.connect(self._show_query_page)
        self._stack.addWidget(self._results_page)     # index 1

        # Page 2 — query mode
        self._query_page = QueryPage(results_folder=self._defaults.results_folder)
        self._query_page.back_requested.connect(lambda: self._stack.setCurrentIndex(1))
        self._query_page.status_message.connect(
            lambda msg: self.statusBar().showMessage(msg)
        )
        self._stack.addWidget(self._query_page)       # index 2

    # ── Parameter panel ───────────────────────────────────────────────────────

    def _build_param_panel(self, parent: QWidget) -> None:
        layout = QVBoxLayout(parent)
        layout.setAlignment(Qt.AlignTop)

        # ── I/O ──────────────────────────────────────────────────────────────
        io = QGroupBox("Input / Output")
        g = QGridLayout(io)

        g.addWidget(QLabel("LAS/LAZ Folder:"), 0, 0)
        self._input_folder = QLineEdit()
        self._input_folder.setPlaceholderText("Folder containing .las / .laz tiles")
        self._input_folder.editingFinished.connect(
            lambda: self._auto_populate_names(self._input_folder.text().strip())
            if self._input_folder.text().strip() else None
        )
        g.addWidget(self._input_folder, 0, 1)
        b_in = QPushButton("Browse…")
        b_in.clicked.connect(self._browse_input)
        g.addWidget(b_in, 0, 2)

        g.addWidget(QLabel("Results Folder:"), 1, 0)
        self._results_folder = QLineEdit(self._defaults.results_folder)
        g.addWidget(self._results_folder, 1, 1)
        b_out = QPushButton("Browse…")
        b_out.clicked.connect(self._browse_results)
        g.addWidget(b_out, 1, 2)

        g.addWidget(QLabel("Cylinder Filename:"), 2, 0)
        self._cylinder_filename = QLineEdit(self._defaults.cylinder_filename)
        g.addWidget(self._cylinder_filename, 2, 1, 1, 2)

        layout.addWidget(io)

        # ── Tile subdivision ──────────────────────────────────────────────────
        tg = QGroupBox("Tile Subdivision")
        tg.setCheckable(True)
        tg.setChecked(False)
        tg.setToolTip("Check the title to enable editing of this section")
        tl = QGridLayout(tg)

        tl.addWidget(QLabel("Cube Size (m):"), 0, 0)
        self._cube_size = self._dspin(0.5, 500.0, self._defaults.cube_size, 1)
        tl.addWidget(self._cube_size, 0, 1)

        tl.addWidget(QLabel("Metre Conversion:"), 1, 0)
        self._meter_conversion = self._dspin(0.001, 1000.0, self._defaults.meter_conversion, 4)
        tl.addWidget(self._meter_conversion, 1, 1)

        reset_sub_btn = QPushButton("Reset to Defaults")
        reset_sub_btn.clicked.connect(self._reset_subdivision_defaults)
        tl.addWidget(reset_sub_btn, 2, 0, 1, 2)

        layout.addWidget(tg)

        # ── Ground filtering (CSF) ────────────────────────────────────────────
        cg = QGroupBox("Ground Filtering (CSF)")
        cg.setCheckable(True)
        cg.setChecked(False)
        cg.setToolTip("Check the title to enable editing of this section")
        cl = QGridLayout(cg)

        cl.addWidget(QLabel("Band Size:"), 0, 0)
        self._csf_band_size = self._dspin(0.01, 10.0, self._defaults.csf_band_size, 3)
        cl.addWidget(self._csf_band_size, 0, 1)

        cl.addWidget(QLabel("Threshold:"), 1, 0)
        self._csf_threshold = self._spin(1, 1000, self._defaults.csf_threshold)
        cl.addWidget(self._csf_threshold, 1, 1)

        cl.addWidget(QLabel("Offset:"), 2, 0)
        self._csf_offset = self._dspin(0.0, 10.0, self._defaults.csf_offset, 3)
        cl.addWidget(self._csf_offset, 2, 1)

        self._remove_underground = QCheckBox("Remove Underground Points")
        self._remove_underground.setChecked(self._defaults.remove_under_ground)
        cl.addWidget(self._remove_underground, 3, 0, 1, 2)

        reset_csf_btn = QPushButton("Reset to Defaults")
        reset_csf_btn.clicked.connect(self._reset_csf_defaults)
        cl.addWidget(reset_csf_btn, 4, 0, 1, 2)

        layout.addWidget(cg)

        # ── Terrain model ─────────────────────────────────────────────────────
        tmg = QGroupBox("Terrain Model (DEM)")
        tmg.setCheckable(True)
        tmg.setChecked(False)
        tmg.setToolTip("Check the title to enable editing of this section")
        tml = QGridLayout(tmg)

        tml.addWidget(QLabel("Grid Size (m):"), 0, 0)
        self._terrain_grid_size = self._dspin(0.01, 10.0, self._defaults.terrain_grid_size, 3)
        tml.addWidget(self._terrain_grid_size, 0, 1)

        reset_terrain_btn = QPushButton("Reset to Defaults")
        reset_terrain_btn.clicked.connect(self._reset_terrain_defaults)
        tml.addWidget(reset_terrain_btn, 1, 0, 1, 2)

        layout.addWidget(tmg)

        # ── Optional cleanup / denoise ────────────────────────────────────────
        og = QGroupBox("Optional Cleanup / Denoise")
        og.setCheckable(True)
        og.setChecked(False)
        og.setToolTip("Check the title to enable editing of this section")
        ol = QGridLayout(og)

        self._remove_duplicates = QCheckBox("Remove Duplicate Points")
        self._remove_duplicates.setChecked(self._defaults.remove_duplicates)
        ol.addWidget(self._remove_duplicates, 0, 0, 1, 2)

        self._use_denoise = QCheckBox("Run Denoise After Subdivision")
        self._use_denoise.setChecked(self._defaults.use_denoise)
        ol.addWidget(self._use_denoise, 1, 0, 1, 2)

        ol.addWidget(QLabel("Denoise Grid Size:"), 2, 0)
        self._denoise_grid_size = self._dspin(0.01, 10.0, self._defaults.denoise_grid_size, 3)
        ol.addWidget(self._denoise_grid_size, 2, 1)

        ol.addWidget(QLabel("Denoise Min Points:"), 3, 0)
        self._denoise_min_points = self._spin(1, 100000, self._defaults.denoise_min_points)
        ol.addWidget(self._denoise_min_points, 3, 1)

        ol.addWidget(QLabel("Denoise Resolution:"), 4, 0)
        self._denoise_resolution = self._dspin(0.001, 5.0, self._defaults.denoise_resolution, 3)
        ol.addWidget(self._denoise_resolution, 4, 1)

        reset_cleanup_btn = QPushButton("Reset to Defaults")
        reset_cleanup_btn.clicked.connect(self._reset_cleanup_defaults)
        ol.addWidget(reset_cleanup_btn, 5, 0, 1, 2)

        layout.addWidget(og)

        # ── Tree segmentation ─────────────────────────────────────────────────
        sg = QGroupBox("Tree Segmentation")
        sg.setCheckable(True)
        sg.setChecked(False)
        sg.setToolTip("Check the title to enable editing of this section")
        sl = QGridLayout(sg)

        sl.addWidget(QLabel("Intensity Threshold:"), 0, 0)
        self._seg_intensity = self._spin(0, 100000, self._defaults.segment_intensity_threshold)
        sl.addWidget(self._seg_intensity, 0, 1)

        self._save_clusters = QCheckBox("Save Intermediate Clusters")
        self._save_clusters.setChecked(self._defaults.save_clusters)
        sl.addWidget(self._save_clusters, 1, 0, 1, 2)

        reset_seg_btn = QPushButton("Reset to Defaults")
        reset_seg_btn.clicked.connect(self._reset_segmentation_defaults)
        sl.addWidget(reset_seg_btn, 2, 0, 1, 2)

        layout.addWidget(sg)

        # ── QSM ──────────────────────────────────────────────────────────────
        qg = QGroupBox("QSM / Outputs")
        qg.setCheckable(True)
        qg.setChecked(False)
        qg.setToolTip("Check the title to enable editing of this section")
        ql = QGridLayout(qg)

        self._run_qsm = QCheckBox("Run QSM + RGI")
        self._run_qsm.setChecked(self._defaults.run_qsm)
        ql.addWidget(self._run_qsm, 0, 0, 1, 2)

        self._run_leaf_removal = QCheckBox("Run Leaf Removal")
        self._run_leaf_removal.setChecked(getattr(self._defaults, "run_leaf_removal", True))
        ql.addWidget(self._run_leaf_removal, 1, 0, 1, 2)

        ql.addWidget(QLabel("Intensity Threshold:"), 2, 0)
        self._qsm_intensity = self._spin(0, 200000, self._defaults.qsm_intensity_threshold)
        ql.addWidget(self._qsm_intensity, 2, 1)

        self._save_leaf_output = QCheckBox("Save Leaf Removal Output")
        self._save_leaf_output.setChecked(self._defaults.save_leaf_removal_output)
        ql.addWidget(self._save_leaf_output, 3, 0, 1, 2)

        self._create_cylinder_plot = QCheckBox("Create Cylinder Plot")
        self._create_cylinder_plot.setChecked(self._defaults.create_cylinder_plot)
        ql.addWidget(self._create_cylinder_plot, 4, 0, 1, 2)

        reset_qsm_btn = QPushButton("Reset to Defaults")
        reset_qsm_btn.clicked.connect(self._reset_qsm_defaults)
        ql.addWidget(reset_qsm_btn, 5, 0, 1, 2)

        layout.addWidget(qg)

        # ── Cover sets ───────────────────────────────────────────────────────────
        csg = QGroupBox("Cover Sets")
        csg.setCheckable(True)
        csg.setChecked(False)
        csg.setToolTip("Check the title to enable editing of this section")
        csl = QGridLayout(csg)

        cs_fields = [
            ("Patch Diam 1:", "_patch_diam1", 0.001, 5.0, self._defaults.patch_diam1, 4),
            ("Ball Rad 1:", "_ball_rad1", 0.001, 5.0, self._defaults.ball_rad1, 4),
            ("Patch Diam 2 Min:", "_patch_diam2_min", 0.001, 5.0, self._defaults.patch_diam2_min, 4),
            ("Patch Diam 2 Max:", "_patch_diam2_max", 0.001, 5.0, self._defaults.patch_diam2_max, 4),
            ("Ball Rad 2:", "_ball_rad2", 0.001, 5.0, self._defaults.ball_rad2, 4),
        ]
        for row, (lbl, attr, lo, hi, val, dec) in enumerate(cs_fields):
            csl.addWidget(QLabel(lbl), row, 0)
            w = self._dspin(lo, hi, val, dec)
            setattr(self, attr, w)
            csl.addWidget(w, row, 1)

        n = len(cs_fields)
        csl.addWidget(QLabel("Nmin 1:"), n, 0)
        self._nmin1 = self._spin(1, 1000, self._defaults.nmin1)
        csl.addWidget(self._nmin1, n, 1)

        reset_cover_btn = QPushButton("Reset to Defaults")
        reset_cover_btn.clicked.connect(self._reset_cover_defaults)
        csl.addWidget(reset_cover_btn, n + 1, 0, 1, 2)

        layout.addWidget(csg)

        # ── RGI leaf/wood separation ──────────────────────────────────────────
        rg = QGroupBox("RGI Leaf/Wood Separation")
        rg.setCheckable(True)
        rg.setChecked(False)
        rg.setToolTip("Check the title to enable editing of this section")
        rl = QGridLayout(rg)

        rgi_fields = [
            ("Noise Percentile:", "_rgi_noise_percentile", "spin", 0, 100, self._defaults.rgi_noise_percentile, 0),
            ("Angle (deg):", "_rgi_angle_deg", "dspin", 0.1, 90.0, self._defaults.rgi_angle_deg, 2),
            ("Curvature Thresh:", "_rgi_curv_thresh", "dspin", 1e-4, 1.0, self._defaults.rgi_curv_thresh, 4),
            ("Residual Thresh:", "_rgi_resid_thresh", "dspin", 1e-4, 1.0, self._defaults.rgi_resid_thresh, 4),
            ("K (neighbours):", "_rgi_k", "spin", 5, 1000, self._defaults.rgi_k, 0),
            ("Min Cluster Size:", "_rgi_min_cluster_size", "spin", 1, 100000, self._defaults.rgi_min_cluster_size, 0),
            ("Max Cluster Size:", "_rgi_max_cluster_size", "spin", 1, 10000000, self._defaults.rgi_max_cluster_size, 0),
        ]
        for row, (lbl, attr, kind, lo, hi, val, dec) in enumerate(rgi_fields):
            rl.addWidget(QLabel(lbl), row, 0)
            if kind == "spin":
                w = self._spin(int(lo), int(hi), int(val))
            else:
                w = self._dspin(lo, hi, val, dec)
            setattr(self, attr, w)
            rl.addWidget(w, row, 1)

        off = len(rgi_fields)
        self._rgi_smooth_mode = QCheckBox("Smooth Mode")
        self._rgi_smooth_mode.setChecked(self._defaults.rgi_smooth_mode)
        rl.addWidget(self._rgi_smooth_mode, off, 0, 1, 2)

        self._rgi_use_residual = QCheckBox("Use Residual Test")
        self._rgi_use_residual.setChecked(self._defaults.rgi_use_residual_test)
        rl.addWidget(self._rgi_use_residual, off + 1, 0, 1, 2)

        self._rgi_use_curvature = QCheckBox("Use Curvature Test")
        self._rgi_use_curvature.setChecked(self._defaults.rgi_use_curvature_test)
        rl.addWidget(self._rgi_use_curvature, off + 2, 0, 1, 2)

        reset_rgi_btn = QPushButton("Reset to Defaults")
        reset_rgi_btn.clicked.connect(self._reset_rgi_defaults)
        rl.addWidget(reset_rgi_btn, off + 3, 0, 1, 2)

        layout.addWidget(rg)

        # ── Checkpoint ────────────────────────────────────────────────────────
        pkg = QGroupBox("Checkpoint")
        pkg.setCheckable(True)
        pkg.setChecked(False)
        pkg.setToolTip("Check the title to enable editing of this section")
        pkl = QGridLayout(pkg)

        pkl.addWidget(QLabel("Name:"), 0, 0)
        self._checkpoint_name = QLineEdit(self._defaults.checkpoint_name)
        pkl.addWidget(self._checkpoint_name, 0, 1)

        pkl.addWidget(QLabel("Folder:"), 1, 0)
        self._checkpoint_folder = QLineEdit(self._defaults.checkpoint_folder)
        pkl.addWidget(self._checkpoint_folder, 1, 1)
        browse_folder_btn = QPushButton("…")
        browse_folder_btn.setMaximumWidth(30)
        browse_folder_btn.clicked.connect(self._browse_checkpoint_folder)
        pkl.addWidget(browse_folder_btn, 1, 2)

        self._save_checkpoint = QCheckBox("Save Checkpoints During Run")
        self._save_checkpoint.setChecked(getattr(self._defaults, "save_checkpoint", False))
        pkl.addWidget(self._save_checkpoint, 2, 0, 1, 3)

        self._use_checkpoint = QCheckBox("Resume from Checkpoint")
        self._use_checkpoint.setChecked(self._defaults.use_checkpoint)
        pkl.addWidget(self._use_checkpoint, 3, 0, 1, 3)

        pkl.addWidget(QLabel("Resume File:"), 4, 0)
        self._checkpoint_resume_file = QLineEdit(
            getattr(self._defaults, "checkpoint_resume_file", ""))
        self._checkpoint_resume_file.setPlaceholderText("path/to/checkpoint.pickle")
        self._checkpoint_resume_file.setEnabled(self._defaults.use_checkpoint)
        pkl.addWidget(self._checkpoint_resume_file, 4, 1)
        browse_resume_btn = QPushButton("…")
        browse_resume_btn.setMaximumWidth(30)
        browse_resume_btn.setEnabled(self._defaults.use_checkpoint)
        browse_resume_btn.clicked.connect(self._browse_checkpoint_resume_file)
        pkl.addWidget(browse_resume_btn, 4, 2)
        self._browse_resume_btn = browse_resume_btn   # keep reference to enable/disable

        self._use_checkpoint.toggled.connect(self._checkpoint_resume_file.setEnabled)
        self._use_checkpoint.toggled.connect(self._browse_resume_btn.setEnabled)

        pkg.toggled.connect(self._on_checkpoint_section_toggled)

        layout.addWidget(pkg)

    # ── Right panel ───────────────────────────────────────────────────────────

    def _build_right_panel(self, parent: QWidget) -> None:
        layout = QVBoxLayout(parent)

        btn_row = QHBoxLayout()
        self._run_btn = QPushButton("Start")
        self._run_btn.setStyleSheet("font-weight: bold; padding: 6px;")
        self._run_btn.clicked.connect(self._start_pipeline)
        btn_row.addWidget(self._run_btn)

        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._stop_pipeline)
        btn_row.addWidget(self._stop_btn)

        self._results_btn = QPushButton("Results →")
        self._results_btn.setToolTip("View results")
        self._results_btn.clicked.connect(self._show_results_page)
        btn_row.addWidget(self._results_btn)

        self._query_nav_btn = QPushButton("Query →")
        self._query_nav_btn.setToolTip("Find nearest tree by world coordinate")
        self._query_nav_btn.clicked.connect(self._show_query_page)
        btn_row.addWidget(self._query_nav_btn)
        layout.addLayout(btn_row)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 1)
        self._progress_bar.setValue(0)
        self._progress_bar.setFormat("Step %v / %m")
        layout.addWidget(self._progress_bar)

        self._progress_label = QLabel("Ready")
        layout.addWidget(self._progress_label)

        log_layout = QVBoxLayout()
        log_layout.addWidget(QLabel("Pipeline Log:"))
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFontFamily("Courier")
        log_layout.addWidget(self._log, stretch=1)
        clear_btn = QPushButton("Clear Log")
        clear_btn.clicked.connect(self._log.clear)
        log_layout.addWidget(clear_btn)
        layout.addLayout(log_layout, stretch=1)

    # ── Config building ───────────────────────────────────────────────────────

    def _build_config(self) -> EcomodelConfig:
        return EcomodelConfig(
            input_folder=self._input_folder.text().strip(),
            results_folder=self._results_folder.text().strip(),
            cylinder_filename=self._cylinder_filename.text().strip(),
            cube_size=self._cube_size.value(),
            meter_conversion=self._meter_conversion.value(),
            csf_band_size=self._csf_band_size.value(),
            csf_threshold=self._csf_threshold.value(),
            csf_offset=self._csf_offset.value(),
            remove_under_ground=self._remove_underground.isChecked(),
            terrain_grid_size=self._terrain_grid_size.value(),
            use_denoise=self._use_denoise.isChecked(),
            denoise_grid_size=self._denoise_grid_size.value(),
            denoise_min_points=self._denoise_min_points.value(),
            denoise_resolution=self._denoise_resolution.value(),
            remove_duplicates=self._remove_duplicates.isChecked(),
            segment_intensity_threshold=self._seg_intensity.value(),
            save_clusters=self._save_clusters.isChecked(),
            qsm_intensity_threshold=self._qsm_intensity.value(),
            save_leaf_removal_output=self._save_leaf_output.isChecked(),
            run_qsm=self._run_qsm.isChecked(),
            run_leaf_removal=self._run_leaf_removal.isChecked(),
            create_cylinder_plot=self._create_cylinder_plot.isChecked(),
            patch_diam1=self._patch_diam1.value(),
            ball_rad1=self._ball_rad1.value(),
            nmin1=self._nmin1.value(),
            patch_diam2_min=self._patch_diam2_min.value(),
            patch_diam2_max=self._patch_diam2_max.value(),
            ball_rad2=self._ball_rad2.value(),
            rgi_noise_percentile=self._rgi_noise_percentile.value(),
            rgi_angle_deg=self._rgi_angle_deg.value(),
            rgi_curv_thresh=self._rgi_curv_thresh.value(),
            rgi_resid_thresh=self._rgi_resid_thresh.value(),
            rgi_k=self._rgi_k.value(),
            rgi_min_cluster_size=self._rgi_min_cluster_size.value(),
            rgi_max_cluster_size=self._rgi_max_cluster_size.value(),
            rgi_smooth_mode=self._rgi_smooth_mode.isChecked(),
            rgi_use_residual_test=self._rgi_use_residual.isChecked(),
            rgi_use_curvature_test=self._rgi_use_curvature.isChecked(),
            checkpoint_name=self._checkpoint_name.text().strip(),
            checkpoint_folder=self._checkpoint_folder.text().strip(),
            use_checkpoint=self._use_checkpoint.isChecked(),
            save_checkpoint=self._save_checkpoint.isChecked(),
            checkpoint_resume_file=self._checkpoint_resume_file.text().strip(),
        )

    # ── Pipeline control ──────────────────────────────────────────────────────

    def _start_pipeline(self) -> None:
        config = self._build_config()

        # When resuming from a specific checkpoint file the original LAS/LAZ
        # folder may no longer exist — the pickled Ecomodel already contains
        # all the loaded point data.  Only validate the input folder when a
        # fresh run is needed (not resuming from an explicit checkpoint file).
        _needs_input_folder = not (
            config.use_checkpoint and config.checkpoint_resume_file
        )
        if _needs_input_folder:
            if not config.input_folder:
                QMessageBox.warning(self, "Missing Input", "Please select a LAS/LAZ input folder.")
                return
            if not Path(config.input_folder).is_dir():
                QMessageBox.warning(
                    self,
                    "Folder Not Found",
                    f"Input folder does not exist:\n{config.input_folder}",
                )
                return

        self._log.clear()
        self._progress_bar.setValue(0)
        self._progress_label.setText("Starting...")
        self._run_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)

        self._thread = QThread()
        self._worker = EcomodelWorker(config)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.log.connect(self._append_log)
        self._worker.progress.connect(self._update_progress)
        self._worker.tile_update.connect(self._on_tile_update)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.finished.connect(self._on_thread_done)

        self._thread.start()

    def _stop_pipeline(self) -> None:
        if self._worker:
            self._worker.request_stop()
            self._stop_btn.setEnabled(False)
            self._append_log("[GUI] Stop requested — pipeline will halt after the current step.\n")

    # ── Reset handlers ────────────────────────────────────────────────────────

    def _reset_io_defaults(self) -> None:
        d = self._defaults
        self._results_folder.setText(d.results_folder)
        self._cylinder_filename.setText(d.cylinder_filename)

    def _reset_subdivision_defaults(self) -> None:
        d = self._defaults
        self._cube_size.setValue(d.cube_size)
        self._meter_conversion.setValue(d.meter_conversion)

    def _reset_csf_defaults(self) -> None:
        d = self._defaults
        self._csf_band_size.setValue(d.csf_band_size)
        self._csf_threshold.setValue(d.csf_threshold)
        self._csf_offset.setValue(d.csf_offset)
        self._remove_underground.setChecked(d.remove_under_ground)

    def _reset_terrain_defaults(self) -> None:
        d = self._defaults
        self._terrain_grid_size.setValue(d.terrain_grid_size)

    def _reset_cleanup_defaults(self) -> None:
        d = self._defaults
        self._remove_duplicates.setChecked(d.remove_duplicates)
        self._use_denoise.setChecked(d.use_denoise)
        self._denoise_grid_size.setValue(d.denoise_grid_size)
        self._denoise_min_points.setValue(d.denoise_min_points)
        self._denoise_resolution.setValue(d.denoise_resolution)

    def _reset_segmentation_defaults(self) -> None:
        d = self._defaults
        self._seg_intensity.setValue(d.segment_intensity_threshold)
        self._save_clusters.setChecked(d.save_clusters)

    def _reset_qsm_defaults(self) -> None:
        d = self._defaults
        self._run_qsm.setChecked(d.run_qsm)
        self._run_leaf_removal.setChecked(getattr(d, "run_leaf_removal", True))
        self._qsm_intensity.setValue(d.qsm_intensity_threshold)
        self._save_leaf_output.setChecked(d.save_leaf_removal_output)
        self._create_cylinder_plot.setChecked(d.create_cylinder_plot)

    def _reset_cover_defaults(self) -> None:
        d = self._defaults
        self._patch_diam1.setValue(d.patch_diam1)
        self._ball_rad1.setValue(d.ball_rad1)
        self._nmin1.setValue(d.nmin1)
        self._patch_diam2_min.setValue(d.patch_diam2_min)
        self._patch_diam2_max.setValue(d.patch_diam2_max)
        self._ball_rad2.setValue(d.ball_rad2)

    def _reset_rgi_defaults(self) -> None:
        d = self._defaults
        self._rgi_noise_percentile.setValue(d.rgi_noise_percentile)
        self._rgi_angle_deg.setValue(d.rgi_angle_deg)
        self._rgi_curv_thresh.setValue(d.rgi_curv_thresh)
        self._rgi_resid_thresh.setValue(d.rgi_resid_thresh)
        self._rgi_k.setValue(d.rgi_k)
        self._rgi_min_cluster_size.setValue(d.rgi_min_cluster_size)
        self._rgi_max_cluster_size.setValue(d.rgi_max_cluster_size)
        self._rgi_smooth_mode.setChecked(d.rgi_smooth_mode)
        self._rgi_use_residual.setChecked(d.rgi_use_residual_test)
        self._rgi_use_curvature.setChecked(d.rgi_use_curvature_test)

    def _reset_checkpoint_defaults(self) -> None:
        d = self._defaults
        self._checkpoint_name.setText(d.checkpoint_name)
        self._checkpoint_folder.setText(d.checkpoint_folder)
        self._use_checkpoint.setChecked(d.use_checkpoint)
        self._save_checkpoint.setChecked(getattr(d, "save_checkpoint", False))
        self._checkpoint_resume_file.setText(getattr(d, "checkpoint_resume_file", ""))

    def _on_checkpoint_section_toggled(self, checked: bool) -> None:
        """Re-apply the resume-file enable state after the section is unlocked."""
        if checked:
            resume_on = self._use_checkpoint.isChecked()
            self._checkpoint_resume_file.setEnabled(resume_on)
            self._browse_resume_btn.setEnabled(resume_on)

    def _browse_checkpoint_folder(self) -> None:
        from PySide6.QtWidgets import QFileDialog
        folder = QFileDialog.getExistingDirectory(
            self, "Select Checkpoint Folder",
            self._checkpoint_folder.text() or "."
        )
        if folder:
            self._checkpoint_folder.setText(folder)

    def _browse_checkpoint_resume_file(self) -> None:
        from PySide6.QtWidgets import QFileDialog
        start = (
            self._checkpoint_folder.text().strip()
            or self._results_folder.text().strip()
            or "."
        )
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Checkpoint File", start,
            "Pickle files (*.pickle);;All files (*)"
        )
        if not path:
            return
        self._checkpoint_resume_file.setText(path)
        # Auto-populate folder and name from the selected file
        p = Path(path)
        self._checkpoint_folder.setText(str(p.parent))
        stem = p.stem   # e.g. "ecomodel_checkpoint_post_normalize"
        for tag in ("_post_normalize", "_post_segmentation", "_post_qsm"):
            if stem.endswith(tag):
                stem = stem[: -len(tag)]
                break
        self._checkpoint_name.setText(stem)

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _append_log(self, msg: str) -> None:
        self._log.insertPlainText(msg)
        self._log.moveCursor(QTextCursor.End)

    def _update_progress(self, current: int, total: int, label: str) -> None:
        self._progress_bar.setMaximum(total)
        self._progress_bar.setValue(current)
        self._progress_label.setText(label)
        self.statusBar().showMessage(f"Step {current}/{total}: {label}")

    def _on_finished(self, eco) -> None:
        if eco is None:
            self._append_log("[GUI] Pipeline was stopped by user.\n")
            self._progress_label.setText("Stopped")
            self._progress_bar.setValue(0)
            self.statusBar().showMessage("Pipeline stopped")
        else:
            results = Path(eco.results_folder).resolve()
            self._append_log("[GUI] Pipeline completed successfully.\n")
            self._append_log(f"[GUI] Results written to: {results}\n")
            self._progress_label.setText("Done")
            self.statusBar().showMessage(f"Done — results in {results}")
            page = self._show_results_page()
            page.notify_run_complete(results)

    def _on_error(self, tb: str) -> None:
        self._append_log(f"[ERROR]\n{tb}\n")
        QMessageBox.critical(self, "Pipeline Error", tb[:600])
        self._progress_label.setText("Error")

    def _on_thread_done(self) -> None:
        self._run_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        if self._thread:
            self._thread.deleteLater()
            self._thread = None
        if self._worker:
            self._worker.deleteLater()
            self._worker = None

    def closeEvent(self, event) -> None:
        """Stop the pipeline worker and drain all background tasks cleanly."""
        # Stop the pipeline if it is running
        if self._worker is not None:
            self._worker.request_stop()
            if self._thread is not None:
                self._thread.quit()
                self._thread.wait(5_000)

        # Cancel and wait for every in-flight BgTask (scan / load / render)
        from gui.worker import cancel_all_bg_tasks
        cancel_all_bg_tasks(3_000)

        super().closeEvent(event)

    def _on_tile_update(self, tile_name: str, status: str, cyl_count: int, output_path: str) -> None:
        parts = [f"[Tile] {tile_name}: {status}"]
        if cyl_count >= 0:
            parts.append(f"{cyl_count} cylinders")
        if output_path:
            parts.append(output_path)
        self._append_log(" — ".join(parts) + "\n")

    # ── Results page navigation ───────────────────────────────────────────────

    def _show_results_page(self) -> ResultsPage:
        """Sync the results folder and flip to the results page."""
        folder = self._results_folder.text().strip()
        self._results_page.set_results_folder(folder)
        self._query_page.set_results_folder(folder)
        self._stack.setCurrentIndex(1)
        return self._results_page

    def _show_query_page(self) -> QueryPage:
        """Sync the results folder and flip to the query page."""
        folder = self._results_folder.text().strip()
        self._query_page.set_results_folder(folder)
        self._stack.setCurrentIndex(2)
        return self._query_page

    # ── File dialogs ──────────────────────────────────────────────────────────

    def _browse_input(self) -> None:
        start = self._input_folder.text().strip() or "."
        path, _ = QFileDialog.getOpenFileName(
            self, "Select a LAS/LAZ File", start,
            "LiDAR files (*.las *.laz);;All files (*)"
        )
        if path:
            folder = str(Path(path).parent)
            self._input_folder.setText(folder)
            self._auto_populate_names(folder)

    def _auto_populate_names(self, folder: str) -> None:
        """
        Derive cylinder filename and checkpoint name from the input folder.

        Uses the LAS/LAZ file stem if there is exactly one file in the folder,
        otherwise uses the folder name itself.  Replaces non-word characters
        with underscores so the names are safe to use as file/folder names.
        """
        import os
        import re
        from pathlib import Path

        p = Path(folder)
        try:
            las_files = [f for f in os.listdir(folder)
                         if f.lower().endswith((".las", ".laz"))]
        except OSError:
            las_files = []

        stem = Path(las_files[0]).stem if len(las_files) == 1 else p.name
        stem = re.sub(r"[^\w]", "_", stem).strip("_") or "ecomodel"

        self._cylinder_filename.setText(stem)
        self._checkpoint_name.setText(f"{stem}_checkpoint")

    def _browse_results(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Results Folder")
        if folder:
            self._results_folder.setText(folder)

    # ── Widget factory helpers ────────────────────────────────────────────────

    @staticmethod
    def _spin(lo: int, hi: int, val: int) -> QSpinBox:
        w = QSpinBox()
        w.setRange(lo, hi)
        w.setValue(val)
        return w

    @staticmethod
    def _dspin(lo: float, hi: float, val: float, dec: int = 3) -> QDoubleSpinBox:
        w = QDoubleSpinBox()
        w.setRange(lo, hi)
        w.setValue(val)
        w.setDecimals(dec)
        return w