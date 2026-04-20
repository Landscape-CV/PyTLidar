"""
Query Mode page for EcomodelMainWindow.

Classes
-------
QueryPage
    QWidget page (lives at QStackedWidget index 2) that lets the user pick a
    completed run, enter a world coordinate and voxel size, and get a full
    vegetation breakdown for that region.

Threading model
---------------
All file I/O and computation runs on background threads (BgTask):

  BgTask(_bg_scan_runs, folder)      → scan results folder (shared with ResultsPage)
  BgTask(_bg_engine_load, run_dir)   → QueryEngine.load() (np.load × 2 + np.loadtxt)
  BgTask(_bg_query, engine, ...)     → query_voxel() + build_voxel_query_meshes()

Only EmbeddedPlotWidget.show_pyvista_meshes() touches the plotter on the
main thread.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from gui.results_widgets import EmbeddedPlotWidget


# ── Background worker free functions ─────────────────────────────────────────

def _bg_build_cloud_meshes(cloud) -> list:
    """Build point-cloud mesh_list from a numpy array (no disk I/O)."""
    from plotting.pv_rendering import build_point_cloud_meshes
    mesh_list, _ = build_point_cloud_meshes(cloud, None, "Height (Z)")
    return mesh_list


def _bg_build_cyl_meshes(cyls: dict) -> dict:
    """Build cylinder mesh_list from an already-loaded cyls dict (no disk I/O)."""
    from plotting.pv_rendering import build_cylinder_meshes
    mesh_list, starts, ends, radii, lengths = build_cylinder_meshes(cyls)
    return {"mesh_list": mesh_list, "starts": starts, "ends": ends,
            "radii": radii, "lengths": lengths}


def _bg_scan_runs(results_folder: str) -> list:
    """Re-use the same scanner as ResultsPage (returns list of (label, Path))."""
    from gui.results_io import find_run_dirs, load_run_metadata
    runs = find_run_dirs(results_folder) if results_folder else []
    items = []
    for run_dir in runs:
        meta = load_run_metadata(run_dir)
        label = f"{run_dir.name}  ({meta.timestamp})" if meta else run_dir.name
        items.append((label, run_dir))
    return items


def _bg_engine_load(run_dir: Path) -> tuple:
    """Load a QueryEngine on the background thread.  Returns (engine, error_str|None)."""
    from gui.query_engine import QueryEngine
    engine = QueryEngine(run_dir)
    err = engine.load()
    return engine, err


def _bg_query(engine, wx, wy, wz, voxel_size,
              coloring="segment", cover_sets=None) -> tuple:
    """Run query_voxel and build voxel meshes on the background thread."""
    result = engine.query_voxel(wx, wy, wz, voxel_size)
    from plotting.pv_rendering import build_voxel_query_meshes
    mesh_list = build_voxel_query_meshes(
        engine.cloud, engine.labels, result, wx, wy, wz,
        cover_sets=cover_sets,
        coloring=coloring,
    )
    return result, mesh_list


# ── QueryPage ─────────────────────────────────────────────────────────────────

class QueryPage(QWidget):
    """
    Inline voxel-query page.

    Emits ``back_requested`` when the user clicks "← Back".
    Emits ``status_message(str)`` for status-bar updates.
    """

    back_requested = Signal()
    status_message = Signal(str)

    def __init__(self, results_folder: str = "results", parent=None) -> None:
        super().__init__(parent)
        self._results_folder: str = results_folder
        self._run_dirs: list[Path] = []
        self._engine = None

        # Monotonic counters to discard stale background results
        self._load_seq:   int = 0
        self._render_seq: int = 0

        # Keep task references alive
        self._scan_task   = None
        self._load_task   = None
        self._render_task = None

        # View-layer caches
        self._cloud_cache:  "list | None" = None   # cached point-cloud mesh_list
        self._cyls_cache:   "dict | None" = None   # cached {mesh_list, …} from _bg_build_cyl_meshes
        self._bg_meshes:    "list | None" = None   # currently active background layer
        self._query_meshes: "list | None" = None   # latest voxel-query overlay

        # Lon/lat input mode
        self._lonlat_mode: bool = False

        # Point coloring mode
        self._coloring: str = "segment"   # "segment" | "cover_set"

        self._build_ui()
        self._refresh_runs()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)

        # ── Shared top bar (Back + Run selector) ──────────────────────────────
        top_row = QHBoxLayout()
        self._back_btn = QPushButton("← Back")
        self._back_btn.setToolTip("Return to the results page")
        self._back_btn.clicked.connect(self.back_requested)
        top_row.addWidget(self._back_btn)

        top_row.addSpacing(8)
        top_row.addWidget(QLabel("Run:"))

        self._run_combo = QComboBox()
        self._run_combo.setToolTip("Select a completed pipeline run to query")
        self._run_combo.setMinimumWidth(260)
        self._run_combo.currentIndexChanged.connect(self._on_run_changed)
        top_row.addWidget(self._run_combo, stretch=1)

        self._refresh_btn = QPushButton("↺")
        self._refresh_btn.setFixedWidth(28)
        self._refresh_btn.setToolTip("Rescan results folder")
        self._refresh_btn.clicked.connect(self._refresh_runs)
        top_row.addWidget(self._refresh_btn)

        root.addLayout(top_row)

        self._load_status_label = QLabel()
        self._load_status_label.setWordWrap(True)
        self._load_status_label.setStyleSheet("color: #888; font-size: 11px;")
        root.addWidget(self._load_status_label)

        # ── Tabs: Single Point / Batch CSV ────────────────────────────────────
        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_single_point_tab(), "Single Point")
        self._tabs.addTab(self._build_batch_tab(), "Batch CSV")
        root.addWidget(self._tabs, stretch=1)

    def _build_single_point_tab(self) -> QWidget:
        page = QWidget()
        page_layout = QHBoxLayout(page)
        page_layout.setContentsMargins(0, 4, 0, 0)

        splitter = QSplitter(Qt.Horizontal)
        page_layout.addWidget(splitter)

        # ── Left pane ─────────────────────────────────────────────────────────
        left = QWidget()
        left.setMinimumWidth(230)
        left.setMaximumWidth(320)
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 4, 0)

        coord_box = QGroupBox("Query Region")
        coord_form = QFormLayout(coord_box)

        # ── XY spinboxes (always present) ─────────────────────────────────
        self._spin_x = self._dspin()
        self._spin_y = self._dspin()
        self._spin_z = self._dspin()

        self._xy_x_row_label = QLabel("X:")
        self._xy_y_row_label = QLabel("Y:")
        coord_form.addRow(self._xy_x_row_label, self._spin_x)
        coord_form.addRow(self._xy_y_row_label, self._spin_y)

        # ── Lon/Lat spinboxes (shown when lon/lat mode is active) ─────────
        self._spin_lon = QDoubleSpinBox()
        self._spin_lon.setRange(-180.0, 180.0)
        self._spin_lon.setDecimals(8)
        self._spin_lon.setSingleStep(0.0001)
        self._spin_lon.setToolTip("WGS84 Longitude (°E)")

        self._spin_lat = QDoubleSpinBox()
        self._spin_lat.setRange(-90.0, 90.0)
        self._spin_lat.setDecimals(8)
        self._spin_lat.setSingleStep(0.0001)
        self._spin_lat.setToolTip("WGS84 Latitude (°N)")

        self._lon_row_label = QLabel("Lon (°E):")
        self._lat_row_label = QLabel("Lat (°N):")
        coord_form.addRow(self._lon_row_label, self._spin_lon)
        coord_form.addRow(self._lat_row_label, self._spin_lat)
        # Hide lon/lat rows until toggled
        self._lon_row_label.setVisible(False)
        self._spin_lon.setVisible(False)
        self._lat_row_label.setVisible(False)
        self._spin_lat.setVisible(False)

        coord_form.addRow("Z:", self._spin_z)

        self._spin_voxel = QDoubleSpinBox()
        self._spin_voxel.setRange(0.1, 500.0)
        self._spin_voxel.setDecimals(2)
        self._spin_voxel.setValue(1.0)
        self._spin_voxel.setSingleStep(0.5)
        self._spin_voxel.setSuffix(" m")
        self._spin_voxel.setToolTip("Side length of the axis-aligned query cube")
        coord_form.addRow("Voxel size:", self._spin_voxel)

        # ── Lon/Lat toggle ────────────────────────────────────────────────
        self._lonlat_toggle = QCheckBox("Use Lon/Lat (WGS84)")
        self._lonlat_toggle.setToolTip(
            "Enter geographic coordinates instead of projected X/Y.\n"
            "Requires the source LAS files to contain an embedded CRS."
        )
        self._lonlat_toggle.setEnabled(False)   # enabled once CRS is confirmed
        self._lonlat_toggle.toggled.connect(self._on_lonlat_toggled)
        coord_form.addRow("", self._lonlat_toggle)

        # CRS status line
        self._crs_label = QLabel("")
        self._crs_label.setStyleSheet("color: #888; font-size: 10px;")
        self._crs_label.setWordWrap(True)
        coord_form.addRow("", self._crs_label)

        lv.addWidget(coord_box)

        self._query_btn = QPushButton("Query Voxel")
        self._query_btn.setStyleSheet("font-weight: bold; padding: 5px;")
        self._query_btn.setEnabled(False)
        self._query_btn.clicked.connect(self._run_query)
        lv.addWidget(self._query_btn)

        self._result_label = QLabel("Select a run and enter a coordinate.")
        self._result_label.setWordWrap(True)
        self._result_label.setStyleSheet(
            "color: #333; font-size: 11px; font-family: monospace;"
        )
        self._result_label.setAlignment(Qt.AlignTop)
        lv.addWidget(self._result_label, stretch=1)

        # ── Point Coloring group ──────────────────────────────────────────
        coloring_box = QGroupBox("Point Coloring")
        coloring_layout = QVBoxLayout(coloring_box)

        self._chk_cover_sets = QCheckBox("Color by Cover Sets")
        self._chk_cover_sets.setToolTip(
            "Color interior voxel points by vegetation cover-set ID.\n"
            "Available only when cover_sets.npy exists for this run."
        )
        self._chk_cover_sets.setEnabled(False)
        self._chk_cover_sets.toggled.connect(self._on_coloring_changed)
        coloring_layout.addWidget(self._chk_cover_sets)

        lv.addWidget(coloring_box)

        splitter.addWidget(left)

        # ── Right pane ────────────────────────────────────────────────────────
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(4, 0, 0, 0)

        # ── Background layer buttons ──────────────────────────────────────
        view_btn_row = QHBoxLayout()
        view_btn_row.addWidget(QLabel("Background:"))
        self._btn_cloud = QPushButton("Point Cloud")
        self._btn_cloud.setToolTip("Show point cloud as background (query overlays on top)")
        self._btn_cyls  = QPushButton("Cylinders")
        self._btn_cyls.setToolTip("Show QSM cylinders as background (query overlays on top)")
        for _b in (self._btn_cloud, self._btn_cyls):
            _b.setEnabled(False)
            view_btn_row.addWidget(_b)
        view_btn_row.addStretch()
        rv.addLayout(view_btn_row)

        self._plot = EmbeddedPlotWidget()
        self._plot._placeholder.setText(
            "Select a run, enter a coordinate, and click 'Query Voxel'."
        )
        rv.addWidget(self._plot, stretch=1)

        self._info_label = QLabel("")
        self._info_label.setStyleSheet("color: #666; font-size: 11px;")
        rv.addWidget(self._info_label)

        splitter.addWidget(right)
        splitter.setSizes([270, 830])

        self._btn_cloud.clicked.connect(self._show_cloud)
        self._btn_cyls.clicked.connect(self._show_cylinders)

        return page

    # ── Batch CSV tab ─────────────────────────────────────────────────────────

    def _build_batch_tab(self) -> QWidget:
        """
        CSV-driven batch-query page.

        Runs ``engine.query_voxel_lonlat`` for every row × every radius and
        writes a results CSV. Intended for workflows like the Lizard Island
        study: thousands of GPS observations, each characterised by
        vegetation structure within 1m / 2m / 3m.
        """
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(6, 8, 6, 6)

        self._batch_worker: "BatchQueryWorker | None" = None
        self._batch_thread: "QThread | None"          = None

        # ── File pickers ──────────────────────────────────────────────────────
        io_box = QGroupBox("CSV Files")
        io_form = QFormLayout(io_box)

        in_row = QHBoxLayout()
        self._batch_in_edit = QLineEdit()
        self._batch_in_edit.setPlaceholderText("Input CSV with observation_id, lon, lat, [height_agl]")
        btn_in = QPushButton("Browse…")
        btn_in.clicked.connect(self._on_batch_pick_input)
        in_row.addWidget(self._batch_in_edit, stretch=1)
        in_row.addWidget(btn_in)
        io_form.addRow("Input CSV:", self._wrap(in_row))

        out_row = QHBoxLayout()
        self._batch_out_edit = QLineEdit()
        self._batch_out_edit.setPlaceholderText("Output CSV (one row per observation, metric blocks per radius)")
        btn_out = QPushButton("Browse…")
        btn_out.clicked.connect(self._on_batch_pick_output)
        out_row.addWidget(self._batch_out_edit, stretch=1)
        out_row.addWidget(btn_out)
        io_form.addRow("Output CSV:", self._wrap(out_row))

        layout.addWidget(io_box)

        # ── Parameters ────────────────────────────────────────────────────────
        params_box = QGroupBox("Parameters")
        params_form = QFormLayout(params_box)

        radii_row = QHBoxLayout()
        self._batch_radii: list[QDoubleSpinBox] = []
        for default in (1.0, 2.0, 3.0):
            sb = QDoubleSpinBox()
            sb.setRange(0.1, 500.0)
            sb.setDecimals(2)
            sb.setSingleStep(0.5)
            sb.setSuffix(" m")
            sb.setValue(default)
            self._batch_radii.append(sb)
            radii_row.addWidget(sb)
        radii_row.addStretch()
        params_form.addRow("Radii (cube sides):", self._wrap(radii_row))

        self._batch_h_agl = QDoubleSpinBox()
        self._batch_h_agl.setRange(-100.0, 100.0)
        self._batch_h_agl.setDecimals(2)
        self._batch_h_agl.setSingleStep(0.5)
        self._batch_h_agl.setSuffix(" m")
        self._batch_h_agl.setValue(2.0)
        self._batch_h_agl.setToolTip(
            "Default height above terrain, used when the CSV has no "
            "'height_agl' column or leaves a cell blank."
        )
        params_form.addRow("Default height AGL:", self._batch_h_agl)

        layout.addWidget(params_box)

        # ── Run controls ──────────────────────────────────────────────────────
        run_row = QHBoxLayout()
        self._batch_run_btn = QPushButton("Run Batch")
        self._batch_run_btn.setStyleSheet("font-weight: bold; padding: 5px;")
        self._batch_run_btn.clicked.connect(self._on_batch_run_clicked)
        run_row.addWidget(self._batch_run_btn)

        self._batch_stop_btn = QPushButton("Stop")
        self._batch_stop_btn.setEnabled(False)
        self._batch_stop_btn.clicked.connect(self._on_batch_stop_clicked)
        run_row.addWidget(self._batch_stop_btn)

        self._batch_progress = QProgressBar()
        self._batch_progress.setRange(0, 0)   # indeterminate until a count is known
        self._batch_progress.setValue(0)
        self._batch_progress.setFormat("Idle")
        run_row.addWidget(self._batch_progress, stretch=1)

        layout.addLayout(run_row)

        # ── Log pane ──────────────────────────────────────────────────────────
        self._batch_log = QPlainTextEdit()
        self._batch_log.setReadOnly(True)
        self._batch_log.setPlaceholderText(
            "Batch log — warnings, skipped rows, and final summary appear here."
        )
        self._batch_log.setStyleSheet("font-family: monospace; font-size: 11px;")
        layout.addWidget(self._batch_log, stretch=1)

        return page

    @staticmethod
    def _wrap(layout) -> QWidget:
        """Wrap a QHBoxLayout in a QWidget so it can fit into a QFormLayout row."""
        w = QWidget()
        layout.setContentsMargins(0, 0, 0, 0)
        w.setLayout(layout)
        return w

    # ── Public API ────────────────────────────────────────────────────────────

    def set_results_folder(self, folder: str) -> None:
        if folder != self._results_folder:
            self._results_folder = folder
            self._refresh_runs()

    # ── Run scanning ──────────────────────────────────────────────────────────

    def _refresh_runs(self) -> None:
        """Rescan results folder on a background thread."""
        from gui.worker import BgTask

        prev_text = self._run_combo.currentText()
        self._run_combo.blockSignals(True)
        self._run_combo.clear()
        self._run_combo.blockSignals(False)
        self._load_status_label.setText("Scanning…")

        folder = self._results_folder
        task = BgTask(_bg_scan_runs, folder)
        task.result.connect(lambda items: self._populate_run_combo(items, prev_text))
        task.error.connect(lambda _tb: self._load_status_label.setText("Scan error"))
        task.start()
        self._scan_task = task

    def _populate_run_combo(self, items: list, prev_text: str) -> None:
        self._run_combo.blockSignals(True)
        self._run_dirs = []
        restore_idx = 0
        for i, (label, run_dir) in enumerate(items):
            self._run_combo.addItem(label)
            self._run_dirs.append(run_dir)
            if label == prev_text:
                restore_idx = i
        self._run_combo.blockSignals(False)

        if items:
            self._run_combo.setCurrentIndex(restore_idx)
            self._on_run_changed(restore_idx)
        else:
            self._query_btn.setEnabled(False)
            self._load_status_label.setText("No runs found in results folder.")

    # ── Engine loading ────────────────────────────────────────────────────────

    def _on_run_changed(self, index: int) -> None:
        if index < 0 or index >= len(self._run_dirs):
            return
        from gui.worker import BgTask

        run_dir = self._run_dirs[index]
        self._engine = None
        self._plot.clear()
        self._result_label.setText("Loading run data…")
        self._query_btn.setEnabled(False)
        self._load_status_label.setText("Loading…")

        self._load_seq += 1
        seq = self._load_seq

        task = BgTask(_bg_engine_load, run_dir)
        task.result.connect(lambda r: self._on_engine_loaded(seq, r[0], r[1], run_dir))
        task.error.connect(lambda tb: self._on_engine_error(seq, tb))
        task.start()
        self._load_task = task

    def _on_engine_loaded(self, seq: int, engine, err, run_dir: Path) -> None:
        if seq != self._load_seq:
            return
        if err:
            self._load_status_label.setText(f"⚠ {err}")
            self._result_label.setText("Could not load run data.")
            return

        self._engine = engine
        self._query_btn.setEnabled(True)

        # Background layer buttons
        self._btn_cloud.setEnabled(True)
        self._btn_cyls.setEnabled(engine.has_cyls)

        # Clear per-run caches
        self._cloud_cache  = None
        self._cyls_cache   = None
        self._bg_meshes    = None
        self._query_meshes = None

        # CRS / lon-lat toggle
        self._lonlat_mode = False
        self._lonlat_toggle.blockSignals(True)
        self._lonlat_toggle.setChecked(False)
        self._lonlat_toggle.blockSignals(False)
        self._on_lonlat_toggled(False)   # ensure XY rows visible

        if engine.has_crs:
            self._lonlat_toggle.setEnabled(True)
            self._crs_label.setText(f"CRS: EPSG:{engine.epsg}")
        else:
            self._lonlat_toggle.setEnabled(False)
            self._crs_label.setText("No CRS in source LAS — X/Y only")

        # Point coloring checkboxes
        self._chk_cover_sets.setEnabled(engine.has_cover_sets)
        self._chk_cover_sets.blockSignals(True)
        self._chk_cover_sets.setChecked(False)
        self._chk_cover_sets.blockSignals(False)
        self._coloring = "segment"

        flags = []
        if engine.has_mean:
            mx, my, mz = engine.mean
            self._spin_x.setValue(mx)
            self._spin_y.setValue(my)
            self._spin_z.setValue(mz)
            flags.append(f"Origin: ({mx:.1f}, {my:.1f}, {mz:.1f})")
        else:
            flags.append("⚠ No origin saved — enter raw normalised coords")

        n_pts = len(engine.cloud)
        flags.append(f"{n_pts:,} pts")
        if engine.has_labels:
            flags.append("labels ✓")
        if engine.has_cyls:
            flags.append("cylinders ✓")

        self._load_status_label.setText("Loaded  ·  " + "\n".join(flags))
        self._result_label.setText(
            "Enter a coordinate and voxel size,\nthen click 'Query Voxel'."
        )
        self.status_message.emit(f"Query ready: {run_dir.name}")

        # Auto-load the point cloud as background so it's ready before the
        # first query runs.
        self._show_cloud()

    def _on_engine_error(self, seq: int, tb: str) -> None:
        if seq != self._load_seq:
            return
        self._load_status_label.setText("⚠ Load failed")
        self._result_label.setText(f"Load error:\n{tb[:200]}")

    # ── Lon/Lat toggle ────────────────────────────────────────────────────────

    def _on_lonlat_toggled(self, checked: bool) -> None:
        """Show lon/lat spinboxes and hide X/Y rows (or vice-versa)."""
        self._lonlat_mode = checked
        self._xy_x_row_label.setVisible(not checked)
        self._spin_x.setVisible(not checked)
        self._xy_y_row_label.setVisible(not checked)
        self._spin_y.setVisible(not checked)
        self._lon_row_label.setVisible(checked)
        self._spin_lon.setVisible(checked)
        self._lat_row_label.setVisible(checked)
        self._spin_lat.setVisible(checked)

    # ── Point coloring ────────────────────────────────────────────────────────

    def _on_coloring_changed(self, _checked: bool) -> None:
        """Switch between segment and cover-set coloring."""
        self._coloring = "cover_set" if self._chk_cover_sets.isChecked() else "segment"
        # Re-run query immediately if a result is already showing
        if self._query_meshes is not None:
            self._run_query()

    # ── Query execution ───────────────────────────────────────────────────────

    def _run_query(self) -> None:
        if self._engine is None or not self._engine.is_loaded:
            return
        from gui.worker import BgTask

        voxel_size = self._spin_voxel.value()
        wz = self._spin_z.value()

        if self._lonlat_mode:
            try:
                wx, wy = self._engine.lonlat_to_norm(
                    self._spin_lon.value(), self._spin_lat.value()
                )
            except Exception as exc:
                self._result_label.setText(f"Coordinate error:\n{exc}")
                return
        else:
            wx = self._spin_x.value()
            wy = self._spin_y.value()

        self._query_btn.setEnabled(False)
        self._result_label.setText("Computing query…")

        self._render_seq += 1
        seq = self._render_seq
        engine = self._engine

        cover_sets = engine.cover_sets if self._coloring == "cover_set" else None

        task = BgTask(_bg_query, engine, wx, wy, wz, voxel_size,
                      self._coloring, cover_sets)
        task.result.connect(
            lambda r: self._on_query_done(seq, r[0], r[1], wx, wy, wz)
        )
        task.error.connect(lambda tb: self._on_query_error(seq, tb))
        task.start()
        self._render_task = task

    def _on_query_done(
        self, seq: int, result, mesh_list: list,
        wx: float, wy: float, wz: "float | None",
    ) -> None:
        if seq != self._render_seq:
            return
        self._query_meshes = mesh_list
        self._query_btn.setEnabled(True)
        self._show_result(result, wx, wy, wz)
        self._render_combined()

    def _on_query_error(self, seq: int, tb: str) -> None:
        if seq != self._render_seq:
            return
        self._query_btn.setEnabled(True)
        self._result_label.setText(f"Query error:\n{tb[:200]}")

    # ── Background layer + overlay rendering ────────────────────────────────

    def _render_combined(self) -> None:
        """Re-render background + voxel-query overlay in a single plotter call."""
        combined: list = []
        if self._bg_meshes:
            combined.extend(self._bg_meshes)
        if self._query_meshes:
            combined.extend(self._query_meshes)
        if combined:
            self._plot.show_pyvista_meshes(combined)

    def _show_cloud(self) -> None:
        """Switch background to the full-run point cloud; keep any query overlay."""
        if self._engine is None:
            return
        if self._cloud_cache is not None:
            self._bg_meshes = self._cloud_cache
            self._render_combined()
            return
        from gui.worker import BgTask
        self._render_seq += 1
        seq = self._render_seq
        task = BgTask(_bg_build_cloud_meshes, self._engine.cloud)
        task.result.connect(lambda ml: self._on_cloud_ready(seq, ml))
        task.error.connect(lambda _tb: None)
        task.start()
        self._render_task = task

    def _on_cloud_ready(self, seq: int, mesh_list: list) -> None:
        if seq != self._render_seq:
            return
        self._cloud_cache = mesh_list
        self._bg_meshes   = mesh_list
        self._render_combined()

    def _show_cylinders(self) -> None:
        """Switch background to QSM cylinders; keep any query overlay."""
        if self._engine is None or not self._engine.has_cyls:
            return
        if self._cyls_cache is not None:
            self._bg_meshes = self._cyls_cache["mesh_list"]
            self._render_combined()
            return
        from gui.worker import BgTask
        self._render_seq += 1
        seq = self._render_seq

        # Cylinder starts are stored in world space in the .txt file.
        # The point cloud snapshot and voxel-query bounds are in normalised
        # space (world - cloud_mean), so we subtract the mean here to align.
        raw = self._engine.cyls
        mean = np.array(self._engine.mean, dtype=np.float64)
        cyls_norm = dict(raw)
        cyls_norm["start"] = raw["start"] - mean

        task = BgTask(_bg_build_cyl_meshes, cyls_norm)
        task.result.connect(lambda r: self._on_cyls_ready(seq, r))
        task.error.connect(lambda _tb: None)
        task.start()
        self._render_task = task

    def _on_cyls_ready(self, seq: int, result) -> None:
        if seq != self._render_seq or result is None:
            return
        self._cyls_cache = result
        self._bg_meshes  = result["mesh_list"]
        self._render_combined()

    # ── Result text ───────────────────────────────────────────────────────────

    def _show_result(
        self,
        result,
        wx: float,
        wy: float,
        wz: "float | None",
    ) -> None:
        """Update text labels with query statistics."""
        from gui.query_engine import VoxelQueryResult
        z_str     = f", {wz:.3f}" if wz is not None else ""
        coord_str = f"({wx:.3f}, {wy:.3f}{z_str})"
        size_str  = f"{result.voxel_size:.2f} m"

        lines = [
            f"Voxel {size_str}  ·  {coord_str}",
            "",
            "── Point Cloud ──────────────────",
            f"  Total points:   {result.point_count:>8,}",
        ]

        if result.point_count > 0:
            lines += [
                f"  Tree points:    {result.tree_point_count:>8,}"
                f"  ({result.veg_cover * 100:.1f}%)",
                f"  Ground/other:   {result.ground_point_count:>8,}"
                f"  ({(1 - result.veg_cover) * 100:.1f}%)",
                f"  Veg cover:      {result.veg_cover * 100:.1f}%",
            ]
        else:
            lines.append("  (no points in this voxel)")

        if result.segment_ids:
            lines += ["", f"── Segments ({len(result.segment_ids)} found) ──────"]
            top = sorted(result.segment_counts.items(), key=lambda x: -x[1])
            for seg_id, cnt in top[:8]:
                pct = cnt / result.point_count * 100
                lines.append(f"  Seg {seg_id:>4d}:  {cnt:>6,} pts  ({pct:.1f}%)")
            if len(top) > 8:
                lines.append(f"  … and {len(top) - 8} more")

        if result.cyl_count > 0:
            lines += [
                "",
                f"── Branches ({result.cyl_count} cylinders) ───",
                f"  Mean radius:  {result.mean_branch_radius * 100:.2f} cm",
                f"  Max radius:   {result.max_branch_radius * 100:.2f} cm",
                f"  Total length: {result.total_branch_length:.2f} m",
            ]
        elif self._engine is not None and self._engine.has_cyls:
            lines += ["", "── Branches ─────────────────────",
                      "  No cylinders in this voxel"]
        else:
            lines += ["", "── Branches ─────────────────────",
                      "  (no QSM data for this run)"]

        self._result_label.setText("\n".join(lines))

        self._info_label.setText(
            f"Voxel {size_str} @ {coord_str}  ·  "
            f"{result.point_count:,} pts  ·  "
            f"veg {result.veg_cover * 100:.1f}%"
            + (f"  ·  {result.cyl_count} cyls" if result.cyl_count > 0 else "")
        )
        self.status_message.emit(
            f"Voxel query: {result.point_count:,} pts, "
            f"veg cover {result.veg_cover * 100:.1f}%"
            + (f", {result.cyl_count} cylinders" if result.cyl_count > 0 else "")
        )

    # ── Batch handlers ────────────────────────────────────────────────────────

    def _on_batch_pick_input(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select input CSV", "", "CSV files (*.csv);;All files (*)"
        )
        if path:
            self._batch_in_edit.setText(path)
            # Auto-suggest an output path next to the input
            if not self._batch_out_edit.text().strip():
                p = Path(path)
                self._batch_out_edit.setText(str(p.with_name(p.stem + "_veg.csv")))

    def _on_batch_pick_output(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Select output CSV", self._batch_out_edit.text() or "",
            "CSV files (*.csv);;All files (*)"
        )
        if path:
            self._batch_out_edit.setText(path)

    def _batch_log_line(self, text: str) -> None:
        self._batch_log.appendPlainText(text.rstrip("\n"))

    def _on_batch_run_clicked(self) -> None:
        # Pre-flight validation
        if self._engine is None or not self._engine.is_loaded:
            self._batch_log_line("⚠ Select a run first (top bar).")
            return
        if not self._engine.has_crs:
            self._batch_log_line(
                "⚠ This run has no CRS embedded in the source LAS — "
                "lon/lat conversion unavailable.  Cannot run batch."
            )
            return

        input_csv  = self._batch_in_edit.text().strip()
        output_csv = self._batch_out_edit.text().strip()
        if not input_csv:
            self._batch_log_line("⚠ Choose an input CSV.")
            return
        if not Path(input_csv).exists():
            self._batch_log_line(f"⚠ Input CSV not found: {input_csv}")
            return
        if not output_csv:
            self._batch_log_line("⚠ Choose an output CSV path.")
            return
        out_parent = Path(output_csv).parent
        if not out_parent.exists():
            self._batch_log_line(f"⚠ Output folder does not exist: {out_parent}")
            return

        radii = [sb.value() for sb in self._batch_radii if sb.value() > 0]
        if not radii:
            self._batch_log_line("⚠ At least one radius must be > 0.")
            return

        self._batch_log.clear()
        self._batch_log_line(f"Input : {input_csv}")
        self._batch_log_line(f"Output: {output_csv}")
        self._batch_log_line(f"Radii : {radii} m,  default AGL {self._batch_h_agl.value()} m")

        # Launch worker on a dedicated QThread (mirrors EcomodelWorker pattern)
        from gui.worker import BatchQueryWorker
        self._batch_thread = QThread(self)
        self._batch_worker = BatchQueryWorker(
            engine         = self._engine,
            input_csv      = input_csv,
            output_csv     = output_csv,
            radii          = radii,
            default_h_agl  = self._batch_h_agl.value(),
        )
        self._batch_worker.moveToThread(self._batch_thread)
        self._batch_thread.started.connect(self._batch_worker.run)
        self._batch_worker.log.connect(self._batch_log_line)
        self._batch_worker.progress.connect(self._on_batch_progress)
        self._batch_worker.finished_ok.connect(self._on_batch_finished_ok)
        self._batch_worker.failed.connect(self._on_batch_failed)
        # Clean up thread after either terminal signal
        self._batch_worker.finished_ok.connect(self._batch_thread.quit)
        self._batch_worker.failed.connect(self._batch_thread.quit)

        self._batch_run_btn.setEnabled(False)
        self._batch_stop_btn.setEnabled(True)
        self._batch_progress.setRange(0, 0)        # busy until first progress tick
        self._batch_progress.setValue(0)
        self._batch_progress.setFormat("Starting…")

        self._batch_thread.start()

    def _on_batch_progress(self, current: int, total: int, obs_id: str) -> None:
        if self._batch_progress.maximum() != total:
            self._batch_progress.setRange(0, max(total, 1))
        self._batch_progress.setValue(current)
        self._batch_progress.setFormat(f"Row {current} / {total}")
        # Only log every 50th row (plus the very first and very last) to keep
        # the log pane readable for a 4000-row batch.
        if current == 1 or current == total or current % 50 == 0:
            self._batch_log_line(f"[{current}/{total}] {obs_id}")

    def _on_batch_stop_clicked(self) -> None:
        if self._batch_worker is not None:
            self._batch_worker.request_stop()
            self._batch_log_line("Stop requested — finishing current row…")
            self._batch_stop_btn.setEnabled(False)

    def _on_batch_finished_ok(self, output_path: str) -> None:
        self._batch_log_line(f"✓ Done.  Wrote {output_path}")
        self._batch_progress.setFormat("Done")
        self._batch_run_btn.setEnabled(True)
        self._batch_stop_btn.setEnabled(False)
        self.status_message.emit(f"Batch query finished: {output_path}")

    def _on_batch_failed(self, message: str) -> None:
        self._batch_log_line("⚠ Batch failed:")
        self._batch_log_line(message)
        self._batch_progress.setFormat("Failed")
        self._batch_run_btn.setEnabled(True)
        self._batch_stop_btn.setEnabled(False)

    # ── Widget factory ────────────────────────────────────────────────────────

    @staticmethod
    def _dspin() -> QDoubleSpinBox:
        w = QDoubleSpinBox()
        w.setRange(-1e8, 1e8)
        w.setDecimals(3)
        w.setValue(0.0)
        w.setSingleStep(1.0)
        return w
