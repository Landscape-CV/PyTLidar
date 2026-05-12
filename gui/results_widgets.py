"""
Results viewer widgets for EcomodelMainWindow.

Classes
-------
BatchResultsTable
    QTableWidget showing per-tile processing status, cylinder count,
    and output path.  Populated live from the worker's tile_update signal.

EmbeddedPlotWidget
    Swappable display area that can show either a matplotlib FigureCanvas
    or a pyvistaqt QtInteractor for native 3-D OpenGL rendering.
    Falls back to a matplotlib placeholder if pyvistaqt is not installed.

ResultsPage
    QWidget page for browsing and viewing pipeline results.  All file I/O
    and mesh-building runs on background threads (via BgTask) so the GUI
    never blocks.  Only plotter.add_mesh() calls happen on the main thread.

Threading model
---------------
  BgTask(_bg_load_run, run_dir)        → loads .npy files
  BgTask(_bg_scan_runs, folder)        → scans results folder
  BgTask(_bg_build_cloud_meshes, ...)  → builds pv.PolyData for point cloud
  BgTask(_bg_build_segment_meshes, ...)→ builds pv.PolyData for segments
  BgTask(_bg_cylinders, cyl_path)      → loadtxt + builds cylinder meshes
  BgTask(_bg_metrics, cyl_path)        → loadtxt + builds matplotlib Figure
  BgTask(_bg_voxel_meshes, ...)        → builds pv.PolyData for voxel query

All bg functions are module-level free functions (not methods) so they can
be pickled / referenced cleanly from the thread.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

try:
    from pyvistaqt import QtInteractor
    _PYVISTAQT_AVAILABLE = True
except ImportError:
    _PYVISTAQT_AVAILABLE = False
    QtInteractor = None  # type: ignore[assignment,misc]


# ── Background worker free functions ─────────────────────────────────────────
# These run on BgTask threads.  They must not touch any Qt widget or signal.

def _bg_scan_runs(results_folder: str) -> list:
    """Scan folder for run dirs and read their metadata.  Returns list of (label, Path)."""
    from gui.results_io import find_run_dirs, load_run_metadata
    runs = find_run_dirs(results_folder) if results_folder else []
    items = []
    for run_dir in runs:
        meta = load_run_metadata(run_dir)
        if meta:
            label = f"{run_dir.name}\n{meta.timestamp}  ·  {meta.cylinder_count} cyls"
        else:
            label = run_dir.name
        items.append((label, run_dir))
    return items


def _bg_load_run(run_dir: Path) -> dict:
    """Load all array snapshots for a run directory.  Returns a plain dict."""
    from gui.results_io import (
        load_run_metadata,
        POINT_CLOUD_FILE, SEGMENT_LABELS_FILE, COVER_SETS_FILE, POINT_FIELDS_FILE,
    )

    cloud = labels = cover_sets = None
    point_fields: dict = {}
    cyl_path = None

    meta = load_run_metadata(run_dir)

    npy = run_dir / POINT_CLOUD_FILE
    if npy.exists():
        try:
            cloud = np.load(npy)
        except Exception:
            pass

    lbl = run_dir / SEGMENT_LABELS_FILE
    if lbl.exists():
        try:
            labels = np.load(lbl)
        except Exception:
            pass

    cov = run_dir / COVER_SETS_FILE
    if cov.exists():
        try:
            cover_sets = np.load(cov)
        except Exception:
            pass

    pf = run_dir / POINT_FIELDS_FILE
    if pf.exists():
        try:
            loaded = np.load(pf)
            n = len(cloud) if cloud is not None else -1
            point_fields = {
                k: loaded[k] for k in loaded.files if len(loaded[k]) == n
            }
        except Exception:
            pass

    if meta:
        cyl = run_dir / meta.cylinder_file
        if cyl.exists() and cyl.stat().st_size > 0:
            cyl_path = cyl
    if cyl_path is None:
        txts = sorted(run_dir.glob("*.txt"))
        if txts:
            cyl_path = txts[0]

    return {
        "cloud": cloud,
        "labels": labels,
        "cover_sets": cover_sets,
        "point_fields": point_fields,
        "cyl_path": cyl_path,
        "meta": meta,
    }


def _bg_build_cloud_meshes(cloud, fields, active_field) -> list:
    from reporting.pv_rendering import build_point_cloud_meshes
    mesh_list, _ = build_point_cloud_meshes(cloud, fields, active_field)
    return mesh_list


def _bg_build_segment_meshes(cloud, cover, labels, view_mode) -> list:
    from reporting.pv_rendering import build_segment_meshes
    mesh_list, _ = build_segment_meshes(cloud, cover, labels, view_mode)
    return mesh_list


def _bg_cylinders(cyl_path: Path) -> "dict | None":
    """Load cylinder .txt and build all render meshes.  Returns dict or None."""
    try:
        data = np.loadtxt(cyl_path)
    except Exception:
        return None
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.ndim != 2 or data.shape[1] != 8 or data.shape[0] == 0:
        return None

    cyls = {
        "start":  data[:, 0:3],
        "radius": data[:, 3],
        "axis":   data[:, 4:7],
        "length": data[:, 7],
    }

    from reporting.pv_rendering import build_cylinder_meshes
    mesh_list, starts, ends, radii, lengths = build_cylinder_meshes(cyls)
    return {
        "mesh_list": mesh_list,
        "starts": starts,
        "ends":   ends,
        "radii":  radii,
        "lengths": lengths,
    }


def _bg_metrics(cyl_path: Path) -> Figure:
    """Load cylinder .txt and build a matplotlib metrics Figure."""
    fig = Figure(figsize=(10, 7), tight_layout=True)

    try:
        data = np.loadtxt(cyl_path)
    except Exception:
        data = None

    if data is None or data.ndim == 1 and data.shape[0] != 8:
        ax = fig.add_subplot(111)
        ax.text(
            0.5, 0.5,
            "No cylinder data — QSM not run or pipeline stopped early",
            ha="center", va="center", transform=ax.transAxes,
        )
        ax.axis("off")
        return fig

    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.ndim != 2 or data.shape[1] != 8 or data.shape[0] == 0:
        ax = fig.add_subplot(111)
        ax.text(
            0.5, 0.5, "Cylinder file has unexpected format",
            ha="center", va="center", transform=ax.transAxes,
        )
        ax.axis("off")
        return fig

    radii   = data[:, 3].ravel()
    lengths = data[:, 7].ravel()
    axes_z  = data[:, 4:7][:, 2].ravel()
    zenith  = np.degrees(np.arccos(np.clip(np.abs(axes_z), 0, 1)))

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.hist(radii * 100, bins=30, color="#4caf50", edgecolor="white")
    ax1.set_title("Cylinder Radius Distribution")
    ax1.set_xlabel("Radius (cm)")
    ax1.set_ylabel("Count")
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.hist(lengths, bins=30, color="#1976d2", edgecolor="white")
    ax2.set_title("Cylinder Length Distribution")
    ax2.set_xlabel("Length (m)")
    ax2.set_ylabel("Count")
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.hist(zenith, bins=18, range=(0, 90), color="#f57c00", edgecolor="white")
    ax3.set_title("Branch Zenith Angle Distribution")
    ax3.set_xlabel("Zenith angle (°)  [0° = vertical]")
    ax3.set_ylabel("Count")
    ax3.axvline(45, color="red", linestyle="--", linewidth=0.8, label="45°")
    ax3.legend(fontsize="small")
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(2, 2, 4)
    total_vol_L = float(np.sum(np.pi * radii**2 * lengths) * 1000)
    trunk_mask  = zenith < 30
    stats_labels = [
        "Total cylinders",
        "Trunk cyls  (zen<30°)",
        "Branch cyls (zen≥30°)",
        "Mean radius (cm)",
        "Mean length (m)",
        "Total volume (L)",
    ]
    stats_values = [
        len(radii),
        int(trunk_mask.sum()),
        int((~trunk_mask).sum()),
        round(float(radii.mean()) * 100, 3),
        round(float(lengths.mean()), 3),
        round(total_vol_L, 2),
    ]
    y_pos = np.arange(len(stats_labels))
    bars = ax4.barh(y_pos, stats_values, color="#7b1fa2", alpha=0.8)
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(stats_labels, fontsize=8)
    ax4.set_title("Summary")
    ax4.set_xlabel("Value")
    ax4.grid(True, alpha=0.3, axis="x")
    for bar, val in zip(bars, stats_values):
        ax4.text(
            bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
            str(val), va="center", fontsize=7,
        )

    return fig


# ── Batch Results Table ───────────────────────────────────────────────────────

class BatchResultsTable(QTableWidget):
    """
    Four-column table: Tile Name | Status | Cylinder Count | Output Path.

    Rows are added dynamically as the pipeline reports tile progress via
    add_tile_row() / update_tile_status().
    """

    COLUMNS = ["Tile Name", "Status", "Cylinder Count", "Output Path"]

    def __init__(self, parent=None) -> None:
        super().__init__(0, len(self.COLUMNS), parent)
        self.setHorizontalHeaderLabels(self.COLUMNS)
        self.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.setSelectionBehavior(QTableWidget.SelectRows)
        self.setEditTriggers(QTableWidget.NoEditTriggers)
        self.setMaximumHeight(140)

    def add_tile_row(self, tile_name: str) -> int:
        """Appends a row for tile_name with 'Pending' status. Returns the row index."""
        row = self.rowCount()
        self.insertRow(row)
        for col, text in enumerate([tile_name, "Pending", "-", ""]):
            self.setItem(row, col, QTableWidgetItem(text))
        return row

    def update_tile_status(
        self,
        row: int,
        status: str,
        cyl_count: int = -1,
        output_path: str = "",
    ) -> None:
        """Updates status, cylinder count, and path for the given row."""
        self.item(row, 1).setText(status)
        if cyl_count >= 0:
            self.item(row, 2).setText(str(cyl_count))
        if output_path:
            self.item(row, 3).setText(output_path)

        color: QColor | None = None
        if status == "Done":
            color = QColor("#c8e6c9")
        elif status == "Error":
            color = QColor("#ffcdd2")
        elif status == "Running":
            color = QColor("#fff9c4")

        if color:
            for col in range(self.columnCount()):
                item = self.item(row, col)
                if item:
                    item.setBackground(color)

    def clear_rows(self) -> None:
        """Removes all data rows (called at the start of each pipeline run)."""
        self.setRowCount(0)


# ── Embedded Plot Widget ──────────────────────────────────────────────────────

class EmbeddedPlotWidget(QWidget):
    """
    Swappable single-view area for results visualisation.

    show_pyvista(populate_fn)        – clear the plotter, call populate_fn(plotter),
                                       then reset_camera.  (All on the main thread.)
    show_pyvista_meshes(mesh_list)   – apply pre-built meshes to the plotter on
                                       the main thread; mesh_list was built on a
                                       bg thread via one of the build_* functions.
    show_matplotlib_figure(fig)      – embed a matplotlib Figure inline.
    clear()                          – return to placeholder state.
    get_plotter()                    – return the underlying QtInteractor or None.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)

        self._plotter: "QtInteractor | None" = None   # lazy-init
        self._canvas:  "FigureCanvas | None" = None   # recreated per matplotlib call

        self._placeholder = QLabel("Run the pipeline to see results here.")
        self._placeholder.setAlignment(Qt.AlignCenter)
        self._layout.addWidget(self._placeholder)

    # ── Public API ────────────────────────────────────────────────────────

    def _ensure_plotter(self) -> None:
        """Lazy-init the QtInteractor with conservative render rates."""
        if self._plotter is not None:
            return
        self._plotter = QtInteractor(self)
        self._plotter.enable_terrain_style()
        # Keep render rates low — high rates hammer the GPU driver on Windows
        # and can cause TDR (Timeout Detection and Recovery) → BSOD for large
        # point clouds.  5 fps during interaction is smooth enough to navigate
        # and gentle enough not to trigger a driver timeout.
        try:
            iren = self._plotter.iren
            iren.SetDesiredUpdateRate(5.0)    # fps during mouse interaction
            iren.SetStillUpdateRate(0.0)      # no idle renders
        except Exception:
            pass
        # Disable advanced rendering passes that are common BSOD vectors on
        # Windows GPUs (FXAA, depth peeling, shadows).  We only need plain
        # forward rendering for point clouds and cylinders.
        try:
            ren = self._plotter.renderer
            ren.SetUseFXAA(False)
            ren.SetUseDepthPeeling(False)
            ren.SetUseShadows(False)
        except Exception:
            pass
        try:
            rw = self._plotter.render_window
            rw.SetMultiSamples(0)             # no MSAA
            rw.SetPointSmoothing(False)
            rw.SetLineSmoothing(False)
            rw.SetPolygonSmoothing(False)
        except Exception:
            pass
        self._layout.addWidget(self._plotter)

    def show_pyvista(self, populate_fn) -> None:
        """
        Clear the 3-D scene and call ``populate_fn(plotter)`` to add actors.

        Falls back to a matplotlib placeholder if pyvistaqt is not installed.
        The entire call happens on the calling (main) thread.
        """
        if not _PYVISTAQT_AVAILABLE:
            self.show_matplotlib_figure(
                self._make_fallback_figure(
                    "pyvistaqt is required for 3-D rendering.\n\n"
                    "pip install pyvistaqt"
                )
            )
            return

        self._placeholder.hide()
        if self._canvas is not None:
            self._canvas.hide()

        self._ensure_plotter()
        self._plotter.clear()
        self._plotter.show()

        try:
            populate_fn(self._plotter)
        except Exception as exc:
            self._plotter.clear()
            self.show_matplotlib_figure(
                self._make_fallback_figure(f"Render error:\n{exc}")
            )
            return

        self._plotter.reset_camera()

    def show_pyvista_meshes(self, mesh_list: list, post_fn=None) -> None:
        """
        Apply pre-built meshes to the plotter.  Must be called on the main thread.

        Parameters
        ----------
        mesh_list : list of (pv.DataSet, add_mesh_kwargs) tuples
            Built on a background thread via one of the ``build_*_meshes()``
            functions in ``plotting.pv_rendering``.
        post_fn : callable(plotter) or None
            Optional hook called after all meshes are added but before
            reset_camera().  Use it for plotter.track_click_position() etc.
        """
        if not _PYVISTAQT_AVAILABLE:
            self.show_matplotlib_figure(
                self._make_fallback_figure(
                    "pyvistaqt is required for 3-D rendering.\n\n"
                    "pip install pyvistaqt"
                )
            )
            return

        self._placeholder.hide()
        if self._canvas is not None:
            self._canvas.hide()

        self._ensure_plotter()

        self._plotter.clear()
        self._plotter.show()

        try:
            from reporting.pv_rendering import apply_meshes_to_plotter
            apply_meshes_to_plotter(self._plotter, mesh_list)
            if post_fn is not None:
                post_fn(self._plotter)
        except Exception as exc:
            self._plotter.clear()
            self.show_matplotlib_figure(
                self._make_fallback_figure(f"Render error:\n{exc}")
            )
            return

        self._plotter.reset_camera()

    def show_matplotlib_figure(self, fig: Figure) -> None:
        """Replace current content with a FigureCanvasQTAgg wrapping fig."""
        self._placeholder.hide()
        if self._plotter is not None:
            self._plotter.hide()

        if self._canvas is not None:
            self._layout.removeWidget(self._canvas)
            self._canvas.deleteLater()
        self._canvas = FigureCanvas(fig)
        self._layout.addWidget(self._canvas)
        self._canvas.show()

    def clear(self) -> None:
        """Remove current content and re-show the placeholder."""
        if self._plotter is not None:
            self._plotter.clear()
            self._plotter.hide()
        if self._canvas is not None:
            self._layout.removeWidget(self._canvas)
            self._canvas.deleteLater()
            self._canvas = None
        self._placeholder.show()

    def get_plotter(self) -> "QtInteractor | None":
        """Return the underlying QtInteractor, or None if not yet initialised."""
        return self._plotter

    # ── Private helpers ───────────────────────────────────────────────────

    @staticmethod
    def _make_fallback_figure(message: str) -> Figure:
        fig = Figure(figsize=(4, 2), tight_layout=True)
        ax = fig.add_subplot(111)
        ax.text(
            0.5, 0.5, message,
            ha="center", va="center",
            transform=ax.transAxes,
            wrap=True,
            fontsize=9,
        )
        ax.axis("off")
        return fig

    def closeEvent(self, event) -> None:
        """Cleanly shut down the VTK render window on widget close."""
        if self._plotter is not None:
            try:
                self._plotter.close()
            except Exception:
                pass
        super().closeEvent(event)


# ── Results Page ─────────────────────────────────────────────────────────────

class ResultsPage(QWidget):
    """
    Full-window results viewer, designed to live inside a QStackedWidget.

    All heavy work (file I/O, numpy ops, PyVista mesh building) runs on
    background threads.  Only plotter.add_mesh() calls happen on the main
    thread inside show_pyvista_meshes().

    A monotonic ``_load_seq`` counter guards against stale results when the
    user rapidly switches runs.  A separate ``_render_seq`` counter does the
    same for view changes.
    """

    back_requested  = Signal()
    status_message  = Signal(str)
    query_requested = Signal()

    def __init__(self, results_folder: str = "results", parent=None) -> None:
        super().__init__(parent)

        self._results_folder: str = results_folder
        self._run_dir:  Path | None = None
        self._cloud                 = None
        self._labels                = None
        self._cover_sets            = None
        self._point_fields: dict    = {}
        self._cyl_path: Path | None = None

        self._active_view: str = ""

        # Monotonic counters to discard stale background results
        self._load_seq:   int = 0
        self._render_seq: int = 0

        # Keep task references alive until done
        self._scan_task   = None
        self._load_task   = None
        self._render_task = None

        self._build_ui()
        self.refresh()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        # ── Left pane: run list ───────────────────────────────────────────
        left = QWidget()
        left.setMinimumWidth(200)
        left.setMaximumWidth(300)
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 4, 0)

        self._back_btn = QPushButton("← Back")
        self._back_btn.setToolTip("Return to the pipeline")
        self._back_btn.clicked.connect(self.back_requested)
        lv.addWidget(self._back_btn)

        hdr = QHBoxLayout()
        hdr.addWidget(QLabel("Runs"))
        self._refresh_btn = QPushButton("↺")
        self._refresh_btn.setFixedWidth(28)
        self._refresh_btn.setToolTip("Rescan results folder")
        self._refresh_btn.clicked.connect(self.refresh)
        hdr.addWidget(self._refresh_btn)
        self._open_btn = QPushButton("Open…")
        self._open_btn.setToolTip("Browse to any run folder")
        self._open_btn.clicked.connect(self._browse_run)
        hdr.addWidget(self._open_btn)
        self._query_btn = QPushButton("Query →")
        self._query_btn.setToolTip("Find nearest tree by world coordinate")
        self._query_btn.clicked.connect(self.query_requested)
        hdr.addWidget(self._query_btn)
        lv.addLayout(hdr)

        self._run_list = QListWidget()
        self._run_list.setAlternatingRowColors(True)
        self._run_list.setSpacing(2)
        self._run_list.currentRowChanged.connect(self._on_run_selected)
        lv.addWidget(self._run_list, stretch=1)

        self._meta_label = QLabel()
        self._meta_label.setWordWrap(True)
        self._meta_label.setStyleSheet("color: #555; font-size: 11px;")
        lv.addWidget(self._meta_label)

        splitter.addWidget(left)

        # ── Right pane: viewer ────────────────────────────────────────────
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(4, 0, 0, 0)

        btn_row = QHBoxLayout()
        self._btn_cloud   = QPushButton("Point Cloud")
        self._btn_segs    = QPushButton("Segments")
        self._btn_cyls    = QPushButton("Cylinders")
        self._btn_metrics = QPushButton("Tree Metrics")
        for btn in (self._btn_cloud, self._btn_segs, self._btn_cyls, self._btn_metrics):
            btn.setEnabled(False)
            btn_row.addWidget(btn)
        rv.addLayout(btn_row)

        # ── View-specific toolbar ─────────────────────────────────────────
        self._view_bar = QWidget()
        vb_layout = QHBoxLayout(self._view_bar)
        vb_layout.setContentsMargins(0, 0, 0, 2)
        vb_layout.setSpacing(6)

        self._field_label = QLabel("Colour by:")
        self._field_combo = QComboBox()
        self._field_combo.setMinimumWidth(140)
        self._field_combo.setToolTip("Select scalar field for point colouring")
        self._field_combo.currentTextChanged.connect(self._on_field_changed)
        vb_layout.addWidget(self._field_label)
        vb_layout.addWidget(self._field_combo)

        self._view_label = QLabel("View:")
        self._view_combo = QComboBox()
        self._view_combo.setMinimumWidth(120)
        self._view_combo.setToolTip("Select segmentation colouring mode")
        self._view_combo.currentTextChanged.connect(self._on_view_mode_changed)
        vb_layout.addWidget(self._view_label)
        vb_layout.addWidget(self._view_combo)

        vb_layout.addStretch()
        self._view_bar.setVisible(False)
        rv.addWidget(self._view_bar)

        self._plot = EmbeddedPlotWidget()
        self._plot._placeholder.setText("Select a run from the list to view results.")
        rv.addWidget(self._plot, stretch=1)

        self._volume_label = QLabel("")
        self._volume_label.setStyleSheet(
            "color: #1565c0; font-size: 11px; padding: 3px 0;"
        )
        self._volume_label.setVisible(False)
        rv.addWidget(self._volume_label)

        self._info_label = QLabel("")
        self._info_label.setStyleSheet("color: #666; font-size: 11px;")
        rv.addWidget(self._info_label)

        splitter.addWidget(right)
        splitter.setSizes([240, 860])

        self._btn_cloud.clicked.connect(self._show_cloud)
        self._btn_segs.clicked.connect(self._show_segments)
        self._btn_cyls.clicked.connect(self._show_cylinders)
        self._btn_metrics.clicked.connect(self._show_metrics)

    # ── Public API ────────────────────────────────────────────────────────────

    def set_results_folder(self, folder: str) -> None:
        if folder != self._results_folder:
            self._results_folder = folder
            self.refresh()

    def refresh(self) -> None:
        """Rescan the results folder on a background thread."""
        from gui.worker import BgTask

        prev_item = self._run_list.currentItem()
        prev_run_dir = prev_item.data(Qt.UserRole) if prev_item else None

        # Clear list immediately; repopulated when scan finishes
        self._run_list.blockSignals(True)
        self._run_list.clear()
        self._run_list.blockSignals(False)

        folder = self._results_folder
        task = BgTask(_bg_scan_runs, folder)
        task.result.connect(lambda items: self._populate_run_list(items, prev_run_dir))
        task.error.connect(lambda _tb: None)   # silent on scan error
        task.start()
        self._scan_task = task

    def notify_run_complete(self, run_dir: Path) -> None:
        """Called by the main window after a pipeline run completes."""
        run_dir = Path(run_dir)
        self.refresh()
        # Auto-select after refresh (the list repopulates asynchronously, so
        # _populate_run_list will be called after the scan; we encode the
        # desired target so it can be selected there).
        self._pending_select_dir = run_dir

    # ── Run list population ───────────────────────────────────────────────────

    def _populate_run_list(self, items: list, prev_run_dir) -> None:
        """Called on main thread when the background scan finishes."""
        self._run_list.blockSignals(True)
        self._run_list.clear()
        restore_row = -1

        for i, (label, run_dir) in enumerate(items):
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, run_dir)
            self._run_list.addItem(item)
            if prev_run_dir is not None and run_dir == prev_run_dir:
                restore_row = i

        # notify_run_complete: prefer the newly-completed run
        pending = getattr(self, "_pending_select_dir", None)
        if pending is not None:
            for i, (_, run_dir) in enumerate(items):
                if run_dir == pending:
                    restore_row = i
                    break
            self._pending_select_dir = None

        self._run_list.blockSignals(False)

        if restore_row >= 0:
            self._run_list.setCurrentRow(restore_row)

    # ── Load a run ────────────────────────────────────────────────────────────

    def _on_run_selected(self, row: int) -> None:
        if row < 0:
            return
        item = self._run_list.item(row)
        if item is not None:
            run_dir = item.data(Qt.UserRole)
            if run_dir is not None:
                self._start_load(Path(run_dir))

    def _browse_run(self) -> None:
        from PySide6.QtWidgets import QFileDialog, QMessageBox
        from gui.results_io import load_run_metadata

        folder = QFileDialog.getExistingDirectory(
            self, "Select Run Folder", self._results_folder or "."
        )
        if not folder:
            return
        run_dir = Path(folder)
        if load_run_metadata(run_dir) is None:
            QMessageBox.warning(
                self, "Not a Run Folder",
                f"No metadata.json found in:\n{run_dir}\n\n"
                "Please select a folder created by the Ecomodel pipeline."
            )
            return
        self._start_load(run_dir)

    def _start_load(self, run_dir: Path) -> None:
        """Kick off a background load for run_dir."""
        from gui.worker import BgTask

        self._load_seq += 1
        seq = self._load_seq
        self._run_dir = run_dir

        # Immediately update UI to "loading" state
        for btn in (self._btn_cloud, self._btn_segs, self._btn_cyls, self._btn_metrics):
            btn.setEnabled(False)
        self._active_view = ""
        self._plot.clear()
        self._view_bar.setVisible(False)
        self._volume_label.setVisible(False)
        self._meta_label.setText("Loading…")
        self._info_label.setText("")

        task = BgTask(_bg_load_run, run_dir)
        task.result.connect(lambda data: self._on_run_loaded(seq, data))
        task.error.connect(
            lambda tb: self._meta_label.setText("Load error")
            if seq == self._load_seq else None
        )
        task.start()
        self._load_task = task

    def _on_run_loaded(self, seq: int, data: dict) -> None:
        """Main-thread slot: run data has arrived from the bg thread."""
        if seq != self._load_seq:
            return  # user switched run before this one finished

        self._cloud        = data["cloud"]
        self._labels       = data["labels"]
        self._cover_sets   = data["cover_sets"]
        self._point_fields = data["point_fields"]
        self._cyl_path     = data["cyl_path"]
        meta               = data["meta"]

        self._btn_cloud.setEnabled(self._cloud is not None)
        self._btn_segs.setEnabled(
            self._cloud is not None and self._labels is not None
        )
        self._btn_cyls.setEnabled(self._cyl_path is not None)
        self._btn_metrics.setEnabled(self._cyl_path is not None)

        if meta:
            self._meta_label.setText(
                f"{self._run_dir.name}\n{meta.timestamp}"
                f"\n{meta.cylinder_count} cylinders  ·  {meta.point_count:,} pts"
            )
            self._info_label.setText(
                f"Run: {self._run_dir.name}  |  {meta.timestamp}"
                f"  |  {meta.cylinder_count} cyls  |  {meta.point_count:,} pts"
                + ("  |  QSM" if meta.qsm_enabled else "")
            )
            self.status_message.emit(
                f"Loaded: {self._run_dir.name}  ({meta.timestamp})"
            )
        else:
            self._meta_label.setText(self._run_dir.name)
            self._info_label.setText(str(self._run_dir))
            self.status_message.emit(f"Loaded: {self._run_dir.name}")

    # ── View toolbar helpers ──────────────────────────────────────────────────

    def _show_toolbar_for(self, mode: str) -> None:
        show_field = mode == "cloud"
        show_view  = mode == "segments"

        self._field_label.setVisible(show_field)
        self._field_combo.setVisible(show_field)
        self._view_label.setVisible(show_view)
        self._view_combo.setVisible(show_view)
        self._view_bar.setVisible(show_field or show_view)
        self._volume_label.setVisible(mode == "cylinders")

    # ── View slots ────────────────────────────────────────────────────────────

    def _show_cloud(self) -> None:
        if self._cloud is None:
            return
        self._active_view = "cloud"
        self._show_toolbar_for("cloud")

        all_fields: dict = {"Height (Z)": None}
        if self._point_fields:
            all_fields.update(self._point_fields)
        default = next(
            (k for k in all_fields if "intensity" in k.lower()),
            list(all_fields.keys())[0],
        )

        self._field_combo.blockSignals(True)
        self._field_combo.clear()
        for name in all_fields:
            self._field_combo.addItem(name)
        self._field_combo.setCurrentText(default)
        self._field_combo.blockSignals(False)

        self._render_cloud(default)

    def _render_cloud(self, active_field: str) -> None:
        if self._cloud is None:
            return
        from gui.worker import BgTask

        cloud  = self._cloud
        fields = dict(self._point_fields) if self._point_fields else None
        self._render_seq += 1
        seq = self._render_seq

        task = BgTask(_bg_build_cloud_meshes, cloud, fields, active_field)
        task.result.connect(lambda ml: self._apply_meshes(seq, ml))
        task.error.connect(
            lambda tb: self._plot.show_matplotlib_figure(
                EmbeddedPlotWidget._make_fallback_figure(f"Render error:\n{tb[:200]}")
            ) if seq == self._render_seq else None
        )
        task.start()
        self._render_task = task

    def _on_field_changed(self, field_name: str) -> None:
        if self._active_view == "cloud" and self._cloud is not None and field_name:
            self._render_cloud(field_name)

    def _show_segments(self) -> None:
        if self._cloud is None or self._labels is None:
            return
        self._active_view = "segments"
        self._show_toolbar_for("segments")

        modes = ["Segment"]
        if self._cover_sets is not None:
            modes.append("Cover Set")
        self._view_combo.blockSignals(True)
        self._view_combo.clear()
        for m in modes:
            self._view_combo.addItem(m)
        self._view_combo.setCurrentText("Segment")
        self._view_combo.blockSignals(False)

        self._render_segments("Segment")

    def _render_segments(self, view_mode: str) -> None:
        if self._cloud is None or self._labels is None:
            return
        from gui.worker import BgTask

        n = min(
            len(self._cloud),
            len(self._labels),
            len(self._cover_sets) if self._cover_sets is not None else len(self._labels),
        )
        cloud  = self._cloud[:n]
        labels = self._labels[:n]
        cover  = self._cover_sets[:n] if self._cover_sets is not None else None

        self._render_seq += 1
        seq = self._render_seq

        task = BgTask(_bg_build_segment_meshes, cloud, cover, labels, view_mode)
        task.result.connect(lambda ml: self._apply_meshes(seq, ml))
        task.error.connect(
            lambda tb: self._plot.show_matplotlib_figure(
                EmbeddedPlotWidget._make_fallback_figure(f"Render error:\n{tb[:200]}")
            ) if seq == self._render_seq else None
        )
        task.start()
        self._render_task = task

    def _on_view_mode_changed(self, mode: str) -> None:
        if self._active_view == "segments" and mode:
            self._render_segments(mode)

    def _show_cylinders(self) -> None:
        if self._cyl_path is None:
            self._plot.show_matplotlib_figure(
                EmbeddedPlotWidget._make_fallback_figure(
                    "No cylinder data.\n"
                    "Enable 'Run QSM + RGI' and re-run the pipeline."
                )
            )
            return
        from gui.worker import BgTask

        self._active_view = "cylinders"
        self._show_toolbar_for("cylinders")
        self._volume_label.setText("← Click on a cylinder to calculate volume within 0.5 m")

        self._render_seq += 1
        seq = self._render_seq
        cyl_path = self._cyl_path

        task = BgTask(_bg_cylinders, cyl_path)
        task.result.connect(lambda r: self._apply_cylinder_render(seq, r))
        task.error.connect(
            lambda tb: self._plot.show_matplotlib_figure(
                EmbeddedPlotWidget._make_fallback_figure(f"Cylinder render error:\n{tb[:200]}")
            ) if seq == self._render_seq else None
        )
        task.start()
        self._render_task = task

    def _apply_cylinder_render(self, seq: int, result) -> None:
        if seq != self._render_seq:
            return
        if result is None:
            self._plot.show_matplotlib_figure(
                EmbeddedPlotWidget._make_fallback_figure(
                    "No cylinder data.\n"
                    "Enable 'Run QSM + RGI' and re-run the pipeline."
                )
            )
            return

        mesh_list = result["mesh_list"]
        starts    = result["starts"]
        ends      = result["ends"]
        radii     = result["radii"]
        lengths   = result["lengths"]
        threshold = 0.5
        volume_label = self._volume_label

        def _on_click(pos) -> None:
            pt = np.array(pos, dtype=np.float64)
            d_start = np.linalg.norm(starts - pt, axis=1)
            d_end   = np.linalg.norm(ends   - pt, axis=1)
            matched = np.minimum(d_start, d_end) < threshold
            total_vol = float(np.sum(np.pi * radii[matched] ** 2 * lengths[matched]))
            n_matched = int(matched.sum())
            x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
            volume_label.setText(
                f"Volume within {threshold} m of ({x:.2f}, {y:.2f}, {z:.2f}):  "
                f"{total_vol:.4f} m³  "
                f"({n_matched} cylinder{'s' if n_matched != 1 else ''})"
            )

        def post_fn(plotter):
            plotter.track_click_position(callback=_on_click, side="left")

        self._plot.show_pyvista_meshes(mesh_list, post_fn=post_fn)

    def _show_metrics(self) -> None:
        if self._cyl_path is None:
            fig = Figure(figsize=(10, 7), tight_layout=True)
            ax = fig.add_subplot(111)
            ax.text(
                0.5, 0.5,
                "No cylinder data — QSM not run or pipeline stopped early",
                ha="center", va="center", transform=ax.transAxes,
            )
            ax.axis("off")
            self._plot.show_matplotlib_figure(fig)
            return
        from gui.worker import BgTask

        self._active_view = "metrics"
        self._show_toolbar_for("metrics")

        self._render_seq += 1
        seq = self._render_seq
        cyl_path = self._cyl_path

        task = BgTask(_bg_metrics, cyl_path)
        task.result.connect(
            lambda fig: self._plot.show_matplotlib_figure(fig)
            if seq == self._render_seq else None
        )
        task.error.connect(
            lambda tb: self._plot.show_matplotlib_figure(
                EmbeddedPlotWidget._make_fallback_figure(f"Metrics error:\n{tb[:200]}")
            ) if seq == self._render_seq else None
        )
        task.start()
        self._render_task = task

    # ── Generic mesh-apply slot ───────────────────────────────────────────────

    def _apply_meshes(self, seq: int, mesh_list: list) -> None:
        """Apply pre-built meshes if the render request is still current."""
        if seq != self._render_seq:
            return
        try:
            self._plot.show_pyvista_meshes(mesh_list)
        except Exception as e:
            self._plot.show_matplotlib_figure(
                EmbeddedPlotWidget._make_fallback_figure(str(e))
            )

