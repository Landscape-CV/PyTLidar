"""
Worker classes for running tasks off the GUI thread.

Classes
-------
BgTask
    Generic single-shot background task.  Runs any callable on a private
    QThread and emits result/error signals back to the main thread.

    Usage::

        task = BgTask(fn, arg1, kwarg=val)
        task.result.connect(my_slot)       # receives the return value
        task.error.connect(my_error_slot)  # receives a formatted traceback
        task.start()

EcomodelWorker
    Runs run_ecomodel_pipeline() on a QThread with rich progress/log signals.

    Usage::

        thread = QThread()
        worker = EcomodelWorker(config)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(thread.quit)
        thread.start()

    Stop cleanly:
        worker.request_stop()
"""

from __future__ import annotations
from PySide6.QtCore import QObject, QThread, Signal, Slot

from gui.config import EcomodelConfig


# ── Global task registry ──────────────────────────────────────────────────────
# Keeps every running BgTask object alive (strong Python reference) so that
# the GC never destroys a task—and its QThread—while the thread is still
# running.  Tasks remove themselves from this set when their thread finishes.

_live_tasks: set = set()


def cancel_all_bg_tasks(msecs: int = 3_000) -> None:
    """
    Cancel every in-flight BgTask and wait up to *msecs* for each to finish.

    Call this from the main window's closeEvent() before the app exits.
    """
    for task in list(_live_tasks):
        task.cancel()
    for task in list(_live_tasks):
        task.wait(msecs)


class _BgThread(QThread):
    """Private thread that drives one BgTask."""

    def __init__(self, task: "BgTask") -> None:
        super().__init__()
        self._task = task

    def run(self) -> None:
        task = self._task
        try:
            res = task._fn(*task._args, **task._kwargs)
            if not task._cancelled:
                task.result.emit(res)
        except Exception:
            if not task._cancelled:
                import traceback
                task.error.emit(traceback.format_exc())
        finally:
            # Always emit finished, then release the registry reference.
            # finished is emitted from the worker thread; Qt auto-queues it
            # to any main-thread slots because BgTask lives on the main thread.
            task.finished.emit()
            _live_tasks.discard(task)


class BgTask(QObject):
    """
    Run ``fn(*args, **kwargs)`` on a private QThread.

    BgTask itself stays on the **main thread** (no moveToThread).  Signals
    are emitted from the worker thread but Qt automatically queues them for
    delivery on the main thread because sender and receiver are on different
    threads.

    Signals
    -------
    result(object)  — callable's return value on success.
    error(str)      — formatted traceback on exception.
    finished()      — always emitted last (success or error).

    Cancellation
    ------------
    cancel()  — suppress result/error emission; the function still runs to
                completion (numpy / file-IO cannot be interrupted).
    wait(ms)  — block until the thread exits; use in closeEvent().
    """

    result   = Signal(object)
    error    = Signal(str)
    finished = Signal()

    def __init__(self, fn, *args, **kwargs) -> None:
        super().__init__()          # stays on the main thread
        self._fn        = fn
        self._args      = args
        self._kwargs    = kwargs
        self._cancelled = False
        self._thread    = _BgThread(self)

    def start(self) -> None:
        """Register in the live-task set and start the thread."""
        _live_tasks.add(self)
        self._thread.start()

    def cancel(self) -> None:
        """Suppress result/error signals (thread still runs to completion)."""
        self._cancelled = True

    def wait(self, msecs: int = 5_000) -> bool:
        """Block until the thread exits.  Returns True if it finished in time."""
        return self._thread.wait(msecs)


class EcomodelWorker(QObject):
    """Runs the ecomodel pipeline off the GUI thread."""

    log         = Signal(str)           # Plain-text log messages → log panel
    progress    = Signal(int, int, str) # (current_step, total_steps, label) → progress bar
    finished    = Signal(object)        # Ecomodel instance on success, None if stopped
    error       = Signal(str)           # Full traceback string on unexpected exception
    tile_update = Signal(str, str, int, str)  # (tile_name, status, cyl_count, output_path)

    def __init__(self, config: EcomodelConfig, parent=None):
        super().__init__(parent)
        self._config = config
        self._stop_requested = False

    @Slot()
    def run(self) -> None:
        """Entry point called by QThread.started — do not call directly."""
        # Import inside run() so the heavy ecomodel import happens on the
        # worker thread, not the GUI thread, keeping startup instant.
        from gui.pipeline import PipelineStopped, run_ecomodel_pipeline
        try:
            result = run_ecomodel_pipeline(
                config=self._config,
                log_callback=self.log.emit,
                progress_callback=self.progress.emit,
                should_stop=lambda: self._stop_requested,
                tile_update_callback=self.tile_update.emit,
            )
            self.finished.emit(result)
        except PipelineStopped as exc:
            # Graceful user-initiated stop (should_stop returned True)
            self.log.emit(f"[Stopped] {exc}\n")
            self.finished.emit(None)
        except Exception:
            import traceback
            self.error.emit(traceback.format_exc())

    def request_stop(self) -> None:
        """
        Thread-safe stop request.

        The pipeline checks this flag at each step boundary and raises
        PipelineStopped to abort cleanly.  The current step will complete
        before the pipeline exits.
        """
        self._stop_requested = True


class BatchQueryWorker(QObject):
    """
    Run a CSV-driven batch of voxel queries off the GUI thread.

    Input CSV columns (header-driven):
        observation_id, lon, lat, height_agl    (height_agl optional)

    For every row we call ``engine.query_voxel_lonlat(lon, lat, h, r)`` for
    each radius in ``radii`` and write a flat row to the output CSV with
    per-radius metric blocks (see QueryPage plan for the exact columns).

    Signals
    -------
    log           — free-form status lines for the GUI log pane
    progress(i,n,obs_id)  — after each row is written
    finished_ok(path)     — on success; path is the output CSV
    failed(str)           — on exception; string is a user-facing message
    """

    log         = Signal(str)
    progress    = Signal(int, int, str)
    finished_ok = Signal(str)
    failed      = Signal(str)

    # Per-radius metric column order — kept in one place so the header and
    # per-row emission cannot drift apart.
    _METRIC_COLS = (
        "point_count",
        "tree_points",
        "veg_cover",
        "n_segments",
        "cyl_count",
        "mean_branch_radius",
        "total_branch_length",
    )

    def __init__(
        self,
        engine,
        input_csv: str,
        output_csv: str,
        radii,
        default_h_agl: float,
        parent=None,
    ):
        super().__init__(parent)
        self._engine        = engine
        self._input_csv     = input_csv
        self._output_csv    = output_csv
        self._radii         = [float(r) for r in radii]
        self._default_h_agl = float(default_h_agl)
        self._stop_requested = False

    def request_stop(self) -> None:
        """Request a graceful stop between rows."""
        self._stop_requested = True

    @classmethod
    def _header(cls, radii) -> list[str]:
        cols = ["observation_id", "lon", "lat", "height_agl"]
        for idx, _r in enumerate(radii, 1):
            prefix = f"r{idx}_"
            cols.extend(prefix + c for c in cls._METRIC_COLS)
        return cols

    @staticmethod
    def _metrics_for(res) -> list:
        """Return cell values in the order declared by _METRIC_COLS."""
        return [
            res.point_count,
            res.tree_point_count,
            f"{res.veg_cover:.6f}" if res.point_count else 0,
            len(res.segment_ids),
            res.cyl_count,
            (f"{res.mean_branch_radius:.6f}"
             if res.mean_branch_radius is not None else ""),
            (f"{res.total_branch_length:.6f}"
             if res.total_branch_length is not None else ""),
        ]

    @Slot()
    def run(self) -> None:
        import csv
        try:
            with open(self._input_csv, "r", newline="") as fin:
                reader = csv.DictReader(fin)
                if reader.fieldnames is None or \
                        "lon" not in reader.fieldnames or \
                        "lat" not in reader.fieldnames:
                    self.failed.emit(
                        "Input CSV must have 'lon' and 'lat' columns "
                        "(got: %s)." % (reader.fieldnames,)
                    )
                    return
                rows = list(reader)

            has_h_col = any("height_agl" in (r or {}) for r in rows)
            if not has_h_col:
                self.log.emit(
                    f"[info] CSV has no 'height_agl' column — using "
                    f"default {self._default_h_agl} m for every row.\n"
                )

            total = len(rows)
            self.log.emit(f"[info] Processing {total} rows at radii "
                          f"{self._radii} m …\n")

            with open(self._output_csv, "w", newline="") as fout:
                writer = csv.writer(fout)
                writer.writerow(self._header(self._radii))

                for i, row in enumerate(rows, 1):
                    if self._stop_requested:
                        self.log.emit(f"[stopped] after {i - 1}/{total} rows.\n")
                        break

                    obs_id = (row.get("observation_id")
                              or row.get("id")
                              or f"row_{i}")
                    try:
                        lon = float(row["lon"])
                        lat = float(row["lat"])
                    except (KeyError, TypeError, ValueError) as exc:
                        self.log.emit(
                            f"[skip] row {i} ({obs_id}): bad lon/lat — {exc}\n"
                        )
                        continue

                    h_raw = row.get("height_agl") if has_h_col else None
                    try:
                        h = float(h_raw) if h_raw not in (None, "") \
                            else self._default_h_agl
                    except ValueError:
                        h = self._default_h_agl

                    out = [obs_id, lon, lat, h]
                    for r in self._radii:
                        try:
                            res = self._engine.query_voxel_lonlat(
                                lon, lat, h, r,
                            )
                        except Exception as exc:
                            self.log.emit(
                                f"[skip] row {i} ({obs_id}) r={r}: {exc}\n"
                            )
                            out.extend([""] * len(self._METRIC_COLS))
                            continue
                        out.extend(self._metrics_for(res))

                    writer.writerow(out)
                    fout.flush()           # crash-safe: each row on disk
                    self.progress.emit(i, total, str(obs_id))

            self.finished_ok.emit(self._output_csv)

        except Exception:
            import traceback
            self.failed.emit(traceback.format_exc())
