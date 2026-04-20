"""
query_engine.py — Spatial voxel query for a completed pipeline run.

Loads the saved point-cloud snapshot, segment labels, and (optionally) the
QSM cylinder file for a run directory, then answers axis-aligned voxel
queries: all points inside a user-defined cube, broken down by vegetation
cover, tree-segment membership, and cylinder (branch) statistics.

All queries operate in the run's **normalised space** (XY minus eco.mean[:2],
Z terrain-subtracted).  World coordinates are converted via world_to_norm()
using the cloud_mean stored in metadata.json.  Older runs without cloud_mean
will accept raw normalised coordinates (has_mean == False).

Cylinder start positions are stored in world space in the .txt file; they are
converted to normalised space at query time by subtracting cloud_mean.

Public API
----------
VoxelQueryResult    dataclass — result of a single voxel query
QueryEngine         class     — loads run data once, answers repeated queries
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gui.results_io import (
    POINT_CLOUD_FILE,
    SEGMENT_LABELS_FILE,
    load_run_metadata,
)


def _try_read_epsg(input_folder: Path) -> "int | None":
    """
    Try to read an EPSG code from the first LAS/LAZ file in *input_folder*.

    Requires laspy + pyproj.  Returns None on any failure (missing files,
    no CRS embedded, pyproj not installed, non-EPSG CRS, …).
    """
    try:
        import laspy
        las_files = (
            list(input_folder.glob("*.las")) +
            list(input_folder.glob("*.laz"))
        )
        if not las_files:
            return None
        with laspy.open(las_files[0]) as f:
            crs = f.header.parse_crs()
            if crs is None:
                return None
            epsg = crs.to_epsg()
            return int(epsg) if epsg is not None else None
    except Exception:
        return None


@dataclass
class VoxelQueryResult:
    """Result of a single axis-aligned voxel query."""

    # ── Geometry ──────────────────────────────────────────────────────────────
    voxel_size:          float             # side length of the cube (metres)
    center_norm:         np.ndarray        # (3,) centre in normalised space
    bounds:              tuple             # (x0,x1, y0,y1, z0,z1) normalised

    # ── Point cloud breakdown ─────────────────────────────────────────────────
    point_count:         int               # total points inside the voxel
    tree_point_count:    int               # label >= 0
    ground_point_count:  int               # label == -1 (non-tree / ground)
    veg_cover:           float             # tree_point_count / point_count (0–1)

    # ── Per-segment breakdown ─────────────────────────────────────────────────
    segment_ids:         list              # sorted list of tree-segment IDs found
    segment_counts:      dict              # {segment_id: point_count}

    # ── Cylinder / branch stats (None when no QSM output loaded) ─────────────
    cyl_count:           int               # cylinders with start-point in voxel
    mean_branch_radius:  "float | None"    # metres
    max_branch_radius:   "float | None"    # metres
    total_branch_length: "float | None"    # metres
    branch_radii:        "np.ndarray | None"   # (M,) per-cylinder radii
    branch_lengths:      "np.ndarray | None"   # (M,) per-cylinder lengths

    # ── Mask back into the loaded cloud ──────────────────────────────────────
    point_mask:          np.ndarray        # (N,) bool


class QueryEngine:
    """
    Loads a single run's snapshot arrays and provides spatial voxel queries.

    Usage::

        engine = QueryEngine(run_dir)
        err = engine.load()
        if err:
            print(f"Load failed: {err}")
        else:
            result = engine.query_voxel(wx, wy, wz, voxel_size=1.0)
    """

    def __init__(self, run_dir: Path) -> None:
        self._run_dir = Path(run_dir)
        self._cloud:  np.ndarray | None = None   # (N, 3) float32 normalised
        self._labels: np.ndarray | None = None   # (N,) int32
        self._cyls:   dict | None = None          # {start, radius, axis, length}
        self._mean:   list[float] = [0.0, 0.0, 0.0]
        self._cover_sets: np.ndarray | None = None   # int32 cover-set IDs
        self._epsg:   int | None = None               # projected CRS of the source LAS data
        self._loaded: bool = False

    # ── Public API ────────────────────────────────────────────────────────────

    def load(self) -> str | None:
        """
        Load point_cloud.npy, segment_labels.npy, and the cylinder .txt file.

        segment_labels and cylinder data are optional — the engine loads them
        when present and silently skips them when not.

        Returns None on success, or a human-readable error string on failure.
        Only the point cloud is strictly required; a missing cloud is fatal.
        """
        self._loaded = False

        # ── cloud_mean + CRS from metadata ───────────────────────────────────
        meta = load_run_metadata(self._run_dir)
        if meta is not None and hasattr(meta, "cloud_mean"):
            self._mean = list(meta.cloud_mean)

        # Try to read the projected CRS from the original LAS/LAZ files so
        # that lon/lat queries can be converted to the run's coordinate space.
        if meta is not None and meta.input_folder:
            self._epsg = _try_read_epsg(Path(meta.input_folder))

        # ── Point cloud (required) ────────────────────────────────────────────
        cloud_path = self._run_dir / POINT_CLOUD_FILE
        if not cloud_path.exists():
            return f"Point cloud snapshot not found:\n{cloud_path}"
        try:
            self._cloud = np.load(cloud_path)
        except Exception as exc:
            return f"Could not load point cloud:\n{exc}"
        if self._cloud.ndim != 2 or self._cloud.shape[1] != 3:
            return f"Unexpected point cloud shape: {self._cloud.shape} (expected N×3)"

        # ── Segment labels (optional) ─────────────────────────────────────────
        self._labels = None
        labels_path  = self._run_dir / SEGMENT_LABELS_FILE
        if labels_path.exists():
            try:
                lbl = np.load(labels_path).astype(np.int32)
                if len(lbl) == len(self._cloud):
                    self._labels = lbl
            except Exception:
                pass

        # ── Cover sets (optional) ────────────────────────────────────────────
        self._cover_sets = None
        from gui.results_io import COVER_SETS_FILE

        cov_path = self._run_dir / COVER_SETS_FILE
        if cov_path.exists():
            try:
                cov = np.load(cov_path).astype(np.int32)
                if len(cov) == len(self._cloud):
                    self._cover_sets = cov
            except Exception:
                pass

        # ── Cylinder data (optional — QSM output) ────────────────────────────
        self._cyls = None
        if meta is not None:
            cyl_path = self._run_dir / meta.cylinder_file
            if cyl_path.exists() and cyl_path.stat().st_size > 0:
                try:
                    data = np.loadtxt(cyl_path)
                    if data.ndim == 1:
                        data = data.reshape(1, -1)
                    if data.ndim == 2 and data.shape[1] == 8 and data.shape[0] > 0:
                        self._cyls = {
                            "start":  data[:, 0:3],
                            "radius": data[:, 3],
                            "axis":   data[:, 4:7],
                            "length": data[:, 7],
                        }
                except Exception:
                    pass

        self._loaded = True
        return None

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        """True after a successful load()."""
        return self._loaded

    @property
    def has_mean(self) -> bool:
        """True when cloud_mean was saved for this run (non-zero origin)."""
        return any(v != 0.0 for v in self._mean)

    @property
    def has_labels(self) -> bool:
        """True when segment labels were loaded successfully."""
        return self._labels is not None

    @property
    def has_cyls(self) -> bool:
        """True when QSM cylinder data was loaded successfully."""
        return self._cyls is not None

    @property
    def has_cover_sets(self) -> bool:
        """True when cover_sets.npy was loaded successfully."""
        return self._cover_sets is not None

    @property
    def cover_sets(self) -> "np.ndarray | None":
        """int32 cover-set ID per cloud point, or None."""
        return self._cover_sets

    @property
    def cloud(self) -> np.ndarray | None:
        return self._cloud

    @property
    def labels(self) -> np.ndarray | None:
        return self._labels

    @property
    def cyls(self) -> "dict | None":
        """Raw cylinder dict {start, radius, axis, length}, or None if no QSM data."""
        return self._cyls

    @property
    def mean(self) -> list[float]:
        return self._mean

    @property
    def epsg(self) -> "int | None":
        """EPSG code of the source LAS data's projected CRS, or None."""
        return self._epsg

    @property
    def has_crs(self) -> bool:
        """True when a projected CRS was successfully read from the source LAS files."""
        return self._epsg is not None

    # ── Coordinate conversion ─────────────────────────────────────────────────

    def lonlat_to_norm(
        self,
        lon: float,
        lat: float,
    ) -> "tuple[float, float]":
        """
        Convert WGS84 lon/lat (degrees) to normalised XY coordinates.

        Requires pyproj and that the engine loaded a valid CRS from the
        source LAS files (``has_crs == True``).

        Raises
        ------
        RuntimeError  — if no CRS is available.
        ImportError   — if pyproj is not installed.
        """
        if not self.has_crs:
            raise RuntimeError(
                "Cannot convert lon/lat: no projected CRS was found in the "
                "source LAS files.  Enter X/Y coordinates directly instead."
            )
        from pyproj import Transformer
        t = Transformer.from_crs("EPSG:4326", f"EPSG:{self._epsg}", always_xy=True)
        wx, wy = t.transform(lon, lat)
        return wx - self._mean[0], wy - self._mean[1]

    def world_to_norm(
        self,
        wx: float,
        wy: float,
        wz: float | None = None,
    ) -> np.ndarray:
        """
        Convert world coordinates to normalised (snapshot) space.

        wz=None sets the Z component to 0 (terrain level in normalised space).
        """
        nz = (wz - self._mean[2]) if wz is not None else 0.0
        return np.array([wx - self._mean[0], wy - self._mean[1], nz], dtype=np.float64)

    # ── Voxel query ───────────────────────────────────────────────────────────

    def query_voxel_lonlat(
        self,
        lon: float,
        lat: float,
        height_agl: float,
        voxel_size: float = 1.0,
    ) -> VoxelQueryResult:
        """
        Convenience wrapper for batch GPS-based queries.

        Centres an axis-aligned cube of side ``voxel_size`` at WGS84
        (lon, lat) and ``height_agl`` metres above the terrain, then returns
        the same :class:`VoxelQueryResult` as :meth:`query_voxel`.

        Because the stored cloud is terrain-subtracted, centring at normalised
        Z == ``height_agl`` means "height_agl metres above ground".  We pass
        ``wz = self._mean[2] + height_agl`` so :meth:`world_to_norm` folds it
        to exactly that Z.

        Requires ``has_crs == True`` — raises ``RuntimeError`` otherwise via
        :meth:`lonlat_to_norm`.
        """
        nx, ny = self.lonlat_to_norm(lon, lat)
        # Re-add cloud_mean to hand world coords back to query_voxel,
        # which will subtract it again inside world_to_norm().
        wx = nx + self._mean[0]
        wy = ny + self._mean[1]
        wz = self._mean[2] + float(height_agl)
        return self.query_voxel(wx, wy, wz, voxel_size)

    def query_voxel(
        self,
        wx: float,
        wy: float,
        wz: float | None = None,
        voxel_size: float = 1.0,
    ) -> VoxelQueryResult:
        """
        Return all data for an axis-aligned cube of side ``voxel_size`` (m)
        centred on the world coordinate (wx, wy[, wz]).

        wz is optional — when omitted, the cube is centred at Z=0 in
        normalised space (terrain level).

        Raises RuntimeError when called before load() succeeds.
        """
        if not self._loaded:
            raise RuntimeError("QueryEngine.load() must succeed before querying.")

        centre = self.world_to_norm(wx, wy, wz)
        half   = voxel_size / 2.0
        x0, x1 = centre[0] - half, centre[0] + half
        y0, y1 = centre[1] - half, centre[1] + half
        z0, z1 = centre[2] - half, centre[2] + half

        cloud = self._cloud
        point_mask = (
            (cloud[:, 0] >= x0) & (cloud[:, 0] <= x1) &
            (cloud[:, 1] >= y0) & (cloud[:, 1] <= y1) &
            (cloud[:, 2] >= z0) & (cloud[:, 2] <= z1)
        )
        point_count = int(point_mask.sum())

        # ── Point cloud breakdown ─────────────────────────────────────────────
        if self._labels is not None and point_count > 0:
            labels_in    = self._labels[point_mask]
            tree_mask_in = labels_in >= 0
            tree_count   = int(tree_mask_in.sum())
            ground_count = point_count - tree_count
            veg_cover    = tree_count / point_count

            if tree_count > 0:
                unique_segs, seg_cnts = np.unique(
                    labels_in[tree_mask_in], return_counts=True
                )
                segment_ids    = [int(s) for s in unique_segs]
                segment_counts = {int(s): int(c)
                                  for s, c in zip(unique_segs, seg_cnts)}
            else:
                segment_ids    = []
                segment_counts = {}
        else:
            tree_count     = 0
            ground_count   = point_count
            veg_cover      = 0.0
            segment_ids    = []
            segment_counts = {}

        # ── Cylinder breakdown ────────────────────────────────────────────────
        cyl_count           = 0
        mean_branch_radius  = None
        max_branch_radius   = None
        total_branch_length = None
        branch_radii        = None
        branch_lengths      = None

        if self._cyls is not None:
            # Cylinder starts are in world space — convert to normalised.
            starts_norm = (
                self._cyls["start"] - np.array(self._mean, dtype=np.float64)
            )
            in_voxel = (
                (starts_norm[:, 0] >= x0) & (starts_norm[:, 0] <= x1) &
                (starts_norm[:, 1] >= y0) & (starts_norm[:, 1] <= y1) &
                (starts_norm[:, 2] >= z0) & (starts_norm[:, 2] <= z1)
            )
            cyl_count = int(in_voxel.sum())
            if cyl_count > 0:
                branch_radii        = self._cyls["radius"][in_voxel]
                branch_lengths      = self._cyls["length"][in_voxel]
                mean_branch_radius  = float(branch_radii.mean())
                max_branch_radius   = float(branch_radii.max())
                total_branch_length = float(branch_lengths.sum())

        return VoxelQueryResult(
            voxel_size          = voxel_size,
            center_norm         = centre,
            bounds              = (x0, x1, y0, y1, z0, z1),
            point_count         = point_count,
            tree_point_count    = tree_count,
            ground_point_count  = ground_count,
            veg_cover           = veg_cover,
            segment_ids         = segment_ids,
            segment_counts      = segment_counts,
            cyl_count           = cyl_count,
            mean_branch_radius  = mean_branch_radius,
            max_branch_radius   = max_branch_radius,
            total_branch_length = total_branch_length,
            branch_radii        = branch_radii,
            branch_lengths      = branch_lengths,
            point_mask          = point_mask,
        )
