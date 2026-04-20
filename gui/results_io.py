"""
Results I/O helpers for Ecomodel GUI.

Responsibilities
----------------
- Define the on-disk layout for a single pipeline run:

      {results_folder}/
          {run_name}/                   ← created by make_run_dir()
              metadata.json             ← RunMetadata (always written)
              {cylinder_filename}.txt   ← cylinder data  (written by pipeline)
              point_cloud.npy           ← subsampled cloud snapshot (≤100 k pts)
              segment_labels.npy        ← matching segment labels
              pre_rgi_tile_*.xyz        ← intermediate files (written by ecomodel)
              ...

- Provide thin helpers used by pipeline.py and main_window.py.

Public API
----------
make_run_dir(results_folder, run_name) -> Path
    Returns a unique Path inside results_folder.  Appends a timestamp
    suffix when the name already exists.

write_run_metadata(run_dir, config, cylinder_count, point_count)
    Serialises a RunMetadata instance to metadata.json.

load_run_metadata(run_dir) -> RunMetadata | None
    Reads metadata.json; returns None on any error.

find_run_dirs(results_folder) -> list[Path]
    All subfolders that contain metadata.json, newest-first.

save_point_cloud_snapshot(run_dir, cloud, max_points)
    Saves a subsampled float32 cloud to point_cloud.npy.

save_segment_labels_snapshot(run_dir, labels)
    Saves int32 labels matching the (un-subsampled) cloud to
    segment_labels.npy.  Call after save_point_cloud_snapshot so the
    indices line up.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────

METADATA_FILE         = "metadata.json"
POINT_CLOUD_FILE      = "point_cloud.npy"
SEGMENT_LABELS_FILE   = "segment_labels.npy"
COVER_SETS_FILE       = "cover_sets.npy"
POINT_FIELDS_FILE     = "point_fields.npz"   # named scalar fields (intensity, etc.)

_MAX_SNAPSHOT_POINTS  = 300_000


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class RunMetadata:
    """Everything needed to re-open a completed run without the Ecomodel object."""

    run_name:            str
    timestamp:           str          # ISO-8601 e.g. "2026-04-12T14:30:00"
    input_folder:        str
    run_dir:             str          # absolute path to the run subfolder
    cylinder_file:       str          # filename only, e.g. "ecomodel_cylinder_data.txt"
    cylinder_count:      int
    point_count:         int
    qsm_enabled:         bool
    has_point_cloud:     bool         # point_cloud.npy present
    has_segment_labels:  bool         # segment_labels.npy present
    config_snapshot:     dict[str, Any] = field(default_factory=dict)
    cloud_mean:          list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])


# ── Directory helpers ─────────────────────────────────────────────────────────

def make_run_dir(results_folder: str, run_name: str) -> Path:
    """
    Returns a Path for a new run subfolder.

    If {results_folder}/{run_name} does not yet exist, returns it unchanged.
    Otherwise appends a timestamp to avoid clobbering a previous run.
    """
    base = Path(results_folder) / run_name
    if not base.exists():
        return base
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(results_folder) / f"{run_name}_{ts}"


def find_run_dirs(results_folder: str) -> list[Path]:
    """
    Returns all immediate subdirectories of results_folder that contain
    a metadata.json file, sorted newest-first by filesystem mtime.
    """
    p = Path(results_folder)
    if not p.is_dir():
        return []
    return sorted(
        [d for d in p.iterdir() if d.is_dir() and (d / METADATA_FILE).exists()],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )


# ── Metadata read / write ─────────────────────────────────────────────────────

def write_run_metadata(
    run_dir: Path,
    config,               # EcomodelConfig — imported lazily to avoid circular dep
    cylinder_count: int,
    point_count: int,
    cloud_mean: list[float] | None = None,
) -> None:
    """Writes metadata.json into run_dir."""
    try:
        config_dict = asdict(config)
    except Exception:
        config_dict = {}

    _mean = list(cloud_mean) if cloud_mean is not None else [0.0, 0.0, 0.0]

    meta = RunMetadata(
        run_name           = run_dir.name,
        timestamp          = datetime.now().isoformat(timespec="seconds"),
        input_folder       = str(getattr(config, "input_folder", "")),
        run_dir            = str(run_dir.resolve()),
        cylinder_file      = f"{getattr(config, 'cylinder_filename', 'ecomodel_cylinder_data')}.txt",
        cylinder_count     = cylinder_count,
        point_count        = point_count,
        qsm_enabled        = bool(getattr(config, "run_qsm", False)),
        has_point_cloud    = (run_dir / POINT_CLOUD_FILE).exists(),
        has_segment_labels = (run_dir / SEGMENT_LABELS_FILE).exists(),
        config_snapshot    = config_dict,
        cloud_mean         = _mean,
    )

    with open(run_dir / METADATA_FILE, "w", encoding="utf-8") as fh:
        json.dump(asdict(meta), fh, indent=2)


def load_run_metadata(run_dir: Path) -> RunMetadata | None:
    """
    Reads metadata.json from run_dir.
    Returns None if the file is missing or malformed.
    """
    meta_path = run_dir / METADATA_FILE
    if not meta_path.exists():
        return None
    try:
        with open(meta_path, encoding="utf-8") as fh:
            data = json.load(fh)
        return RunMetadata(**data)
    except Exception:
        return None


# ── Array snapshots ───────────────────────────────────────────────────────────

def _voxel_subsample(cloud: np.ndarray, max_points: int) -> np.ndarray:
    """
    Spatially-uniform subsampling via a voxel grid.

    Computes a voxel size that produces roughly max_points occupied voxels,
    then keeps one point per voxel (the one closest to its voxel centre).
    Falls back to random sampling if the voxel approach yields too few or too
    many points (e.g. very uniform clouds where every point is in its own
    voxel at the target resolution).

    Returns an index array into the original cloud.
    """
    n = len(cloud)
    if n <= max_points:
        return np.arange(n)

    # Estimate voxel size from the bounding box so we get ≈ max_points voxels.
    extents = cloud.max(axis=0) - cloud.min(axis=0)
    volume  = float(np.prod(np.maximum(extents, 1e-6)))
    voxel_size = (volume / max_points) ** (1.0 / 3.0)

    # Assign each point to a voxel key.
    mins  = cloud.min(axis=0)
    keys  = ((cloud - mins) / voxel_size).astype(np.int32)
    # Pack (ix, iy, iz) into a single integer for fast grouping.
    max_k = keys.max(axis=0) + 1
    flat  = keys[:, 0] * int(max_k[1]) * int(max_k[2]) + keys[:, 1] * int(max_k[2]) + keys[:, 2]

    # For each voxel keep the point whose flat-key appears first (stable sort).
    order        = np.argsort(flat, kind="stable")
    sorted_keys  = flat[order]
    _, first_idx = np.unique(sorted_keys, return_index=True)
    voxel_indices = order[first_idx]

    # If voxel grid gave too many or too few points, fall back to random.
    if not (max_points // 2 <= len(voxel_indices) <= max_points * 2):
        return np.random.choice(n, min(n, max_points), replace=False)

    # If still over budget, random-sample the voxel representatives.
    if len(voxel_indices) > max_points:
        voxel_indices = np.random.choice(voxel_indices, max_points, replace=False)

    return voxel_indices


def save_point_cloud_snapshot(
    run_dir: Path,
    cloud: np.ndarray,
    max_points: int = _MAX_SNAPSHOT_POINTS,   # kept for back-compat, ignored
) -> np.ndarray:
    """
    Saves the full float32 point cloud to point_cloud.npy.

    Returns np.arange(len(cloud)) so callers can pass the same indices to
    save_segment_labels_snapshot — guaranteeing row-for-row alignment.
    """
    if cloud is None or len(cloud) == 0:
        return np.array([], dtype=np.int64)
    idx = np.arange(len(cloud))
    np.save(run_dir / POINT_CLOUD_FILE, cloud.astype(np.float32))
    return idx


def save_segment_labels_snapshot(
    run_dir: Path,
    labels: np.ndarray,
    indices: np.ndarray | None = None,
) -> None:
    """
    Saves segment labels to segment_labels.npy.

    Pass the index array returned by save_point_cloud_snapshot so that the
    saved labels are guaranteed to align row-for-row with the saved cloud.
    Pass indices=None only if labels already match the (un-subsampled) cloud
    and you want to save every label.
    """
    if labels is None or len(labels) == 0:
        return
    out = labels[indices] if indices is not None else labels
    np.save(run_dir / SEGMENT_LABELS_FILE, out.astype(np.int32))


def save_cover_sets_snapshot(
    run_dir: Path,
    cover_sets: np.ndarray,
    indices: np.ndarray | None = None,
) -> None:
    """
    Saves cover-set IDs to cover_sets.npy using the same index array as the
    point cloud snapshot so all three files (cloud, labels, cover_sets) are
    guaranteed to align row-for-row.
    """
    if cover_sets is None or len(cover_sets) == 0:
        return
    out = cover_sets[indices] if indices is not None else cover_sets
    np.save(run_dir / COVER_SETS_FILE, out.astype(np.int32))


def save_point_fields_snapshot(
    run_dir: Path,
    fields: "dict[str, np.ndarray]",
    indices: np.ndarray | None = None,
) -> None:
    """
    Save named per-point scalar fields to point_fields.npz.

    ``fields`` is a dict mapping a short display name (e.g. "Intensity",
    "Return Number") to a 1-D array of length N aligned with the full cloud.
    The same ``indices`` used for save_point_cloud_snapshot are applied so
    every file in the run folder is row-for-row consistent.

    Loading back::

        data = np.load(run_dir / POINT_FIELDS_FILE)
        intensity = data["Intensity"]
    """
    if not fields:
        return
    to_save = {}
    for name, arr in fields.items():
        if arr is None or len(arr) == 0:
            continue
        out = arr[indices] if indices is not None else arr
        to_save[name] = out.astype(np.float32)
    if to_save:
        np.savez(run_dir / POINT_FIELDS_FILE, **to_save)
