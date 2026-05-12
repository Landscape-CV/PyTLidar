"""
PyVista rendering functions for the EcomodelMainWindow GUI.

Threading model
---------------
All ``build_*_meshes()`` functions are **thread-safe**: they only create
``pv.PolyData`` / ``pv.DataSet`` objects (pure VTK data, no OpenGL).  Call
them on a background thread to avoid blocking the GUI.

``apply_meshes_to_plotter()`` and all ``render_*()`` wrappers call
``plotter.add_mesh()`` and must be called on the **main thread**.

Public API
----------
build_point_cloud_meshes(points, fields, active_field)
    -> (mesh_list, field_names)

build_segment_meshes(points, cover_sets, labels, view_mode)
    -> (mesh_list, mode_names)

build_cylinder_meshes(cylinders, line_threshold)
    -> (mesh_list, starts, ends, radii, lengths)

build_voxel_query_meshes(cloud, labels, result, wx, wy, wz)
    -> mesh_list

apply_meshes_to_plotter(plotter, mesh_list, background)
    Add pre-built meshes to *plotter* (main thread only).

render_point_cloud(plotter, points, fields, active_field) -> list[str]
render_segments(plotter, points, cover_sets, labels, view_mode) -> list[str]
render_cylinders(plotter, cylinders, line_threshold) -> (starts, ends, radii, lengths)
render_voxel_query(plotter, cloud, labels, result, wx, wy, wz)
    Legacy wrappers — call build_* then apply_meshes_to_plotter on the
    same (main) thread.
"""

from __future__ import annotations

import numpy as np
import pyvista as pv


# ── Helpers ────────────────────────────────────────────────────────────────────

def _hex_to_rgb01(hex_color: str) -> tuple[float, float, float]:
    """Convert '#rrggbb' to (r, g, b) in [0, 1]."""
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]


def _turbo_color(t: float) -> str:
    """Sample the Turbo colormap at t ∈ [0, 1], return hex string."""
    import matplotlib.pyplot as plt
    rgba = plt.get_cmap("turbo")(float(np.clip(t, 0.0, 1.0)))
    return "#{:02x}{:02x}{:02x}".format(
        int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
    )


# ── Radius-class colour scheme (matches utils/plot_tools.ResultsPlotter) ──────

_RADIUS_CLASSES: list[tuple[float, float, str]] = [
    (0.00, 0.02, "red"),       # twig       0–2 cm
    (0.02, 0.05, "green"),     # small      2–5 cm
    (0.05, 0.10, "steelblue"), # medium     5–10 cm
    (0.10, 1e9,  "orange"),    # large      10+ cm
]


def _radius_class_color(radius: float) -> str:
    for lo, hi, color in _RADIUS_CLASSES:
        if lo < radius <= hi:
            return color
    return "orange"


# ── Thread-safe mesh builders ─────────────────────────────────────────────────

def build_point_cloud_meshes(
    points: np.ndarray,
    fields: "dict[str, np.ndarray] | None" = None,
    active_field: "str | None" = None,
) -> "tuple[list, list[str]]":
    """
    Build PolyData for a point cloud view.  Thread-safe.

    Returns
    -------
    (mesh_list, field_names)
        mesh_list  — list of (pv.PolyData, add_mesh_kwargs) tuples
        field_names — ordered list of available field names
    """
    if points is None or len(points) == 0:
        return [], []

    cloud = pv.PolyData(points.astype(np.float32))

    all_fields: dict[str, np.ndarray] = {
        "Height (Z)": points[:, 2].astype(np.float32)
    }
    if fields:
        for name, arr in fields.items():
            if arr is not None and len(arr) == len(points):
                all_fields[name] = np.asarray(arr, dtype=np.float32)

    for name, arr in all_fields.items():
        cloud.point_data[name] = arr

    if active_field is None or active_field not in all_fields:
        active_field = next(
            (k for k in all_fields if "intensity" in k.lower()),
            list(all_fields.keys())[0],
        )

    cmap = "plasma" if "intensity" in active_field.lower() else "viridis"

    mesh_list = [(cloud, dict(
        scalars=active_field,
        cmap=cmap,
        point_size=3,
        render_points_as_spheres=False,
        show_scalar_bar=True,
        scalar_bar_args={
            "title": active_field, "vertical": True,
            "n_labels": 5, "fmt": "%.2f",
        },
    ))]
    return mesh_list, list(all_fields.keys())


def build_segment_meshes(
    points: np.ndarray,
    cover_sets: "np.ndarray | None",
    labels: "np.ndarray | None",
    view_mode: str = "Segment",
) -> "tuple[list, list[str]]":
    """
    Build PolyData objects for a segmentation view.  Thread-safe.

    Returns
    -------
    (mesh_list, mode_names)
    """
    if points is None or len(points) == 0:
        return [], []

    pts = points.astype(np.float32)
    n = len(pts)

    if labels is None:
        labels = np.full(n, -1, dtype=np.int32)

    mode_names: list[str] = ["Segment"]
    if cover_sets is not None:
        mode_names.append("Cover Set")

    if view_mode == "Cover Set" and cover_sets is not None:
        ids = np.asarray(cover_sets, dtype=np.int32)
        cmap_name = "tab20b"
    else:
        ids = np.asarray(labels, dtype=np.int32)
        cmap_name = "turbo"

    mesh_list = []

    bg_mask = ids < 0
    if bg_mask.any():
        bg = pv.PolyData(pts[bg_mask])
        mesh_list.append((bg, dict(
            color="grey", opacity=0.35, point_size=1.5,
            render_points_as_spheres=False,
        )))

    fg_mask = ~bg_mask
    if fg_mask.any():
        fg_pts = pts[fg_mask]
        fg_ids = ids[fg_mask]
        unique_ids = np.unique(fg_ids)
        n_unique = max(len(unique_ids), 1)

        norm_scalar = np.empty(fg_mask.sum(), dtype=np.float32)
        for rank, uid in enumerate(unique_ids):
            norm_scalar[fg_ids == uid] = (rank + 0.5) / n_unique

        fg_cloud = pv.PolyData(fg_pts)
        fg_cloud.point_data["colour"] = norm_scalar
        mesh_list.append((fg_cloud, dict(
            scalars="colour",
            cmap=cmap_name,
            clim=[0.0, 1.0],
            point_size=2,
            render_points_as_spheres=False,
            show_scalar_bar=False,
        )))

    return mesh_list, mode_names


def build_cylinder_meshes(
    cylinders: dict,
    line_threshold: float = 0.1,
) -> "tuple[list, np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    """
    Build VTK meshes for a QSM cylinder view.  Thread-safe.

    Returns
    -------
    (mesh_list, starts, ends, radii, lengths)
        mesh_list — list of (mesh, add_mesh_kwargs) tuples
        starts, ends — (N, 3) arrays for click-volume callback
        radii, lengths — (N,) arrays for click-volume callback
    """
    starts  = np.asarray(cylinders["start"],  dtype=np.float64)
    axes    = np.asarray(cylinders["axis"],   dtype=np.float64)
    radii   = np.asarray(cylinders["radius"], dtype=np.float64)
    lengths = np.asarray(cylinders["length"], dtype=np.float64)
    n = len(radii)

    if n == 0:
        empty = np.empty((0, 3), dtype=np.float64)
        return [], empty, empty, np.empty(0), np.empty(0)

    norms = np.linalg.norm(axes, axis=1, keepdims=True)
    safe = norms.ravel() > 1e-9
    axis_norm = np.where(norms > 1e-9, axes / np.maximum(norms, 1e-9), axes)
    ends = starts + axis_norm * lengths[:, None]

    mesh_list = []

    # ── Thin branches → single PolyData with line cells ───────────────────────
    thin_mask = (radii <= line_threshold) & safe
    if thin_mask.any():
        n_thin = int(thin_mask.sum())
        thin_starts = starts[thin_mask]
        thin_ends   = ends[thin_mask]
        line_pts = np.empty((n_thin * 2, 3), dtype=np.float32)
        line_pts[0::2] = thin_starts.astype(np.float32)
        line_pts[1::2] = thin_ends.astype(np.float32)
        cells = np.empty(n_thin * 3, dtype=np.int_)
        cells[0::3] = 2
        cells[1::3] = np.arange(n_thin) * 2
        cells[2::3] = np.arange(n_thin) * 2 + 1
        line_mesh = pv.PolyData()
        line_mesh.points = line_pts
        line_mesh.lines = cells
        mesh_list.append((line_mesh, dict(
            color="steelblue", line_width=1.5, opacity=0.85,
        )))

    # ── Thick cylinders → batch by radius class ────────────────────────────────
    thick_mask = (~thin_mask) & safe
    if thick_mask.any():
        thick_idx = np.where(thick_mask)[0]
        class_buckets: dict[str, list[int]] = {c: [] for _, _, c in _RADIUS_CLASSES}
        for i in thick_idx:
            class_buckets[_radius_class_color(radii[i])].append(i)

        for color, idx_list in class_buckets.items():
            if not idx_list:
                continue
            meshes: list[pv.PolyData] = []
            for i in idx_list:
                center = starts[i] + (lengths[i] / 2.0) * axis_norm[i]
                try:
                    mesh = pv.Cylinder(
                        center=center.tolist(),
                        direction=axis_norm[i].tolist(),
                        radius=float(radii[i]),
                        height=float(lengths[i]),
                        resolution=8,
                    )
                    meshes.append(mesh)
                except Exception:
                    continue
            if meshes:
                combined = pv.merge(meshes) if len(meshes) > 1 else meshes[0]
                mesh_list.append((combined, dict(color=color, opacity=0.85)))

    return mesh_list, starts, ends, radii, lengths


def build_voxel_query_meshes(
    cloud: np.ndarray,
    labels: "np.ndarray | None",
    result,
    wx: float,
    wy: float,
    wz: "float | None",
    cover_sets: "np.ndarray | None" = None,
    coloring: str = "segment",   # "segment" | "cover_set"
) -> list:
    """
    Build all VTK meshes for a voxel query visualisation.  Thread-safe.

    Returns
    -------
    list of (mesh, add_mesh_kwargs) tuples
    """
    if cloud is None or len(cloud) == 0:
        return []

    mask  = result.point_mask
    x0, x1, y0, y1, z0, z1 = result.bounds

    mesh_list = []

    # ── Interior points ───────────────────────────────────────────────────────
    # (Background cloud is intentionally omitted here — callers that want
    #  spatial context should add their own background layer, e.g. the full
    #  point cloud or the QSM cylinders.)
    fg_pts = cloud[mask].astype(np.float32)
    if len(fg_pts) > 0:

        if coloring == "cover_set" and cover_sets is not None:
            # ── Cover-set coloring ────────────────────────────────────────
            fg_cs = cover_sets[mask].astype(np.int32)
            unique_cs = np.unique(fg_cs[fg_cs >= 0])
            n_cs = max(len(unique_cs), 1)
            bg_m = fg_cs < 0
            if bg_m.any():
                mesh_list.append((pv.PolyData(fg_pts[bg_m]), dict(
                    color="grey", opacity=0.35, point_size=1.5,
                    render_points_as_spheres=False,
                )))
            for rank, cs_id in enumerate(unique_cs):
                cs_mask = fg_cs == cs_id
                t = (rank + 0.5) / n_cs
                mesh_list.append((pv.PolyData(fg_pts[cs_mask]), dict(
                    color=_turbo_color(t), opacity=0.90, point_size=3,
                    render_points_as_spheres=False,
                )))

        else:
            # ── Default: segment coloring ─────────────────────────────────
            if labels is not None:
                fg_labels = labels[mask]

                ground_mask = fg_labels < 0
                if ground_mask.any():
                    gp = pv.PolyData(fg_pts[ground_mask])
                    mesh_list.append((gp, dict(
                        color="#795548", opacity=0.70, point_size=2.5,
                        render_points_as_spheres=False,
                    )))

                if result.segment_ids:
                    n_segs = max(len(result.segment_ids), 1)
                    for rank, seg_id in enumerate(result.segment_ids):
                        seg_mask = fg_labels == seg_id
                        if not seg_mask.any():
                            continue
                        t = (rank + 0.5) / n_segs
                        color = _turbo_color(t)
                        sp = pv.PolyData(fg_pts[seg_mask])
                        mesh_list.append((sp, dict(
                            color=color, opacity=0.90, point_size=3,
                            render_points_as_spheres=False,
                        )))
            else:
                fp = pv.PolyData(fg_pts)
                mesh_list.append((fp, dict(
                    color="#ff6f00", opacity=0.90, point_size=3,
                    render_points_as_spheres=False,
                )))

    # ── 4. Voxel wireframe box ────────────────────────────────────────────────
    box = pv.Box(bounds=(x0, x1, y0, y1, z0, z1))
    mesh_list.append((box, dict(
        style="wireframe", color="#0288d1", line_width=3, opacity=1.0,
    )))

    # ── 5. Query-centre marker ────────────────────────────────────────────────
    qn = np.asarray(result.center_norm, dtype=np.float32)
    sphere_r = float(result.voxel_size) * 0.04
    sphere = pv.Sphere(radius=max(sphere_r, 0.01), center=qn.tolist())
    mesh_list.append((sphere, dict(color="#e53935", opacity=1.0)))

    return mesh_list


# ── Main-thread apply helper ──────────────────────────────────────────────────

def apply_meshes_to_plotter(
    plotter,
    mesh_list: list,
    background: str = "white",
) -> None:
    """
    Add pre-built meshes to *plotter*.  Must be called on the main thread.

    Parameters
    ----------
    plotter : QtInteractor
        Already cleared by the caller (e.g. EmbeddedPlotWidget.show_pyvista_meshes).
    mesh_list : list of (mesh, add_mesh_kwargs) tuples
    background : str
        Background colour passed to plotter.set_background().
    """
    for mesh, kwargs in mesh_list:
        plotter.add_mesh(mesh, **kwargs)
    plotter.add_axes(line_width=2)
    plotter.set_background(background)


# ── Legacy wrappers (main-thread, build + apply in one call) ──────────────────

def render_point_cloud(
    plotter,
    points: np.ndarray,
    fields: "dict[str, np.ndarray] | None" = None,
    active_field: "str | None" = None,
) -> "list[str]":
    """Build and apply a point cloud render.  Main thread only."""
    mesh_list, field_names = build_point_cloud_meshes(points, fields, active_field)
    apply_meshes_to_plotter(plotter, mesh_list)
    return field_names


def render_segments(
    plotter,
    points: np.ndarray,
    cover_sets: "np.ndarray | None",
    labels: "np.ndarray | None",
    view_mode: str = "Segment",
) -> "list[str]":
    """Build and apply a segmentation render.  Main thread only."""
    mesh_list, mode_names = build_segment_meshes(points, cover_sets, labels, view_mode)
    apply_meshes_to_plotter(plotter, mesh_list)
    return mode_names


def render_cylinders(
    plotter,
    cylinders: dict,
    line_threshold: float = 0.1,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    """Build and apply a cylinder render.  Main thread only."""
    mesh_list, starts, ends, radii, lengths = build_cylinder_meshes(
        cylinders, line_threshold
    )
    apply_meshes_to_plotter(plotter, mesh_list)
    return starts, ends, radii, lengths


def render_voxel_query(
    plotter,
    cloud: np.ndarray,
    labels: "np.ndarray | None",
    result,
    wx: float,
    wy: float,
    wz: "float | None",
) -> None:
    """Build and apply a voxel query render.  Main thread only."""
    mesh_list = build_voxel_query_meshes(cloud, labels, result, wx, wy, wz)
    apply_meshes_to_plotter(plotter, mesh_list)
