"""Tests for the stem triangulation path (Tria=1).

`curve_based_triangulation` reconstructs a triangular-mesh surface for the lower
stem. On a synthetic cylinder of known radius/height the enclosed mesh volume
should recover the analytic volume, and the returned mesh must be non-empty and
well-formed. This exercises the faithfulness fixes in the triangulation modules
without the cost of a full QSM run.
"""
import os
import sys

import numpy as np
import pytest

from PyTLidar.TreeQSMSteps.triangulation import curve_based_triangulation
from PyTLidar.TreeQSMSteps.triangulation.boundary_curve import boundary_curve
from PyTLidar.TreeQSMSteps.triangulation.curve_based_triangulation import _triangulate_polygon, _polyarea


def _cylinder_surface(radius, height, n_theta=180, n_z=200, seed=0):
    """Dense points on the lateral surface of a vertical cylinder centered at origin."""
    rng = np.random.default_rng(seed)
    z = rng.uniform(0.0, height, n_theta * n_z)
    theta = rng.uniform(0.0, 2 * np.pi, n_theta * n_z)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return np.column_stack([x, y, z])


def test_curve_based_triangulation_recovers_cylinder_volume():
    radius, height = 0.20, 2.0
    P = _cylinder_surface(radius, height)

    tria_height = 0.10
    tria = curve_based_triangulation(P, tria_height, tria_height)

    # A mesh must actually be produced (empty dict / falsy means it gave up)
    assert tria, "triangulation returned no model for a clean cylinder"

    vert = np.asarray(tria["vert"], float)
    facet = np.asarray(tria["facet"])
    assert vert.shape[0] > 0 and facet.shape[0] > 0, "empty mesh"
    assert vert.shape[1] == 3 and facet.shape[1] == 3

    # tria['volume'] is in litres (m^3 * 1000); analytic = pi r^2 h
    analytic_litres = np.pi * radius**2 * height * 1000.0
    assert tria["volume"] > 0
    rel_err = abs(tria["volume"] - analytic_litres) / analytic_litres
    assert rel_err < 0.20, (
        f"mesh volume {tria['volume']:.1f} L vs analytic {analytic_litres:.1f} L "
        f"(rel err {rel_err:.2%})"
    )

    # Height extent of the reconstructed section should be sane
    assert 0.0 <= tria["bottom"] < tria["top"] <= height + tria_height


def test_curve_based_triangulation_returns_expected_keys():
    P = _cylinder_surface(0.25, 1.5)
    tria = curve_based_triangulation(P, 0.1, 0.1)
    assert tria
    for key in ("vert", "facet", "volume", "SideArea", "bottom", "top", "triah", "triaw"):
        assert key in tria, f"missing key {key} in triangulation output"
        assert np.all(np.isfinite(np.asarray(tria[key], float)))


def _ring_seeds_and_points(nc, empty_seeds, radius=0.5, z_seed=1.0, z_pts=0.9, per_seed=12, seed=0):
    """Seeds on a ring; a tight cluster of points under every seed except those in
    `empty_seeds`, so exactly those segments come out empty in boundary_curve."""
    rng = np.random.default_rng(seed)
    th = np.linspace(0, 2 * np.pi, nc, endpoint=False)
    Curve0 = np.column_stack([radius * np.cos(th), radius * np.sin(th), np.full(nc, z_seed)])
    pts = []
    for i in range(nc):
        if i in empty_seeds:
            continue
        c = Curve0[i, :2]
        pts.append(np.column_stack([c[0] + 0.005 * rng.standard_normal(per_seed),
                                    c[1] + 0.005 * rng.standard_normal(per_seed),
                                    np.full(per_seed, z_pts)]))
    return np.vstack(pts), Curve0


@pytest.mark.parametrize("empty_seeds", [
    {0},                 # only the first seed empty: wrap-around case with b == 1
    {0, 39},             # first and last empty
    {0, 1, 38, 39},      # gap spanning the wrap-around on both sides
    {0, 1, 2, 3, 4, 5},  # long leading gap (takes the Curve0-copy branch)
])
def test_boundary_curve_fills_every_empty_segment(empty_seeds):
    """Empty segments must be interpolated, never left as the zero rows the curve array
    is initialised with. A single unfilled row collapses the whole layer to z=0 (the
    height is flattened with min()) and puts a vertex at the origin, which produced
    tangled stem meshes on real trees."""
    nc = 40
    P, Curve0 = _ring_seeds_and_points(nc, empty_seeds)
    tria_width = 0.1
    Curve, Ind = boundary_curve(P, Curve0, 2 * tria_width, 1.5 * tria_width)

    assert Curve.shape[0] > 0
    # no row may remain all-zero (unfilled)
    assert not np.any(~Curve.any(axis=1)), f"unfilled rows: {np.where(~Curve.any(axis=1))[0].tolist()}"
    # every curve point stays near the ring (no point pulled towards the origin)
    r = np.hypot(Curve[:, 0], Curve[:, 1])
    assert np.all(np.abs(r - 0.5) < 0.1), f"radial range {r.min():.3f}..{r.max():.3f}"
    # layer height is flat and at the points' height, not at zero
    assert np.allclose(Curve[:, 2], Curve[0, 2])
    assert abs(Curve[0, 2] - 0.9) < 1e-6


@pytest.mark.parametrize("n_side", [1, 3, 6])
def test_triangulate_polygon_handles_collinear_runs(n_side):
    """The cap triangulator must not give up on a simple polygon that has runs of
    exactly-collinear vertices (boundary_curve's linear interpolation of empty
    segments produces those). MATLAB's constrained Delaunay accepts such caps, so a
    spurious failure here would send the port down a different retry path."""
    # Square with n_side evenly spaced extra vertices on every edge (all collinear)
    corners = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    pts = []
    for a, b in zip(corners, np.roll(corners, -1, axis=0)):
        for t in np.linspace(0.0, 1.0, n_side + 1, endpoint=False):
            pts.append(a + t * (b - a))
    pts = np.array(pts)
    tri, ok = _triangulate_polygon(pts)
    assert ok, "collinear-run polygon was rejected"
    assert tri.shape[0] == len(pts) - 2
    # covered area equals the polygon area (zero-area facets are harmless)
    x, y = pts[:, 0], pts[:, 1]
    area = 0.0
    for t in tri:
        a, b, c = pts[t[0]], pts[t[1]], pts[t[2]]
        area += 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
    assert abs(area - _polyarea(x, y)) < 1e-9


def _collinear_ring_polygon(trial, seed=0):
    """Noisy ring polygon (simple by construction) with boundary_curve-style runs of
    exactly collinear interpolated vertices; replays the generator to `trial`."""
    rng = np.random.default_rng(seed)
    for t in range(trial + 1):
        n = rng.integers(20, 80)
        th = np.sort(rng.uniform(0, 2 * np.pi, n))
        r = 1.0 + 0.15 * rng.standard_normal(n)
        pts = np.column_stack([r * np.cos(th), r * np.sin(th)])
        out = []
        for i in range(n):
            out.append(pts[i])
            if rng.random() < 0.15:
                k = rng.integers(1, 5)
                nxt = pts[(i + 1) % n]
                for j in range(k):
                    out.append(pts[i] + (j + 1) / (k + 1) * (nxt - pts[i]))
    return np.array(out)


def test_triangulate_polygon_does_not_stall_on_collinear_ring():
    """A simple polygon the strict ear test used to stall on (no strict ear left,
    only collinear vertices), which made the cap gate reject a valid layer."""
    P = _collinear_ring_polygon(529)
    tri, ok = _triangulate_polygon(P)
    assert ok, "simple polygon with collinear runs was rejected"
    assert tri.shape[0] == len(P) - 2
    a = 0.0
    for t in tri:
        A, B, C = P[t[0]], P[t[1]], P[t[2]]
        a += 0.5 * abs((B[0] - A[0]) * (C[1] - A[1]) - (B[1] - A[1]) * (C[0] - A[0]))
    assert abs(a - _polyarea(P[:, 0], P[:, 1])) < 1e-3 * a
