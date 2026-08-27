import numpy as np
import pytest

from pytlidar_cc.mesh import cylinder_mesh


def test_empty_input():
    vertices, triangles, vertex_cylinder = cylinder_mesh(
        np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0), np.zeros(0))
    assert vertices.shape == (0, 3)
    assert triangles.shape == (0, 3)
    assert vertex_cylinder.shape == (0,)


def test_counts_and_index_ranges():
    n, facets = 3, 8
    rng = np.random.default_rng(0)
    starts = rng.normal(size=(n, 3))
    axes = rng.normal(size=(n, 3))
    lengths = rng.uniform(0.1, 2.0, n)
    radii = rng.uniform(0.01, 0.5, n)

    vertices, triangles, vertex_cylinder = cylinder_mesh(
        starts, axes, lengths, radii, facets=facets)

    assert vertices.shape == (n * 2 * facets, 3)
    assert triangles.shape == (n * 2 * facets, 3)
    assert vertex_cylinder.shape == (n * 2 * facets,)
    assert triangles.min() >= 0
    assert triangles.max() < len(vertices)
    # Every triangle stays within its own cylinder's vertex block.
    tri_cyl = vertex_cylinder[triangles]
    assert (tri_cyl == tri_cyl[:, :1]).all()
    assert np.array_equal(np.unique(vertex_cylinder), np.arange(n))


def test_vertical_cylinder_geometry():
    length, radius, facets = 2.0, 0.25, 12
    vertices, _, _ = cylinder_mesh(
        np.array([[1.0, -2.0, 5.0]]), np.array([[0.0, 0.0, 1.0]]),
        np.array([length]), np.array([radius]), facets=facets)

    base, top = vertices[:facets], vertices[facets:]
    assert np.allclose(base[:, 2], 5.0)
    assert np.allclose(top[:, 2], 5.0 + length)
    for ring in (base, top):
        assert np.allclose(np.hypot(ring[:, 0] - 1.0, ring[:, 1] + 2.0), radius)


def test_tilted_cylinder_ring_radius():
    axis = np.array([[1.0, 1.0, 1.0]]) / np.sqrt(3.0)
    start = np.array([[0.0, 0.0, 0.0]])
    radius = 0.1
    vertices, _, _ = cylinder_mesh(start, axis, np.array([1.0]),
                                   np.array([radius]), facets=8)
    # Perpendicular distance of every vertex from the axis line is the radius.
    along = vertices @ axis[0]
    perp = vertices - np.outer(along, axis[0])
    assert np.allclose(np.linalg.norm(perp, axis=1), radius)


def test_degenerate_axis_no_nan():
    vertices, triangles, _ = cylinder_mesh(
        np.zeros((1, 3)), np.zeros((1, 3)), np.array([1.0]), np.array([0.1]))
    assert np.isfinite(vertices).all()
    assert len(triangles) == 16
