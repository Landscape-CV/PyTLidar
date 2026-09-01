import numpy as np
import pytest

from PyTLidar.TreeQSMSteps.alpha_shape import alphashape, _circumradii, _circumradius


def _notched_square(n=400):
    rng = np.random.Generator(np.random.Philox(0))
    P = rng.uniform(0, 10, (n, 2))
    # cut a deep notch out of the right hand side
    keep = ~((P[:, 0] > 5) & (np.abs(P[:, 1] - 5) < 1.5))
    return P[keep]


def test_batched_circumradii_match_single():
    rng = np.random.Generator(np.random.Philox(1))
    for K in (2, 3):
        coords = rng.normal(size=(60, K))
        from scipy.spatial import Delaunay
        S = Delaunay(coords).simplices
        r = _circumradii(coords, S)
        single = np.array([_circumradius(coords[s]) for s in S])
        assert np.array_equal(r, single)


def test_concave_hull_is_smaller_than_convex_hull():
    P = _notched_square()
    convex = alphashape(P, 0.0)
    concave = alphashape(P, 1.0 / 0.8)
    assert concave.area < convex.area
    assert concave.area > 0.5 * convex.area


def test_large_alpha_radius_approaches_the_convex_hull():
    P = _notched_square()
    convex = alphashape(P, 0.0)
    loose = alphashape(P, 1.0 / 100.0)
    assert loose.area == pytest.approx(convex.area, rel=1e-2)
    assert loose.area <= convex.area


def test_three_points_return_the_hull():
    P = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    assert alphashape(P, 1.0).area == pytest.approx(0.5)


def test_3d_mesh_volume():
    rng = np.random.Generator(np.random.Philox(2))
    P = rng.uniform(0, 1, (500, 3))
    mesh = alphashape(P, 1.0 / 2.0)
    assert abs(mesh.volume) == pytest.approx(1.0, rel=0.15)
