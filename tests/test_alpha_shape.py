import numpy as np
import pytest
from scipy.spatial import Delaunay

from PyTLidar.TreeQSMSteps.alpha_shape import alpha_area, alpha_volume, _circumradii, _circumradius


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
        S = Delaunay(coords).simplices
        r = _circumradii(coords, S)
        single = np.array([_circumradius(coords[s]) for s in S])
        assert np.array_equal(r, single)


def test_concave_area_is_smaller_than_convex_area():
    P = _notched_square()
    convex = alpha_area(P, 0.0)
    concave = alpha_area(P, 0.8)
    assert concave < convex
    assert concave > 0.5 * convex


def test_large_alpha_radius_approaches_the_convex_hull():
    P = _notched_square()
    convex = alpha_area(P, 0.0)
    loose = alpha_area(P, 100.0)
    assert loose == pytest.approx(convex, rel=1e-2)
    assert loose <= convex


def test_three_points_give_the_triangle_area():
    P = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    assert alpha_area(P, 1.0) == pytest.approx(0.5)


def test_cube_volume():
    rng = np.random.Generator(np.random.Philox(2))
    P = rng.uniform(0, 1, (2000, 3))
    assert alpha_volume(P, 0.5) == pytest.approx(1.0, rel=0.15)


def test_interior_void_is_filled():
    rng = np.random.Generator(np.random.Philox(3))
    P = rng.uniform(0, 1, (4000, 3))
    hollow = P[np.any((P < 0.3) | (P > 0.7), axis=1)]
    full = alpha_volume(P, 0.4)
    assert alpha_volume(hollow, 0.4) == pytest.approx(full, rel=0.05)
