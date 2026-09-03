import numpy as np
import pytest
from numba import njit


@njit
def _seed_numba(n):
    np.random.seed(n)


@pytest.fixture
def seeded():
    np.random.seed(0)
    _seed_numba(0)


def _cylinder_points(rng, start, axis, length, radius, n):
    axis = axis / np.linalg.norm(axis)
    u = np.cross(axis, [1.0, 0.0, 0.0])
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    t = rng.uniform(0, length, n)
    a = rng.uniform(0, 2 * np.pi, n)
    r = radius + rng.normal(0, 0.002, n)
    return start + np.outer(t, axis) + np.outer(r * np.cos(a), u) + np.outer(r * np.sin(a), v)


@pytest.fixture
def cylinder_points():
    return _cylinder_points


@pytest.fixture
def small_tree():
    """A 4 m trunk with one branch, dense enough to model in a few seconds."""
    rng = np.random.default_rng(1)
    trunk = _cylinder_points(rng, np.array([0, 0, 0.0]), np.array([0, 0, 1.0]), 4.0, 0.12, 6000)
    branch = _cylinder_points(rng, np.array([0, 0, 2.5]), np.array([1.0, 0.3, 0.8]), 2.0, 0.05, 2500)
    return np.vstack([trunk, branch])
