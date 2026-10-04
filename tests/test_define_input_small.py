import warnings

import numpy as np
import pytest

from PyTLidar.Utils.define_input import define_input


def _stem(radius, height, n=2000):
    rng = np.random.default_rng(0)
    a = rng.uniform(0, 2 * np.pi, n)
    return np.column_stack([radius * np.cos(a), radius * np.sin(a), rng.uniform(0, height, n)])


def test_a_branch_sized_cloud_warns_about_its_patch_sizes():
    with pytest.warns(UserWarning, match="PatchDiam1 is 5 mm"):
        define_input(_stem(0.015, 2.0), 1, 1, 1)


def test_a_tree_sized_stem_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        define_input(_stem(0.2, 10.0), 1, 1, 1)


@pytest.mark.parametrize("n, seed, expected", [
    (5, 1, "5 points"),
    (5, 0, "1 of them in the stem section"),  # this seed leaves one point between 2% and 10% of the height
    (3, 0, "Tree_1"),
])
def test_too_few_stem_points_raise_a_clear_error(n, seed, expected):
    P = np.random.default_rng(seed).uniform([0, 0, 0], [0.4, 0.4, 3.0], (n, 3))
    with pytest.raises(ValueError, match=expected):
        define_input(P, 1, 1, 1)
