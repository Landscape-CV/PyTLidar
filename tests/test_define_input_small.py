import numpy as np
import pytest

from PyTLidar.Utils.define_input import define_input


@pytest.mark.parametrize("n, seed, expected", [
    (5, 1, "5 points"),
    (5, 0, "1 of them in the stem section"),  # this seed leaves one point between 2% and 10% of the height
    (3, 0, "Tree_1"),
])
def test_too_few_stem_points_raise_a_clear_error(n, seed, expected):
    P = np.random.default_rng(seed).uniform([0, 0, 0], [0.4, 0.4, 3.0], (n, 3))
    with pytest.raises(ValueError, match=expected):
        define_input(P, 1, 1, 1)
