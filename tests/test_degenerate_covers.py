"""Small and degenerate covers: the steps either carry on the way TreeQSM does or stop with a clear error."""
import numpy as np
import pytest

from PyTLidar.Utils import Utils
from PyTLidar.TreeQSMSteps.cover_sets import cover_sets
from PyTLidar.TreeQSMSteps.tree_sets import tree_sets
from PyTLidar.TreeQSMSteps.tree_data import stem_taper


def test_logical_subset_of_three_sets_is_not_read_as_indices():
    Nei = Utils.object_array([np.array([1]), np.array([0]), np.zeros(0, dtype=int)])
    comps, sizes = Utils.connected_components_array(Nei, np.array([True, True, True]), 1, np.zeros(3, dtype=bool))
    assert [sorted(c.tolist()) for c in comps] == [[0, 1], [2]]
    assert list(sizes) == [2, 1]


INPUTS = {'BallRad1': 0.1, 'PatchDiam1': 0.05, 'nmin1': 1, 'OnlyTree': 1}
COLUMN = np.array([[0, 0, 0], [0, 0, 1.0], [0, 0, 2.0]])


def test_isolated_sets_keep_one_empty_neighbor_list_each():
    Nei = cover_sets(COLUMN, dict(INPUTS))['neighbor']
    assert Nei.shape == (3,)
    assert all(len(n) == 0 and n.dtype == np.int64 for n in Nei)


def test_tree_sets_runs_on_isolated_sets(seeded):
    cover, Base, Forb = tree_sets(COLUMN, cover_sets(COLUMN, dict(INPUTS)), dict(INPUTS))
    assert cover['center'][Base].tolist() == [0]  # the set at the lowest point


def test_tree_sets_stops_clearly_on_an_empty_cover():
    cover = {'center': np.zeros(0, dtype=np.int64), 'ball': [], 'neighbor': Utils.object_array([]),
             'sets': np.zeros(3, dtype=int) - 1}
    with pytest.raises(ValueError, match="no sets"):
        tree_sets(np.zeros((3, 3)), cover, dict(INPUTS))


def test_tree_sets_stops_clearly_on_a_flat_cover():
    P = np.array([[0, 0, 0], [1.0, 0, 0], [2.0, 0, 0]])
    with pytest.raises(ValueError, match="one height"):
        tree_sets(P, cover_sets(P, dict(INPUTS)), dict(INPUTS))


def test_second_pass_stops_clearly_when_no_stem_set_is_at_the_bottom():
    segment = {'SegmentOfPoint': np.array([1, 0, 0])}
    with pytest.raises(ValueError, match="base of the trunk"):
        tree_sets(COLUMN, cover_sets(COLUMN, dict(INPUTS)), dict(INPUTS), segment)


def test_stem_taper_with_and_without_trunk_cylinders():
    Rad = np.array([0.1, 0.08, 0.05])
    Len = np.array([1.0, 1.0, 1.0])
    assert np.allclose(stem_taper(Rad, Len, np.array([True, True, False])), [[0, 1, 2], [0.2, 0.16, 0.16]])
    assert np.array_equal(stem_taper(Rad, Len, np.zeros(3, dtype=bool)), np.zeros((2, 1)))
