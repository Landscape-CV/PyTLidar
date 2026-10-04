"""Small and degenerate covers: the steps either carry on the way TreeQSM does or stop with a clear error."""
import numpy as np

from PyTLidar.Utils import Utils
from PyTLidar.TreeQSMSteps.cover_sets import cover_sets
from PyTLidar.TreeQSMSteps.tree_sets import tree_sets


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
