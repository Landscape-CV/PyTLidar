"""Small and degenerate covers: the steps either carry on the way TreeQSM does or stop with a clear error."""
import numpy as np

from PyTLidar.Utils import Utils


def test_logical_subset_of_three_sets_is_not_read_as_indices():
    Nei = Utils.object_array([np.array([1]), np.array([0]), np.zeros(0, dtype=int)])
    comps, sizes = Utils.connected_components_array(Nei, np.array([True, True, True]), 1, np.zeros(3, dtype=bool))
    assert [sorted(c.tolist()) for c in comps] == [[0, 1], [2]]
    assert list(sizes) == [2, 1]
