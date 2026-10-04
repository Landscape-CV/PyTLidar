"""Segment and child lists must stay one dimensional object arrays whatever shape their items have."""
import numpy as np

from PyTLidar.Utils import Utils
from PyTLidar.TreeQSMSteps.correct_segments import remove_small
from PyTLidar.TreeQSMSteps.segments import segments


def test_object_array_keeps_equal_length_items_apart():
    layers = [np.array([3, 4], dtype=np.uint32), np.array([5, 6], dtype=np.uint32)]
    a = Utils.object_array(layers)
    assert a.shape == (2,)
    assert a[0].dtype == np.uint32
    assert np.array_equal(a[1], [5, 6])
    assert Utils.object_array([]).shape == (0,)


def test_remove_small_keeps_a_tree_that_is_one_bare_segment():
    Ce = np.column_stack([np.zeros(5), np.zeros(5), np.arange(5, dtype=float)])
    Segs = [[np.array([i], dtype=np.uint32) for i in range(5)]]
    SPar = np.array([[-1, -1]])
    SChi = [np.zeros(0, dtype=np.uint32)]
    Segs, SPar, SChi = remove_small(Ce, Segs, SPar, SChi)
    assert len(Segs) == 1 and len(Segs[0]) == 5
    assert SPar.shape == (1, 2)
    assert len(SChi) == 1 and len(SChi[0]) == 0


def test_a_stem_of_single_set_layers_survives_segments_and_remove_small():
    n = 12
    cover = {"neighbor": [[j for j in (i - 1, i + 1) if 0 <= j < n] for i in range(n)], "sets": np.arange(n)}
    segment = segments(cover, np.array([0]), np.array([], dtype=int))
    stem = segment["segments"][0]
    assert len(segment["segments"]) == 1 and stem.ndim == 1 and len(stem) == n
    assert stem[0].dtype == np.uint32
    Ce = np.column_stack([np.zeros(n), np.zeros(n), np.arange(n, dtype=float)])
    Segs, SPar, SChi = remove_small(Ce, segment["segments"], segment["ParentSegment"], segment["ChildSegment"])
    assert len(Segs) == 1 and len(Segs[0]) == n
    assert len(SChi[0]) == 0
