"""A cloud far from the origin must give the same model as the same cloud near it. The cylinder
arrays are single precision like MATLAB's, so treeqsm shifts the cloud internally and undoes the
shift on the output."""
import numpy as np
from numba import njit

from PyTLidar.pipeline import build_inputs, run_qsm

OFFSET = np.array([740000.0, 3462400.0, 100.0])


@njit
def _seed_numba(n):
    np.random.seed(n)


def _model(P):
    np.random.seed(0)
    _seed_numba(0)
    inputs = build_inputs(P, n_patchdiam=(1, 1, 1), savetxt=0, plot=0, disp=0, savepdf=0)[0]
    models, _ = run_qsm(P, inputs)
    return models[0]


def test_georeferenced_cloud_gives_the_same_model(small_tree):
    local = _model(small_tree)
    shifted = _model(small_tree + OFFSET)
    a, b = local["cylinder"], shifted["cylinder"]

    assert a["radius"].shape == b["radius"].shape
    assert np.array_equal(a["parent"], b["parent"])
    assert np.array_equal(a["branch"], b["branch"])
    assert np.allclose(a["radius"], b["radius"], rtol=1e-5)
    assert np.allclose(a["length"], b["length"], rtol=1e-5)
    assert np.allclose(a["axis"], b["axis"], atol=1e-5)
    # Output positions are back in the input frame, single precision at UTM scale
    assert np.allclose(b["start"] - OFFSET, a["start"], atol=0.3)
    assert np.allclose(shifted["points"][:5] - OFFSET, local["points"][:5], atol=1e-6)
    assert np.allclose(shifted["treedata"]["location"] - OFFSET, local["treedata"]["location"], atol=0.3)
    assert float(shifted["treedata"]["TotalVolume"]) == float(local["treedata"]["TotalVolume"])
    assert float(shifted["treedata"]["CrownDiamAve"]) == float(local["treedata"]["CrownDiamAve"])


def test_local_cloud_is_not_shifted(small_tree):
    model = _model(small_tree)
    assert np.array_equal(model["xy_offset"], np.zeros(2))
