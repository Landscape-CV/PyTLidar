"""Empty branch orders give 0 in the distance statistics, as in MATLAB; an empty trunk gives
NaN. calculate_optimal must never pick a model on a NaN metric."""
import numpy as np
import pytest

from PyTLidar.Utils.define_input import define_input
from PyTLidar.pipeline import run_qsm
from PyTLidar.treeqsm import calculate_optimal


def _model(P, name, tmp_path):
    inp = define_input(P, 1, 1, 1)[0]
    inp.update(savemat=0, savepdf=0, savetxt=0, plot=0, disp=0, name=name)
    return run_qsm(P, inp, results_dir=str(tmp_path))[0][0]


def test_empty_orders_and_nan_safe_optimum(tmp_path, seeded, small_tree, cylinder_points, capsys):
    rng = np.random.default_rng(2)
    trunk_only = cylinder_points(rng, np.array([0, 0, 0.0]), np.array([0, 0, 1.0]), 4.0, 0.12, 6000)
    trunk = _model(trunk_only, "trunk", tmp_path)
    tree = _model(small_tree, "tree", tmp_path)
    pm = trunk["pmdistance"]
    assert np.isfinite(pm["TrunkMean"])
    assert pm["Branch1Mean"] == 0.0 and pm["Branch2Mean"] == 0.0    # MATLAB sets empty orders to 0

    best, value, _ = calculate_optimal([trunk, tree], "trunk_mean_dis")
    assert np.isfinite(value)

    trunk["pmdistance"]["TrunkMean"] = float("nan")
    best, value, _ = calculate_optimal([trunk, tree], "trunk_mean_dis")
    assert best == 1 and np.isfinite(value)

    tree["pmdistance"]["TrunkMean"] = float("nan")
    with pytest.raises(ValueError, match="undefined for every model"):
        calculate_optimal([trunk, tree], "trunk_mean_dis")
