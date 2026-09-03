"""growth_volume_correction reads parent as 0-based with -1 for the base cylinder and
extension as 0-based with 0 for a tip, which is how cylinders.py builds them."""
import numpy as np
import pytest
import matplotlib

matplotlib.use("Agg")

from PyTLidar.Utils import Utils
from PyTLidar.Utils.define_input import define_input
from PyTLidar.pipeline import run_qsm


def test_growth_volumes_follow_the_cylinder_tree(monkeypatch):
    # base 0 continues into 1 then tip 2; tip 3 is a side branch off the base
    cyl = {
        "radius": np.array([0.10, 0.08, 0.05, 0.03]),
        "length": np.array([1.0, 1.0, 1.0, 0.5]),
        "parent": np.array([-1, 0, 1, 0]),
        "extension": np.array([1, 2, 0, 0]),
    }
    own = np.pi * cyl["radius"] ** 2 * cyl["length"]
    seen = {}

    def fake_fit(f, gv, rad, p0, maxfev):
        seen["gv"] = gv.copy()
        return np.array([1.0, 0.0, 0.0]), None   # predicted radius 1 everywhere

    monkeypatch.setattr(Utils, "curve_fit", fake_fit)
    Utils.growth_volume_correction(cyl, {"GrowthVolFac": 1e9})
    gv = seen["gv"]
    assert gv[2] == pytest.approx(own[2]) and gv[3] == pytest.approx(own[3])
    assert gv[1] == pytest.approx(own[1] + own[2])
    assert gv[0] == pytest.approx(own.sum())          # the base counts its side branch
    assert gv[0] >= gv[1] >= gv[2]                     # non-increasing along the chain


def test_correction_on_a_small_tree_keeps_radii_finite(tmp_path, seeded, small_tree, capsys):
    inp = define_input(small_tree, 1, 1, 1)[0]
    inp.update(savemat=0, savepdf=0, savetxt=0, plot=0, disp=0, name="synth",
               GrowthVolCor=1, GrowthVolFac=1.5)
    models, _ = run_qsm(small_tree, inp, results_dir=str(tmp_path))
    r = np.asarray(models[0]["cylinder"]["radius"], float)
    r0 = np.asarray(models[0]["cylinder"]["UnmodRadius"], float)
    assert np.all(np.isfinite(r)) and np.all(r > 0)
    assert len(r) == len(r0)
    assert np.all(r <= 1.5 * r0.max())
