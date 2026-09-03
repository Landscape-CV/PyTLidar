"""treeqsm changes into results_location while it runs; it must change back on every exit."""
import multiprocessing as mp
import os

import numpy as np
import pytest
from numba import njit

from PyTLidar.Utils.define_input import define_input
from PyTLidar.treeqsm import treeqsm
from PyTLidar.pipeline import run_qsm


@njit
def _seed_numba(n):
    np.random.seed(n)


def _failing_cloud():
    """Sixty random points in a 0.4 x 0.4 x 3 m box: too sparse to model, treeqsm fails
    inside the cylinder fitting with a zero-size reduction."""
    rng = np.random.default_rng(0)
    return rng.uniform([0, 0, 0], [0.4, 0.4, 3.0], (60, 3))


def _inputs(P):
    inp = define_input(P, 1, 1, 1)[0]
    inp["PatchDiam1"] = [0.05, 0.1]
    inp["PatchDiam2Min"] = [0.04, 0.05]
    inp["PatchDiam2Max"] = [0.12, 0.14]
    inp["BallRad1"] = [d + 0.01 for d in inp["PatchDiam1"]]
    inp["BallRad2"] = [d + 0.01 for d in inp["PatchDiam2Max"]]
    inp.update(savemat=0, savepdf=0, savetxt=0, plot=0, disp=0, name="tiny")
    return inp


def _cylinder_points(rng, start, axis, length, radius, n):
    axis = axis / np.linalg.norm(axis)
    u = np.cross(axis, [1.0, 0.0, 0.0])
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    t = rng.uniform(0, length, n)
    a = rng.uniform(0, 2 * np.pi, n)
    r = radius + rng.normal(0, 0.002, n)
    return start + np.outer(t, axis) + np.outer(r * np.cos(a), u) + np.outer(r * np.sin(a), v)


def _small_tree():
    """A 4 m trunk with one branch, dense enough to model in a few seconds."""
    rng = np.random.default_rng(1)
    trunk = _cylinder_points(rng, np.array([0, 0, 0.0]), np.array([0, 0, 1.0]), 4.0, 0.12, 6000)
    branch = _cylinder_points(rng, np.array([0, 0, 2.5]), np.array([1.0, 0.3, 0.8]), 2.0, 0.05, 2500)
    return np.vstack([trunk, branch])


@pytest.fixture
def seeded():
    np.random.seed(0)
    _seed_numba(0)


def test_failed_run_returns_error_and_restores_the_working_directory(tmp_path, seeded, capsys):
    P = _failing_cloud()
    home = os.getcwd()
    assert treeqsm(P, _inputs(P), results_location=str(tmp_path)) == ("ERROR", "ERROR")
    assert os.getcwd() == home
    queue = mp.Queue()
    assert treeqsm(P, _inputs(P), 3, queue, results_location=str(tmp_path)) == ("ERROR", "ERROR")
    assert queue.get(timeout=5) == [3, "ERROR", "ERROR"]
    assert os.getcwd() == home
    with pytest.raises(RuntimeError, match="treeqsm failed"):
        run_qsm(P, _inputs(P), results_dir=str(tmp_path))
    assert os.getcwd() == home


def test_successful_run_writes_the_model_files(tmp_path, seeded, capsys):
    P = _small_tree()
    inp = define_input(P, 1, 1, 1)[0]
    inp.update(savemat=0, savepdf=0, savetxt=1, plot=0, disp=0, name="synth")
    home = os.getcwd()
    models, htmls = run_qsm(P, inp, results_dir=str(tmp_path))
    assert os.getcwd() == home
    assert len(models[0]["cylinder"]["radius"]) > 0
    assert [p.name for p in (tmp_path / "results").glob("cylinder_synth_t1_m1*.txt")] == ["cylinder_synth_t1_m1.txt"]
