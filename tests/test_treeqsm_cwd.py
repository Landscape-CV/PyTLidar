"""treeqsm changes into results_location while it runs; it must change back on every exit."""
import multiprocessing as mp
import os

import numpy as np
import pytest

from PyTLidar.Utils.define_input import define_input
from PyTLidar.treeqsm import treeqsm
from PyTLidar.pipeline import run_qsm


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


def test_successful_run_writes_the_model_files(tmp_path, seeded, small_tree, capsys):
    inp = define_input(small_tree, 1, 1, 1)[0]
    inp.update(savemat=0, savepdf=0, savetxt=1, plot=0, disp=0, name="synth")
    home = os.getcwd()
    models, htmls = run_qsm(small_tree, inp, results_dir=str(tmp_path))
    assert os.getcwd() == home
    assert len(models[0]["cylinder"]["radius"]) > 0
    assert [p.name for p in (tmp_path / "results").glob("cylinder_synth_t1_m1*.txt")] == ["cylinder_synth_t1_m1.txt"]
