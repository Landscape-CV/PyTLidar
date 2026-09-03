import os

import numpy as np
import pytest

from PyTLidar import pipeline


def _cloud(n=50, seed=0):
    rng = np.random.default_rng(seed)
    P = rng.uniform(-1, 1, size=(n, 3))
    P[:, 2] += 5.0  # keep it well above zero so the z shift is visible
    return P


def test_centre_subtracts_mean_then_floors_z():
    P = _cloud()
    Q = pipeline.centre(P)
    assert np.allclose(Q[:, :2].mean(axis=0), 0.0)
    assert Q[:, 2].min() == 0.0
    # same expressions the GUI uses, applied in the same order
    expected = P - np.mean(P, axis=0)
    expected[:, 2] = expected[:, 2] - np.min(expected[:, 2], axis=0)
    assert np.array_equal(Q, expected)


def test_centre_xy_only_leaves_height_alone():
    P = _cloud()
    Q = pipeline.centre(P, z=False)
    assert np.array_equal(Q, P - np.mean(P, axis=0))
    assert Q[:, 2].min() != 0.0


def test_centre_does_not_mutate_input_and_handles_empty():
    P = _cloud()
    before = P.copy()
    pipeline.centre(P)
    pipeline.centre(P, xy=False)
    assert np.array_equal(P, before)
    empty = pipeline.centre(np.empty((0, 3)))
    assert empty.shape == (0, 3)


def test_build_inputs_custom_values_and_ball_radius_rule():
    P = _cloud(200)
    pd1, pd2min, pd2max = [0.05, 0.08], [0.03], [0.10, 0.12]
    inp = pipeline.build_inputs(P, custom=(pd1, pd2min, pd2max), names=["pine"],
                                savemat=1, savetxt=0, plot=0, disp=2)[0]
    assert inp["PatchDiam1"] == pd1 and inp["PatchDiam2Min"] == pd2min and inp["PatchDiam2Max"] == pd2max
    assert inp["BallRad1"] == [d + .01 for d in pd1]
    assert inp["BallRad2"] == [d + .01 for d in pd2max]
    assert inp["name"] == "pine"
    assert (inp["savemat"], inp["savetxt"], inp["plot"], inp["disp"]) == (1, 0, 0, 2)
    assert inp["savepdf"] == 1   # define_input's default, untouched unless asked
    assert inp["Tria"] == 0


def test_build_inputs_generated_values_and_savepdf_override():
    P = _cloud(200)
    inp = pipeline.build_inputs(P, n_patchdiam=(2, 1, 1), savepdf=0)[0]
    assert len(inp["PatchDiam1"]) == 2
    assert inp["savepdf"] == 0
    assert inp["name"] == "Tree_1"
    assert (inp["savemat"], inp["savetxt"], inp["plot"], inp["disp"]) == (0, 1, 0, 0)


def test_build_inputs_drops_empty_clouds_and_keeps_names_aligned():
    clouds = [_cloud(100, 1), np.empty((0, 3)), _cloud(100, 2)]
    inputs = pipeline.build_inputs(clouds, names=["a", "b", "c"])
    assert [inp["name"] for inp in inputs] == ["a", "c"]
    # define_input numbers trees by position in the list it receives, so the empty cloud no
    # longer consumes a number
    assert [inp["tree"] for inp in inputs] == [1, 2]
    assert pipeline.build_inputs([np.empty((0, 3))]) == []
    with pytest.raises(ValueError):
        pipeline.build_inputs(clouds, names=["a", "b"])


def test_run_qsm_raises_on_failure_and_restores_cwd(monkeypatch, tmp_path):
    start = os.getcwd()

    def failing_treeqsm(P, inputs, batch=0, processing_queue=None, results_location=None):
        os.chdir(tmp_path)          # treeqsm only restores cwd on success
        return "ERROR", "ERROR"

    monkeypatch.setattr(pipeline, "treeqsm", failing_treeqsm)
    with pytest.raises(RuntimeError):
        pipeline.run_qsm(_cloud(), {"name": "x"})
    assert os.getcwd() == start


def test_run_qsm_passes_through_results(monkeypatch):
    seen = {}

    def fake_treeqsm(P, inputs, batch=0, processing_queue=None, results_location=None):
        seen.update(batch=batch, queue=processing_queue, results=results_location)
        return ["model"], ["html"]

    monkeypatch.setattr(pipeline, "treeqsm", fake_treeqsm)
    models, htmls = pipeline.run_qsm(_cloud(), {"name": "x"}, results_dir="out")
    assert models == ["model"] and htmls == ["html"]
    assert seen == dict(batch=0, queue=None, results="out")


def test_run_batch_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        pipeline.run_batch([_cloud()], [], n_workers=1)


def test_build_inputs_rejects_a_single_empty_cloud():
    with pytest.raises(ValueError, match="empty"):
        pipeline.build_inputs(np.empty((0, 3)))
