"""run_batch reads every worker's result off its queue. A worker that exits without putting one
must be reported as a failed tree, not leave the caller waiting forever."""
import os
import queue
import threading

import numpy as np
import pytest

from PyTLidar import pipeline


def _fake_treeqsm(P, inputs, batch=0, processing_queue=None, results_location=None):
    """Stands in for treeqsm in the worker processes, which import it from this module by name.
    The tree's name says what to do."""
    if inputs["name"] == "dies":
        os._exit(3)
    if inputs["name"] == "fails":
        processing_queue.put([batch, "ERROR", "ERROR"])
        return "ERROR", "ERROR"
    models = [{"name": inputs["name"], "points": P.shape[0], "results": results_location}]
    processing_queue.put([batch, models, ["html"]])
    return models, ["html"]


def _batch(names, n_workers, results_dir=None, timeout=60):
    """run_batch on small random clouds with the fake treeqsm, in a thread so that a hang fails
    the test instead of stalling the suite. Returns the results, messages and on_result calls."""
    rng = np.random.default_rng(0)
    clouds = [rng.uniform(0, 1, (10 + k, 3)) for k in range(len(names))]
    inputs = [{"name": name} for name in names]
    messages, finished, out = [], [], {}

    def run():
        out["results"] = pipeline.run_batch(
            clouds, inputs, n_workers, results_dir, on_message=messages.append,
            on_result=lambda i, models, cyl_htmls: finished.append(i))

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(timeout)
    assert not thread.is_alive(), f"run_batch was still waiting after {timeout} s"
    return out["results"], messages, finished


@pytest.fixture
def fake_treeqsm(monkeypatch):
    monkeypatch.setattr(pipeline, "treeqsm", _fake_treeqsm)


def test_run_batch_returns_each_result_in_order(fake_treeqsm, tmp_path):
    results, messages, finished = _batch(["a", "b", "c"], 2, results_dir=str(tmp_path))
    assert [models[0]["name"] for models, _ in results] == ["a", "b", "c"]
    assert [models[0]["points"] for models, _ in results] == [10, 11, 12]
    assert all(models[0]["results"] == str(tmp_path) for models, _ in results)
    assert all(cyl_htmls == ["html"] for _, cyl_htmls in results)
    assert finished == [0, 1, 2]
    assert not any("error" in m for m in messages)


def test_run_batch_carries_on_past_a_worker_that_exits_without_a_result(fake_treeqsm):
    results, messages, finished = _batch(["a", "dies", "fails", "b"], 2)
    assert results[1] is None and results[2] is None
    assert results[0][0][0]["name"] == "a" and results[3][0][0]["name"] == "b"
    assert finished == [0, 3]
    died = [m for m in messages if "dies" in m and "error" in m]
    assert len(died) == 1 and "code 3" in died[0]
    assert any("fails" in m and "error" in m for m in messages)


def test_run_batch_rejects_a_missing_results_dir(fake_treeqsm, tmp_path):
    with pytest.raises(ValueError, match="results_dir"):
        pipeline.run_batch([np.zeros((10, 3))], [{"name": "a"}], 1, str(tmp_path / "missing"),
                           on_message=None)


class _Exited:
    def is_alive(self):
        return False


def test_wait_for_worker_reads_a_message_put_just_before_the_worker_exited():
    q = queue.Queue()
    q.put([0, ["model"], ["html"]])
    assert pipeline._wait_for_worker(q, _Exited(), timeout=0.1) == [0, ["model"], ["html"]]
    assert pipeline._wait_for_worker(q, _Exited(), timeout=0.1) is None
