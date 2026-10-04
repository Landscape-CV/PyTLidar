import json

import numpy as np

from pytlidar_cc.worker import run_info


def fake_models():
    # rundata inputs are the caller's whole grid, shared by every model.
    grid = {
        "PatchDiam1": np.array([0.05, 0.08, 0.10]),
        "PatchDiam2Min": np.array([0.01, 0.02, 0.03]),
        "PatchDiam2Max": np.array([0.04, 0.06, 0.07]),
        "Tria": 0,
    }
    models = []
    for k in range(3):
        models.append({
            "rundata": {"inputs": grid},
            "treedata": {"TotalVolume": np.array([100.0 + k])},
            "PatchDiam1": grid["PatchDiam1"][k],
            "PatchDiam2Min": grid["PatchDiam2Min"][k],
            "PatchDiam2Max": grid["PatchDiam2Max"][k],
        })
    return models


def test_run_info_records_the_chosen_model():
    models = fake_models()
    for best in (1, 2):
        info = run_info(models[best], "tree", 3, 1000, "all_mean_dis", 12.34)
        assert info["PatchDiam1"] == models[best]["PatchDiam1"]
        assert info["PatchDiam2Min"] == models[best]["PatchDiam2Min"]
        assert info["PatchDiam2Max"] == models[best]["PatchDiam2Max"]
        assert info["TotalVolume_L"] == 100.0 + best
        json.dumps(info)


def test_run_info_fields():
    info = run_info(fake_models()[0], "tree", "3", 1000, "all_mean_dis", 12.34)
    assert info["source"] == "tree"
    assert info["run"] == 3
    assert info["points"] == 1000
    assert info["Tria"] == 0
    assert info["metric"] == "all_mean_dis"
    assert info["elapsed_s"] == 12.3
    assert info["PatchDiam1"] == 0.05
