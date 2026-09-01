"""
PLY loading in load_point_cloud, on small synthetic files: a plain xyz cloud
and one with a Reflectance vertex field, plus a CloudCompare style
scalar_Intensity export.
"""

import numpy as np
import pytest
from plyfile import PlyData, PlyElement

from PyTLidar.Utils.Utils import load_point_cloud, list_scalar_fields


def _write_ply(path, extra=None, n=100):
    rng = np.random.Generator(np.random.Philox(1))
    fields = [("x", "f8"), ("y", "f8"), ("z", "f8")] + [(name, "f4") for name in (extra or [])]
    data = np.empty(n, dtype=fields)
    data["x"] = rng.uniform(0, 10, n)
    data["y"] = rng.uniform(0, 10, n)
    data["z"] = rng.uniform(0, 5, n)
    if extra:
        for name in extra:
            data[name] = rng.uniform(-20.0, 0.0, n).astype(np.float32)
    PlyData([PlyElement.describe(data, "vertex")]).write(str(path))


def test_plain_xyz_ply_loads_with_zero_intensity(tmp_path):
    p = tmp_path / "plain.ply"
    _write_ply(p)
    cloud, pd = load_point_cloud(str(p), 0, True)
    assert cloud.shape == (100, 3)
    assert np.all(pd[:, 3] == 0)


def test_list_scalar_fields_ply(tmp_path):
    p = tmp_path / "refl.ply"
    _write_ply(p, ["Reflectance"])
    fields = list_scalar_fields(str(p))
    assert "x" in fields
    assert "Reflectance" in fields


def test_ply_reflectance_case_insensitive(tmp_path):
    p = tmp_path / "refl.ply"
    _write_ply(p, ["Reflectance"])
    _, pd = load_point_cloud(str(p), intensity_threshold=-100, full_data=True,
                             scalar_field="reflectance")
    assert pd[:, 3].min() < 0
    assert len(pd) == 100


def test_ply_normalized_reflectance(tmp_path):
    p = tmp_path / "refl.ply"
    _write_ply(p, ["Reflectance"])
    _, pd = load_point_cloud(str(p), intensity_threshold=0, full_data=True,
                             scalar_field="Reflectance", normalize_scalar=True)
    assert pd[:, 3].min() >= 0.0
    assert pd[:, 3].max() == pytest.approx(65535.0)
    assert len(pd) == 100


def test_ply_scalar_prefix_matches_cloudcompare_export(tmp_path):
    """CloudCompare writes fields as scalar_<name>; the default 'intensity'
    finds scalar_Intensity."""
    p = tmp_path / "cc.ply"
    _write_ply(p, ["scalar_Intensity"])
    _, pd = load_point_cloud(str(p), intensity_threshold=-100, full_data=True)
    assert pd[:, 3].min() < 0


def test_ply_missing_field_raises(tmp_path):
    p = tmp_path / "plain.ply"
    _write_ply(p)
    with pytest.raises(ValueError) as excinfo:
        load_point_cloud(str(p), 0, True, scalar_field="Reflectance")
    assert "Reflectance" in str(excinfo.value)


def test_ply_without_intensity_rejects_a_threshold(tmp_path):
    p = tmp_path / "plain.ply"
    _write_ply(p)
    with pytest.raises(ValueError) as excinfo:
        load_point_cloud(str(p), 100)
    assert "no intensity field" in str(excinfo.value)


def test_ply_without_vertex_element_raises(tmp_path):
    p = tmp_path / "faces.ply"
    data = np.zeros(3, dtype=[("a", "f4")])
    PlyData([PlyElement.describe(data, "face")]).write(str(p))
    with pytest.raises(ValueError) as excinfo:
        load_point_cloud(str(p))
    assert "vertex" in str(excinfo.value)
    assert list_scalar_fields(str(p)) == []


def test_ply_under_a_txt_named_folder(tmp_path):
    d = tmp_path / "exports.txt"
    d.mkdir()
    p = d / "cloud.ply"
    _write_ply(p)
    assert load_point_cloud(str(p)).shape == (100, 3)
