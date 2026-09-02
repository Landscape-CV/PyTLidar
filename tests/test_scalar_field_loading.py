"""
Scalar-field selection in load_point_cloud, on a small synthetic LAS file
shaped like a RIEGL export: zero intensity, return strength in a Reflectance
extra dimension.
"""

import numpy as np
import pytest
import laspy

from PyTLidar.Utils.Utils import load_point_cloud, list_las_scalar_fields


def _write_fixture(path, n=200):
    """Small LAS (point format 6) with all-zero intensity and a
    negative-valued Reflectance extra dimension."""
    header = laspy.LasHeader(version="1.4", point_format=6)
    header.add_extra_dim(laspy.ExtraBytesParams(name="Reflectance", type=np.float32))
    las = laspy.LasData(header)

    rng = np.random.Generator(np.random.Philox(0))
    las.x = rng.uniform(0, 10, n)
    las.y = rng.uniform(0, 10, n)
    las.z = rng.uniform(0, 5, n)
    las.intensity = np.zeros(n, dtype=np.uint16)
    las.Reflectance = rng.uniform(-20.0, 0.0, n).astype(np.float32)
    las.write(str(path))


@pytest.fixture
def las_file(tmp_path):
    p = tmp_path / "fixture.las"
    _write_fixture(p)
    return str(p)


def test_list_scalar_fields_includes_extra_dim(las_file):
    fields = list_las_scalar_fields(las_file)
    assert "intensity" in fields
    assert "Reflectance" in fields


def test_default_intensity_is_all_zero(las_file):
    """Default behaviour (scalar_field='intensity') is unchanged."""
    _, pd = load_point_cloud(las_file, 0, True)
    assert pd.shape[1] == 4
    assert np.all(pd[:, 3] == 0)


def test_reflectance_without_normalize_is_negative(las_file):
    """Raw reflectance carries through as negative dB values."""
    _, pd = load_point_cloud(las_file, intensity_threshold=-100, full_data=True,
                             scalar_field="Reflectance")
    assert pd[:, 3].min() < 0
    assert pd[:, 3].max() <= 0.0


def test_reflectance_normalized_to_intensity_range(las_file):
    """Normalized reflectance lands in 0-65535 and survives threshold 0."""
    _, pd = load_point_cloud(las_file, intensity_threshold=0, full_data=True,
                             scalar_field="Reflectance", normalize_scalar=True)
    col = pd[:, 3]
    assert col.min() >= 0.0
    assert col.max() == pytest.approx(65535.0)
    assert len(pd) == 200


def test_case_insensitive_field_match(las_file):
    """'reflectance' resolves the 'Reflectance' extra dim."""
    _, pd = load_point_cloud(las_file, intensity_threshold=-100, full_data=True,
                             scalar_field="reflectance")
    assert pd[:, 3].min() < 0


def test_normalize_ignores_nan(tmp_path):
    p = tmp_path / "nan.las"
    _write_fixture(p, n=50)
    las = laspy.read(str(p))
    refl = np.asarray(las.Reflectance, dtype=np.float32)
    refl[7] = np.nan
    las.Reflectance = refl
    las.write(str(p))
    cloud, pd = load_point_cloud(str(p), intensity_threshold=0, full_data=True,
                                 scalar_field="Reflectance", normalize_scalar=True)
    assert len(cloud) == 49
    assert pd[:, 3].max() == pytest.approx(65535.0)


def test_missing_field_raises_clear_error(las_file):
    with pytest.raises(ValueError) as excinfo:
        load_point_cloud(las_file, 0, True, scalar_field="Nonexistent")
    msg = str(excinfo.value)
    assert "Nonexistent" in msg
    assert "Available" in msg
