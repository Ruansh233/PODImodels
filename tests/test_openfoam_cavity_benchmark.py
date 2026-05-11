import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "openfoam_cavity"
    / "benchmark_cavity_models.py"
)


def _load_benchmark_module():
    spec = importlib.util.spec_from_file_location("cavity_benchmark", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_update_moving_wall_velocity_localized():
    mod = _load_benchmark_module()
    original = """boundaryField
{
    movingWall
    {
        type            fixedValue;
        value           uniform (1 0 0);
    }

    fixedWalls
    {
        type            noSlip;
    }
}
"""
    updated = mod.update_moving_wall_velocity(original, 2.5)
    assert "value           uniform (2.5 0 0);" in updated
    assert "type            noSlip;" in updated
    assert updated.count("movingWall") == 1


def test_flatten_velocity_snapshot_shape():
    mod = _load_benchmark_module()
    arr = np.array([[1.0, 0.0, 0.0], [0.5, 0.1, 0.2]])
    flat = mod.flatten_velocity_snapshot(arr)
    assert flat.shape == (6,)
    assert np.allclose(flat, np.array([1.0, 0.0, 0.0, 0.5, 0.1, 0.2]))


def test_update_kinematic_viscosity_localized():
    mod = _load_benchmark_module()
    original = """transportModel  Newtonian;
nu              [0 2 -1 0 0 0 0] 0.01;
"""
    updated = mod.update_kinematic_viscosity(original, 0.005)
    assert "[0 2 -1 0 0 0 0] 0.005;" in updated
    assert "transportModel  Newtonian;" in updated


def test_deterministic_split_indices_is_reproducible():
    mod = _load_benchmark_module()
    train_a, test_a = mod.deterministic_split_indices(8, 0.75, seed=42)
    train_b, test_b = mod.deterministic_split_indices(8, 0.75, seed=42)
    assert np.array_equal(train_a, train_b)
    assert np.array_equal(test_a, test_b)
    assert len(train_a) == 6
    assert len(test_a) == 2


def test_load_foam_to_python_missing_dependency(monkeypatch):
    mod = _load_benchmark_module()
    monkeypatch.setitem(sys.modules, "foamToPython", None)

    import importlib as _importlib

    def _fake_import(name):
        if name == "foamToPython":
            raise ModuleNotFoundError("No module named 'foamToPython'")
        return _importlib.import_module(name)

    monkeypatch.setattr(mod.importlib, "import_module", _fake_import)
    with pytest.raises(RuntimeError, match="Missing optional dependency 'foamToPython'"):
        mod.load_foam_to_python()


def test_benchmark_models_rows_and_metrics():
    mod = _load_benchmark_module()

    class _DummyModel:
        def fit(self, x, y):
            self._mean = y.mean(axis=0, keepdims=True)

        def predict(self, x):
            return np.repeat(self._mean, x.shape[0], axis=0)

    builders = {
        "A": lambda: _DummyModel(),
        "B": lambda: _DummyModel(),
    }
    x_train = np.array([[0.0], [1.0], [2.0]], dtype=float)
    y_train = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]], dtype=float)
    x_test = np.array([[3.0], [4.0]], dtype=float)
    y_test = np.array([[3.0, 4.0], [4.0, 5.0]], dtype=float)

    rows = mod.benchmark_models(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        rank=5,
        ann_epochs=1,
        builders=builders,
    )

    assert len(rows) == 2
    for row in rows:
        assert set(mod.RESULT_COLUMNS).issubset(row.keys())
        assert isinstance(row["relative_frobenius_error"], float)
        assert isinstance(row["fit_seconds"], float)
        assert isinstance(row["predict_seconds"], float)
        assert row["train_cases"] == 3
        assert row["test_cases"] == 2


def test_model_builders_contains_all_models():
    mod = _load_benchmark_module()
    builders = mod.model_builders(rank=5, ann_epochs=1)
    expected = {
        "fieldsLinear",
        "fieldsRidge",
        "fieldsGPR",
        "fieldsRidgeGPR",
        "fieldsRBF",
        "fieldsRidgeRBF",
        "PODLinear",
        "PODRidge",
        "PODRBF",
        "PODRidgeRBF",
        "PODGPR",
        "PODRidgeGPR",
        "PODANN",
    }
    assert set(builders.keys()) == expected


def test_select_time_dirs_respects_mode_and_cap(tmp_path):
    mod = _load_benchmark_module()
    for name in ["0", "0.1", "0.2", "0.3", "constant", "system"]:
        (tmp_path / name).mkdir()

    final_only = mod.select_time_dirs(
        case_dir=tmp_path,
        snapshot_mode="final",
        max_snapshots_per_case=2,
    )
    assert len(final_only) == 1
    assert final_only[0][0] == pytest.approx(0.3)

    capped = mod.select_time_dirs(
        case_dir=tmp_path,
        snapshot_mode="all",
        max_snapshots_per_case=2,
    )
    assert len(capped) == 2
    assert capped[0][0] == pytest.approx(0.1)
    assert capped[-1][0] == pytest.approx(0.3)
