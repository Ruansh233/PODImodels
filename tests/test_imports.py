import importlib

import PODImodels


def test_top_level_all_exports_are_valid():
    for symbol in PODImodels.__all__:
        assert hasattr(PODImodels, symbol), f"Missing symbol in __all__: {symbol}"


def test_top_level_exports_include_pod_linear_and_pod_ridge():
    assert hasattr(PODImodels, "PODLinear")
    assert hasattr(PODImodels, "PODRidge")


def test_legacy_module_import_paths_still_work():
    legacy_models = importlib.import_module("PODImodels.PODImodels")
    legacy_data = importlib.import_module("PODImodels.PODdata")
    legacy_base = importlib.import_module("PODImodels.podImodelabstract")

    assert hasattr(legacy_models, "PODLinear")
    assert hasattr(legacy_models, "fieldsGPR")
    assert hasattr(legacy_data, "PODDataSet")
    assert hasattr(legacy_base, "PODImodelAbstract")
