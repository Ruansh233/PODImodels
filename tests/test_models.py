import numpy as np
import torch.nn as nn

from PODImodels import PODANN, PODLinear, PODRidge, fieldsLinear


def _toy_data():
    x = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.5, 0.25],
            [0.25, 0.75],
        ],
        dtype=float,
    )
    y = np.array(
        [
            [0.0, 1.0, 2.0, 3.0, 4.0],
            [0.5, 1.5, 2.5, 3.5, 4.5],
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [1.5, 2.5, 3.5, 4.5, 5.5],
            [0.75, 1.75, 2.75, 3.75, 4.75],
            [0.6, 1.6, 2.6, 3.6, 4.6],
        ],
        dtype=float,
    )
    return x, y


def test_fields_linear_fit_predict_shape():
    x, y = _toy_data()
    model = fieldsLinear()
    model.fit(x, y)
    prediction = model.predict(x[:2])
    assert prediction.shape == (2, y.shape[1])


def test_pod_linear_fit_predict_no_pod_crash():
    x, y = _toy_data()
    model = PODLinear(rank=2)
    model.fit(x, y)
    prediction = model.predict(x[:2])
    assert prediction.shape == (2, y.shape[1])


def test_pod_ridge_fit_predict_no_pod_crash():
    x, y = _toy_data()
    model = PODRidge(rank=2)
    model.fit(x, y)
    prediction = model.predict(x[:2])
    assert prediction.shape == (2, y.shape[1])


def test_pod_ann_supports_gelu_activation():
    model = PODANN(
        activation_function_name="gelu",
        num_epochs=1,
    )
    activation = model._get_activation_function()
    assert isinstance(activation, nn.GELU)
