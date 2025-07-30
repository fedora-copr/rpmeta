# flake8: noqa: N803, N806

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import joblib
import numpy as np

from rpmeta.regressor import TransformedTargetRegressor
from test.helpers import power_of_10


class SimpleRegressor:
    """A simple regressor that just predicts the mean of the target."""

    def __init__(self):
        self.mean = None

    def fit(self, X, y):
        self.mean = np.mean(y)
        return self

    def predict(self, X):
        return np.full(len(X), self.mean)


def test_custom_transformed_target_regressor_basic():
    X = np.array([[1], [2], [3], [4]])
    y = np.array([10, 100, 1000, 10000])

    regressor = SimpleRegressor()
    ttr = TransformedTargetRegressor(
        regressor=regressor,
        func=np.log10,
        inverse_func=power_of_10,
    )

    ttr.fit(X, y)

    expected_prediction = 10 ** (np.mean(np.log10(y)))

    predictions = ttr.predict(X)
    assert len(predictions) == len(X)
    assert np.isclose(predictions[0], expected_prediction)


def test_joblib_serialization():
    X = np.array([[1], [2], [3], [4]])
    y = np.array([10, 100, 1000, 10000])

    regressor = SimpleRegressor()
    ttr = TransformedTargetRegressor(
        regressor=regressor,
        func=np.log10,
        inverse_func=power_of_10,
    )
    ttr.fit(X, y)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "model.joblib"
        joblib.dump(ttr, model_path)

        loaded_ttr = joblib.load(model_path)

        predictions = loaded_ttr.predict(X)
        expected_prediction = 10 ** (np.mean(np.log10(y)))
        assert len(predictions) == len(X)
        assert np.isclose(predictions[0], expected_prediction)


def test_create_regressor(base_model_subclass, example_config):
    model = base_model_subclass("test_model", config=example_config)

    regressor = model.create_regressor({"param1": 100, "param2": 200})

    assert isinstance(regressor, TransformedTargetRegressor)
    assert regressor.regressor.name == "mock_regressor"
    assert regressor.func == np.log1p
    assert regressor.inverse_func == np.expm1
    assert regressor.regressor.param1 == 100
    assert regressor.regressor.param2 == 200


def test_create_regressor_params_passed_to_regressor(base_model_subclass, example_config):
    model = base_model_subclass("test_model", config=example_config)
    test_params = {"param1": 999, "param2": 888}

    with patch.object(model, "_make_regressor") as mock_make_regressor:
        mock_regressor = MagicMock()
        mock_make_regressor.return_value = mock_regressor

        model.create_regressor(test_params)
        mock_make_regressor.assert_called_once_with(test_params)
