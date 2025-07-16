# flake8: noqa: N803, N806

import tempfile
from pathlib import Path

import joblib
import numpy as np

from rpmeta.train.transformers import TransformedTargetRegressor
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


def test_custom_transformed_target_regressor_with_pipeline():
    class SimplePipeline:
        def __init__(self, steps):
            self.steps = steps
            self.final_step = steps[-1][1]

        def fit(self, X, y):
            self.final_step.fit(X, y)
            return self

        def predict(self, X):
            return self.final_step.predict(X)

    X = np.array([[1], [2], [3], [4]])
    y = np.array([10, 100, 1000, 10000])

    regressor = SimpleRegressor()
    ttr = TransformedTargetRegressor(
        regressor=regressor,
        func=np.log10,
        inverse_func=power_of_10,
    )

    pipeline = SimplePipeline(
        [
            ("regressor", ttr),
        ],
    )

    pipeline.fit(X, y)
    predictions = pipeline.predict(X)

    expected_prediction = 10 ** (np.mean(np.log10(y)))
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
