import numpy as np
import pandas as pd
import pytest

from rpmeta.config import Config
from rpmeta.constants import CATEGORICAL_FEATURES, NUMERICAL_FEATURES
from rpmeta.model import LightGBMModel, XGBoostModel, get_model_by_name


def _create_test_data(
    n_samples: int = 100,
    n_categories: int = 5,
) -> tuple[pd.DataFrame, pd.Series]:
    np.random.seed(42)
    data = {}

    for col in CATEGORICAL_FEATURES:
        data[col] = pd.Categorical(
            np.random.choice([f"cat_{i}" for i in range(n_categories)], n_samples),
        )

    for col in NUMERICAL_FEATURES:
        data[col] = np.random.rand(n_samples) * 100

    X = pd.DataFrame(data)
    y = pd.Series(np.random.rand(n_samples) * 100 + 1, name="build_duration")
    return X, y


class TestModelPersistence:
    @pytest.fixture
    def test_data(self):
        return _create_test_data(n_samples=200, n_categories=10)

    @pytest.fixture
    def config(self, tmp_path):
        return Config(result_dir=tmp_path)

    def test_lightgbm_save_load_predictions_match(self, test_data, config, tmp_path):
        X, y = test_data
        model = LightGBMModel(config)

        regressor = model.create_regressor({"n_estimators": 10, "learning_rate": 0.1})
        regressor.fit(X, y)

        predictions_before = regressor.predict(X)

        save_path = tmp_path / "lightgbm_model"
        save_path.mkdir()
        model.save_regressor(regressor, save_path)

        loaded_regressor = model.load_regressor(save_path)

        predictions_after = loaded_regressor.predict(X)

        np.testing.assert_allclose(
            predictions_before,
            predictions_after,
            rtol=1e-5,
            err_msg="LightGBM predictions differ after save/load!",
        )

    def test_xgboost_save_load_predictions_match(self, test_data, config, tmp_path):
        X, y = test_data
        model = XGBoostModel(config)

        regressor = model.create_regressor({"n_estimators": 10, "learning_rate": 0.1})
        regressor.fit(X, y)

        predictions_before = regressor.predict(X)

        save_path = tmp_path / "xgboost_model"
        save_path.mkdir()
        model.save_regressor(regressor, save_path)

        loaded_regressor = model.load_regressor(save_path)

        predictions_after = loaded_regressor.predict(X)

        np.testing.assert_allclose(
            predictions_before,
            predictions_after,
            rtol=1e-5,
            err_msg="XGBoost predictions differ after save/load!",
        )

    @pytest.mark.parametrize("model_name", ["lightgbm", "xgboost"])
    def test_model_by_name_save_load_predictions_match(
        self,
        model_name,
        test_data,
        config,
        tmp_path,
    ):
        X, y = test_data
        model = get_model_by_name(model_name, config)

        regressor = model.create_regressor({"n_estimators": 10, "learning_rate": 0.1})
        regressor.fit(X, y)

        predictions_before = regressor.predict(X)

        save_path = tmp_path / f"{model_name}_model"
        save_path.mkdir()
        model.save_regressor(regressor, save_path)

        fresh_model = get_model_by_name(model_name, config)
        loaded_regressor = fresh_model.load_regressor(save_path)

        predictions_after = loaded_regressor.predict(X)

        np.testing.assert_allclose(
            predictions_before,
            predictions_after,
            rtol=1e-5,
            err_msg=f"{model_name} predictions differ after save/load!",
        )

    @pytest.mark.parametrize("model_name", ["lightgbm", "xgboost"])
    def test_multiple_save_load_cycles(self, model_name, test_data, config, tmp_path):
        X, y = test_data
        model = get_model_by_name(model_name, config)

        regressor = model.create_regressor({"n_estimators": 10, "learning_rate": 0.1})
        regressor.fit(X, y)

        original_predictions = regressor.predict(X)

        current_regressor = regressor
        for i in range(3):
            save_path = tmp_path / f"{model_name}_cycle_{i}"
            save_path.mkdir()
            model.save_regressor(current_regressor, save_path)
            current_regressor = model.load_regressor(save_path)

        final_predictions = current_regressor.predict(X)

        np.testing.assert_allclose(
            original_predictions,
            final_predictions,
            rtol=1e-5,
            err_msg=f"{model_name} predictions degraded after multiple save/load cycles!",
        )

    @pytest.mark.parametrize("model_name", ["lightgbm", "xgboost"])
    def test_transformed_target_regressor_preserved(self, model_name, test_data, config, tmp_path):
        X, y = test_data
        model = get_model_by_name(model_name, config)

        regressor = model.create_regressor({"n_estimators": 10, "learning_rate": 0.1})
        regressor.fit(X, y)

        assert regressor.func is not None, "func (log1p) should be set"
        assert regressor.inverse_func is not None, "inverse_func (expm1) should be set"

        save_path = tmp_path / f"{model_name}_transformer"
        save_path.mkdir()
        model.save_regressor(regressor, save_path)
        loaded_regressor = model.load_regressor(save_path)

        assert loaded_regressor.func is not None, "func (log1p) should be preserved after load"
        assert loaded_regressor.inverse_func is not None, (
            "inverse_func (expm1) should be preserved after load"
        )

        test_value = np.array([100.0])
        original_transformed = regressor.func(test_value)
        loaded_transformed = loaded_regressor.func(test_value)
        np.testing.assert_array_equal(original_transformed, loaded_transformed)

    @pytest.mark.parametrize("model_name", ["lightgbm", "xgboost"])
    def test_prediction_shape_preserved(self, model_name, test_data, config, tmp_path):
        X, y = test_data
        model = get_model_by_name(model_name, config)

        regressor = model.create_regressor({"n_estimators": 10, "learning_rate": 0.1})
        regressor.fit(X, y)

        predictions_before = regressor.predict(X)

        save_path = tmp_path / f"{model_name}_shape"
        save_path.mkdir()
        model.save_regressor(regressor, save_path)
        loaded_regressor = model.load_regressor(save_path)

        predictions_after = loaded_regressor.predict(X)

        assert predictions_before.shape == predictions_after.shape, (
            f"Prediction shape changed: {predictions_before.shape} -> {predictions_after.shape}"
        )

    @pytest.mark.parametrize("model_name", ["lightgbm", "xgboost"])
    def test_single_sample_prediction(self, model_name, test_data, config, tmp_path):
        X, y = test_data
        model = get_model_by_name(model_name, config)

        regressor = model.create_regressor({"n_estimators": 10, "learning_rate": 0.1})
        regressor.fit(X, y)

        single_sample = X.iloc[[0]]
        prediction_before = regressor.predict(single_sample)

        save_path = tmp_path / f"{model_name}_single"
        save_path.mkdir()
        model.save_regressor(regressor, save_path)
        loaded_regressor = model.load_regressor(save_path)

        prediction_after = loaded_regressor.predict(single_sample)

        np.testing.assert_allclose(
            prediction_before,
            prediction_after,
            rtol=1e-5,
            err_msg=f"{model_name} single sample prediction differs after save/load!",
        )
