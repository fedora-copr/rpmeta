import numpy as np
import pandas as pd
import pytest

from rpmeta.config import Config
from rpmeta.constants import CATEGORICAL_FEATURES, NUMERICAL_FEATURES
from rpmeta.model import LightGBMModel, Model, XGBoostModel, get_model_by_name


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


def _category_maps_from_df(df: pd.DataFrame) -> dict[str, list[str]]:
    return {col: list(df[col].cat.categories) for col in CATEGORICAL_FEATURES}


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
        category_maps = _category_maps_from_df(X)

        regressor = model.make_regressor({"n_estimators": 10, "learning_rate": 0.1})
        regressor.fit(X, Model.TARGET_FUNC(y))
        predictions_before = Model.INVERSE_FUNC(regressor.predict(X))

        save_path = tmp_path / "lightgbm_model"
        save_path.mkdir()
        model.save_regressor(regressor, save_path)

        model.load_regressor(save_path)
        model.prepare_for_prediction(category_maps)
        predictions_after = model.predict(X)

        np.testing.assert_allclose(
            predictions_before,
            predictions_after,
            rtol=1e-5,
            err_msg="LightGBM predictions differ after save/load!",
        )

    def test_xgboost_save_load_predictions_match(self, test_data, config, tmp_path):
        X, y = test_data
        model = XGBoostModel(config)
        category_maps = _category_maps_from_df(X)

        regressor = model.make_regressor({"n_estimators": 10, "learning_rate": 0.1})
        regressor.fit(X, Model.TARGET_FUNC(y))
        predictions_before = Model.INVERSE_FUNC(regressor.predict(X))

        save_path = tmp_path / "xgboost_model"
        save_path.mkdir()
        model.save_regressor(regressor, save_path)

        model.load_regressor(save_path)
        model.prepare_for_prediction(category_maps)
        predictions_after = model.predict(X)

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
        category_maps = _category_maps_from_df(X)

        regressor = model.make_regressor({"n_estimators": 10, "learning_rate": 0.1})
        regressor.fit(X, Model.TARGET_FUNC(y))
        predictions_before = Model.INVERSE_FUNC(regressor.predict(X))

        save_path = tmp_path / f"{model_name}_model"
        save_path.mkdir()
        model.save_regressor(regressor, save_path)

        fresh_model = get_model_by_name(model_name, config)
        fresh_model.load_regressor(save_path)
        fresh_model.prepare_for_prediction(category_maps)
        predictions_after = fresh_model.predict(X)

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
        category_maps = _category_maps_from_df(X)

        regressor = model.make_regressor({"n_estimators": 10, "learning_rate": 0.1})
        regressor.fit(X, Model.TARGET_FUNC(y))
        original_predictions = Model.INVERSE_FUNC(regressor.predict(X))

        save_path = tmp_path / f"{model_name}_cycle_0"
        save_path.mkdir()
        model.save_regressor(regressor, save_path)
        model.load_regressor(save_path)

        for i in range(1, 3):
            cycle_path = tmp_path / f"{model_name}_cycle_{i}"
            cycle_path.mkdir()
            native_path = cycle_path / model.native_model_filename
            model.save_model(model._native_model, native_path)
            model.load_regressor(cycle_path)

        model.prepare_for_prediction(category_maps)
        final_predictions = model.predict(X)

        np.testing.assert_allclose(
            original_predictions,
            final_predictions,
            rtol=1e-5,
            err_msg=f"{model_name} predictions degraded after multiple save/load cycles!",
        )

    @pytest.mark.parametrize("model_name", ["lightgbm", "xgboost"])
    def test_target_transform_functions(self, model_name, config):
        model = get_model_by_name(model_name, config)

        test_value = np.array([100.0])
        transformed = model.TARGET_FUNC(test_value)
        restored = model.INVERSE_FUNC(transformed)

        np.testing.assert_allclose(
            test_value,
            restored,
            rtol=1e-10,
            err_msg="TARGET_FUNC and INVERSE_FUNC are not inverses of each other!",
        )

    @pytest.mark.parametrize("model_name", ["lightgbm", "xgboost"])
    def test_prediction_shape_preserved(self, model_name, test_data, config, tmp_path):
        X, y = test_data
        model = get_model_by_name(model_name, config)
        category_maps = _category_maps_from_df(X)

        regressor = model.make_regressor({"n_estimators": 10, "learning_rate": 0.1})
        regressor.fit(X, Model.TARGET_FUNC(y))
        predictions_before = Model.INVERSE_FUNC(regressor.predict(X))

        save_path = tmp_path / f"{model_name}_shape"
        save_path.mkdir()
        model.save_regressor(regressor, save_path)

        model.load_regressor(save_path)
        model.prepare_for_prediction(category_maps)
        predictions_after = model.predict(X)

        assert predictions_before.shape == predictions_after.shape, (
            f"Prediction shape changed: {predictions_before.shape} -> {predictions_after.shape}"
        )

    @pytest.mark.parametrize("model_name", ["lightgbm", "xgboost"])
    def test_single_sample_prediction(self, model_name, test_data, config, tmp_path):
        X, y = test_data
        model = get_model_by_name(model_name, config)
        category_maps = _category_maps_from_df(X)

        regressor = model.make_regressor({"n_estimators": 10, "learning_rate": 0.1})
        regressor.fit(X, Model.TARGET_FUNC(y))

        single_sample = X.iloc[[0]]
        prediction_before = Model.INVERSE_FUNC(regressor.predict(single_sample))

        save_path = tmp_path / f"{model_name}_single"
        save_path.mkdir()
        model.save_regressor(regressor, save_path)

        model.load_regressor(save_path)
        model.prepare_for_prediction(category_maps)
        prediction_after = model.predict(single_sample)

        np.testing.assert_allclose(
            prediction_before,
            prediction_after,
            rtol=1e-5,
            err_msg=f"{model_name} single sample prediction differs after save/load!",
        )
