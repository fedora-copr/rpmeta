from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pandas as pd

from rpmeta.model import Predictor


@patch("rpmeta.model.joblib.load")
@patch(
    "builtins.open",
    new_callable=mock_open,
    read_data='{"package_name": ["pkg1", "pkg2"], "feature": ["a", "b"]}',
)
def test_predictor_load(mock_file, mock_load):
    mock_model = MagicMock()
    mock_load.return_value = mock_model
    predictor = Predictor.load("model_path", "category_maps_path")
    assert isinstance(predictor, Predictor)
    assert predictor.model == mock_model
    assert predictor.category_maps == {"package_name": ["pkg1", "pkg2"], "feature": ["a", "b"]}
    mock_load.assert_called_once_with("model_path")
    mock_file.assert_called_once_with("category_maps_path")


def test_preprocess_applies_category_and_round():
    category_maps = {"feature": ["a", "b"], "package_name": ["pkg1", "pkg2"]}
    mock_input = MagicMock()
    mock_input.to_data_frame.return_value = {
        "feature": "a",
        "package_name": "pkg1",
        "ram": 123456,
        "swap": 567890,
    }
    predictor = Predictor(MagicMock(), category_maps)
    df = predictor._preprocess(mock_input)
    assert isinstance(df, pd.DataFrame)

    # because of the DIVIDER constant
    assert df.loc[0, "ram"] == 1
    assert df.loc[0, "swap"] == 6

    assert df.loc[0, "feature"] == "a"
    assert df.loc[0, "package_name"] == "pkg1"
    assert isinstance(df["feature"].dtype, pd.CategoricalDtype)
    assert isinstance(df["package_name"].dtype, pd.CategoricalDtype)


def test_predict_returns_prediction():
    category_maps = {"package_name": ["pkg1"], "feature": ["a"]}
    mock_input = MagicMock()
    mock_input.package_name = "pkg1"
    mock_input.to_data_frame.return_value = {
        "feature": "a",
        "package_name": "pkg1",
        "ram": 1000,
        "swap": 2000,
    }
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([42])
    predictor = Predictor(mock_model, category_maps)
    result = predictor.predict(mock_input)
    assert result == 42
    mock_model.predict.assert_called_once()


def test_predict_unknown_package_name_logs_and_returns_minus_one(caplog):
    category_maps = {"package_name": ["pkg1"], "feature": ["a"]}
    mock_input = MagicMock()
    mock_input.package_name = "unknown"
    predictor = Predictor(MagicMock(), category_maps)
    with caplog.at_level("ERROR"):
        result = predictor.predict(mock_input)

    assert result == -1
    assert "is not known" in caplog.text


@patch("rpmeta.model.save_joblib")
@patch("builtins.open", new_callable=mock_open)
@patch("pathlib.Path.exists", return_value=False)
def test_predictor_save_calls_save_joblib_and_writes_json(mock_exists, mock_file, mock_save_joblib):
    category_maps = {"package_name": ["pkg1"], "feature": ["a"]}
    mock_model = MagicMock()
    predictor = Predictor(mock_model, category_maps)
    result_dir = Path("result_dir")
    predictor.save(result_dir, model_name="mymodel", category_maps_name="mycats")
    mock_save_joblib.assert_called_once_with(mock_model, result_dir, "mymodel")
    mock_file.assert_called_once_with(result_dir / "mycats.json", "w")
    handle = mock_file()
    handle.write.assert_called()
