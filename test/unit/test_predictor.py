from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pandas as pd

from rpmeta.predictor import Predictor


@patch("rpmeta.store.ModelStorage.get_model")
@patch("joblib.load")
@patch(
    "builtins.open",
    new_callable=mock_open,
    read_data='{"package_name": ["pkg1", "pkg2"], "feature": ["a", "b"]}',
)
def test_predictor_load(mock_file, mock_load, mock_get_model, example_config):
    mock_model = MagicMock()
    mock_load.return_value = mock_model
    mock_get_model.return_value = mock_model

    predictor = Predictor.load(
        model_path=Path("model_path"),
        model_name="test_model",
        category_maps_path=Path("category_maps_path"),
        config=example_config,
    )
    assert isinstance(predictor, Predictor)
    assert predictor.model == mock_model
    assert predictor.category_maps == {"package_name": ["pkg1", "pkg2"], "feature": ["a", "b"]}


def test_predict_returns_prediction(example_config):
    category_maps = {"package_name": ["pkg1"], "feature": ["a"]}
    mock_input = MagicMock()
    mock_input.package_name = "pkg1"
    df_dict = {
        "feature": "a",
        "package_name": "pkg1",
        "ram": 1000,
        "swap": 2000,
    }
    mock_input.to_data_frame.return_value = pd.DataFrame([df_dict])
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([42])
    predictor = Predictor(mock_model, category_maps, example_config)
    result = predictor.predict(mock_input, example_config.model.behavior)
    assert result == 42
    mock_model.predict.assert_called_once()


def test_predict_unknown_package_name_logs_and_returns_minus_one(caplog, example_config):
    category_maps = {"package_name": ["pkg1"], "feature": ["a"]}
    mock_input = MagicMock()
    mock_input.package_name = "unknown"
    predictor = Predictor(MagicMock(), category_maps, example_config)
    with caplog.at_level("ERROR"):
        result = predictor.predict(mock_input, example_config.model.behavior)

    assert result == -1
    assert "is not known" in caplog.text


@patch("rpmeta.predictor.save_joblib")
@patch("builtins.open", new_callable=mock_open)
@patch("pathlib.Path.exists", return_value=False)
def test_predictor_save_calls_save_joblib_and_writes_json(
    mock_exists,
    mock_file,
    mock_save_joblib,
    example_config,
):
    category_maps = {"package_name": ["pkg1"], "feature": ["a"]}
    mock_model = MagicMock()
    predictor = Predictor(mock_model, category_maps, example_config)
    result_dir = Path("result_dir")
    predictor.save(result_dir, model_name="mymodel", category_maps_name="mycats")
    mock_save_joblib.assert_called_once_with(mock_model, result_dir, "mymodel")
    mock_file.assert_called_once_with(result_dir / "mycats.json", "w")
    handle = mock_file()
    handle.write.assert_called()
