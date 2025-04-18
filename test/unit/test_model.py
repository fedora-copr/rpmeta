from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rpmeta.model import load_model, make_prediction, save_model


@patch("rpmeta.model.joblib.load")
def test_load_model(mock_load):
    mock_load.return_value = "mocked_pipeline"
    result = load_model("mock_model_path")
    mock_load.assert_called_once_with("mock_model_path")
    assert result == "mocked_pipeline"


@patch("rpmeta.model.Path.exists", return_value=True)
def test_save_model_file_exists(mock_exists):
    mock_model = MagicMock()
    with pytest.raises(ValueError, match="already exists"):
        save_model(mock_model, "mock_model_path")
    mock_exists.assert_called_once()


@patch("rpmeta.model.Path.exists", return_value=False)
@patch("rpmeta.model.joblib.dump")
def test_save_model_success(mock_dump, mock_exists):
    mock_model = MagicMock()
    save_model(mock_model, "mock_model_path")
    mock_exists.assert_called_once()
    mock_dump.assert_called_once_with(mock_model, "mock_model_path")


@patch("sklearn.pipeline.Pipeline", autospec=True)
def test_make_prediction(mock_pipeline):
    mock_pipeline = MagicMock(predict=lambda: np.array([1]))
    mock_input = MagicMock(to_data_frame=lambda: {"feature": [2]})

    result = make_prediction(mock_pipeline.return_value, mock_input)
    mock_pipeline.return_value.predict.assert_called_once()
    assert result == 1
