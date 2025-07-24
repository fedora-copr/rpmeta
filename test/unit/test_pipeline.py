from unittest.mock import MagicMock, patch

from sklearn.pipeline import Pipeline


def test_create_pipeline_with_preprocessor(base_model_subclass, example_config):
    model = base_model_subclass("test_model", config=example_config)
    pipeline = model.create_pipeline({"param1": 100, "param2": 200})

    assert isinstance(pipeline, Pipeline)
    assert len(pipeline.steps) == 2
    assert pipeline.steps[0][0] == "preprocessor"
    assert pipeline.steps[1][0] == "regressor"


def test_create_pipeline_without_preprocessor(base_model_subclass, example_config):
    model = base_model_subclass("test_model", use_preprocessor=False, config=example_config)

    pipeline = model.create_pipeline({"param1": 100, "param2": 200})

    assert not isinstance(pipeline, Pipeline) or len(pipeline.steps) == 1


def test_create_pipeline_params_passed_to_regressor(base_model_subclass, example_config):
    model = base_model_subclass("test_model", config=example_config)
    test_params = {"param1": 999, "param2": 888}

    with patch.object(model, "_make_regressor") as mock_make_regressor:
        mock_regressor = MagicMock()
        mock_make_regressor.return_value = mock_regressor

        model.create_pipeline(test_params)
        mock_make_regressor.assert_called_once_with(test_params)
