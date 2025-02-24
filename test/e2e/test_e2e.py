import json
from pathlib import Path

import requests

from test.helpers import run_rpmeta_cli


def test_train(trained_model):
    assert trained_model.exists(), "Model was not saved"
    assert trained_model.stat().st_size > 0, "Model is empty"


def test_predict(trained_model):
    dataset_path = Path(__file__).parent.parent / "data" / "dataset_predict.json"
    cmd = ["predict", "--model-path", str(trained_model), "--data", str(dataset_path)]

    result = run_rpmeta_cli([*cmd, "--output-type", "json"])

    assert result.returncode == 0

    response = json.loads(result.stdout)
    assert len(response) == 1
    assert isinstance(response["prediction"], int)

    result = run_rpmeta_cli([*cmd, "--output-type", "text"])
    assert result.returncode == 0
    # check if output is Prediction: <number>
    assert result.stdout.startswith("Prediction: ")
    assert result.stdout.split(":")[1].strip().isnumeric()


def test_api_server(api_server):
    response = requests.get("http://localhost:9876")
    assert response.status_code == 404
    assert response.text == "Not Found"

    response = requests.post(
        "http://localhost:9876/predict",
        json=json.loads(
            (Path(__file__).parent.parent / "data" / "dataset_predict.json").read_text(),
        ),
    )
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert isinstance(response.json()["prediction"], int)

    # TODO: this is where validation like pydantic comes handy instead of making
    # wrappers around the code to return 404
    bad_response = requests.post(
        "http://localhost:9876/predict",
        json={"foo": "bar"},
    )
    assert bad_response.status_code == 500
