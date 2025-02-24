from pathlib import Path
from subprocess import Popen
from time import sleep

import pytest

from test.helpers import run_rpmeta_cli


@pytest.fixture
def trained_model(tmp_path_factory):
    dataset_path = Path(__file__).parent / "data" / "dataset_train.json"
    model_path = tmp_path_factory.mktemp("models") / "model.joblib"
    result = run_rpmeta_cli(
        ["train", "--dataset-path", str(dataset_path), "--destination-path", str(model_path)],
    )
    assert result.returncode == 0, result.stderr or result.stdout
    return model_path


@pytest.fixture
def api_server(trained_model):
    api_server = Popen(
        [
            "python3",
            "-m",
            "rpmeta.cli",
            "serve",
            "--model-path",
            str(trained_model),
            "--port",
            "9876",
            "--host",
            "0.0.0.0",
        ],
    )
    # Wait for the server to start
    sleep(3)
    yield
    api_server.terminate()
