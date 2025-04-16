import json
from pathlib import Path
from subprocess import Popen
from time import sleep

import pytest

from rpmeta.dataset import HwInfo, Record
from test.helpers import run_rpmeta_cli


@pytest.fixture
def trained_model(tmp_path_factory):
    dataset_path = Path(__file__).parent / "data" / "dataset_train.json"
    model_path = tmp_path_factory.mktemp("models") / "model.joblib"
    result = run_rpmeta_cli(
        ["train", "--dataset", str(dataset_path), "--destination", str(model_path)],
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
            "--model",
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


@pytest.fixture
def koji_build():
    with open(Path(__file__).parent / "data" / "koji_build.json") as f:
        return json.load(f)


@pytest.fixture
def koji_task_descendant():
    with open(Path(__file__).parent / "data" / "koji_task_descendant.json") as f:
        return json.load(f)


@pytest.fixture
def hw_info():
    return HwInfo(
        cpu_arch="x86_64",
        cpu_cores=4,
        cpu_model="12",
        cpu_model_name="silny procak",
        ram=16,
        swap=8,
    )


@pytest.fixture
def dataset_record(hw_info, koji_build):
    return Record(
        package_name=koji_build[0]["name"],
        version=koji_build[0]["version"],
        release=koji_build[0]["release"],
        epoch=0,
        mock_chroot="fedora-43-x86_64",
        build_duration=893,
        hw_info=hw_info,
    )
