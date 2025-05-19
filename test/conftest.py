import json
from pathlib import Path
from subprocess import Popen
from time import sleep

import pytest

from rpmeta.dataset import HwInfo, Record
from test.helpers import run_rpmeta_cli


@pytest.fixture
def model_and_types(tmp_path_factory):
    dataset_path = Path(__file__).parent / "data" / "dataset_train.json"
    result_dir = tmp_path_factory.mktemp("models")

    result = run_rpmeta_cli(
        [
            "train",
            "--dataset",
            str(dataset_path),
            "--result-dir",
            str(result_dir),
            "--model-allowlist",
            "lightgbm",
            "run",
        ],
    )
    assert result.returncode == 0, result.stderr or result.stdout

    cat_dtypes_files = list(result_dir.glob("*.json"))
    assert len(cat_dtypes_files) == 1, "Category dtypes file not found"
    assert cat_dtypes_files[0].stat().st_size > 0, "Category dtypes file is empty"
    assert cat_dtypes_files[0].read_text(), "Category dtypes file is empty"
    assert cat_dtypes_files[0].suffix == ".json", "Category dtypes file is not a JSON file"

    return Path(result.stdout.strip()), cat_dtypes_files[0]


@pytest.fixture
def api_server(model_and_types):
    trained_model_file, category_dtypes_file = model_and_types
    api_server = Popen(
        [
            "python3",
            "-m",
            "rpmeta.cli.main",
            "model",
            "--model",
            str(trained_model_file),
            "--categories",
            str(category_dtypes_file),
            "serve",
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
        epoch=0,
        mock_chroot="fedora-43-x86_64",
        build_duration=893,
        hw_info=hw_info,
    )
