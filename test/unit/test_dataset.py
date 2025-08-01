from pathlib import Path

import pandas as pd

from rpmeta.dataset import HwInfo, Record


def test_hwinfo_parse_from_lscpu():
    with open(Path(__file__).parent.parent / "data" / "hw_info.log") as f:
        lscpu_output = f.read()

    hw_info = HwInfo.parse_from_lscpu(lscpu_output)
    assert hw_info.cpu_model_name == "AMD EPYC 7302 16-Core Processor"
    assert hw_info.cpu_arch == "x86_64"
    assert hw_info.cpu_model == "49"
    assert hw_info.cpu_cores == 2
    assert hw_info.ram == 16369604
    assert hw_info.swap == 147284256


def test_inputrecord_to_model_dict(hw_info, input_record):
    assert input_record.to_model_dict() == {
        "package_name": "test-package",
        "epoch": 0,
        "version": "1.0.0",
        "os": "fedora",
        "os_family": "fedora",
        "os_version": "35",
        "os_arch": "x86_64",
        "hw_info": {
            "cpu_model_name": "silny procak",
            "cpu_arch": "x86_64",
            "cpu_model": "12",
            "cpu_cores": 4,
            "ram": 16000000,
            "swap": 8000000,
        },
    }


def test_record_to_model_dict(hw_info):
    record = Record(
        package_name="test-package",
        epoch=0,
        version="1.0.0",
        mock_chroot="fedora-35-x86_64",
        hw_info=hw_info,
        build_duration=120,
    )
    model_dict = record.to_model_dict()

    assert model_dict == {
        "package_name": "test-package",
        "epoch": 0,
        "version": "1.0.0",
        "os": "fedora",
        "os_family": "fedora",
        "os_version": "35",
        "os_arch": "x86_64",
        "build_duration": 2,  # must be in minutes
        "hw_info": {
            "cpu_model_name": "silny procak",
            "cpu_arch": "x86_64",
            "cpu_model": "12",
            "cpu_cores": 4,
            "ram": 16000000,
            "swap": 8000000,
        },
    }


def test_inputrecord_to_data_frame(input_record):
    df = input_record.to_data_frame(
        category_maps={
            "package_name": ["test-package"],
            "version": ["1.0.0"],
            "os": ["fedora"],
            "os_family": ["fedora"],
            "os_version": ["35"],
            "os_arch": ["x86_64"],
            "hw_info.cpu_model_name": ["silny procak"],
            "hw_info.cpu_arch": ["x86_64"],
            "hw_info.cpu_model": ["12"],
        },
    )
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (1, 13)
    assert df.iloc[0]["package_name"] == "test-package"
    assert df.iloc[0]["epoch"] == 0
    assert df.iloc[0]["version"] == "1.0.0"
    assert df.iloc[0]["os"] == "fedora"
    assert df.iloc[0]["os_family"] == "fedora"
    assert df.iloc[0]["os_version"] == "35"
    assert df.iloc[0]["os_arch"] == "x86_64"
    assert df.iloc[0]["hw_info.cpu_model_name"] == "silny procak"
    assert df.iloc[0]["hw_info.cpu_arch"] == "x86_64"
    assert df.iloc[0]["hw_info.cpu_model"] == "12"
    assert df.iloc[0]["hw_info.cpu_cores"] == 4
    # because of DIVIDER
    assert df.iloc[0]["hw_info.ram"] == 160
    assert df.iloc[0]["hw_info.swap"] == 80
