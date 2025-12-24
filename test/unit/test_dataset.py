from pathlib import Path

import pandas as pd
import pytest

from rpmeta.dataset import HwInfo, Record


def test_hwinfo_parse_from_lscpu():
    with open(Path(__file__).parent.parent / "data" / "hw_info.log") as f:
        lscpu_output = f.read()

    hw_info = HwInfo.parse_from_lscpu(lscpu_output)
    assert hw_info.cpu_model_name == "AMD EPYC 7302 16-Core Processor"
    assert hw_info.cpu_arch == "x86_64"
    assert hw_info.cpu_model == "49"
    assert hw_info.cpu_cores == 2
    assert hw_info.ram == 15.6
    assert hw_info.swap == 140.5


def test_hwinfo_parse_human_readable_format():
    lscpu_output = """CPU info:
Architecture:                         x86_64
CPU op-mode(s):                       32-bit, 64-bit
CPU(s):                               8
Model name:                           Intel(R) Core(TM) i7-8550U CPU @ 1.80GHz
Model:                                142
Memory:
               total        used        free      shared  buff/cache   available
Mem:            54Gi       9,5Gi        10Gi       154Mi        34Gi        45Gi
Swap:           63Gi          0B        63Gi
"""
    hw_info = HwInfo.parse_from_lscpu(lscpu_output)
    assert hw_info.cpu_model_name == "Intel(R) Core(TM) i7-8550U CPU @ 1.80GHz"
    assert hw_info.cpu_arch == "x86_64"
    assert hw_info.cpu_model == "142"
    assert hw_info.cpu_cores == 8
    assert hw_info.ram == 54.0
    assert hw_info.swap == 63.0


@pytest.mark.parametrize(
    "value,expected_gi",
    [
        ("54Gi", 54.0),
        ("63Gi", 63.0),
        ("154Mi", 0.2),
        ("9,5Gi", 9.5),
        ("1024Mi", 1.0),
        ("512Mi", 0.5),
        ("2Ti", 2048.0),
        ("1048576Ki", 1.0),
        ("1073741824B", 1.0),
        ("0B", 0.0),
        ("57195144", 54.5),
    ],
)
def test_convert_human_readable_to_gi(value, expected_gi):
    result = HwInfo._convert_human_readable_to_gi(value)
    assert result == expected_gi


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
            "ram": 15.3,
            "swap": 7.6,
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
            "ram": 15.3,
            "swap": 7.6,
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
    assert df.iloc[0]["hw_info.ram"] == 15.3
    assert df.iloc[0]["hw_info.swap"] == 7.6
