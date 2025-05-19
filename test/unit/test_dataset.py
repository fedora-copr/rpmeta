from pathlib import Path

from rpmeta.dataset import HwInfo, InputRecord, Record


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


def test_inputrecord_from_data_frame():
    data = {
        "package_name": "test-package",
        "epoch": 0,
        "version": "1.0.0",
        "mock_chroot": "fedora-35-x86_64",
        "cpu_model_name": "Intel(R) Core(TM) i7-8550U CPU @ 1.80GHz",
        "cpu_arch": "x86_64",
        "cpu_model": "142",
        "cpu_cores": 8,
        "ram": 16384,
        "swap": 8192,
        "os": "centos-stream",
        "os_family": "centos",
        "os_version": "9",
        "os_arch": "x86_64",
    }
    input_record = InputRecord.from_data_frame(data)

    assert input_record.package_name == "test-package"
    assert input_record.epoch == 0
    assert input_record.version == "1.0.0"
    assert input_record.mock_chroot == "centos-stream-9-x86_64"
    assert input_record.hw_info.cpu_model_name == "Intel(R) Core(TM) i7-8550U CPU @ 1.80GHz"
    assert input_record.hw_info.cpu_arch == "x86_64"
    assert input_record.hw_info.cpu_model == "142"
    assert input_record.hw_info.cpu_cores == 8
    assert input_record.hw_info.ram == 16384
    assert input_record.hw_info.swap == 8192


def test_inputrecord_to_data_frame():
    hw_info = HwInfo(
        cpu_model_name="Intel(R) Core(TM) i7-8550U CPU @ 1.80GHz",
        cpu_arch="x86_64",
        cpu_model="142",
        cpu_cores=8,
        ram=16384,
        swap=8192,
    )
    input_record = InputRecord(
        package_name="test-package",
        epoch=0,
        version="1.0.0",
        mock_chroot="fedora-35-x86_64",
        hw_info=hw_info,
    )
    data_frame = input_record.to_data_frame()

    assert data_frame == {
        "package_name": "test-package",
        "epoch": 0,
        "version": "1.0.0",
        "os": "fedora",
        "os_family": "fedora",
        "os_version": "35",
        "os_arch": "x86_64",
        "cpu_model_name": "Intel(R) Core(TM) i7-8550U CPU @ 1.80GHz",
        "cpu_arch": "x86_64",
        "cpu_model": "142",
        "cpu_cores": 8,
        "ram": 16384,
        "swap": 8192,
    }


def test_record_to_data_frame():
    hw_info = HwInfo(
        cpu_model_name="Intel(R) Core(TM) i7-8550U CPU @ 1.80GHz",
        cpu_arch="x86_64",
        cpu_model="142",
        cpu_cores=8,
        ram=16384,
        swap=8192,
    )
    record = Record(
        package_name="test-package",
        epoch=0,
        version="1.0.0",
        mock_chroot="fedora-35-x86_64",
        hw_info=hw_info,
        build_duration=120,
    )
    data_frame = record.to_data_frame()

    assert data_frame == {
        "package_name": "test-package",
        "epoch": 0,
        "version": "1.0.0",
        "os": "fedora",
        "os_family": "fedora",
        "os_version": "35",
        "os_arch": "x86_64",
        "cpu_model_name": "Intel(R) Core(TM) i7-8550U CPU @ 1.80GHz",
        "cpu_arch": "x86_64",
        "cpu_model": "142",
        "cpu_cores": 8,
        "ram": 16384,
        "swap": 8192,
        "build_duration": 120,
    }
