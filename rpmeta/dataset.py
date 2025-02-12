import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class HwInfo:
    """
    Hardware information of the build system.
    """

    cpu_model_name: str
    cpu_arch: str
    cpu_model: str
    cpu_cores: str
    ram: str
    swap: str
    bogomips: str

    @classmethod
    def parse_from_lscpu(cls, content: str) -> "HwInfo":
        logger.debug(f"lscpu output: {content}")
        hw_info = {}
        for line in content.splitlines():
            if line.startswith("Model name:"):
                hw_info["cpu_model_name"] = line.split(":")[1].strip()
            elif line.startswith("Architecture:"):
                hw_info["cpu_arch"] = line.split(":")[1].strip()
            elif line.startswith("Model:"):
                hw_info["cpu_model"] = line.split(":")[1].strip()
            elif line.startswith("CPU(s):"):
                hw_info["cpu_cores"] = line.split(":")[1].strip()
            elif line.startswith("Mem:"):
                hw_info["ram"] = line.split()[1]
            elif line.startswith("Swap:"):
                hw_info["swap"] = line.split()[1]
            elif line.startswith("BogoMIPS:"):
                hw_info["bogomips"] = line.split(":")[1].strip()

        logger.debug(f"Extracted hardware info: {hw_info}")
        return cls(**hw_info)

    def to_dict(self) -> dict:
        """
        Convert the hardware information to dictionary with only interesting fields for the models.
        """
        return {
            "cpu_model_name": self.cpu_model_name,
            "cpu_arch": self.cpu_arch,
            "cpu_model": self.cpu_model,
            "cpu_cores": self.cpu_cores,
            "ram": self.ram,
            "swap": self.swap,
            "bogomips": self.bogomips,
        }


@dataclass
class Record:
    """
    A record of a successful build in build system in dataset.
    """

    package_name: str
    epoch: int
    version: str
    release: str
    # TODO: probably drop this since I can't parse every record
    mock_chroot_name: Optional[str]
    start_ts: int
    end_ts: int
    hw_info: HwInfo

    @property
    def nevra(self) -> str:
        return f"{self.package_name}-{self.epoch}:{self.version}-{self.release}"

    @property
    def nvr(self) -> str:
        return f"{self.package_name}-{self.version}-{self.release}"

    @property
    def build_duration(self) -> int:
        return self.end_ts - self.start_ts

    def to_data_frame(self) -> dict:
        """
        Convert the record to dictionary that the trained model understands.
        """
        return {
            "package_name": self.package_name,
            "epoch": self.epoch,
            "version": self.version,
            "reslease": self.release,
            "mock_chroot_name": self.mock_chroot_name,
            "build_duration": self.build_duration,
            **self.hw_info.to_dict(),
        }

    @classmethod
    def from_data_frame(cls, data: dict) -> "Record":
        """
        Create a record from the dictionary that the trained model understands to the Record.
        """
        logger.debug(f"Creating Record from data: {data}")
        return cls(
            package_name=data["package_name"],
            epoch=data["epoch"],
            version=data["version"],
            release=data["release"],
            mock_chroot_name=data["mock_chroot_name"],
            start_ts=data["start_ts"],
            end_ts=data["end_ts"],
            hw_info=HwInfo(
                cpu_model_name=data["cpu_model_name"],
                cpu_arch=data["cpu_arch"],
                cpu_model=data["cpu_model"],
                cpu_cores=data["cpu_cores"],
                ram=data["ram"],
                swap=data["swap"],
                bogomips=data["bogomips"],
            ),
        )

    def to_dict(self) -> dict:
        """
        Convert the record to dictionary.
        """
        return {
            "package_name": self.package_name,
            "epoch": self.epoch,
            "version": self.version,
            "release": self.release,
            "mock_chroot_name": self.mock_chroot_name,
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
            "hw_info": self.hw_info.to_dict(),
        }
