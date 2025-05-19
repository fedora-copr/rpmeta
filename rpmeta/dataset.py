# TODO: this will be useful if I decide to go with pydantic for validation, but that's too huge
# dependency for this small thingy, so will I benefit from it? Other usage may be on copr's side
# when working with the tool and parsing data like HW info and data for model. If neither of this
# is the case, just drop the biolerplate and use plain dicsts


import logging
from dataclasses import dataclass
from typing import Optional

from rpmeta.constants import ALL_FEATURES

logger = logging.getLogger(__name__)


@dataclass
class HwInfo:
    """
    Hardware information of the build system.
    """

    cpu_model_name: str
    cpu_arch: str
    cpu_model: str
    cpu_cores: int
    ram: int
    swap: int

    @classmethod
    def parse_from_lscpu(cls, content: str) -> "HwInfo":
        logger.debug(f"lscpu output: {content}")
        hw_info: dict[str, int | str] = {}
        for line in content.splitlines():
            if line.startswith("Model name:"):
                hw_info["cpu_model_name"] = line.split(":")[1].strip()
            elif line.startswith("Architecture:"):
                hw_info["cpu_arch"] = line.split(":")[1].strip()
            elif line.startswith("Model:"):
                hw_info["cpu_model"] = line.split(":")[1].strip()
            elif line.startswith("CPU(s):"):
                hw_info["cpu_cores"] = int(line.split(":")[1].strip())
            elif line.startswith("Mem:"):
                hw_info["ram"] = int(line.split()[1])
            elif line.startswith("Swap:"):
                hw_info["swap"] = int(line.split()[1])

        if hw_info.get("cpu_model") is None:
            hw_info["cpu_model"] = "unknown"

        logger.debug(f"Extracted hardware info: {hw_info}")
        return cls(**hw_info)  # type: ignore

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
        }


@dataclass
class InputRecord:
    package_name: str
    epoch: int
    version: str
    hw_info: HwInfo
    mock_chroot: Optional[str]

    @property
    def neva(self) -> str:
        """
        Name, Epoch, Version, Architecture; Release is (intentionally) missing in data.
        """
        return f"{self.package_name}-{self.epoch}:{self.version}-{self.os_arch}"

    @property
    def os(self) -> Optional[str]:
        if self.mock_chroot is None:
            return None

        return self.mock_chroot.rsplit("-", 2)[0]

    @property
    def os_family(self) -> Optional[str]:
        if self.os is None:
            return None

        return self.os.rsplit("-")[0]

    @property
    def os_version(self) -> Optional[str]:
        if self.mock_chroot is None:
            return None

        return self.mock_chroot.rsplit("-", 2)[1]

    @property
    def os_arch(self) -> Optional[str]:
        if self.mock_chroot is None:
            return None

        return self.mock_chroot.rsplit("-", 2)[2]

    @classmethod
    def from_data_frame(cls, data: dict) -> "InputRecord":
        """
        Create a record from the dictionary that the trained model understands to the Record.
        """
        logger.debug(f"Creating InputRecord from data: {data}")
        mock_chroot = None
        if data.get("os") is not None:
            mock_chroot = f"{data['os']}-{data['os_version']}-{data['os_arch']}"

        return cls(
            package_name=data["package_name"],
            epoch=data["epoch"],
            version=data["version"],
            mock_chroot=mock_chroot,
            hw_info=HwInfo(
                cpu_model_name=data["cpu_model_name"],
                cpu_arch=data["cpu_arch"],
                cpu_model=data["cpu_model"],
                cpu_cores=data["cpu_cores"],
                ram=data["ram"],
                swap=data["swap"],
            ),
        )

    def to_data_frame(self) -> dict:
        """
        Convert the record to dictionary that the _trained model_ understands.
        """
        result = {
            "package_name": self.package_name,
            "epoch": self.epoch,
            "version": self.version,
            "os": self.os,
            "os_family": self.os_family,
            "os_version": self.os_version,
            "os_arch": self.os_arch,
            **self.hw_info.to_dict(),
        }
        # ensuring that all features are in expected order for the model
        return {k: result[k] for k in ALL_FEATURES}


@dataclass
class Record(InputRecord):
    """
    A record of a successful build in build system in dataset.
    """

    build_duration: int

    def to_data_frame(self) -> dict:
        """
        Convert the record to dictionary that the model _to be trained_ understands + has the
         target feature.
        """
        return {
            **super().to_data_frame(),
            "build_duration": self.build_duration,
        }
