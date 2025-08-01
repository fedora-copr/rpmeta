import logging
from typing import Any, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from rpmeta.constants import ALL_FEATURES, DIVIDER
from rpmeta.helpers import to_minutes_rounded

logger = logging.getLogger(__name__)


class HwInfo(BaseModel):
    """
    Hardware information of the build system.

    This model represents the hardware specifications of the machine used for building packages.
    It includes CPU details, architecture, and memory information.
    """

    cpu_model_name: str = Field(
        description=(
            "The full name/model of the CPU (e.g., 'Intel(R) Core(TM) i7-8550U CPU @ 1.80GHz')"
        ),
    )
    cpu_arch: str = Field(
        description="CPU architecture (e.g., 'x86_64', 'aarch64')",
    )
    cpu_model: str = Field(
        description="CPU model number (e.g., '142', '49')",
        examples=["85", "142", "49"],
    )
    cpu_cores: int = Field(
        description="Number of CPU cores",
        gt=0,
    )
    ram: int = Field(
        description="Total RAM in KB",
        gt=0,
    )
    swap: int = Field(
        description="Total swap space in KB",
        ge=0,
    )

    model_config = ConfigDict(
        validate_assignment=True,
    )

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
        return cls(**hw_info)

    @field_validator("cpu_model", mode="before")
    @classmethod
    def validate_cpu_model(cls, v):
        if v is None:
            # setting to unknown for the model to be able to handle missing cpu models
            return "unknown"

        return v


class InputRecord(BaseModel):
    """
    Input record for the model training and prediction.

    This record is used to pass the input data to the model for training or prediction.
    Exactly the same data structure is provided by user for the API or CLI.

    ## Example
    ```json
    {
      "package_name": "rust-winit",
      "epoch": 0,
      "version": "0.30.8",
      "mock_chroot": "fedora-41-x86_64",
      "hw_info": {
        "cpu_model_name": "Intel Xeon Processor (Cascadelake)",
        "cpu_arch": "x86_64",
        "cpu_model": "85",
        "cpu_cores": 6,
        "ram": 15324520,
        "swap": 8388604
      }
    }
    ```
    """

    package_name: str = Field(
        description="Name of the RPM package to build",
        examples=["rust-winit", "python-numpy"],
    )
    epoch: int = Field(
        description="Package epoch (usually 0)",
        ge=0,
        examples=[0, 1],
    )
    version: str = Field(
        description="Package version string",
        examples=["0.30.8", "1.24.3"],
    )
    hw_info: HwInfo = Field(
        description="Hardware information of the build system",
    )
    mock_chroot: Optional[str] = Field(
        default=None,
        description="Mock chroot in format '<distro>-<version>-<arch>' (e.g., 'fedora-41-x86_64')",
        examples=["fedora-41-x86_64", "centos-stream-9-x86_64"],
    )

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

    def to_model_dict(self) -> dict[str, Any]:
        """
        Convert the record to dictionary with data that the _trained model_ expects.
        """
        return {
            "package_name": self.package_name,
            "epoch": self.epoch,
            "version": self.version,
            "os": self.os,
            "os_family": self.os_family,
            "os_version": self.os_version,
            "os_arch": self.os_arch,
            "hw_info": self.hw_info.model_dump(),
        }

    def to_data_frame(self, category_maps: dict[str, list[str]]) -> pd.DataFrame:
        """
        Convert the record to a pandas DataFrame that the model understands.
        This is used for prediction.
        """
        df = pd.json_normalize(self.model_dump())
        df["os"] = self.os
        df["os_family"] = self.os_family
        df["os_version"] = self.os_version
        df["os_arch"] = self.os_arch

        # preprocess
        for col, cat_list in category_maps.items():
            dtype = pd.CategoricalDtype(categories=cat_list, ordered=False)
            df[col] = df[col].astype(dtype)

        df["hw_info.ram"] = np.round(df["hw_info.ram"] / DIVIDER).astype(int)
        df["hw_info.swap"] = np.round(df["hw_info.swap"] / DIVIDER).astype(int)

        # ensure all features are in the expected order for the model
        return df[ALL_FEATURES]


class Record(InputRecord):
    """
    A record of a successful build in build system in dataset.

    This extends InputRecord to include the actual build duration,
    which is used as the target variable for model training.
    """

    build_duration: int = Field(
        description="Actual build duration in seconds",
        gt=0,
        examples=[5, 60, 720],
    )

    def to_model_dict(self) -> dict[str, Any]:
        """
        Convert the record to dictionary that the model _to be trained_ understands + has the
         target feature.
        """
        return {
            **super().to_model_dict(),
            # the fetched data is in seconds, but the model gives better results in minutes
            "build_duration": to_minutes_rounded(self.build_duration),
        }
