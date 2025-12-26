import os
from enum import StrEnum
from pathlib import Path

# constants

KOJI_HUB_URL = "https://koji.fedoraproject.org/kojihub"
COPR_URL = "https://copr.fedorainfracloud.org"

# DO NOT TOUCH THE ORDER of these features, it is important for the model
# If you are changing the order, you need to retrain the model
# ideally with optuna to fine tune the parameters once more
CATEGORICAL_FEATURES = [
    "package_name",
    "version",
    "os",
    "os_family",
    "os_version",
    "os_arch",
    "hw_info.cpu_model_name",
    "hw_info.cpu_arch",
    "hw_info.cpu_model",
]
NUMERICAL_FEATURES = ["epoch", "hw_info.cpu_cores", "hw_info.ram", "hw_info.swap"]
ALL_FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
TARGET = "build_duration"


# config defaults

USER_RESULT_DIR = (
    Path(os.environ.get("XDG_DATA_HOME", os.path.expanduser("~/.local/share"))) / "rpmeta"
)
GLOBAL_RESULT_DIR = Path("/var/lib/rpmeta")
# order is important! user overrides global result dir
RESULT_DIR_LOCATIONS = [USER_RESULT_DIR, GLOBAL_RESULT_DIR]

# Configuration file locations (order matters)
CONFIG_LOCATIONS = [
    Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "rpmeta",
    Path("/etc/rpmeta"),
]

STATIC_DIR = Path(__file__).parent / "server" / "static"
FAVICON_PATH = STATIC_DIR / "favicon.ico"

# enums


class ModelEnum(StrEnum):
    LIGHTGBM = "lightgbm"
    XGBOOST = "xgboost"

    @classmethod
    def get_all_model_names(cls) -> list[str]:
        return [model.value for model in cls]


class TimeFormat(StrEnum):
    """Valid time formats for prediction output"""

    SECONDS = "seconds"
    MINUTES = "minutes"
    HOURS = "hours"

    @classmethod
    def get_all_formats(cls) -> list[str]:
        return [fmt.value for fmt in cls]


class ModelStorageBaseNames(StrEnum):
    NATIVE_MODEL = "native_model"
    SKELETON_NAME = "model_skeleton"


class ModelFileExtensions(StrEnum):
    LIGHTGBM = "txt"
    XGBOOST = "ubj"
