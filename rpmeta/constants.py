import os
from pathlib import Path

# constants

KOJI_HUB_URL = "https://koji.fedoraproject.org/kojihub"

DIVIDER = 100000

# DO NOT TOUCH THE ORDER of these features, it is important for the model
# If you are changing the order, you need to retrain the model
CATEGORICAL_FEATURES = [
    "package_name",
    "version",
    "os",
    "os_family",
    "os_version",
    "os_arch",
    "cpu_model_name",
    "cpu_arch",
    "cpu_model",
]
NUMERICAL_FEATURES = ["epoch", "cpu_cores", "ram", "swap"]
ALL_FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
TARGET = "build_duration"


# config defaults

HOST = os.environ.get("HOST", "localhost")
PORT = int(os.environ.get("PORT", 44882))

USER_RESULT_DIR = (
    Path(os.environ.get("XDG_DATA_HOME", os.path.expanduser("~/.local/share"))) / "rpmeta"
)
GLOBAL_RESULT_DIR = Path("/var/lib/rpmeta")
# order is important! user overrides global result dir
RESULT_DIR_LOCATIONS = [USER_RESULT_DIR, GLOBAL_RESULT_DIR]
