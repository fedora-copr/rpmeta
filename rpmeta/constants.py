from os import environ

# constants

KOJI_HUB_URL = "https://koji.fedoraproject.org/kojihub"

DIVIDER = 100000

RESULTS_DIR = "/etc/rpmeta/results"

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


# config of model/server

HOST = environ.get("HOST", "localhost")
PORT = int(environ.get("PORT", 44882))
