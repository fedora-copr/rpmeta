import logging
import tomllib
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator

from rpmeta.constants import (
    CONFIG_LOCATIONS,
    KOJI_HUB_URL,
    RESULT_DIR_LOCATIONS,
    TimeFormat,
)

logger = logging.getLogger(__name__)


class Api(BaseModel):
    """API server configuration"""

    host: str = Field(default="localhost", description="Hostname to bind the API server to")
    port: int = Field(default=44882, description="Port to bind the API server to", gt=0, lt=65536)
    debug: bool = Field(default=False, description="Enable debug mode")


class Koji(BaseModel):
    """Koji client configuration"""

    hub_url: str = Field(default=KOJI_HUB_URL, description="Koji hub URL")


class Copr(BaseModel):
    """Copr client configuration"""

    api_url: str = Field(
        default="https://copr.fedorainfracloud.org/api_3",
        description="Copr API URL",
    )


class ModelParams(BaseModel):
    """Optional model parameters"""

    params: Optional[dict[str, Any]] = Field(
        default=None,
        description="Optional model parameters",
        examples=[
            {
                "objective": "reg:squarederror",
                "tree_method": "hist",
            },
        ],
    )


class XGBoostParams(ModelParams):
    """XGBoost model parameters"""

    n_estimators: int = Field(
        default=1003,
        description="Number of boosting rounds",
    )
    learning_rate: float = Field(
        default=0.2415,
        description="Step size shrinkage used to prevent overfitting",
    )
    max_depth: int = Field(
        default=5,
        description="Maximum depth of a tree",
    )
    subsample: float = Field(
        default=0.5693,
        description="Subsample ratio of the training instances",
    )
    colsample_bytree: float = Field(
        default=0.6181,
        description="Subsample ratio of columns when constructing each tree",
    )
    reg_alpha: float = Field(
        default=0.0305,
        description="L1 regularization term on weights",
    )
    reg_lambda: float = Field(
        default=7.6076,
        description="L2 regularization term on weights",
    )
    min_child_weight: float = Field(
        default=1.101,
        description="Minimum sum of instance weight needed in a child",
    )
    gamma: float = Field(
        default=1.904,
        description="Minimum loss reduction required to make a further partition on a leaf node",
    )
    early_stopping_rounds: Optional[int] = Field(
        default=None,
        description="Number of rounds for early stopping",
        examples=[10, 20, 50],
    )


class LightGBMParams(ModelParams):
    """LightGBM model parameters"""

    n_estimators: int = Field(
        default=1208,
        description="Number of boosting rounds",
    )
    learning_rate: float = Field(
        default=0.2319,
        description="Step size shrinkage used to prevent overfitting",
    )
    max_depth: int = Field(
        default=10,
        description="Maximum depth of a tree",
    )
    num_leaves: int = Field(
        default=849,
        description="Maximum tree leaves for base learners",
    )
    min_child_samples: int = Field(
        default=57,
        description="Minimum number of data needed in a leaf",
    )
    subsample: float = Field(
        default=0.6354,
        description="Subsample ratio of the training instances",
    )
    colsample_bytree: float = Field(
        default=0.9653,
        description="Subsample ratio of columns when constructing each tree",
    )
    lambda_l1: float = Field(
        default=0.0005,
        description="L1 regularization term on weights",
    )
    lambda_l2: float = Field(
        default=0.0001,
        description="L2 regularization term on weights",
    )
    max_bin: int = Field(
        default=282,
        description="Max number of bins that feature values will be bucketed in",
    )
    early_stopping_rounds: Optional[int] = Field(
        default=None,
        description="Number of rounds for early stopping",
        examples=[10, 20, 50],
    )


class ModelBehavior(BaseModel):
    """Model behavior configuration"""

    time_format: TimeFormat = Field(
        default=TimeFormat.MINUTES,
        description="Format for predicted time output",
        examples=TimeFormat.get_all_formats(),
    )


class Model(BaseModel):
    """Machine learning model configuration"""

    random_state: int = Field(
        default=42,
        description="Random state seed for reproducibility",
    )
    n_jobs: int = Field(
        default=-1,
        description="Number of jobs for parallel processing, -1 means use all available cores",
    )
    test_size: float = Field(
        default=0.2,
        description="Fraction of data to use for testing",
        gt=0.0,
        lt=1.0,
    )
    verbose: bool = Field(
        default=False,
        description="Enable verbose output during model training and evaluation",
    )
    behavior: ModelBehavior = Field(
        default_factory=ModelBehavior,
        description="Model behavior configuration",
    )
    xgboost: XGBoostParams = Field(
        default_factory=XGBoostParams,
        description="XGBoost model parameters",
    )
    lightgbm: LightGBMParams = Field(
        default_factory=LightGBMParams,
        description="LightGBM model parameters",
    )


class Logging(BaseModel):
    """Logging configuration"""

    format: str = Field(
        default="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        description="Log format string",
    )
    datefmt: str = Field(
        default="%Y-%m-%d %H:%M:%S",
        description="Date format for log timestamps",
    )
    file: Optional[Path] = Field(
        default=None,
        description="Path to the log file. If None, logs will be written to stderr",
        examples=["/var/log/rpmeta.log"],
    )


class Config(BaseModel):
    """Complete configuration model for RPMeta"""

    result_dir: Path = Field(
        default=RESULT_DIR_LOCATIONS[0],
        description="Directory for storing results and trained models",
        examples=[RESULT_DIR_LOCATIONS[1]],
    )
    api: Api = Field(default_factory=Api, description="API server configuration")
    koji: Koji = Field(default_factory=Koji, description="Koji client configuration")
    copr: Copr = Field(default_factory=Copr, description="Copr client configuration")
    model: Model = Field(default_factory=Model, description="Machine learning model configuration")
    logging: Logging = Field(default_factory=Logging, description="Logging configuration")

    @model_validator(mode="after")
    def ensure_result_dir_exists(self) -> "Config":
        if not self.result_dir.exists():
            self.result_dir.mkdir(parents=True, exist_ok=True)
            logger.debug("Created result directory: %s", self.result_dir)
        return self


class ConfigManager:
    """
    Configuration manager for RPMeta

    This class handles loading configuration from TOML files in the following order of precedence:
    1. Direct parameters passed to get_config()
    2. Configuration files
    3. Default values

    Configuration files are searched in the following locations defined in CONFIG_LOCATIONS:
    - $XDG_CONFIG_HOME/rpmeta/config.toml
    - $HOME/.config/rpmeta/config.toml
    - /etc/rpmeta/config.toml
    """

    @staticmethod
    def _get_result_dir() -> Path:
        for location in RESULT_DIR_LOCATIONS:
            if location.exists():
                logger.debug("Using result dir: %s", location)
                return location

            logger.debug("Result dir does not exist: %s", location)

        # user location does not exist, create the first one
        default_location = RESULT_DIR_LOCATIONS[0]
        default_location.mkdir(parents=True, exist_ok=True)
        logger.debug("Created result dir and using: %s", default_location)
        return default_location

    @staticmethod
    def _find_config_file() -> Optional[Path]:
        for location in CONFIG_LOCATIONS:
            if not location.exists():
                continue

            toml_config = location / "config.toml"
            if toml_config.exists():
                return toml_config

        return None

    @classmethod
    def _load_from_file(cls, config_file: Path) -> dict[str, Any]:
        if config_file.suffix.lower() != ".toml":
            logger.warning(
                "Unsupported config file format: %s. Only toml is supported.",
                config_file.suffix.lower(),
            )
            return {}

        with open(config_file, "rb") as f:
            return tomllib.load(f)

    @classmethod
    def get_config(
        cls,
        result_dir: Optional[Path] = None,
        config_file: Optional[Path] = None,
    ) -> Config:
        """
        Get the configuration object

        Args:
            result_dir: Results directory (overrides other sources)
            config_file: Path to a specific config file to load (overrides auto-detection)

        Returns:
            Config: A configured Config object
        """
        config_data = {}
        if config_file:
            logger.info("Loading configuration from specified file: %s", config_file)
            config_data = cls._load_from_file(config_file)
        else:
            location = cls._find_config_file()
            if location:
                logger.info("Loading configuration from detected file: %s", location)
                config_data = cls._load_from_file(location)
            else:
                logger.info("No configuration file found, using defaults")

        if result_dir:
            logger.info("Using explicitly provided result directory: %s", result_dir)

        result_dir = result_dir or config_data.get("result_dir", cls._get_result_dir())
        logger.debug("Setting result directory to: %s", result_dir)
        config_data.setdefault("result_dir", result_dir)

        logger.debug("Constructed configuration data: %s", config_data)
        return Config(**config_data)
