import logging
from pathlib import Path
from typing import Optional

from rpmeta.constants import HOST, PORT, RESULT_DIR_LOCATIONS

logger = logging.getLogger(__name__)


class Config:
    """
    Configuration class
    """

    def __init__(self, host: str, port: int, result_dir: Path) -> None:
        self.host = host
        self.port = port
        self.result_dir = result_dir

    @staticmethod
    def _get_result_dir() -> Path:
        for location in RESULT_DIR_LOCATIONS:
            if location.exists():
                logger.debug(f"Using result dir: {location}")
                return location

            logger.debug(f"Result dir does not exist: {location}")

        # user location does not exist, create the first one
        default_location = RESULT_DIR_LOCATIONS[0]
        default_location.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created result dir and using: {default_location}")
        return default_location

    @classmethod
    def get_config(
        cls,
        host: Optional[str] = None,
        port: Optional[int] = None,
        result_dir: Optional[Path] = None,
    ) -> "Config":
        """
        Get the configuration object
        """
        # now it's just a simple wrapper, but it'll load also the config from a file in the future
        return cls(
            host=host or HOST,
            port=port or PORT,
            result_dir=result_dir or cls._get_result_dir(),
        )
