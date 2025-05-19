import logging
from pathlib import Path

import joblib

logger = logging.getLogger(__name__)


def save_joblib(obj: object, result_dir: Path, filename: str) -> Path:
    """
    Save an object to a file using joblib.

    Args:
        obj: The object to save
        result_dir: The directory to save the object
        filename: The name of the file to save the object to

    Returns:
        The path to the saved file
    """
    if not result_dir.is_dir():
        raise ValueError(f"{result_dir} is not a directory")

    if not result_dir.exists():
        result_dir.mkdir(parents=True, exist_ok=True)

    path = result_dir / f"{filename}.joblib"
    if path.exists():
        raise ValueError(f"File {path} already exists, won't overwrite it")

    joblib.dump(obj, path)
    logger.info(f"Saved {obj.__class__.__name__} to {path}")
    return path
