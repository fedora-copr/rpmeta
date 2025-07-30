from dataclasses import dataclass
from typing import Optional

from rpmeta.config import Config
from rpmeta.predictor import Predictor
from rpmeta.trainer.trainer import ModelTrainingManager


@dataclass
class Context:
    """
    Context for the CLI commands.
    """

    config: Optional[Config] = None
    predictor: Optional[Predictor] = None
    log_level: Optional[str] = None
    trainer: Optional[ModelTrainingManager] = None
