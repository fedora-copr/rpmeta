import json
import logging
import math
from pathlib import Path

from sklearn.compose import TransformedTargetRegressor

from rpmeta.config import Config, ModelBehavior
from rpmeta.constants import ModelEnum, TimeFormat
from rpmeta.dataset import InputRecord
from rpmeta.model import Model, get_all_models

logger = logging.getLogger(__name__)


class Predictor:
    def __init__(
        self,
        model: TransformedTargetRegressor,
        category_maps: dict[str, list[str]],
        config: Config,
    ) -> None:
        self.model = model
        self.category_maps = category_maps
        self.config = config

    @staticmethod
    def _model_factory(model_name: ModelEnum) -> Model:
        for model in get_all_models():
            if model.name == model_name:
                return model

        raise ValueError(f"Model {model_name} is not supported.")

    @classmethod
    def load(
        cls,
        model_path: Path,
        model_name: ModelEnum,
        category_maps_path: Path,
        config: Config,
    ) -> "Predictor":
        """
        Load the model from the given path and category maps from the given path.

        Args:
            model_path: The path to the model directory
            model_name: The name of the model type
            category_maps_path: The path to the category maps file
            config: The configuration to use

        Returns:
            The loaded Predictor instance with the model and category maps
        """
        logger.info("Loading model %s from %s", model_name, model_path)

        model = cls._model_factory(model_name).load_regressor(model_path)

        logger.info("Loading category maps from %s", category_maps_path)
        with open(category_maps_path) as f:
            category_maps = json.load(f)

        return cls(model, category_maps, config)

    def predict(self, input_data: InputRecord, behavior: ModelBehavior) -> int:
        """
        Make prediction on the input data using the model and category maps.

        Args:
            input_data: The input data to make prediction on
            behavior: The model behavior configuration

        Returns:
            The prediction time in minutes by default
        """
        if input_data.package_name not in self.category_maps["package_name"]:
            logger.error(
                f"Package name {input_data.package_name} is not known. "
                "Please retrain the model with the new package name.",
            )
            return -1

        df = input_data.to_data_frame(self.category_maps)
        pred = self.model.predict(df)
        minutes = int(pred[0].item())

        if minutes <= 0:
            # this really shouldn't happen, since the model is trained on positive values
            # but you never know...
            logger.error("Model prediction returned a negative value, which is invalid.")
            minutes = 1

        if behavior.time_format == TimeFormat.SECONDS:
            return minutes * 60
        if behavior.time_format == TimeFormat.MINUTES:
            return minutes
        if behavior.time_format == TimeFormat.HOURS:
            return math.ceil(minutes / 60)

        logger.error(
            f"Unknown time format {behavior.time_format}. Returning minutes as default.",
        )
        return minutes
