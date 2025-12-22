import json
import logging
import math
from pathlib import Path

from sklearn.compose import TransformedTargetRegressor

from rpmeta.config import Config, ModelBehavior
from rpmeta.constants import ModelEnum, TimeFormat
from rpmeta.dataset import InputRecord
from rpmeta.model import get_model_by_name

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

        model_handler = get_model_by_name(model_name, config)
        model = model_handler.load_regressor(model_path)

        logger.info("Loading category maps from %s", category_maps_path)
        with open(category_maps_path, encoding="utf-8") as f:
            category_maps = json.load(f)

        return cls(model, category_maps, config)

    def predict(self, input_data: InputRecord, behavior: ModelBehavior) -> int:
        """
        Make prediction on the input data using the model and category maps.

        Args:
            input_data: The input data to make prediction on
            behavior: The model behavior configuration

        Returns:
            The prediction time in the configured time format (minutes by default),
            or -1 if the package name is unknown.
        """
        if input_data.package_name not in self.category_maps["package_name"]:
            logger.error(
                "Package name '%s' is not known. "
                "Please retrain the model with the new package name.",
                input_data.package_name,
            )
            return -1

        df = input_data.to_data_frame(self.category_maps)
        pred = self.model.predict(df)

        # extract scalar value from prediction array
        raw_prediction = pred[0]
        if hasattr(raw_prediction, "item"):
            minutes = int(raw_prediction.item())
        else:
            minutes = int(raw_prediction)

        if minutes <= 0:
            # this really shouldn't happen, since the model is trained on positive values
            # but you never know...
            logger.warning(
                "Model prediction returned non-positive value (%d), clamping to 1. Data: %s",
                minutes,
                input_data.to_model_dict(),
            )
            minutes = 1

        if behavior.time_format == TimeFormat.SECONDS:
            return minutes * 60
        if behavior.time_format == TimeFormat.MINUTES:
            return minutes
        if behavior.time_format == TimeFormat.HOURS:
            return math.ceil(minutes / 60)

        logger.warning(
            "Unknown time format '%s'. Returning minutes as default.",
            behavior.time_format,
        )
        return minutes
