import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import joblib
import numpy as np
import pandas as pd

from rpmeta.constants import DIVIDER
from rpmeta.dataset import InputRecord
from rpmeta.helpers import save_joblib

if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline


logger = logging.getLogger(__name__)


class Predictor:
    def __init__(self, model: "Pipeline", category_maps: dict[str, list[str]]) -> None:
        self.model = model
        self.category_maps = category_maps

    @classmethod
    def load(cls, model_path: Path, category_maps_path: Path) -> "Predictor":
        """
        Load the model from the given path and category maps from the given path.

        Args:
            model_path: The path to the model file
            category_maps_path: The path to the category maps file

        Returns:
            The loaded Predictor instance with the model and category maps
        """
        model = joblib.load(model_path)
        logger.info(f"Loaded model from {model_path}")

        with open(category_maps_path) as f:
            category_maps = json.load(f)
            logger.info(f"Loaded category maps from {category_maps_path}")

        return cls(model, category_maps)

    def _preprocess(self, input_data: InputRecord) -> pd.DataFrame:
        df = pd.DataFrame([input_data.to_data_frame()])

        for col, cat_list in self.category_maps.items():
            dtype = pd.CategoricalDtype(categories=cat_list, ordered=False)
            df[col] = df[col].astype(dtype)

        df["ram"] = np.round(df["ram"] / DIVIDER).astype(int)
        df["swap"] = np.round(df["swap"] / DIVIDER).astype(int)
        return df

    def predict(self, input_data: InputRecord) -> int:
        """
        Make prediction on the input data using the model and category maps.

        Args:
            input_data: The input data to make prediction on

        Returns:
            The prediction time in seconds
        """
        if input_data.package_name not in self.category_maps["package_name"]:
            logger.error(
                f"Package name {input_data.package_name} is not known. "
                "Please retrain the model with the new package name.",
            )
            return -1

        df = self._preprocess(input_data)
        pred = self.model.predict(df)
        return int(pred[0].item())

    def save(
        self,
        result_dir: Path,
        model_name: str = "model",
        category_maps_name: str = "category_maps",
    ) -> None:
        """
        Save the model and category maps to the given directory.

        Args:
            result_dir: The directory to save the model and category maps
            model_name: The name of the model file
            category_maps_name: The name of the category maps file
        """
        cat_file = result_dir / f"{category_maps_name}.json"
        if cat_file.exists():
            raise ValueError(f"File {cat_file} already exists, won't overwrite it")

        save_joblib(self.model, result_dir, model_name)

        with open(cat_file, "w") as f:
            json.dump(self.category_maps, f, indent=4)
            logger.info(f"Saved category maps to {cat_file}")
