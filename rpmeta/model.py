from pathlib import Path
from typing import TYPE_CHECKING

import joblib
import pandas as pd

from rpmeta.dataset import InputRecord

if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline


def load_model(model_path: str) -> "Pipeline":
    """
    Load the model from the given path.

    Args:
        model_path: The path to the model file

    Returns:
        The loaded model
    """
    return joblib.load(model_path)


def save_model(model: "Pipeline", model_path: str) -> None:
    """
    Save the model to the given path.

    Args:
        model: The model to save
        model_path: The path to save the model
    """
    if Path(model_path).exists():
        raise ValueError(f"File {model_path} already exists, won't overwrite it")

    joblib.dump(model, model_path)


def make_prediction(model: "Pipeline", input_data: InputRecord) -> int:
    """
    Make prediction on the input data using the model.

    Args:
        model: The trained model to make prediction with
        input_data: The input data to make prediction on

    Returns:
        The prediction time in seconds
    """
    df = pd.DataFrame([input_data.to_data_frame()])
    prediction = model.predict(df, output_margin=True)
    return int(prediction[0].item())
