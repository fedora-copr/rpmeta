from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from rpmeta.dataset import Record


def load_model(model_path: str) -> Pipeline:
    """
    Load the model from the given path.

    Args:
        model_path: The path to the model file

    Returns:
        The loaded model
    """
    return joblib.load(model_path)


def save_model(model: Pipeline, model_path: str) -> None:
    """
    Save the model to the given path.

    Args:
        model: The model to save
        model_path: The path to save the model
    """
    if Path(model_path).exists():
        raise ValueError(f"File {model_path} already exists, won't overwrite it")

    joblib.dump(model, model_path)


def make_prediction(model: Pipeline, input_data: Record) -> tuple[int, float]:
    """
    Make prediction on the input data using the model.

    Args:
        model: The trained model to make prediction with
        input_data: The input data to make prediction on

    Returns:
        The prediction time in seconds and confidence - how much the model is sure about the
        prediction
    """
    df = pd.DataFrame([input_data.to_data_frame()])
    prediction = model.predict(df, output_margin=True)

    std_dev = np.std(prediction)
    if std_dev == 0:
        confidence = 1
    else:
        confidence = (1 / (1 + np.exp(-prediction / std_dev)))[0].item()

    return prediction[0].item(), confidence
