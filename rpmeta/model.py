from pathlib import Path

import joblib
import pandas as pd


def load_model(model_path: str):
    """
    Load the model from the given path.

    Args:
        model_path: The path to the model file

    Returns:
        The loaded model
    """
    return joblib.load(model_path)


def save_model(model, model_path: str):
    """
    Save the model to the given path.

    Args:
        model: The model to save
        model_path: The path to save the model
    """
    if Path(model_path).exists():
        raise ValueError(f"File {model_path} already exists, won't overwrite it")

    joblib.dump(model, model_path)


def make_prediction(model, input_data) -> tuple[int, float]:
    """
    Make prediction on the input data using the model.

    Args:
        model: The trained model to make prediction with
        input_data: The input data to make prediction on

    Returns:
        The prediction time in seconds and certainty - how much the model is sure about the
        prediction
    """
    df = pd.DataFrame([input_data])
    prediction = model.predict(df)
    probability = model.predict_proba(df)
    return prediction[0], probability[0]
