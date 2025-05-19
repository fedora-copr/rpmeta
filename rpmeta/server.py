import logging

from flask import Flask, jsonify, request

from rpmeta.dataset import InputRecord
from rpmeta.model import Predictor

logger = logging.getLogger(__name__)

app = Flask(__name__)

# TODO: no validation and error handling yet. Implement it or use FastAPI??


@app.route("/predict", methods=["GET"])
def predict_endpoint():
    """
    Endpoint to make prediction on the input data
    """
    if request.content_type != "application/json":
        return "Invalid Content-Type", 400

    try:
        data = request.get_json()
        logger.debug(f"Received data: {data}")

        input_record = InputRecord.from_data_frame(data)
        prediction = predictor.predict(input_record)
        return jsonify({"prediction": prediction})
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return "Internal Server Error", 500


predictor = None


def reload_predictor(new_predictor: Predictor) -> None:
    """
    Reload the model and categories map from the given path for the API server.
    """
    global predictor
    logger.info("Reloading predictor")
    predictor = new_predictor
    logger.info("Predictor reloaded")
