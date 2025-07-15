import logging

from fastapi import FastAPI, HTTPException, Request, status

from rpmeta.dataset import InputRecord
from rpmeta.model import Predictor

logger = logging.getLogger(__name__)

app = FastAPI(
    title="RPMeta API",
    description="API for predicting build times for RPM packages",
    version="1.0.0",
)

# Store the predictor instance globally - loaded once when server starts
# This ensures the model is loaded into RAM only once
predictor = None


@app.get("/predict")
async def predict_endpoint(request: Request) -> dict[str, int]:
    """
    Endpoint to make prediction on the input data.
    Uses synchronous processing as async isn't recommended for model inference.
    """
    if request.headers.get("content-type") != "application/json":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid Content-Type, expected application/json",
        )

    try:
        data = await request.json()
        logger.debug(f"Received data: {data}")

        input_record = InputRecord.from_data_frame(data)
        prediction = predictor.predict(input_record)
        return {"prediction": prediction}
    except Exception as e:
        # TODO: better error handling
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal Server Error",
        )


def reload_predictor(new_predictor: Predictor) -> None:
    """
    Reload the model and categories map for the API server.
    """
    # This allows the model to be loaded into RAM only once when the server starts,
    # and can be called to reload the model if needed without restarting the server.
    global predictor
    logger.info("Reloading predictor")
    predictor = new_predictor
    logger.info("Predictor loaded successfully")
