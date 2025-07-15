import logging

from fastapi import APIRouter, FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from rpmeta.dataset import InputRecord
from rpmeta.model import Predictor

logger = logging.getLogger(__name__)

app = FastAPI(
    title="RPMeta API",
    description="""
    API for predicting build times for RPM packages based on hardware information
    and package metadata.

    ## Overview

    This API allows you to predict how long it will take to build an RPM package
    based on the hardware specifications of the build machine and package information.

    ## Usage

    Send a POST request to `/predict` with the necessary package and hardware information.

    ### Example Input
    ```json
    {
      "package_name": "rust-winit",
      "epoch": 0,
      "version": "0.30.8",
      "mock_chroot": "fedora-41-x86_64",
      "hw_info": {
        "cpu_model_name": "Intel Xeon Processor (Cascadelake)",
        "cpu_arch": "x86_64",
        "cpu_model": "85",
        "cpu_cores": 6,
        "ram": 15324520,
        "swap": 8388604
      }
    }
    ```

    ### Example Response
    ```json
    {
      "prediction": 283
    }
    ```

    The prediction is the estimated build time in seconds.
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Store the predictor instance globally - loaded once when server starts
# This ensures the model is loaded into RAM only once
predictor = None


class PredictionResponse(BaseModel):
    """
    Response model for the prediction endpoint.

    This contains the predicted build duration in seconds.
    """

    prediction: int = Field(
        description="Predicted build duration in seconds",
        gt=0,
        examples=[283, 1800, 7200],
    )


# API router for v1 endpoints
v1_router = APIRouter(prefix="/v1")

_responses_v1 = {
    200: {"description": "Successful prediction"},
    422: {"description": "Validation error - Invalid input data"},
    500: {"description": "Server error - Model not initialized"},
}


@v1_router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict build duration",
    description=(
        "Predicts the build duration for an RPM package based on hardware information"
        " and package metadata."
    ),
    responses=_responses_v1,
)
def predict_endpoint_v1(input_record: InputRecord) -> PredictionResponse:
    """
    Predict the build duration for an RPM package.

    This endpoint accepts package information and hardware specs and returns
    the predicted build duration in seconds.
    """

    logger.debug(f"Received request for prediction: {input_record}")

    if predictor is None:
        logger.error("Predictor not initialized")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model not initialized. Server not ready for predictions.",
        )

    prediction = predictor.predict(input_record)
    logger.debug(f"Prediction for {input_record.package_name}: {prediction} seconds")
    return PredictionResponse(prediction=prediction)


app.include_router(v1_router)


# Add an alias endpoint at the root path that redirects to the latest version (currently v1)
@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict build duration (Latest API version)",
    description=(
        "Alias to the latest API version (currently v1). Predicts the build duration for "
        "an RPM package based on hardware information and package metadata."
    ),
    responses=_responses_v1,
)
def predict_endpoint(input_record: InputRecord) -> PredictionResponse:
    """
    Alias to the latest API version (currently v1).

    This is a convenience endpoint that forwards requests to the current stable API version.
    For new implementations, consider using the versioned endpoint directly.
    """
    return predict_endpoint_v1(input_record)


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
