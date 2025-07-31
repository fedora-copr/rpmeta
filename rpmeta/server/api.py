import logging
from typing import Optional

from fastapi import APIRouter, FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from rpmeta import __version__
from rpmeta.config import ModelBehavior
from rpmeta.dataset import InputRecord
from rpmeta.predictor import Predictor

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
      "prediction": 5
    }
    ```

    The prediction is the estimated build time in minutes by default.
    """,
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Store the predictor instance globally - loaded once when server starts
# This ensures the model is loaded into RAM only once
predictor = None


class PredictionResponse(BaseModel):
    """
    Response model for the prediction endpoint.

    This contains the predicted build duration in desired time format.
    """

    prediction: int = Field(
        description="Predicted build duration in desired time format",
        gt=0,
        examples=[5, 30, 120],
    )
    used_configuration: ModelBehavior = Field(
        ...,
        description="Configuration used for the prediction",
    )


class PredictionRequest(InputRecord):
    """
    Input body for the prediction endpoint.

    This is used to pass the input data to the model for prediction with all
    the necessary configurations.
    """

    configuration: Optional[ModelBehavior] = Field(
        default=None,
        description="Optional configuration for the server or model",
    )


# API router for v1 endpoints
v1_router = APIRouter(prefix="/v1")


@v1_router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict build duration",
    description=(
        "Predicts the build duration for an RPM package based on hardware information"
        " and package metadata."
    ),
)
def predict_endpoint_v1(request_data: PredictionRequest) -> PredictionResponse:
    """
    Predict the build duration for an RPM package.

    This endpoint accepts package information and hardware specs and returns
    the predicted build duration in seconds.
    """

    logger.debug("Received request for prediction: %s", request_data)

    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model not initialized. Server not ready for predictions.",
        )

    # original model behavior before any changes
    model_behavior_before = predictor.config.model.behavior

    if request_data.configuration:
        update_behavior = request_data.configuration.model_dump(
            exclude_unset=True,
            exclude_none=True,
        )
        predictor.config.model.behavior = model_behavior_before.model_copy(update=update_behavior)

    package_data = InputRecord.model_validate(request_data)
    prediction = predictor.predict(package_data)
    logger.debug(
        "Prediction for %s: %s %s",
        package_data.package_name,
        prediction,
        predictor.config.model.behavior.time_format,
    )
    resp = PredictionResponse(
        prediction=prediction,
        used_configuration=predictor.config.model.behavior,
    )

    # restore the original behavior after prediction
    predictor.config.model.behavior = model_behavior_before
    return resp


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
)
def predict_endpoint(request_data: PredictionRequest) -> PredictionResponse:
    """
    Alias to the latest API version (currently v1).

    This is a convenience endpoint that forwards requests to the current stable API version.
    For new implementations, consider using the versioned endpoint directly.
    """
    return predict_endpoint_v1(request_data)


def reload_predictor(new_predictor: Predictor) -> None:
    """
    Reload the model and categories map for the API server.

    Args:
        new_predictor: The predictor instance to use
    """
    # This allows the model to be loaded into RAM only once when the server starts,
    # and can be called to reload the model if needed without restarting the server.
    global predictor
    logger.info("Reloading predictor")
    predictor = new_predictor
    logger.info("Predictor loaded successfully")
