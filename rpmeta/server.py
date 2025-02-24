import asyncio
import logging

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.routing import Route

from rpmeta.dataset import InputRecord
from rpmeta.model import load_model, make_prediction

logger = logging.getLogger(__name__)

# TODO: no validation and error handling yet. Implement it or use FastAPI??


async def predict_endpoint(request: Request) -> JSONResponse:
    """
    Endpoint to make prediction on the input data
    """
    if request.headers.get("Content-Type") != "application/json":
        return PlainTextResponse("Invalid Content-Type", status_code=400)

    data = await request.json()
    logger.debug(f"Received data: {data}")
    # not actually async, but this way it won't block the starlette's event loop
    prediction = await asyncio.to_thread(
        make_prediction,
        model,
        InputRecord.from_data_frame(data),
    )
    return JSONResponse({"prediction": prediction})


routes = [Route("/predict", predict_endpoint, methods=["POST"])]
app = Starlette(routes=routes)


model = None


def reload_model(model_path: str) -> None:
    """
    Reload the model from the given path for the API server
    """
    global model
    logger.info(f"Reloading model from: {model_path}")
    model = load_model(model_path)
    logger.info(f"Model reloaded from: {model_path}")
