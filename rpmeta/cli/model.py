import json
import logging
from pathlib import Path
from typing import Optional

import click
from click import Path as ClickPath

from rpmeta.config import Config
from rpmeta.dataset import InputRecord
from rpmeta.model import Predictor

logger = logging.getLogger(__name__)


@click.group("model")
@click.option(
    "-m",
    "--model",
    type=ClickPath(exists=True, dir_okay=False, resolve_path=True, file_okay=True, path_type=Path),
    required=True,
    help="Path to the model file",
)
@click.option(
    "-c",
    "--categories",
    type=ClickPath(exists=True, dir_okay=False, resolve_path=True, file_okay=True, path_type=Path),
    required=True,
    help="Path to the categories file",
)
@click.pass_context
def model(ctx, model: Path, categories: Path):
    """
    Subcommand to collect model-related commands.

    The model is expected to be a joblib file, and the categories are expected to be in JSON
    format. The categories file should contain a mapping of categorical features to their
    possible values.

    The response of the model is a single integer representing the predicted build time duration
    in seconds. If the model does not recognize the package, it will return -1 as a prediction
    and log an error message.
    """
    ctx.ensure_object(dict)
    ctx.obj["predictor"] = Predictor.load(model, categories)


@model.command("serve")
@click.option("--host", type=str, default=None, help="Host to serve the API on")
@click.option(
    "-p",
    "--port",
    type=int,
    default=None,
    help="Port to serve the API on",
)
@click.pass_context
def serve(ctx, host: Optional[str], port: Optional[int]):
    """
    Start the API server on specified host and port.

    The server will accept HTTP GET requests with JSON payloads containing the input data in
    format:
        {
            "cpu_model_name": "AMD Ryzen 7 PRO 7840HS",
            "cpu_arch": "x86_64",
            "cpu_model": "116",
            "cpu_cores": 8,
            "ram": 123456789,
            "swap": 123456789,
            "package_name": "example-package",
            "epoch": 0,
            "version": "1.0.0",
            "mock_chroot": "fedora-42-x86_64"
        }

    The server will return a JSON response with the predicted build time duration in seconds in
    format:
        {
            "prediction": 1234
        }
    or -1 as the prediction if the package name is not recognized.
    """
    from rpmeta.server import app, reload_predictor

    reload_predictor(ctx.obj["predictor"])

    config = Config.get_config(host=host, port=port)

    logger.info(f"Serving on: {config.host}:{config.port}")
    app.run(host=config.host, port=config.port)


@model.command("predict")
@click.option(
    "-d",
    "--data",
    type=str,
    required=True,
    help="Input data to make prediction on (file path or JSON string)",
)
@click.option(
    "--output-type",
    type=click.Choice(["json", "text"], case_sensitive=False),
    default="text",
    show_default=True,
    help="Output type for the prediction",
)
@click.pass_context
def predict(ctx, data: str, output_type: str):
    """
    Make single prediction on the input data.

    WARNING: The model must be loaded into memory on each query. This mode is extremely
    inefficient for frequent real-time queries.

    The command accepts raw string in JSON format or Linux path to the JSON file in format:
        {
            "cpu_model_name": "AMD Ryzen 7 PRO 7840HS",
            "cpu_arch": "x86_64",
            "cpu_model": "116",
            "cpu_cores": 8,
            "ram": 123456789,
            "swap": 123456789,
            "package_name": "example-package",
            "epoch": 0,
            "version": "1.0.0",
            "mock_chroot": "fedora-42-x86_64"
        }

    Command response is in seconds.
    """
    if Path(data).exists():
        with open(data) as f:
            input_data = json.load(f)
    else:
        input_data = json.loads(data)

    logger.debug(f"Input data received: {input_data}")

    predictor = ctx.obj["predictor"]
    prediction = predictor.predict(InputRecord.from_data_frame(input_data))

    if output_type == "json":
        print(json.dumps({"prediction": prediction}))
    else:
        print(f"Prediction: {prediction}")
