import json
import logging
from pathlib import Path
from typing import Optional

import click
from click import Path as ClickPath

from rpmeta.cli.ctx import Context
from rpmeta.constants import ModelEnum
from rpmeta.dataset import InputRecord
from rpmeta.predictor import Predictor

logger = logging.getLogger(__name__)


@click.group("model")
@click.option(
    "-m",
    "--model-dir",
    type=ClickPath(exists=True, dir_okay=True, resolve_path=True, file_okay=False, path_type=Path),
    required=True,
    help="Path to the model directory",
)
@click.option(
    "-n",
    "--model-name",
    type=click.Choice(ModelEnum, case_sensitive=False),
    required=True,
    help="Type of the model to use",
)
@click.option(
    "-c",
    "--categories",
    type=ClickPath(exists=True, dir_okay=False, resolve_path=True, file_okay=True, path_type=Path),
    required=True,
    help="Path to the categories file",
)
@click.pass_context
def model(ctx: click.Context, model_dir: Path, model_name: ModelEnum, categories: Path):
    """
    Subcommand to collect model-related commands.

    The model is expected to be a joblib file, and the categories are expected to be in JSON
    format. The categories file should contain a mapping of categorical features to their
    possible values.

    The response of the model is a single integer representing the predicted build time duration
    in minutes by default. If the model does not recognize the package, it will return -1 as
    a prediction and log an error message.
    """
    ctx.ensure_object(Context)
    ctx.obj.predictor = Predictor.load(model_dir, model_name, categories, ctx.obj.config)


@model.command("serve")
@click.option("--host", type=str, default=None, help="Host to serve the API on")
@click.option(
    "-p",
    "--port",
    type=int,
    default=None,
    help="Port to serve the API on",
)
@click.option(
    "--debug",
    is_flag=True,
    default=False,
    help="Enable debug mode",
)
@click.pass_context
def serve(ctx: click.Context, host: Optional[str], port: Optional[int], debug: bool):
    """
    Start the API server on specified host and port.

    The server will accept HTTP POST requests with JSON payloads containing the input data in
    format:
        {
            "package_name": "example-package",
            "epoch": 0,
            "version": "1.0.0",
            "mock_chroot": "fedora-42-x86_64",
            "hw_info": {
                "cpu_model_name": "Intel Xeon Processor (Cascadelake)",
                "cpu_arch": "x86_64",
                "cpu_model": "85",
                "cpu_cores": 6,
                "ram": 15324520,
                "swap": 8388604
            }
        }

    The server will return a JSON response with the predicted build time duration in minutes
    (by default) in the following format:
        {
            "prediction": 21
        }
    or -1 as the prediction if the package name is not recognized.
    """
    from uvicorn.config import Config as UvicornConfig
    from uvicorn.server import Server

    from rpmeta.server.api import app, reload_predictor

    reload_predictor(ctx.obj.predictor)

    config = ctx.obj.config

    if host:
        config.api.host = host
    if port:
        config.api.port = port
    if debug:
        config.api.debug = debug

    logger.info("Serving on: %s:%s", config.api.host, config.api.port)

    uvicorn_config = UvicornConfig(
        app=app,
        host=config.api.host,
        port=config.api.port,
        log_level=ctx.obj.log_level.lower() if ctx.obj.log_level else "info",
    )
    server = Server(config=uvicorn_config)
    server.run()


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
def predict(ctx: click.Context, data: str, output_type: str):
    """
    Make single prediction on the input data.

    WARNING: The model must be loaded into memory on each query. This mode is extremely
    inefficient for frequent real-time queries.

    The command accepts raw string in JSON format or Linux path to the JSON file in format:
        {
            "package_name": "example-package",
            "epoch": 0,
            "version": "1.0.0",
            "mock_chroot": "fedora-42-x86_64",
            "hw_info": {
                "cpu_model_name": "AMD Ryzen 7 PRO 7840HS",
                "cpu_arch": "x86_64",
                "cpu_model": "116",
                "cpu_cores": 8,
                "ram": 123456789,
                "swap": 123456789
            }
        }

    Command response is in minutes by default.
    """
    if Path(data).exists():
        with open(data) as f:
            input_data = json.load(f)
    else:
        input_data = json.loads(data)

    logger.debug("Input data received: %s", input_data)

    prediction = ctx.obj.predictor.predict(InputRecord(**input_data), ctx.obj.config.model.behavior)

    if output_type == "json":
        print(json.dumps({"prediction": prediction}))
    else:
        print(f"Prediction: {prediction}")
