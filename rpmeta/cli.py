import json
import logging
from datetime import datetime
from os import PathLike
from pathlib import Path
from typing import Any, Optional

import click

from rpmeta.constants import HOST, PORT
from rpmeta.model import load_model, make_prediction

logging.basicConfig()
logger = logging.getLogger()


def _get_context_settings() -> dict[str, Any]:
    return {"help_option_names": ["-h", "--help"]}


@click.group("rpmeta", context_settings=_get_context_settings())
@click.option(
    "-d",
    "--debug",
    is_flag=True,
    default=False,
    help="Enable debugging logs",
)
def entry_point(debug: bool):
    """
    Integration scripts for the RPM build prediction tool
    """
    if debug:
        logger.setLevel(logging.DEBUG)


@entry_point.command("serve")
@click.option("--host", type=str, default=HOST, help="Host to serve the API on")
@click.option("-p", "--port", type=int, default=PORT, help="Port to serve the API on")
@click.option(
    "-m",
    "--model-path",
    type=PathLike,
    required=True,
    help="Path to the model file",
)
def serve(host: str, port: int, model_path: str):
    """
    Start the API server
    """
    import uvicorn

    from rpmeta.server import app, reload_model

    reload_model(model_path)

    logger.info(f"Serving on: {host}:{port}")
    uvicorn.run(app, host=host, port=port)


@entry_point.command("predict")
@click.option(
    "-d",
    "--data",
    type=str,
    required=True,
    help="Input data to make prediction on (file path or JSON string)",
)
@click.option(
    "-m",
    "--model-path",
    type=PathLike,
    required=True,
    help="Path to the model file",
)
def predict(data: str, model_path: str):
    """
    Make single prediction on the input data
    """
    if Path(data).exists():
        with open(data) as f:
            input_data = json.load(f)
    else:
        input_data = json.loads(data)

    model = load_model(model_path)
    prediction, certainty = make_prediction(model, input_data)
    print(f"Prediction: {prediction}, Certainty: {certainty}")


@entry_point.command("train")
@click.option(
    "-d",
    "--dataset-path",
    type=PathLike,
    required=True,
    help="Path to the dataset file",
)
@click.option(
    "-s",
    "--destination-path",
    type=PathLike,
    required=True,
    help="Path to save the model",
)
@click.option(
    "-r",
    "--random-state",
    type=int,
    default=42,
    help="Random state for the model",
)
def train(dataset_path: str, destination_path: str, random_state: int):
    """
    Train the model on the input dataset
    """
    from rpmeta.train import Trainer

    trainer = Trainer(dataset_path=dataset_path, random_state=random_state)
    trainer.train()
    trainer.save(destination_path)


@entry_point.command("fetch-data")
@click.option("-p", "--path", type=PathLike, required=True, help="Path to save the data")
@click.option(
    "-s",
    "--start-date",
    type=datetime,
    default=None,
    help="Start date for fetching data",
)
@click.option(
    "-e",
    "--end-date",
    type=datetime,
    default=None,
    help="End date for fetching data",
)
@click.option("--copr", is_flag=True, help="Fetch data from COPR")
@click.option(
    "--is-copr-instance",
    is_flag=True,
    default=False,
    help=(
        "If script is running on Copr instance (e.g, Copr container instance), " "include this flag"
    ),
)
@click.option("--koji", is_flag=True, help="Fetch data from Koji")
def fetch_data(
    path: str,
    start_date: Optional[datetime],
    end_date: Optional[datetime],
    copr: bool,
    is_copr_instance: bool,
    koji: bool,
):
    """
    Fetch the dataset from desired build systems
    """
    from rpmeta.fetcher import CoprFetcher, KojiFetcher

    if Path(path).exists():
        raise click.UsageError(f"File {path} already exists, won't overwrite it")

    if not (copr or koji):
        raise click.UsageError("At least one of --copr or --koji must be provided")

    if not copr and is_copr_instance:
        raise click.UsageError("Flag --is-copr-instance can only be used with --copr")

    fetched_data = []
    if koji:
        koji_fetcher = KojiFetcher(start_date, end_date)
        fetched_data.extend(koji_fetcher.fetch_data())

    if copr:
        copr_fetcher = CoprFetcher(start_date, end_date, is_copr_instance)
        fetched_data.extend(copr_fetcher.fetch_data())

    with open(path, "w") as f:
        logger.info(f"Saving data to: {path}")
        json.dump(fetched_data, f, indent=4)
        logger.info(f"Data saved to: {path}")


if __name__ == "__main__":
    entry_point()
