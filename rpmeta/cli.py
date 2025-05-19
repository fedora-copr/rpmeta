import getpass
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import click
import pandas as pd
from click import DateTime
from click import Path as ClickPath

from rpmeta.constants import HOST, PORT
from rpmeta.dataset import InputRecord, Record
from rpmeta.model import Predictor

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
    Predict build time duration of an RPM build based on available hardware resources.
    """
    if debug:
        logger.setLevel(logging.DEBUG)


@entry_point.command("serve")
@click.option("--host", type=str, default=HOST, show_default=True, help="Host to serve the API on")
@click.option(
    "-p",
    "--port",
    type=int,
    default=PORT,
    show_default=True,
    help="Port to serve the API on",
)
@click.option(
    "-m",
    "--model",
    type=ClickPath(exists=True, dir_okay=False, resolve_path=True, file_okay=True, path_type=Path),
    required=True,
    help="Path (URL or Linux path) to the model file",
)
@click.option(
    "-c",
    "--categories",
    type=ClickPath(exists=True, dir_okay=False, resolve_path=True, file_okay=True, path_type=Path),
    required=True,
    help="Path (URL or Linux path) to the categories file",
)
def serve(host: str, port: int, model: Path, categories: Path):
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
    """
    from rpmeta.server import app, reload_predictor

    reload_predictor(model, categories)

    logger.info(f"Serving on: {host}:{port}")
    app.run(host=host, port=port)


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
    "--model",
    type=ClickPath(exists=True, dir_okay=False, resolve_path=True, file_okay=True, path_type=Path),
    required=True,
    help="Path (URL or Linux path) to the model file",
)
@click.option(
    "-c",
    "--categories",
    type=ClickPath(exists=True, dir_okay=False, resolve_path=True, file_okay=True, path_type=Path),
    required=True,
    help="Path (URL or Linux path) to the categories file",
)
@click.option(
    "--output-type",
    type=click.Choice(["json", "text"], case_sensitive=False),
    default="text",
    show_default=True,
    help="Output type for the prediction",
)
def predict(data: str, model: Path, categories: Path, output_type: str):
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

    predictor = Predictor.load(model, categories)
    prediction = predictor.predict(InputRecord.from_data_frame(input_data))

    if output_type == "json":
        print(json.dumps({"prediction": prediction}))
    else:
        print(f"Prediction: {prediction}")


@entry_point.command("train")
@click.option(
    "-d",
    "--dataset",
    type=ClickPath(exists=True, dir_okay=False, resolve_path=True, file_okay=True, path_type=Path),
    required=True,
    help="Path to the dataset file",
)
@click.option(
    "-s",
    "--result-dir",
    type=ClickPath(
        dir_okay=True,
        file_okay=False,
        writable=True,
        resolve_path=True,
        path_type=Path,
    ),
    required=True,
    help="Result directory to save the model",
)
@click.option(
    "-m",
    "--model-allowlist",
    # NOTE: update this manually, get_all_model_names can't be currently imported due to
    # the modularity
    type=click.Choice(["lightgbm", "xgboost"], case_sensitive=False),
    multiple=True,
    default=["lightgbm", "xgboost"],
    show_default=True,
    callback=lambda _, __, values: set(values) if values else None,
    help="List of models to train",
)
@click.option(
    "--optuna",
    is_flag=True,
    help="Use Optuna for hyperparameter tuning instead of default parameters",
)
def train(dataset: Path, result_dir: Path, model_allowlist: set[str], optuna: bool):
    """
    Train the model on the input dataset.
    """
    from rpmeta.train.trainer import ModelTrainer
    from rpmeta.train.visualizer import ResultsHandler

    trainer = ModelTrainer(
        data=pd.read_json(dataset),
        model_allowlist=model_allowlist,
        result_dir=result_dir,
    )

    if optuna:
        all_results, best_models, studies = trainer.run_all_studies(n_trials=100)
        result_handler = ResultsHandler(
            all_trials=all_results,
            best_models=best_models,
            studies=studies,
            X_test=trainer.X_test,
            y_test=trainer.y_test,
        )
        result_handler.run_all()
    else:
        print(*trainer.run(result_dir), sep="\n")


@entry_point.command("fetch-data")
@click.option(
    "-p",
    "--path",
    type=ClickPath(exists=False, dir_okay=False, resolve_path=True, path_type=Path),
    required=True,
    help="Path to save the fetched data",
)
@click.option(
    "-s",
    "--start-date",
    type=DateTime(formats=["%Y-%m-%d"]),
    default=None,
    help="Start date for fetching data",
)
@click.option(
    "-e",
    "--end-date",
    type=DateTime(formats=["%Y-%m-%d"]),
    default=None,
    help="End date for fetching data",
)
@click.option("--copr", is_flag=True, help="Fetch data from COPR")
@click.option(
    "--is-copr-instance",
    is_flag=True,
    help=(
        "If script is running on Copr instance (e.g. Copr container instance) with current"
        " database dump (https://copr.fedorainfracloud.org/db_dumps/), include this flag"
    ),
)
@click.option("--koji", is_flag=True, help="Fetch data from Koji")
def fetch_data(
    path: Path,
    start_date: Optional[datetime],
    end_date: Optional[datetime],
    copr: bool,
    is_copr_instance: bool,
    koji: bool,
):
    """
    Fetch the dataset from desired build systems (Copr, Koji) and save it to the specified path.
    """
    from rpmeta.fetcher import CoprFetcher, KojiFetcher

    if not (copr or koji):
        raise click.UsageError("At least one of --copr or --koji must be provided")

    if not copr and is_copr_instance:
        raise click.UsageError("Flag --is-copr-instance can only be used with --copr")

    if copr and is_copr_instance and (os.getuid() == 0 or getpass.getuser() != "copr-fe"):
        logger.error("CoprFetcher should be run as the 'copr-fe' user inside Copr instance")
        raise click.UsageError(
            "CoprFetcher should be run as the 'copr-fe' user. Please run:\n"
            "$ sudo -u copr-fe rpmeta fetch-data ...",
        )

    fetched_data = []
    if koji:
        koji_fetcher = KojiFetcher(start_date, end_date)
        fetched_data.extend(koji_fetcher.fetch_data())

    if copr:
        copr_fetcher = CoprFetcher(start_date, end_date, is_copr_instance)
        fetched_data.extend(copr_fetcher.fetch_data())

    with open(path, "w") as f:
        logger.info(f"Saving data to: {path}")
        json.dump(fetched_data, f, indent=4, default=Record.to_data_frame)
        logger.info(f"Data saved to: {path}")


if __name__ == "__main__":
    entry_point()
