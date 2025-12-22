import json
from pathlib import Path

import click
import pandas as pd
from click import Path as ClickPath

from rpmeta.cli.ctx import Context
from rpmeta.constants import ModelEnum
from rpmeta.trainer.trainer import ModelTrainingManager
from rpmeta.trainer.visualizer import ResultsHandler


@click.group("train")
@click.option(
    "-d",
    "--dataset",
    type=ClickPath(exists=True, dir_okay=False, resolve_path=True, file_okay=True, path_type=Path),
    required=True,
    help="Path to the dataset file",
)
@click.option(
    "-r",
    "--result-dir",
    type=ClickPath(
        dir_okay=True,
        file_okay=False,
        writable=True,
        resolve_path=True,
        path_type=Path,
    ),
    default=None,
    help="Result directory to save relevant data",
)
@click.option(
    "-m",
    "--model-allowlist",
    type=click.Choice(ModelEnum, case_sensitive=False),
    multiple=True,
    default=set(ModelEnum.get_all_model_names()),
    show_default=True,
    callback=lambda _, __, values: set(values) if values else None,
    help="List of models to train",
)
@click.pass_context
def train(
    ctx: click.Context,
    dataset: Path,
    result_dir: Path | None,
    model_allowlist: set[ModelEnum],
):
    """
    Subcommand to train the desired models on the input dataset.
    """
    ctx.ensure_object(Context)

    config = ctx.obj.config
    if result_dir:
        config.result_dir = result_dir

    with open(dataset, encoding="utf-8") as f:
        data = json.load(f)

    trainer = ModelTrainingManager(
        data=pd.json_normalize(data),
        model_allowlist=model_allowlist,
        config=config,
    )
    ctx.obj.trainer = trainer


@train.command("tune")
@click.option(
    "-n",
    "--n-trials",
    type=int,
    default=100,
    show_default=True,
    help="Number of trials for Optuna hyperparameter tuning",
)
@click.pass_context
def tune(ctx: click.Context, n_trials: int):
    """
    Run hyperparameter tuning for all models in the allowlist using Optuna framework.
    """
    trainer = ctx.obj.trainer
    all_results, best_models, studies = trainer.run_all_studies(n_trials=n_trials)
    result_handler = ResultsHandler(
        all_trials=all_results,
        best_models=best_models,
        studies=studies,
        X_test=trainer.X_test,
        y_test=trainer.y_test,
        config=ctx.obj.config,
    )
    result_handler.run_all()


@train.command("run")
@click.pass_context
def run(ctx: click.Context):
    """
    Run the model training on pre-defined hyperparameters.
    """
    print(*ctx.obj.trainer.run(), sep="\n")
