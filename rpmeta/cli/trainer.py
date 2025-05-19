from pathlib import Path

import click
import pandas as pd
from click import Path as ClickPath


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
@click.pass_context
def train(ctx, dataset: Path, result_dir: Path, model_allowlist: set[str]):
    """
    Subcommand to train the desired models on the input dataset.
    """
    from rpmeta.train.trainer import ModelTrainer

    ctx.ensure_object(dict)

    trainer = ModelTrainer(
        data=pd.read_json(dataset),
        model_allowlist=model_allowlist,
        result_dir=result_dir,
    )
    ctx.obj["trainer"] = trainer


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
def tune(ctx, n_trials: int):
    """
    Run hyperparameter tuning for all models in the allowlist using Optuna framework.
    """
    from rpmeta.train.visualizer import ResultsHandler

    trainer = ctx.obj["trainer"]
    all_results, best_models, studies = trainer.run_all_studies(n_trials=n_trials)
    result_handler = ResultsHandler(
        all_trials=all_results,
        best_models=best_models,
        studies=studies,
        X_test=trainer.X_test,
        y_test=trainer.y_test,
    )
    result_handler.run_all()


@train.command("run")
@click.pass_context
def run(ctx):
    """
    Run the model training on pre-defined hyperparameters.
    """
    trainer = ctx.obj["trainer"]
    print(*trainer.run(trainer.result_dir), sep="\n")
