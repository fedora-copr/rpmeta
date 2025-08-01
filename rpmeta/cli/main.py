import logging
import sys
from pathlib import Path
from typing import Any, Optional

import click

from rpmeta.cli.ctx import Context
from rpmeta.cli.run import run
from rpmeta.config import ConfigManager


def _get_context_settings() -> dict[str, Any]:
    return {"help_option_names": ["-h", "--help"]}


@click.group("rpmeta", context_settings=_get_context_settings())
@click.option(
    "-l",
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False),
    default="WARNING",
    show_default=True,
    help="Set the logging level",
)
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True, path_type=Path),
    default=None,
    help="Path to config file",
)
@click.pass_context
def entry_point(ctx: click.Context, log_level: str, config: Optional[Path]):
    """
    Predict build time duration of an RPM build based on available hardware resources.
    """
    app_config = ConfigManager.get_config(config_file=config)

    root_logger = logging.getLogger(__name__)
    root_logger.handlers.clear()

    log_level_value = log_level.upper()

    if app_config.logging.file:
        logging.basicConfig(
            level=log_level_value,
            format=app_config.logging.format,
            datefmt=app_config.logging.datefmt,
            filename=app_config.logging.file,
            filemode="a",
        )
        console = logging.StreamHandler()
        console.setLevel(log_level_value)
        console.setFormatter(
            logging.Formatter(app_config.logging.format, datefmt=app_config.logging.datefmt),
        )
        root_logger.addHandler(console)
    else:
        # Configure logging to console only
        logging.basicConfig(
            level=log_level_value,
            format=app_config.logging.format,
            datefmt=app_config.logging.datefmt,
        )

    root_logger.debug("Log level set to %s", log_level_value)
    root_logger.debug("Loaded configuration: %s", app_config.model_dump())

    ctx.ensure_object(Context)
    ctx.obj.config = app_config
    ctx.obj.log_level = log_level_value


def _register_optional_command(import_path: str, command_name: str) -> None:
    """
    Dynamically register a command if the import path exists.
    """
    try:
        logger = logging.getLogger(__name__)
        logger.debug("Attempting to register command: %s from %s", command_name, import_path)
        module = __import__(import_path, fromlist=[command_name])
        command = getattr(module, command_name)
        entry_point.add_command(command)
    except (ImportError, AttributeError):

        @entry_point.command(command_name)
        def placeholder_command():
            """Placeholder command, not available in this installation."""
            click.echo(f"Error: {command_name} command is not available in this installation.")
            click.echo("Install the required package to enable this command.")
            click.echo("Check the documentation for more details on how to install it.")
            sys.exit(1)


# base commands
entry_point.add_command(run)

# optional commands
_register_optional_command("rpmeta.cli.fetcher", "fetch_data")
_register_optional_command("rpmeta.cli.trainer", "train")


if __name__ == "__main__":
    entry_point()
