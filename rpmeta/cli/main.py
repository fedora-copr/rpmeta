import logging
from pathlib import Path
from typing import Any, Optional

import click

from rpmeta.cli.ctx import Context
from rpmeta.cli.fetcher import fetch_data
from rpmeta.cli.model import model
from rpmeta.cli.trainer import train
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
        logging.getLogger("").addHandler(console)
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


entry_point.add_command(fetch_data)
entry_point.add_command(train)
entry_point.add_command(model)


if __name__ == "__main__":
    entry_point()
