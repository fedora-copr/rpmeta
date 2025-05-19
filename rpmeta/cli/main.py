import logging
from typing import Any

import click

from rpmeta.cli.fetcher import fetch_data
from rpmeta.cli.model import model
from rpmeta.cli.trainer import train

logging.basicConfig()
logger = logging.getLogger(__name__)


def _get_context_settings() -> dict[str, Any]:
    return {"help_option_names": ["-h", "--help"]}


@click.group("rpmeta", context_settings=_get_context_settings())
@click.option(
    "-l",
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False),
    default="ERROR",
    show_default=True,
    help="Set the logging level",
)
def entry_point(log_level: str):
    """
    Predict build time duration of an RPM build based on available hardware resources.
    """
    logger.setLevel(log_level.upper())
    logger.debug(f"Log level set to {log_level}")


entry_point.add_command(fetch_data)
entry_point.add_command(train)
entry_point.add_command(model)


if __name__ == "__main__":
    entry_point()
