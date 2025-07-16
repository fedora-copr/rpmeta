import logging
from typing import Any

import click

from rpmeta.cli.fetcher import fetch_data
from rpmeta.cli.model import model
from rpmeta.cli.trainer import train


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
@click.pass_context
def entry_point(ctx, log_level: str):
    """
    Predict build time duration of an RPM build based on available hardware resources.
    """
    root_logger = logging.getLogger(__name__)
    root_logger.handlers.clear()

    logging.basicConfig(level=log_level.upper(), datefmt="[%H:%M:%S]")

    root_logger.debug("Log level set to %s", log_level)

    ctx.ensure_object(dict)
    ctx.obj["log_level"] = log_level.upper()


entry_point.add_command(fetch_data)
entry_point.add_command(train)
entry_point.add_command(model)


if __name__ == "__main__":
    entry_point()
