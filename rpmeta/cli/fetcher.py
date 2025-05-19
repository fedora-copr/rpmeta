import getpass
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import click
from click import DateTime
from click import Path as ClickPath

from rpmeta.config import Config
from rpmeta.dataset import Record

logger = logging.getLogger(__name__)


@click.command("fetch-data")
@click.option(
    "-p",
    "--path",
    type=ClickPath(exists=False, dir_okay=False, resolve_path=True, path_type=Path),
    default=None,
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
@click.option(
    "-l",
    "--limit",
    type=int,
    default=10000,
    show_default=True,
    help="Limit the number of records to fetch on one page",
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
    path: Optional[Path],
    start_date: Optional[datetime],
    end_date: Optional[datetime],
    limit: int,
    copr: bool,
    is_copr_instance: bool,
    koji: bool,
):
    """
    Fetch the dataset from desired build systems (Copr, Koji).

    The dataset output is ready to be fed into the training process.
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
        koji_fetcher = KojiFetcher(start_date, end_date, limit)
        fetched_data.extend(koji_fetcher.fetch_data())

    if copr:
        copr_fetcher = CoprFetcher(start_date, end_date, is_copr_instance, limit)
        fetched_data.extend(copr_fetcher.fetch_data())

    path = (
        path
        or Config.get_config().result_dir
        / f"dataset_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
    )
    with open(path, "w") as f:
        logger.info(f"Saving data to: {path}")
        json.dump(fetched_data, f, indent=4, default=Record.to_data_frame)
        logger.info(f"Data saved to: {path}")
