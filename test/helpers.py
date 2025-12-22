import logging
import subprocess

logger = logging.getLogger(__name__)


def run_rpmeta_cli(params: list[str]) -> subprocess.CompletedProcess:
    cmd = ["python3", "-m", "rpmeta.cli.main", "--log-level", "DEBUG", *params]
    logger.debug("Running command: %s", " ".join(cmd))
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )


def power_of_10(x: int) -> int:
    return 10**x
