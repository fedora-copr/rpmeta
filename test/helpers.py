import logging
import subprocess

logger = logging.getLogger(__name__)


def run_rpmeta_cli(params: list[str]) -> subprocess.CompletedProcess:
    cmd = ["python3", "-m", "rpmeta.cli.main", "--log-level", "DEBUG", *params]
    logger.debug(f"Running command: {' '.join(cmd)}")
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )
