import logging
import math

logger = logging.getLogger(__name__)


def to_minutes_rounded(seconds: int) -> int:
    """
    Convert seconds to minutes, rounding up.

    Args:
        seconds: The time in seconds

    Returns:
        The time in minutes, rounded up
    """
    return math.ceil(seconds / 60)
