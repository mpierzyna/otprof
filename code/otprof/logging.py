"""
Following https://github.com/mCodingLLC/VideosSampleCode/tree/master/videos/135_modern_logging
"""

import contextlib
import enum
import logging
from typing import Optional


class LogContext(enum.Enum):
    """Logging modules for different contexts of the project.
    Do not log by component or submodule."""

    data = "data"
    train = "train"
    eval = "eval"
    report = "report"


def get_logger(context: Optional[LogContext] = None) -> logging.Logger:
    """Get a logger for a context. Avoid making it too granular!"""
    if context is None:
        context = "otprof"
    else:
        context = f"otprof.{context.value}"

    logger = logging.getLogger(context)
    return logger


@contextlib.contextmanager
def warnings_to_logger(logger: logging.Logger):
    """Redirect warnings to the logger.

    See Also
    --------
    https://docs.python.org/3/library/warnings.html#warnings.catch_warnings

    """
    import warnings

    def warning_to_logger(message, category, filename, lineno, file=None, line=None):
        """Redirect warnings to logger with standard formatting."""
        warn_str = warnings.formatwarning(message, category, filename, lineno, line)
        logger.warning(warn_str.strip())

    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.showwarning = warning_to_logger
        yield
