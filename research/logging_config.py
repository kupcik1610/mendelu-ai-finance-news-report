"""
Logging configuration for the research application.
"""

import logging
import sys


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """
    Configure and return the main logger for the research application.

    Args:
        level: Logging level (default INFO)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger('research')

    if logger.handlers:
        return logger

    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger


def get_logger(name: str = None) -> logging.Logger:
    """
    Get a logger instance for a specific module.

    Args:
        name: Optional module name (will be prefixed with 'research.')

    Returns:
        Logger instance
    """
    setup_logging()

    if name:
        return logging.getLogger(f'research.{name}')
    return logging.getLogger('research')
