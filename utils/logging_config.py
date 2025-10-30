"""
Centralized logging configuration for Nautilus Trading Platform.

This module provides logging setup with:
- Rotation by date and size
- Console and file handlers
- Structured log formatting
- Environment-based log levels
"""

import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path
from typing import Optional


def setup_logging(
    log_level: Optional[str] = None,
    log_dir: Optional[str] = None,
    max_bytes: int = 50 * 1024 * 1024,  # 50 MB
    backup_count: int = 5,
    console_output: bool = True,
    file_output: bool = True,
) -> logging.Logger:
    """
    Setup centralized logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
                   Defaults to LOG_LEVEL env var or INFO.
        log_dir: Directory for log files. Defaults to LOG_DIR env var or 'logs'.
        max_bytes: Maximum size of each log file before rotation (default: 50MB).
        backup_count: Number of backup log files to keep (default: 5).
        console_output: Enable console logging (default: True).
        file_output: Enable file logging (default: True).

    Returns:
        Configured root logger instance.

    Example:
        >>> logger = setup_logging(log_level='DEBUG')
        >>> logger.info('Trading system started')
    """
    # Get log level from env or parameter
    if log_level is None:
        log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    # Get log directory from env or parameter
    if log_dir is None:
        log_dir = os.getenv("LOG_DIR", "logs")

    # Create log directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level))

    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-30s | %(funcName)-20s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    simple_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # Console Handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level))
        console_handler.setFormatter(simple_formatter)
        root_logger.addHandler(console_handler)

    # File Handlers
    if file_output:
        # 1. Main rotating file handler (by size)
        main_log_file = log_path / "nautilus.log"
        file_handler = RotatingFileHandler(
            filename=main_log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(getattr(logging, log_level))
        file_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(file_handler)

        # 2. Error-only file handler
        error_log_file = log_path / "errors.log"
        error_handler = RotatingFileHandler(
            filename=error_log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(error_handler)

        # 3. Daily rotating file handler (one file per day)
        daily_log_file = log_path / "nautilus_daily.log"
        daily_handler = TimedRotatingFileHandler(
            filename=daily_log_file,
            when="midnight",
            interval=1,
            backupCount=30,  # Keep 30 days
            encoding="utf-8",
        )
        daily_handler.setLevel(getattr(logging, log_level))
        daily_handler.setFormatter(detailed_formatter)
        daily_handler.suffix = "%Y-%m-%d"
        root_logger.addHandler(daily_handler)

    # Log initial message
    root_logger.info("=" * 80)
    root_logger.info("Logging system initialized")
    root_logger.info(f"Log level: {log_level}")
    root_logger.info(f"Log directory: {log_path.absolute()}")
    root_logger.info(f"Console output: {console_output}")
    root_logger.info(f"File output: {file_output}")
    root_logger.info("=" * 80)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.

    Args:
        name: Name of the logger (typically __name__).

    Returns:
        Logger instance.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info('Module initialized')
    """
    return logging.getLogger(name)


def log_function_call(logger: logging.Logger):
    """
    Decorator to log function calls with arguments and return values.

    Args:
        logger: Logger instance to use.

    Example:
        >>> logger = get_logger(__name__)
        >>> @log_function_call(logger)
        ... def my_function(x, y):
        ...     return x + y
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
            try:
                result = func(*args, **kwargs)
                logger.debug(f"{func.__name__} returned {result}")
                return result
            except Exception as e:
                logger.error(f"{func.__name__} raised {type(e).__name__}: {e}")
                raise

        return wrapper

    return decorator


def log_trade(logger: logging.Logger, trade_info: dict) -> None:
    """
    Log trade information in a structured format.

    Args:
        logger: Logger instance.
        trade_info: Dictionary containing trade information.

    Example:
        >>> logger = get_logger(__name__)
        >>> log_trade(logger, {
        ...     'action': 'BUY',
        ...     'instrument': 'BTCUSDT',
        ...     'quantity': 0.01,
        ...     'price': 50000.0,
        ...     'timestamp': '2024-01-01 12:00:00'
        ... })
    """
    logger.info(
        f"TRADE | "
        f"Action: {trade_info.get('action', 'N/A')} | "
        f"Instrument: {trade_info.get('instrument', 'N/A')} | "
        f"Qty: {trade_info.get('quantity', 'N/A')} | "
        f"Price: {trade_info.get('price', 'N/A')} | "
        f"Time: {trade_info.get('timestamp', 'N/A')}"
    )


def log_performance_metrics(logger: logging.Logger, metrics: dict) -> None:
    """
    Log performance metrics in a structured format.

    Args:
        logger: Logger instance.
        metrics: Dictionary containing performance metrics.

    Example:
        >>> logger = get_logger(__name__)
        >>> log_performance_metrics(logger, {
        ...     'total_return': 0.15,
        ...     'sharpe_ratio': 1.8,
        ...     'max_drawdown': -0.12,
        ...     'win_rate': 0.65
        ... })
    """
    logger.info("=" * 80)
    logger.info("PERFORMANCE METRICS")
    logger.info("-" * 80)
    for key, value in metrics.items():
        logger.info(f"{key.replace('_', ' ').title():<30} : {value}")
    logger.info("=" * 80)


# Initialize logging on module import (optional)
# This will be called when the module is imported
# To customize, call setup_logging() explicitly in your main script
if __name__ != "__main__":
    # Only auto-initialize if not running as main script
    # This allows for customization in main application
    pass
