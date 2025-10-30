"""
Utilities module for Nautilus Trading Platform.

This module provides common utilities including:
- Logging configuration with rotation
- Helper functions for dates, conversions, validations
- File operations
- Statistical calculations
"""

from utils.helpers import (
    calculate_max_drawdown,
    calculate_returns,
    calculate_sharpe_ratio,
    calculate_volatility,
    chunks,
    date_range,
    ensure_directory,
    format_currency,
    format_date,
    format_percentage,
    get_env_var,
    get_project_root,
    get_utc_now,
    is_market_open,
    load_env_file,
    parse_date,
    round_price,
    round_quantity,
    to_decimal,
    to_float,
    to_int,
    validate_instrument,
    validate_percentage,
    validate_positive,
    validate_range,
)
from utils.logging_config import (
    get_logger,
    log_function_call,
    log_performance_metrics,
    log_trade,
    setup_logging,
)

__all__ = [
    # Logging
    "setup_logging",
    "get_logger",
    "log_function_call",
    "log_trade",
    "log_performance_metrics",
    # Date and Time
    "parse_date",
    "format_date",
    "get_utc_now",
    "date_range",
    "is_market_open",
    # Conversions
    "to_decimal",
    "to_float",
    "to_int",
    "round_price",
    "round_quantity",
    # Validations
    "validate_positive",
    "validate_range",
    "validate_percentage",
    "validate_instrument",
    # File Operations
    "ensure_directory",
    "get_project_root",
    "load_env_file",
    "get_env_var",
    # Statistics
    "calculate_returns",
    "calculate_volatility",
    "calculate_sharpe_ratio",
    "calculate_max_drawdown",
    # Formatting
    "format_currency",
    "format_percentage",
    # Misc
    "chunks",
]
