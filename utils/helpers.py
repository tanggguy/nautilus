"""
Utility helper functions for Nautilus Trading Platform.

This module provides common utility functions for:
- Date and time operations
- Data type conversions
- Validation functions
- File operations
- Statistical calculations
"""

import os
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd


# =============================================================================
# DATE AND TIME UTILITIES
# =============================================================================


def parse_date(date_str: str, date_format: str = "%Y-%m-%d") -> datetime:
    """
    Parse a date string into a datetime object.

    Args:
        date_str: Date string to parse.
        date_format: Format of the date string (default: YYYY-MM-DD).

    Returns:
        Datetime object.

    Raises:
        ValueError: If date string cannot be parsed.

    Example:
        >>> dt = parse_date("2024-01-01")
        >>> dt.year
        2024
    """
    try:
        return datetime.strptime(date_str, date_format)
    except ValueError as e:
        raise ValueError(f"Cannot parse date '{date_str}' with format '{date_format}': {e}")


def format_date(dt: datetime, date_format: str = "%Y-%m-%d") -> str:
    """
    Format a datetime object to string.

    Args:
        dt: Datetime object to format.
        date_format: Desired output format (default: YYYY-MM-DD).

    Returns:
        Formatted date string.

    Example:
        >>> dt = datetime(2024, 1, 1)
        >>> format_date(dt)
        '2024-01-01'
    """
    return dt.strftime(date_format)


def get_utc_now() -> datetime:
    """
    Get current UTC time.

    Returns:
        Current UTC datetime.

    Example:
        >>> now = get_utc_now()
        >>> now.tzinfo.tzname(None)
        'UTC'
    """
    return datetime.now(timezone.utc)


def date_range(
    start_date: Union[str, datetime],
    end_date: Union[str, datetime],
    freq: str = "D",
) -> List[datetime]:
    """
    Generate a list of dates between start and end.

    Args:
        start_date: Start date (string or datetime).
        end_date: End date (string or datetime).
        freq: Frequency (D=daily, W=weekly, M=monthly, etc.).

    Returns:
        List of datetime objects.

    Example:
        >>> dates = date_range("2024-01-01", "2024-01-05")
        >>> len(dates)
        5
    """
    if isinstance(start_date, str):
        start_date = parse_date(start_date)
    if isinstance(end_date, str):
        end_date = parse_date(end_date)

    return pd.date_range(start=start_date, end=end_date, freq=freq).to_pydatetime().tolist()


def is_market_open(dt: datetime, market: str = "crypto") -> bool:
    """
    Check if market is open at given datetime.

    Args:
        dt: Datetime to check.
        market: Market type ('crypto', 'forex', 'stock').

    Returns:
        True if market is open, False otherwise.

    Note:
        Crypto markets are always open 24/7.
        Stock and forex have specific hours (simplified implementation).

    Example:
        >>> dt = datetime(2024, 1, 1, 12, 0)
        >>> is_market_open(dt, 'crypto')
        True
    """
    if market.lower() == "crypto":
        return True

    # Simplified: stock market open 9:30 - 16:00 EST on weekdays
    if market.lower() == "stock":
        if dt.weekday() >= 5:  # Saturday or Sunday
            return False
        return 9 <= dt.hour < 16

    # Forex: 24/5 (closed weekends)
    if market.lower() == "forex":
        return dt.weekday() < 5

    return True


# =============================================================================
# DATA TYPE CONVERSIONS
# =============================================================================


def to_decimal(value: Union[int, float, str, Decimal]) -> Decimal:
    """
    Convert value to Decimal for precise financial calculations.

    Args:
        value: Value to convert.

    Returns:
        Decimal representation.

    Example:
        >>> to_decimal(0.1)
        Decimal('0.1')
    """
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def to_float(value: Union[int, float, str, Decimal]) -> float:
    """
    Convert value to float.

    Args:
        value: Value to convert.

    Returns:
        Float representation.

    Example:
        >>> to_float("123.45")
        123.45
    """
    return float(value)


def to_int(value: Union[int, float, str]) -> int:
    """
    Convert value to integer.

    Args:
        value: Value to convert.

    Returns:
        Integer representation.

    Example:
        >>> to_int("123")
        123
    """
    return int(float(value))


def round_price(price: float, decimals: int = 2) -> float:
    """
    Round price to specified decimal places.

    Args:
        price: Price to round.
        decimals: Number of decimal places (default: 2).

    Returns:
        Rounded price.

    Example:
        >>> round_price(123.456789, 2)
        123.46
    """
    return round(price, decimals)


def round_quantity(quantity: float, decimals: int = 8) -> float:
    """
    Round quantity to specified decimal places.

    Args:
        quantity: Quantity to round.
        decimals: Number of decimal places (default: 8 for crypto).

    Returns:
        Rounded quantity.

    Example:
        >>> round_quantity(0.123456789, 8)
        0.12345679
    """
    return round(quantity, decimals)


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================


def validate_positive(value: Union[int, float], name: str = "value") -> None:
    """
    Validate that a value is positive.

    Args:
        value: Value to validate.
        name: Name of the parameter (for error messages).

    Raises:
        ValueError: If value is not positive.

    Example:
        >>> validate_positive(10.5, "price")
        >>> validate_positive(-1, "price")  # Raises ValueError
        Traceback (most recent call last):
        ...
        ValueError: price must be positive, got -1
    """
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def validate_range(
    value: Union[int, float],
    min_value: Union[int, float],
    max_value: Union[int, float],
    name: str = "value",
) -> None:
    """
    Validate that a value is within a specified range.

    Args:
        value: Value to validate.
        min_value: Minimum allowed value.
        max_value: Maximum allowed value.
        name: Name of the parameter (for error messages).

    Raises:
        ValueError: If value is outside the range.

    Example:
        >>> validate_range(5, 0, 10, "score")
        >>> validate_range(15, 0, 10, "score")  # Raises ValueError
        Traceback (most recent call last):
        ...
        ValueError: score must be between 0 and 10, got 15
    """
    if not min_value <= value <= max_value:
        raise ValueError(f"{name} must be between {min_value} and {max_value}, got {value}")


def validate_percentage(value: float, name: str = "percentage") -> None:
    """
    Validate that a value is a valid percentage (0-1 or 0-100).

    Args:
        value: Value to validate.
        name: Name of the parameter (for error messages).

    Raises:
        ValueError: If value is not a valid percentage.

    Example:
        >>> validate_percentage(0.5, "win_rate")
        >>> validate_percentage(150, "win_rate")  # Raises ValueError
        Traceback (most recent call last):
        ...
        ValueError: win_rate must be between 0 and 100, got 150
    """
    if not 0 <= value <= 100:
        raise ValueError(f"{name} must be between 0 and 100, got {value}")


def validate_instrument(instrument: str) -> None:
    """
    Validate instrument symbol format.

    Args:
        instrument: Instrument symbol (e.g., 'BTCUSDT').

    Raises:
        ValueError: If instrument format is invalid.

    Example:
        >>> validate_instrument("BTCUSDT")
        >>> validate_instrument("")  # Raises ValueError
        Traceback (most recent call last):
        ...
        ValueError: Instrument symbol cannot be empty
    """
    if not instrument or not isinstance(instrument, str):
        raise ValueError("Instrument symbol cannot be empty")
    if not instrument.isalnum():
        raise ValueError(f"Instrument symbol must be alphanumeric, got '{instrument}'")


# =============================================================================
# FILE OPERATIONS
# =============================================================================


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if it doesn't.

    Args:
        path: Directory path.

    Returns:
        Path object of the directory.

    Example:
        >>> ensure_directory("data/results")
        PosixPath('data/results')
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_project_root() -> Path:
    """
    Get the project root directory.

    Returns:
        Path to project root.

    Example:
        >>> root = get_project_root()
        >>> root.name
        'nautilus'
    """
    return Path(__file__).parent.parent


def load_env_file(env_file: str = ".env") -> Dict[str, str]:
    """
    Load environment variables from a file.

    Args:
        env_file: Path to .env file (default: .env).

    Returns:
        Dictionary of environment variables.

    Example:
        >>> env_vars = load_env_file()
        >>> 'BINANCE_API_KEY' in env_vars
        True
    """
    env_vars = {}
    env_path = get_project_root() / env_file

    if not env_path.exists():
        return env_vars

    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                if "=" in line:
                    key, value = line.split("=", 1)
                    env_vars[key.strip()] = value.strip()

    return env_vars


def get_env_var(key: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
    """
    Get environment variable with optional default and required flag.

    Args:
        key: Environment variable key.
        default: Default value if not found.
        required: If True, raise error if not found.

    Returns:
        Environment variable value or default.

    Raises:
        ValueError: If required=True and variable not found.

    Example:
        >>> api_key = get_env_var("BINANCE_API_KEY", required=True)
    """
    value = os.getenv(key, default)
    if required and value is None:
        raise ValueError(f"Required environment variable '{key}' not found")
    return value


# =============================================================================
# STATISTICAL CALCULATIONS
# =============================================================================


def calculate_returns(prices: Union[List[float], np.ndarray, pd.Series]) -> np.ndarray:
    """
    Calculate returns from price series.

    Args:
        prices: Price series.

    Returns:
        Array of returns.

    Example:
        >>> prices = [100, 105, 103, 108]
        >>> returns = calculate_returns(prices)
        >>> len(returns)
        3
    """
    if isinstance(prices, pd.Series):
        return prices.pct_change().dropna().values
    prices = np.array(prices)
    return np.diff(prices) / prices[:-1]


def calculate_volatility(returns: Union[List[float], np.ndarray], annualize: bool = True) -> float:
    """
    Calculate volatility (standard deviation of returns).

    Args:
        returns: Return series.
        annualize: If True, annualize the volatility (default: True).

    Returns:
        Volatility value.

    Example:
        >>> returns = [0.01, -0.02, 0.03, -0.01]
        >>> vol = calculate_volatility(returns)
        >>> vol > 0
        True
    """
    returns = np.array(returns)
    vol = np.std(returns)
    if annualize:
        # Assume 365 periods per year for crypto (24/7 trading)
        vol *= np.sqrt(365)
    return vol


def calculate_sharpe_ratio(
    returns: Union[List[float], np.ndarray],
    risk_free_rate: float = 0.02,
    periods_per_year: int = 365,
) -> float:
    """
    Calculate Sharpe ratio.

    Args:
        returns: Return series.
        risk_free_rate: Annual risk-free rate (default: 0.02 = 2%).
        periods_per_year: Number of periods per year (default: 365 for crypto).

    Returns:
        Sharpe ratio.

    Example:
        >>> returns = [0.01, 0.02, -0.01, 0.03]
        >>> sharpe = calculate_sharpe_ratio(returns)
        >>> isinstance(sharpe, float)
        True
    """
    returns = np.array(returns)
    if len(returns) == 0:
        return 0.0

    excess_returns = returns - (risk_free_rate / periods_per_year)
    if np.std(excess_returns) == 0:
        return 0.0

    return np.sqrt(periods_per_year) * np.mean(excess_returns) / np.std(excess_returns)


def calculate_max_drawdown(equity_curve: Union[List[float], np.ndarray, pd.Series]) -> float:
    """
    Calculate maximum drawdown from equity curve.

    Args:
        equity_curve: Equity curve (cumulative returns or portfolio value).

    Returns:
        Maximum drawdown as a positive decimal (e.g., 0.2 for 20% drawdown).

    Example:
        >>> equity = [100, 110, 105, 120, 90, 115]
        >>> mdd = calculate_max_drawdown(equity)
        >>> mdd > 0
        True
    """
    equity_curve = np.array(equity_curve)
    if len(equity_curve) == 0:
        return 0.0

    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - running_max) / running_max
    return abs(np.min(drawdown))


# =============================================================================
# MISCELLANEOUS
# =============================================================================


def format_currency(amount: float, currency: str = "USD", decimals: int = 2) -> str:
    """
    Format amount as currency string.

    Args:
        amount: Amount to format.
        currency: Currency symbol (default: USD).
        decimals: Number of decimal places (default: 2).

    Returns:
        Formatted currency string.

    Example:
        >>> format_currency(1234.56)
        '$1,234.56'
    """
    symbol = {"USD": "$", "EUR": "€", "GBP": "£", "USDT": "₮"}.get(currency.upper(), currency)
    return f"{symbol}{amount:,.{decimals}f}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format value as percentage string.

    Args:
        value: Value to format (0.15 = 15%).
        decimals: Number of decimal places (default: 2).

    Returns:
        Formatted percentage string.

    Example:
        >>> format_percentage(0.1534)
        '15.34%'
    """
    return f"{value * 100:.{decimals}f}%"


def chunks(lst: List[Any], n: int) -> List[List[Any]]:
    """
    Split a list into chunks of size n.

    Args:
        lst: List to split.
        n: Chunk size.

    Returns:
        List of chunks.

    Example:
        >>> chunks([1, 2, 3, 4, 5], 2)
        [[1, 2], [3, 4], [5]]
    """
    return [lst[i : i + n] for i in range(0, len(lst), n)]
