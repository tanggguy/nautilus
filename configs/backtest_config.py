"""
Backtest configuration for Nautilus Trading Platform.

This module provides configuration classes for backtesting with validation using Pydantic.
"""

import os
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class VenueConfig(BaseModel):
    """
    Configuration for a trading venue.

    Attributes:
        name: Venue name (e.g., 'BINANCE', 'BYBIT').
        venue_type: Type of venue ('EXCHANGE', 'BROKERAGE').
        account_type: Account type ('CASH', 'MARGIN').
        base_currency: Base currency for the account (default: 'USDT').
        starting_balances: Dictionary of starting balances per currency.
    """

    name: str = Field(..., description="Venue name")
    venue_type: str = Field(default="EXCHANGE", description="Type of venue")
    account_type: str = Field(default="CASH", description="Account type")
    base_currency: str = Field(default="USDT", description="Base currency")
    starting_balances: Dict[str, float] = Field(
        default_factory=lambda: {"USDT": 10000.0},
        description="Starting balances per currency",
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate venue name is uppercase."""
        return v.upper()

    @field_validator("base_currency")
    @classmethod
    def validate_base_currency(cls, v: str) -> str:
        """Validate base currency is uppercase."""
        return v.upper()


class InstrumentConfig(BaseModel):
    """
    Configuration for a trading instrument.

    Attributes:
        symbol: Instrument symbol (e.g., 'BTCUSDT').
        venue: Venue where instrument is traded.
        instrument_type: Type of instrument ('SPOT', 'FUTURES', 'PERPETUAL').
        price_precision: Number of decimal places for price (default: 2).
        size_precision: Number of decimal places for size (default: 8).
        min_quantity: Minimum order quantity.
        max_quantity: Maximum order quantity (optional).
    """

    symbol: str = Field(..., description="Instrument symbol")
    venue: str = Field(..., description="Trading venue")
    instrument_type: str = Field(default="SPOT", description="Instrument type")
    price_precision: int = Field(default=2, ge=0, le=10, description="Price decimal places")
    size_precision: int = Field(default=8, ge=0, le=10, description="Size decimal places")
    min_quantity: float = Field(default=0.00001, gt=0, description="Minimum order quantity")
    max_quantity: Optional[float] = Field(default=None, description="Maximum order quantity")

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """Validate symbol is alphanumeric and uppercase."""
        if not v.isalnum():
            raise ValueError(f"Symbol must be alphanumeric, got: {v}")
        return v.upper()

    @field_validator("venue")
    @classmethod
    def validate_venue(cls, v: str) -> str:
        """Validate venue is uppercase."""
        return v.upper()


class DataConfig(BaseModel):
    """
    Configuration for historical data.

    Attributes:
        data_path: Path to historical data directory.
        start_date: Start date for backtest (YYYY-MM-DD or datetime).
        end_date: End date for backtest (YYYY-MM-DD or datetime).
        bar_type: Bar aggregation type (e.g., '1-MINUTE', '5-MINUTE', '1-HOUR').
        data_format: Data format ('parquet', 'csv', 'feather').
        validate_data: Whether to validate data quality (default: True).
    """

    data_path: Path = Field(default=Path("data"), description="Path to historical data")
    start_date: str = Field(..., description="Backtest start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="Backtest end date (YYYY-MM-DD)")
    bar_type: str = Field(default="1-MINUTE", description="Bar aggregation type")
    data_format: str = Field(default="parquet", description="Data format")
    validate_data: bool = Field(default=True, description="Validate data quality")

    @field_validator("start_date", "end_date")
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        """Validate date is in YYYY-MM-DD format."""
        try:
            datetime.strptime(v, "%Y-%m-%d")
            return v
        except ValueError:
            raise ValueError(f"Date must be in YYYY-MM-DD format, got: {v}")

    @field_validator("data_format")
    @classmethod
    def validate_data_format(cls, v: str) -> str:
        """Validate data format is supported."""
        supported = ["parquet", "csv", "feather"]
        if v.lower() not in supported:
            raise ValueError(f"Data format must be one of {supported}, got: {v}")
        return v.lower()

    @model_validator(mode="after")
    def validate_date_range(self) -> "DataConfig":
        """Validate end_date is after start_date."""
        start = datetime.strptime(self.start_date, "%Y-%m-%d")
        end = datetime.strptime(self.end_date, "%Y-%m-%d")
        if end <= start:
            raise ValueError(f"end_date ({self.end_date}) must be after start_date ({self.start_date})")
        return self


class RiskConfig(BaseModel):
    """
    Configuration for risk management.

    Attributes:
        max_order_size: Maximum order size as percentage of capital (0-1).
        max_position_size: Maximum position size as percentage of capital (0-1).
        max_open_positions: Maximum number of open positions.
        default_stop_loss: Default stop loss percentage (0-1).
        default_take_profit: Default take profit percentage (0-1).
        max_daily_loss: Maximum daily loss percentage (0-1).
        use_trailing_stop: Enable trailing stop loss (default: False).
        trailing_stop_distance: Trailing stop distance percentage (0-1).
    """

    max_order_size: float = Field(default=0.1, ge=0, le=1, description="Max order size (% of capital)")
    max_position_size: float = Field(default=0.2, ge=0, le=1, description="Max position size (% of capital)")
    max_open_positions: int = Field(default=3, ge=1, description="Max open positions")
    default_stop_loss: float = Field(default=0.02, ge=0, le=1, description="Default stop loss (%)")
    default_take_profit: float = Field(default=0.04, ge=0, le=1, description="Default take profit (%)")
    max_daily_loss: float = Field(default=0.05, ge=0, le=1, description="Max daily loss (%)")
    use_trailing_stop: bool = Field(default=False, description="Enable trailing stop")
    trailing_stop_distance: float = Field(default=0.01, ge=0, le=1, description="Trailing stop distance (%)")


class FeeConfig(BaseModel):
    """
    Configuration for trading fees and commissions.

    Attributes:
        maker_fee: Maker fee percentage (e.g., 0.001 = 0.1%).
        taker_fee: Taker fee percentage (e.g., 0.001 = 0.1%).
        use_fee_tiers: Whether to use tiered fees based on volume (default: False).
    """

    maker_fee: float = Field(default=0.001, ge=0, le=0.1, description="Maker fee %")
    taker_fee: float = Field(default=0.001, ge=0, le=0.1, description="Taker fee %")
    use_fee_tiers: bool = Field(default=False, description="Use tiered fees")


class BacktestConfig(BaseModel):
    """
    Main backtest configuration.

    Attributes:
        name: Name of the backtest run.
        description: Description of the backtest.
        venue: Venue configuration.
        instruments: List of instrument configurations.
        data: Data configuration.
        risk: Risk management configuration.
        fees: Fee configuration.
        strategy_params: Strategy-specific parameters.
        output_path: Path to save backtest results.
        save_results: Whether to save results (default: True).
        run_analysis: Whether to run analysis after backtest (default: True).
    """

    name: str = Field(..., description="Backtest name")
    description: Optional[str] = Field(default="", description="Backtest description")
    venue: VenueConfig = Field(..., description="Venue configuration")
    instruments: List[InstrumentConfig] = Field(..., min_length=1, description="Instrument configurations")
    data: DataConfig = Field(..., description="Data configuration")
    risk: RiskConfig = Field(default_factory=RiskConfig, description="Risk configuration")
    fees: FeeConfig = Field(default_factory=FeeConfig, description="Fee configuration")
    strategy_params: Dict[str, Any] = Field(default_factory=dict, description="Strategy parameters")
    output_path: Path = Field(default=Path("results/backtests"), description="Output path")
    save_results: bool = Field(default=True, description="Save results")
    run_analysis: bool = Field(default=True, description="Run analysis")

    @model_validator(mode="after")
    def validate_output_path(self) -> "BacktestConfig":
        """Ensure output path exists."""
        if self.save_results:
            self.output_path.mkdir(parents=True, exist_ok=True)
        return self

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return self.model_dump()

    @classmethod
    def from_env(cls, **kwargs) -> "BacktestConfig":
        """
        Create configuration from environment variables.

        Args:
            **kwargs: Override parameters.

        Returns:
            BacktestConfig instance.

        Example:
            >>> config = BacktestConfig.from_env(name="MACD Backtest")
        """
        venue_config = VenueConfig(
            name=os.getenv("DEFAULT_VENUE", "BINANCE"),
            starting_balances={os.getenv("BACKTEST_BASE_CURRENCY", "USDT"): float(os.getenv("BACKTEST_INITIAL_CAPITAL", "10000"))},
        )

        instrument_config = InstrumentConfig(
            symbol=os.getenv("DEFAULT_INSTRUMENT", "BTCUSDT"),
            venue=os.getenv("DEFAULT_VENUE", "BINANCE"),
        )

        data_config = DataConfig(
            data_path=Path(os.getenv("DATA_DIR", "data")),
            start_date=os.getenv("BACKTEST_START_DATE", "2023-01-01"),
            end_date=os.getenv("BACKTEST_END_DATE", "2024-12-31"),
            bar_type=os.getenv("BACKTEST_BAR_TYPE", "1-MINUTE"),
        )

        risk_config = RiskConfig(
            max_order_size=float(os.getenv("MAX_POSITION_SIZE", "0.1")),
            max_position_size=float(os.getenv("MAX_POSITION_SIZE", "0.1")),
            default_stop_loss=float(os.getenv("DEFAULT_STOP_LOSS", "0.02")),
            default_take_profit=float(os.getenv("DEFAULT_TAKE_PROFIT", "0.04")),
        )

        fee_config = FeeConfig(
            maker_fee=0.001,  # Binance spot maker fee (0.1%)
            taker_fee=0.001,  # Binance spot taker fee (0.1%)
        )

        return cls(
            venue=venue_config,
            instruments=[instrument_config],
            data=data_config,
            risk=risk_config,
            fees=fee_config,
            **kwargs,
        )


# =============================================================================
# EXAMPLE CONFIGURATIONS
# =============================================================================


def create_default_config() -> BacktestConfig:
    """
    Create a default backtest configuration.

    Returns:
        Default BacktestConfig.

    Example:
        >>> config = create_default_config()
        >>> config.venue.name
        'BINANCE'
    """
    return BacktestConfig(
        name="Default Backtest",
        description="Default backtest configuration for testing",
        venue=VenueConfig(name="BINANCE"),
        instruments=[InstrumentConfig(symbol="BTCUSDT", venue="BINANCE")],
        data=DataConfig(start_date="2023-01-01", end_date="2024-01-01"),
    )


def create_crypto_config(
    symbol: str = "BTCUSDT",
    start_date: str = "2023-01-01",
    end_date: str = "2024-01-01",
    initial_capital: float = 10000.0,
) -> BacktestConfig:
    """
    Create a cryptocurrency backtest configuration.

    Args:
        symbol: Trading symbol (default: BTCUSDT).
        start_date: Start date (default: 2023-01-01).
        end_date: End date (default: 2024-01-01).
        initial_capital: Initial capital in USDT (default: 10000).

    Returns:
        Configured BacktestConfig for crypto.

    Example:
        >>> config = create_crypto_config(symbol="ETHUSDT", initial_capital=5000)
        >>> config.instruments[0].symbol
        'ETHUSDT'
    """
    return BacktestConfig(
        name=f"{symbol} Crypto Backtest",
        description=f"Backtest for {symbol} from {start_date} to {end_date}",
        venue=VenueConfig(name="BINANCE", starting_balances={"USDT": initial_capital}),
        instruments=[InstrumentConfig(symbol=symbol, venue="BINANCE")],
        data=DataConfig(start_date=start_date, end_date=end_date, bar_type="1-MINUTE"),
        risk=RiskConfig(
            max_order_size=0.1,
            max_position_size=0.2,
            max_open_positions=3,
            default_stop_loss=0.02,
            default_take_profit=0.04,
        ),
        fees=FeeConfig(maker_fee=0.001, taker_fee=0.001),
    )


if __name__ == "__main__":
    # Example usage
    config = create_default_config()
    print(config.model_dump_json(indent=2))
