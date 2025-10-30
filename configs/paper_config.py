"""
Paper trading configuration for Nautilus Trading Platform.

This module provides configuration classes for paper trading (simulated live trading)
with real-time data feeds but simulated execution.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

# Reuse components from backtest_config
from configs.backtest_config import FeeConfig, InstrumentConfig, RiskConfig, VenueConfig


class DataProviderConfig(BaseModel):
    """
    Configuration for real-time data provider.

    Attributes:
        provider: Data provider name ('binance', 'binance_testnet', 'bybit').
        use_testnet: Whether to use testnet (default: True for safety).
        api_key: API key for data provider (optional for public data).
        api_secret: API secret for data provider (optional for public data).
        subscribe_bars: Whether to subscribe to bar data (default: True).
        subscribe_ticks: Whether to subscribe to tick data (default: False).
        subscribe_order_book: Whether to subscribe to order book (default: False).
        order_book_depth: Order book depth if subscribed (default: 10).
    """

    provider: str = Field(..., description="Data provider name")
    use_testnet: bool = Field(default=True, description="Use testnet")
    api_key: Optional[str] = Field(default=None, description="API key")
    api_secret: Optional[str] = Field(default=None, description="API secret")
    subscribe_bars: bool = Field(default=True, description="Subscribe to bars")
    subscribe_ticks: bool = Field(default=False, description="Subscribe to ticks")
    subscribe_order_book: bool = Field(default=False, description="Subscribe to order book")
    order_book_depth: int = Field(default=10, ge=1, le=100, description="Order book depth")

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        """Validate provider is supported."""
        supported = ["binance", "binance_testnet", "bybit", "bybit_testnet"]
        if v.lower() not in supported:
            raise ValueError(f"Provider must be one of {supported}, got: {v}")
        return v.lower()


class ExecutionConfig(BaseModel):
    """
    Configuration for simulated execution.

    Attributes:
        simulate_latency: Simulate network latency (default: True).
        latency_ms: Average latency in milliseconds (default: 50).
        latency_std_ms: Standard deviation of latency (default: 20).
        simulate_slippage: Simulate price slippage (default: True).
        slippage_bps: Average slippage in basis points (default: 5 = 0.05%).
        slippage_model: Slippage model ('fixed', 'volume_based', 'volatility_based').
        simulate_rejections: Simulate order rejections (default: False).
        rejection_rate: Order rejection rate 0-1 (default: 0.01 = 1%).
        partial_fills: Allow partial order fills (default: True).
    """

    simulate_latency: bool = Field(default=True, description="Simulate latency")
    latency_ms: int = Field(default=50, ge=0, le=1000, description="Avg latency (ms)")
    latency_std_ms: int = Field(default=20, ge=0, le=500, description="Latency std dev (ms)")
    simulate_slippage: bool = Field(default=True, description="Simulate slippage")
    slippage_bps: float = Field(default=5.0, ge=0, le=100, description="Slippage (bps)")
    slippage_model: str = Field(default="fixed", description="Slippage model")
    simulate_rejections: bool = Field(default=False, description="Simulate rejections")
    rejection_rate: float = Field(default=0.01, ge=0, le=1, description="Rejection rate")
    partial_fills: bool = Field(default=True, description="Allow partial fills")

    @field_validator("slippage_model")
    @classmethod
    def validate_slippage_model(cls, v: str) -> str:
        """Validate slippage model is supported."""
        supported = ["fixed", "volume_based", "volatility_based"]
        if v.lower() not in supported:
            raise ValueError(f"Slippage model must be one of {supported}, got: {v}")
        return v.lower()


class MonitoringConfig(BaseModel):
    """
    Configuration for monitoring and alerting.

    Attributes:
        enable_dashboard: Enable real-time dashboard (default: True).
        dashboard_port: Port for dashboard server (default: 8050).
        enable_alerts: Enable alerting system (default: True).
        alert_on_trade: Alert on trade execution (default: False).
        alert_on_stop_loss: Alert on stop loss hit (default: True).
        alert_on_daily_loss: Alert on daily loss limit (default: True).
        alert_channels: List of alert channels ('telegram', 'email', 'console').
        telegram_token: Telegram bot token (optional).
        telegram_chat_id: Telegram chat ID (optional).
        email_recipient: Email recipient for alerts (optional).
    """

    enable_dashboard: bool = Field(default=True, description="Enable dashboard")
    dashboard_port: int = Field(default=8050, ge=1024, le=65535, description="Dashboard port")
    enable_alerts: bool = Field(default=True, description="Enable alerts")
    alert_on_trade: bool = Field(default=False, description="Alert on trades")
    alert_on_stop_loss: bool = Field(default=True, description="Alert on stop loss")
    alert_on_daily_loss: bool = Field(default=True, description="Alert on daily loss")
    alert_channels: List[str] = Field(default=["console"], description="Alert channels")
    telegram_token: Optional[str] = Field(default=None, description="Telegram token")
    telegram_chat_id: Optional[str] = Field(default=None, description="Telegram chat ID")
    email_recipient: Optional[str] = Field(default=None, description="Email recipient")

    @field_validator("alert_channels")
    @classmethod
    def validate_alert_channels(cls, v: List[str]) -> List[str]:
        """Validate alert channels are supported."""
        supported = ["telegram", "email", "console"]
        for channel in v:
            if channel.lower() not in supported:
                raise ValueError(f"Alert channel must be one of {supported}, got: {channel}")
        return [c.lower() for c in v]


class PersistenceConfig(BaseModel):
    """
    Configuration for data persistence.

    Attributes:
        save_trades: Save all trades to file (default: True).
        save_orders: Save all orders to file (default: True).
        save_positions: Save position snapshots (default: True).
        save_portfolio: Save portfolio snapshots (default: True).
        snapshot_interval_seconds: Interval for snapshots in seconds (default: 60).
        output_path: Path for saving data (default: results/paper_trading).
        data_format: Format for saving data ('parquet', 'csv', 'json').
    """

    save_trades: bool = Field(default=True, description="Save trades")
    save_orders: bool = Field(default=True, description="Save orders")
    save_positions: bool = Field(default=True, description="Save positions")
    save_portfolio: bool = Field(default=True, description="Save portfolio")
    snapshot_interval_seconds: int = Field(default=60, ge=1, description="Snapshot interval (s)")
    output_path: Path = Field(default=Path("results/paper_trading"), description="Output path")
    data_format: str = Field(default="parquet", description="Data format")

    @field_validator("data_format")
    @classmethod
    def validate_data_format(cls, v: str) -> str:
        """Validate data format is supported."""
        supported = ["parquet", "csv", "json"]
        if v.lower() not in supported:
            raise ValueError(f"Data format must be one of {supported}, got: {v}")
        return v.lower()

    @model_validator(mode="after")
    def ensure_output_path(self) -> "PersistenceConfig":
        """Ensure output path exists."""
        self.output_path.mkdir(parents=True, exist_ok=True)
        return self


class PaperTradingConfig(BaseModel):
    """
    Main paper trading configuration.

    Attributes:
        name: Name of the paper trading session.
        description: Description of the session.
        venue: Venue configuration.
        instruments: List of instrument configurations.
        data_provider: Data provider configuration.
        execution: Execution simulation configuration.
        risk: Risk management configuration.
        fees: Fee configuration.
        monitoring: Monitoring and alerting configuration.
        persistence: Data persistence configuration.
        strategy_params: Strategy-specific parameters.
        max_runtime_hours: Maximum runtime in hours (0 = unlimited).
        auto_restart: Auto-restart on error (default: False).
    """

    name: str = Field(..., description="Paper trading session name")
    description: Optional[str] = Field(default="", description="Session description")
    venue: VenueConfig = Field(..., description="Venue configuration")
    instruments: List[InstrumentConfig] = Field(..., min_length=1, description="Instruments")
    data_provider: DataProviderConfig = Field(..., description="Data provider config")
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig, description="Execution config")
    risk: RiskConfig = Field(default_factory=RiskConfig, description="Risk config")
    fees: FeeConfig = Field(default_factory=FeeConfig, description="Fee config")
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig, description="Monitoring config")
    persistence: PersistenceConfig = Field(default_factory=PersistenceConfig, description="Persistence config")
    strategy_params: Dict[str, Any] = Field(default_factory=dict, description="Strategy parameters")
    max_runtime_hours: int = Field(default=0, ge=0, description="Max runtime (hours, 0=unlimited)")
    auto_restart: bool = Field(default=False, description="Auto-restart on error")

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return self.model_dump()

    @classmethod
    def from_env(cls, **kwargs) -> "PaperTradingConfig":
        """
        Create configuration from environment variables.

        Args:
            **kwargs: Override parameters.

        Returns:
            PaperTradingConfig instance.

        Example:
            >>> config = PaperTradingConfig.from_env(name="MACD Paper Trading")
        """
        # Venue configuration
        venue_config = VenueConfig(
            name=os.getenv("DEFAULT_VENUE", "BINANCE"),
            starting_balances={
                os.getenv("PAPER_BASE_CURRENCY", "USDT"): float(
                    os.getenv("PAPER_INITIAL_CAPITAL", "10000")
                )
            },
        )

        # Instrument configuration
        instrument_config = InstrumentConfig(
            symbol=os.getenv("DEFAULT_INSTRUMENT", "BTCUSDT"),
            venue=os.getenv("DEFAULT_VENUE", "BINANCE"),
        )

        # Data provider configuration
        data_provider_config = DataProviderConfig(
            provider=os.getenv("PAPER_DATA_PROVIDER", "binance_testnet"),
            use_testnet=os.getenv("PAPER_DATA_PROVIDER", "binance_testnet").endswith("testnet"),
            api_key=os.getenv("BINANCE_TESTNET_API_KEY" if os.getenv("PAPER_DATA_PROVIDER", "binance_testnet").endswith("testnet") else "BINANCE_API_KEY"),
            api_secret=os.getenv("BINANCE_TESTNET_API_SECRET" if os.getenv("PAPER_DATA_PROVIDER", "binance_testnet").endswith("testnet") else "BINANCE_API_SECRET"),
        )

        # Risk configuration
        risk_config = RiskConfig(
            max_order_size=float(os.getenv("MAX_POSITION_SIZE", "0.1")),
            max_position_size=float(os.getenv("MAX_POSITION_SIZE", "0.1")),
            max_daily_loss=float(os.getenv("MAX_DAILY_LOSS", "0.05")),
        )

        # Monitoring configuration
        monitoring_config = MonitoringConfig(
            telegram_token=os.getenv("TELEGRAM_BOT_TOKEN"),
            telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID"),
            email_recipient=os.getenv("EMAIL_TO"),
        )

        return cls(
            venue=venue_config,
            instruments=[instrument_config],
            data_provider=data_provider_config,
            risk=risk_config,
            monitoring=monitoring_config,
            **kwargs,
        )


# =============================================================================
# EXAMPLE CONFIGURATIONS
# =============================================================================


def create_default_paper_config() -> PaperTradingConfig:
    """
    Create a default paper trading configuration.

    Returns:
        Default PaperTradingConfig.

    Example:
        >>> config = create_default_paper_config()
        >>> config.venue.name
        'BINANCE'
    """
    return PaperTradingConfig(
        name="Default Paper Trading",
        description="Default paper trading configuration with testnet",
        venue=VenueConfig(name="BINANCE", starting_balances={"USDT": 10000.0}),
        instruments=[InstrumentConfig(symbol="BTCUSDT", venue="BINANCE")],
        data_provider=DataProviderConfig(provider="binance_testnet", use_testnet=True),
    )


def create_conservative_paper_config(
    symbol: str = "BTCUSDT",
    initial_capital: float = 5000.0,
) -> PaperTradingConfig:
    """
    Create a conservative paper trading configuration with strict risk limits.

    Args:
        symbol: Trading symbol (default: BTCUSDT).
        initial_capital: Initial capital in USDT (default: 5000).

    Returns:
        Conservative PaperTradingConfig.

    Example:
        >>> config = create_conservative_paper_config(initial_capital=3000)
        >>> config.risk.max_daily_loss
        0.02
    """
    return PaperTradingConfig(
        name=f"{symbol} Conservative Paper Trading",
        description=f"Conservative paper trading for {symbol} with strict risk limits",
        venue=VenueConfig(name="BINANCE", starting_balances={"USDT": initial_capital}),
        instruments=[InstrumentConfig(symbol=symbol, venue="BINANCE")],
        data_provider=DataProviderConfig(provider="binance_testnet", use_testnet=True),
        risk=RiskConfig(
            max_order_size=0.05,  # 5% max per order
            max_position_size=0.1,  # 10% max position
            max_open_positions=2,  # Only 2 positions
            default_stop_loss=0.015,  # 1.5% stop loss
            default_take_profit=0.03,  # 3% take profit
            max_daily_loss=0.02,  # 2% max daily loss
            use_trailing_stop=True,
            trailing_stop_distance=0.01,  # 1% trailing stop
        ),
        monitoring=MonitoringConfig(
            enable_dashboard=True,
            enable_alerts=True,
            alert_on_stop_loss=True,
            alert_on_daily_loss=True,
        ),
    )


if __name__ == "__main__":
    # Example usage
    config = create_default_paper_config()
    print(config.model_dump_json(indent=2))
