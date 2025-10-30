"""
Live trading configuration for Nautilus Trading Platform.

This module provides configuration classes for LIVE TRADING with real money.
USE WITH EXTREME CAUTION - this involves real financial risk!

SAFETY REQUIREMENTS:
- Minimum 3 months successful paper trading before going live
- Start with minimal capital
- Enable all circuit breakers and safety mechanisms
- Monitor continuously during operation
"""

import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

# Reuse components from backtest_config and paper_config
from configs.backtest_config import FeeConfig, InstrumentConfig, RiskConfig, VenueConfig
from configs.paper_config import MonitoringConfig, PersistenceConfig


class ExchangeConfig(BaseModel):
    """
    Configuration for exchange connection (Binance).

    Attributes:
        exchange: Exchange name ('binance', 'bybit', etc.).
        api_key: API key from exchange (REQUIRED).
        api_secret: API secret from exchange (REQUIRED).
        testnet: Whether to use testnet (MUST be False for live, default: False).
        timeout_seconds: Request timeout in seconds (default: 30).
        rate_limit_per_minute: Max requests per minute (default: 1200 for Binance).
        enable_rate_limiter: Enable automatic rate limiting (default: True).
        reconnect_on_disconnect: Auto-reconnect on disconnect (default: True).
        max_reconnect_attempts: Max reconnection attempts (default: 5).
    """

    exchange: str = Field(..., description="Exchange name")
    api_key: str = Field(..., description="API key (REQUIRED)", min_length=10)
    api_secret: str = Field(..., description="API secret (REQUIRED)", min_length=10)
    testnet: bool = Field(default=False, description="Use testnet (MUST be False for live)")
    timeout_seconds: int = Field(default=30, ge=5, le=120, description="Request timeout (s)")
    rate_limit_per_minute: int = Field(default=1200, ge=1, description="Rate limit per minute")
    enable_rate_limiter: bool = Field(default=True, description="Enable rate limiter")
    reconnect_on_disconnect: bool = Field(default=True, description="Auto-reconnect")
    max_reconnect_attempts: int = Field(default=5, ge=1, le=10, description="Max reconnect attempts")

    @field_validator("exchange")
    @classmethod
    def validate_exchange(cls, v: str) -> str:
        """Validate exchange is supported."""
        supported = ["binance", "bybit"]
        if v.lower() not in supported:
            raise ValueError(f"Exchange must be one of {supported}, got: {v}")
        return v.lower()

    @field_validator("api_key", "api_secret")
    @classmethod
    def validate_credentials(cls, v: str) -> str:
        """Validate credentials are not placeholder values."""
        placeholders = [
            "your_api_key",
            "your_api_secret",
            "test",
            "demo",
            "placeholder",
            "example",
        ]
        if any(placeholder in v.lower() for placeholder in placeholders):
            raise ValueError(
                "API credentials cannot be placeholder values. "
                "Please set real credentials in .env file."
            )
        return v

    @model_validator(mode="after")
    def warn_if_live(self) -> "ExchangeConfig":
        """Warn user if testnet is disabled (live trading)."""
        if not self.testnet:
            warnings.warn(
                "\n" + "=" * 80 + "\n"
                "WARNING: LIVE TRADING MODE ENABLED!\n"
                "You are using REAL MONEY on the LIVE exchange.\n"
                "Ensure you have:\n"
                "  1. Tested thoroughly on paper trading for at least 3 months\n"
                "  2. Reviewed all risk management settings\n"
                "  3. Started with minimal capital\n"
                "  4. Enabled all circuit breakers and safety mechanisms\n"
                "=" * 80,
                UserWarning,
                stacklevel=2,
            )
        return self


class CircuitBreakerConfig(BaseModel):
    """
    Configuration for circuit breakers (emergency stop mechanisms).

    Attributes:
        enable_circuit_breakers: Enable circuit breakers (default: True, REQUIRED for live).
        max_daily_loss_pct: Max daily loss % before stopping (default: 5%).
        max_daily_loss_amount: Max daily loss amount before stopping (USDT).
        max_consecutive_losses: Max consecutive losing trades (default: 5).
        min_account_balance: Minimum account balance before stopping (USDT).
        max_drawdown_pct: Max drawdown % before stopping (default: 20%).
        halt_on_connection_loss: Halt on connection loss (default: True).
        halt_on_api_error: Halt on repeated API errors (default: True).
        max_api_errors: Max API errors before halting (default: 10).
        check_interval_seconds: Circuit breaker check interval (default: 60).
    """

    enable_circuit_breakers: bool = Field(default=True, description="Enable circuit breakers")
    max_daily_loss_pct: float = Field(default=0.05, ge=0, le=1, description="Max daily loss %")
    max_daily_loss_amount: Optional[float] = Field(default=None, description="Max daily loss amount")
    max_consecutive_losses: int = Field(default=5, ge=1, description="Max consecutive losses")
    min_account_balance: Optional[float] = Field(default=None, description="Min account balance")
    max_drawdown_pct: float = Field(default=0.20, ge=0, le=1, description="Max drawdown %")
    halt_on_connection_loss: bool = Field(default=True, description="Halt on connection loss")
    halt_on_api_error: bool = Field(default=True, description="Halt on API errors")
    max_api_errors: int = Field(default=10, ge=1, description="Max API errors before halt")
    check_interval_seconds: int = Field(default=60, ge=10, description="Check interval (s)")

    @model_validator(mode="after")
    def validate_circuit_breakers_enabled(self) -> "CircuitBreakerConfig":
        """Ensure circuit breakers are enabled for live trading."""
        if not self.enable_circuit_breakers:
            raise ValueError(
                "Circuit breakers MUST be enabled for live trading! "
                "Set enable_circuit_breakers=True for safety."
            )
        return self


class LiveRiskConfig(RiskConfig):
    """
    Extended risk configuration for live trading with additional safety limits.

    Inherits from RiskConfig and adds live-specific constraints.

    Attributes:
        require_stop_loss: Require stop loss on all positions (default: True).
        require_take_profit: Require take profit on all positions (default: False).
        max_slippage_pct: Max allowed slippage % (reject orders with higher slippage).
        position_timeout_hours: Auto-close positions after N hours (0 = no timeout).
        max_leverage: Maximum leverage allowed (default: 1 = no leverage).
    """

    require_stop_loss: bool = Field(default=True, description="Require stop loss")
    require_take_profit: bool = Field(default=False, description="Require take profit")
    max_slippage_pct: float = Field(default=0.01, ge=0, le=0.1, description="Max slippage %")
    position_timeout_hours: int = Field(default=0, ge=0, description="Position timeout (hours)")
    max_leverage: float = Field(default=1.0, ge=1, le=10, description="Max leverage")

    @model_validator(mode="after")
    def validate_live_risk_settings(self) -> "LiveRiskConfig":
        """Validate risk settings are conservative for live trading."""
        # Enforce conservative defaults
        if self.max_order_size > 0.2:
            warnings.warn(
                f"max_order_size is {self.max_order_size} (20%+ of capital). "
                "Consider reducing for safer live trading.",
                UserWarning,
            )
        if self.max_position_size > 0.3:
            warnings.warn(
                f"max_position_size is {self.max_position_size} (30%+ of capital). "
                "Consider reducing for safer live trading.",
                UserWarning,
            )
        if not self.require_stop_loss:
            warnings.warn(
                "Stop losses are not required! This is very risky for live trading.",
                UserWarning,
            )
        return self


class HealthCheckConfig(BaseModel):
    """
    Configuration for system health monitoring.

    Attributes:
        enable_health_checks: Enable health monitoring (default: True).
        check_balance: Check account balance periodically (default: True).
        check_positions: Check positions consistency (default: True).
        check_connection: Check exchange connection (default: True).
        check_latency: Check API latency (default: True).
        max_latency_ms: Max acceptable latency (default: 1000ms).
        health_check_interval_seconds: Health check interval (default: 300 = 5 min).
        alert_on_health_issue: Alert on health issues (default: True).
    """

    enable_health_checks: bool = Field(default=True, description="Enable health checks")
    check_balance: bool = Field(default=True, description="Check balance")
    check_positions: bool = Field(default=True, description="Check positions")
    check_connection: bool = Field(default=True, description="Check connection")
    check_latency: bool = Field(default=True, description="Check latency")
    max_latency_ms: int = Field(default=1000, ge=100, description="Max latency (ms)")
    health_check_interval_seconds: int = Field(default=300, ge=60, description="Check interval (s)")
    alert_on_health_issue: bool = Field(default=True, description="Alert on issues")


class LiveTradingConfig(BaseModel):
    """
    Main live trading configuration.

    CRITICAL: This configuration is for REAL MONEY trading.
    Review all settings carefully before enabling live trading!

    Attributes:
        name: Name of the live trading session.
        description: Description of the session.
        enabled: Master switch to enable live trading (default: False).
        venue: Venue configuration.
        instruments: List of instruments to trade.
        exchange: Exchange connection configuration.
        risk: Risk management configuration.
        circuit_breakers: Circuit breaker configuration.
        fees: Fee configuration.
        health_checks: Health check configuration.
        monitoring: Monitoring and alerting configuration.
        persistence: Data persistence configuration.
        strategy_params: Strategy-specific parameters.
        dry_run_first: Run in dry-run mode first (default: True).
        require_confirmation: Require manual confirmation to start (default: True).
    """

    name: str = Field(..., description="Live trading session name")
    description: Optional[str] = Field(default="", description="Session description")
    enabled: bool = Field(default=False, description="Enable live trading (Master switch)")
    venue: VenueConfig = Field(..., description="Venue configuration")
    instruments: List[InstrumentConfig] = Field(..., min_length=1, max_length=5, description="Instruments")
    exchange: ExchangeConfig = Field(..., description="Exchange configuration")
    risk: LiveRiskConfig = Field(default_factory=LiveRiskConfig, description="Risk configuration")
    circuit_breakers: CircuitBreakerConfig = Field(
        default_factory=CircuitBreakerConfig, description="Circuit breakers"
    )
    fees: FeeConfig = Field(default_factory=FeeConfig, description="Fee configuration")
    health_checks: HealthCheckConfig = Field(
        default_factory=HealthCheckConfig, description="Health checks"
    )
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig, description="Monitoring")
    persistence: PersistenceConfig = Field(default_factory=PersistenceConfig, description="Persistence")
    strategy_params: Dict[str, Any] = Field(default_factory=dict, description="Strategy parameters")
    dry_run_first: bool = Field(default=True, description="Dry-run first")
    require_confirmation: bool = Field(default=True, description="Require confirmation")

    @model_validator(mode="after")
    def validate_live_trading_config(self) -> "LiveTradingConfig":
        """Comprehensive validation for live trading configuration."""
        if not self.enabled:
            return self  # Skip validation if not enabled

        errors = []

        # 1. Check circuit breakers are enabled
        if not self.circuit_breakers.enable_circuit_breakers:
            errors.append("Circuit breakers MUST be enabled for live trading")

        # 2. Check API credentials are set
        if not self.exchange.api_key or not self.exchange.api_secret:
            errors.append("Exchange API credentials must be set")

        # 3. Check starting balance is reasonable
        total_balance = sum(self.venue.starting_balances.values())
        if total_balance > 10000:
            warnings.warn(
                f"Starting balance is ${total_balance:,.2f}. "
                "For safety, start with smaller amounts (< $1000) until proven.",
                UserWarning,
            )

        # 4. Check max open positions is limited
        if self.risk.max_open_positions > 5:
            errors.append("Max open positions should be <= 5 for live trading")

        # 5. Check instruments count is limited
        if len(self.instruments) > 5:
            errors.append("Trade at most 5 instruments simultaneously for safety")

        # 6. Check monitoring is enabled
        if not self.monitoring.enable_alerts:
            warnings.warn("Alerts are disabled! Enable monitoring for live trading.", UserWarning)

        # 7. Check persistence is enabled
        if not self.persistence.save_trades:
            warnings.warn("Trade saving is disabled! Enable for audit trail.", UserWarning)

        if errors:
            raise ValueError("Live trading validation failed:\n  - " + "\n  - ".join(errors))

        # Final warning
        print("\n" + "=" * 80)
        print("LIVE TRADING CONFIGURATION VALIDATED")
        print("=" * 80)
        print(f"Session: {self.name}")
        print(f"Exchange: {self.exchange.exchange.upper()}")
        print(f"Instruments: {', '.join(i.symbol for i in self.instruments)}")
        print(f"Starting Balance: ${sum(self.venue.starting_balances.values()):,.2f}")
        print(f"Max Daily Loss: {self.circuit_breakers.max_daily_loss_pct * 100:.1f}%")
        print(f"Max Position Size: {self.risk.max_position_size * 100:.1f}%")
        print(f"Circuit Breakers: {'ENABLED' if self.circuit_breakers.enable_circuit_breakers else 'DISABLED'}")
        print("=" * 80)
        print("PROCEED WITH EXTREME CAUTION - REAL MONEY AT RISK!")
        print("=" * 80 + "\n")

        return self

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return self.model_dump()

    @classmethod
    def from_env(cls, **kwargs) -> "LiveTradingConfig":
        """
        Create configuration from environment variables.

        Args:
            **kwargs: Override parameters.

        Returns:
            LiveTradingConfig instance.

        Example:
            >>> config = LiveTradingConfig.from_env(name="MACD Live Trading")
        """
        # Check if live trading is explicitly enabled
        enabled = os.getenv("LIVE_TRADING_ENABLED", "false").lower() == "true"

        if not enabled:
            raise ValueError(
                "Live trading is not enabled in environment. "
                "Set LIVE_TRADING_ENABLED=true in .env to proceed."
            )

        # Venue configuration
        venue_config = VenueConfig(
            name=os.getenv("LIVE_VENUE", "BINANCE"),
            starting_balances={
                "USDT": float(os.getenv("LIVE_INITIAL_CAPITAL", "1000"))  # Default: $1000
            },
        )

        # Instrument configuration
        instruments_str = os.getenv("LIVE_INSTRUMENTS", "BTCUSDT")
        instruments = [
            InstrumentConfig(symbol=symbol.strip(), venue=venue_config.name)
            for symbol in instruments_str.split(",")
        ]

        # Exchange configuration
        exchange_config = ExchangeConfig(
            exchange=venue_config.name.lower(),
            api_key=os.getenv("BINANCE_API_KEY", ""),
            api_secret=os.getenv("BINANCE_API_SECRET", ""),
            testnet=False,  # LIVE trading
        )

        # Risk configuration with conservative defaults
        risk_config = LiveRiskConfig(
            max_order_size=float(os.getenv("MAX_POSITION_SIZE", "0.05")),  # 5% default
            max_position_size=float(os.getenv("MAX_POSITION_SIZE", "0.1")),  # 10% default
            max_open_positions=int(os.getenv("MAX_OPEN_POSITIONS", "2")),  # 2 positions default
            default_stop_loss=float(os.getenv("DEFAULT_STOP_LOSS", "0.02")),
            default_take_profit=float(os.getenv("DEFAULT_TAKE_PROFIT", "0.04")),
            max_daily_loss=float(os.getenv("MAX_DAILY_LOSS", "0.03")),  # 3% default
            require_stop_loss=True,
        )

        # Circuit breaker configuration
        circuit_breaker_config = CircuitBreakerConfig(
            enable_circuit_breakers=True,  # ALWAYS enabled for live
            max_daily_loss_pct=risk_config.max_daily_loss,
            max_consecutive_losses=5,
        )

        return cls(
            enabled=enabled,
            venue=venue_config,
            instruments=instruments,
            exchange=exchange_config,
            risk=risk_config,
            circuit_breakers=circuit_breaker_config,
            **kwargs,
        )


# =============================================================================
# EXAMPLE CONFIGURATIONS (FOR REFERENCE ONLY)
# =============================================================================


def create_minimal_live_config() -> LiveTradingConfig:
    """
    Create a minimal live trading configuration for testing.

    WARNING: This is for REFERENCE only. Configure properly before use!

    Returns:
        Minimal LiveTradingConfig (NOT enabled by default).
    """
    return LiveTradingConfig(
        name="Minimal Live Trading",
        description="Minimal configuration for live trading - configure before use!",
        enabled=False,  # NOT enabled by default
        venue=VenueConfig(name="BINANCE", starting_balances={"USDT": 500.0}),  # Small amount
        instruments=[InstrumentConfig(symbol="BTCUSDT", venue="BINANCE")],
        exchange=ExchangeConfig(
            exchange="binance",
            api_key="YOUR_API_KEY_HERE",  # REPLACE!
            api_secret="YOUR_API_SECRET_HERE",  # REPLACE!
            testnet=False,
        ),
        risk=LiveRiskConfig(
            max_order_size=0.05,  # 5% max
            max_position_size=0.10,  # 10% max
            max_open_positions=1,  # Only 1 position
            default_stop_loss=0.02,  # 2% stop loss
            max_daily_loss=0.03,  # 3% daily loss limit
            require_stop_loss=True,
        ),
        circuit_breakers=CircuitBreakerConfig(
            enable_circuit_breakers=True,
            max_daily_loss_pct=0.03,
            max_consecutive_losses=3,
        ),
    )


if __name__ == "__main__":
    # Example: Show what minimal config looks like (will not enable live trading)
    config = create_minimal_live_config()
    print(config.model_dump_json(indent=2))
