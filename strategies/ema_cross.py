"""
EMA Crossover Strategy for Nautilus Trading Platform.

This strategy uses two Exponential Moving Averages (EMA) - a fast and a slow one.
Trading signals:
- BUY when fast EMA crosses above slow EMA (golden cross)
- SELL when fast EMA crosses below slow EMA (death cross)

This is a trend-following strategy that works best in trending markets.
"""

from decimal import Decimal

from nautilus_trader.indicators.average.ema import ExponentialMovingAverage
from nautilus_trader.model.data import Bar
from nautilus_trader.model.enums import OrderSide

from strategies.base_strategy import BaseStrategy, BaseStrategyConfig


class EMACrossConfig(BaseStrategyConfig):
    """
    Configuration for EMA Crossover strategy.

    Attributes:
        instrument_id: The instrument to trade.
        bar_type: The bar type to subscribe to.
        fast_period: Period for fast EMA (default: 10).
        slow_period: Period for slow EMA (default: 30).
        trade_size: Base trade size in quote currency.
        use_stop_loss: Enable stop loss (default: True).
        stop_loss_pct: Stop loss percentage (default: 0.02 = 2%).
        use_take_profit: Enable take profit (default: False).
        take_profit_pct: Take profit percentage (default: 0.04 = 4%).
    """

    fast_period: int = 10
    slow_period: int = 30


class EMACrossStrategy(BaseStrategy):
    """
    EMA Crossover Strategy.

    Entry signals:
    - LONG: Fast EMA crosses above Slow EMA
    - SHORT: Fast EMA crosses below Slow EMA (if short selling enabled)

    Exit signals:
    - Close LONG: Fast EMA crosses below Slow EMA
    - Close SHORT: Fast EMA crosses above Slow EMA

    Risk Management:
    - Configurable stop loss and take profit
    - Position sizing based on base trade size
    - Optional trailing stop loss

    Best suited for:
    - Trending markets
    - Medium to long-term timeframes (1H, 4H, 1D)
    - Low volatility instruments

    Example usage:
        >>> config = EMACrossConfig(
        ...     instrument_id="BTCUSDT.BINANCE",
        ...     bar_type="BTCUSDT.BINANCE-1-HOUR-LAST-EXTERNAL",
        ...     fast_period=10,
        ...     slow_period=30,
        ...     trade_size=Decimal("100"),
        ...     use_stop_loss=True,
        ...     stop_loss_pct=0.02
        ... )
        >>> strategy = EMACrossStrategy(config)
    """

    def __init__(self, config: EMACrossConfig):
        """
        Initialize EMA Crossover strategy.

        Args:
            config: Strategy configuration.
        """
        super().__init__(config)

        # Strategy-specific configuration
        self.fast_period = config.fast_period
        self.slow_period = config.slow_period

        # Validate periods
        if self.fast_period >= self.slow_period:
            raise ValueError(
                f"fast_period ({self.fast_period}) must be less than "
                f"slow_period ({self.slow_period})"
            )

        # Initialize indicators
        self.fast_ema = ExponentialMovingAverage(period=self.fast_period)
        self.slow_ema = ExponentialMovingAverage(period=self.slow_period)

        # State for crossover detection
        self.previous_fast = None
        self.previous_slow = None
        self.cross_detected = False

    def on_start(self):
        """
        Actions when strategy starts.

        Logs strategy-specific configuration.
        """
        super().on_start()
        self.log.info(f"Fast EMA Period   : {self.fast_period}")
        self.log.info(f"Slow EMA Period   : {self.slow_period}")
        self.log.info("-" * 80)

    def on_bar_update(self, bar: Bar):
        """
        Strategy logic for handling new bars.

        Checks for EMA crossovers and generates trading signals.

        Args:
            bar: The new bar data.
        """
        # Update indicators
        self.fast_ema.handle_bar(bar)
        self.slow_ema.handle_bar(bar)

        # Wait for both EMAs to initialize
        if not self.fast_ema.initialized or not self.slow_ema.initialized:
            return

        # Get current EMA values
        fast_value = self.fast_ema.value
        slow_value = self.slow_ema.value

        # Check for crossover only if we have previous values
        if self.previous_fast is not None and self.previous_slow is not None:
            # Detect golden cross (fast crosses above slow) - BUY signal
            if self._is_golden_cross(fast_value, slow_value):
                self._on_golden_cross(bar, fast_value, slow_value)

            # Detect death cross (fast crosses below slow) - SELL signal
            elif self._is_death_cross(fast_value, slow_value):
                self._on_death_cross(bar, fast_value, slow_value)

        # Update previous values for next bar
        self.previous_fast = fast_value
        self.previous_slow = slow_value

    def _is_golden_cross(self, fast_value: float, slow_value: float) -> bool:
        """
        Check if a golden cross occurred.

        Golden cross: Fast EMA crosses above Slow EMA.

        Args:
            fast_value: Current fast EMA value.
            slow_value: Current slow EMA value.

        Returns:
            True if golden cross detected.
        """
        return (
            self.previous_fast <= self.previous_slow  # Was below or equal
            and fast_value > slow_value  # Now above
        )

    def _is_death_cross(self, fast_value: float, slow_value: float) -> bool:
        """
        Check if a death cross occurred.

        Death cross: Fast EMA crosses below Slow EMA.

        Args:
            fast_value: Current fast EMA value.
            slow_value: Current slow EMA value.

        Returns:
            True if death cross detected.
        """
        return (
            self.previous_fast >= self.previous_slow  # Was above or equal
            and fast_value < slow_value  # Now below
        )

    def _on_golden_cross(self, bar: Bar, fast_value: float, slow_value: float):
        """
        Handle golden cross signal.

        If not in position: Enter LONG
        If in SHORT position: Close SHORT and enter LONG

        Args:
            bar: Current bar.
            fast_value: Fast EMA value.
            slow_value: Slow EMA value.
        """
        self.log.info(
            f"✨ GOLDEN CROSS | "
            f"Fast EMA: {fast_value:.2f} > Slow EMA: {slow_value:.2f} | "
            f"Price: {bar.close:.2f}"
        )

        # If already long, do nothing
        if self.is_long():
            self.log.info("Already in LONG position - no action")
            return

        # If short, close the position first
        if self.is_short():
            self.log.info("Closing SHORT position before entering LONG")
            self.close_position()

        # Enter long position
        if self.can_enter_position():
            self.log.info("📈 ENTERING LONG POSITION")
            self.buy()
            self.total_trades += 1

    def _on_death_cross(self, bar: Bar, fast_value: float, slow_value: float):
        """
        Handle death cross signal.

        If in LONG position: Close LONG
        If not in position and short selling enabled: Enter SHORT

        Args:
            bar: Current bar.
            fast_value: Fast EMA value.
            slow_value: Slow EMA value.
        """
        self.log.info(
            f"💀 DEATH CROSS | "
            f"Fast EMA: {fast_value:.2f} < Slow EMA: {slow_value:.2f} | "
            f"Price: {bar.close:.2f}"
        )

        # If long, close the position
        if self.is_long():
            self.log.info("📉 CLOSING LONG POSITION")
            self.close_position()
            return

        # If short selling is desired, could enter short here
        # For now, we only trade long side
        if not self.has_position():
            self.log.info("No position to close - waiting for next golden cross")

    def on_stop(self):
        """
        Actions when strategy stops.

        Logs final indicator values.
        """
        if self.fast_ema.initialized and self.slow_ema.initialized:
            self.log.info(f"Final Fast EMA: {self.fast_ema.value:.2f}")
            self.log.info(f"Final Slow EMA: {self.slow_ema.value:.2f}")

        super().on_stop()


# =============================================================================
# EXAMPLE CONFIGURATIONS
# =============================================================================


def create_ema_cross_config_conservative() -> EMACrossConfig:
    """
    Create a conservative EMA cross configuration.

    Uses slower EMAs (20/50) for fewer but more reliable signals.

    Returns:
        Conservative EMACrossConfig.
    """
    return EMACrossConfig(
        instrument_id="BTCUSDT.BINANCE",
        bar_type="BTCUSDT.BINANCE-4-HOUR-LAST-EXTERNAL",
        fast_period=20,
        slow_period=50,
        trade_size=Decimal("100"),
        use_stop_loss=True,
        stop_loss_pct=0.03,  # 3% stop loss
        use_take_profit=True,
        take_profit_pct=0.06,  # 6% take profit
    )


def create_ema_cross_config_aggressive() -> EMACrossConfig:
    """
    Create an aggressive EMA cross configuration.

    Uses faster EMAs (5/15) for more frequent signals.

    Returns:
        Aggressive EMACrossConfig.
    """
    return EMACrossConfig(
        instrument_id="BTCUSDT.BINANCE",
        bar_type="BTCUSDT.BINANCE-15-MINUTE-LAST-EXTERNAL",
        fast_period=5,
        slow_period=15,
        trade_size=Decimal("50"),  # Smaller size for higher frequency
        use_stop_loss=True,
        stop_loss_pct=0.015,  # 1.5% stop loss
        use_trailing_stop=True,
        trailing_stop_pct=0.01,  # 1% trailing stop
    )


def create_ema_cross_config_classic() -> EMACrossConfig:
    """
    Create a classic EMA cross configuration.

    Uses traditional 10/30 EMA periods.

    Returns:
        Classic EMACrossConfig.
    """
    return EMACrossConfig(
        instrument_id="BTCUSDT.BINANCE",
        bar_type="BTCUSDT.BINANCE-1-HOUR-LAST-EXTERNAL",
        fast_period=10,
        slow_period=30,
        trade_size=Decimal("100"),
        use_stop_loss=True,
        stop_loss_pct=0.02,  # 2% stop loss
        use_take_profit=False,  # Let profits run
    )
