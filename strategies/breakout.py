"""
Breakout Strategy for Nautilus Trading Platform.

This strategy identifies periods of consolidation (ranges) and trades breakouts
from these ranges. It assumes that strong price movements follow consolidation.

Trading signals:
- BUY when price breaks above the recent high (resistance)
- SELL when price breaks below the recent low (support) or to close long
- Optional volume confirmation for stronger signals

Best suited for:
- Markets transitioning from consolidation to trending
- Instruments with clear support/resistance levels
- Volatile markets with strong momentum
"""

from decimal import Decimal
from typing import Optional

from nautilus_trader.model.data import Bar
from nautilus_trader.model.enums import OrderSide

from strategies.base_strategy import BaseStrategy, BaseStrategyConfig


class BreakoutConfig(BaseStrategyConfig):
    """
    Configuration for Breakout strategy.

    Attributes:
        instrument_id: The instrument to trade.
        bar_type: The bar type to subscribe to.
        lookback_period: Number of bars to look back for range detection (default: 20).
        breakout_threshold: Percentage above high/below low to confirm breakout (default: 0.001 = 0.1%).
        volume_confirm: Require volume confirmation for breakouts (default: False).
        volume_multiplier: Volume must be X times average volume (default: 1.5).
        retest_entry: Wait for price to retest breakout level before entering (default: False).
        trade_size: Base trade size in quote currency.
        use_stop_loss: Enable stop loss (default: True).
        stop_loss_pct: Stop loss percentage (default: 0.02 = 2%).
    """

    lookback_period: int = 20
    breakout_threshold: float = 0.001
    volume_confirm: bool = False
    volume_multiplier: float = 1.5
    retest_entry: bool = False


class BreakoutStrategy(BaseStrategy):
    """
    Breakout Strategy - Trade range breakouts.

    Entry signals:
    - LONG: Price breaks above recent high (resistance) + optional volume confirmation
    - Consolidation detected by analyzing recent high/low range

    Exit signals:
    - Close LONG: Price breaks below recent low OR stop loss hit
    - Close on opposite breakout signal

    Risk Management:
    - Stop loss typically placed at recent low (or below)
    - Take profit at 2x or 3x the range height
    - Tighter stops for failed breakouts

    Features:
    - Adaptive range detection based on lookback period
    - Optional volume confirmation to filter false breakouts
    - Optional retest entry (wait for pullback to breakout level)
    - Breakout strength measurement

    Best suited for:
    - Consolidation breakouts
    - Support/resistance breakouts
    - News-driven volatility
    - Market open hours (traditional markets)

    Not suitable for:
    - Continuous choppy markets
    - Very low volatility periods

    Example usage:
        >>> config = BreakoutConfig(
        ...     instrument_id="BTCUSDT.BINANCE",
        ...     bar_type="BTCUSDT.BINANCE-1-HOUR-LAST-EXTERNAL",
        ...     lookback_period=20,
        ...     breakout_threshold=0.002,
        ...     volume_confirm=True,
        ...     volume_multiplier=1.5,
        ...     trade_size=Decimal("100")
        ... )
        >>> strategy = BreakoutStrategy(config)
    """

    def __init__(self, config: BreakoutConfig):
        """
        Initialize Breakout strategy.

        Args:
            config: Strategy configuration.
        """
        super().__init__(config)

        # Strategy-specific configuration
        self.lookback_period = config.lookback_period
        self.breakout_threshold = config.breakout_threshold
        self.volume_confirm = config.volume_confirm
        self.volume_multiplier = config.volume_multiplier
        self.retest_entry = config.retest_entry

        # Range tracking
        self.range_high: Optional[float] = None
        self.range_low: Optional[float] = None
        self.range_bars: list[Bar] = []

        # Volume tracking
        self.volume_history: list[float] = []
        self.avg_volume: Optional[float] = None

        # Breakout state
        self.breakout_level: Optional[float] = None
        self.breakout_direction: Optional[str] = None  # 'up' or 'down'
        self.awaiting_retest = False

    def on_start(self):
        """
        Actions when strategy starts.

        Logs strategy-specific configuration.
        """
        super().on_start()
        self.log.info(f"Lookback Period   : {self.lookback_period}")
        self.log.info(f"Breakout Thresh   : {self.breakout_threshold * 100:.2f}%")
        self.log.info(f"Volume Confirm    : {self.volume_confirm}")
        if self.volume_confirm:
            self.log.info(f"Volume Multiplier : {self.volume_multiplier}x")
        self.log.info(f"Retest Entry      : {self.retest_entry}")
        self.log.info("-" * 80)

    def on_bar_update(self, bar: Bar):
        """
        Strategy logic for handling new bars.

        Updates range, checks for breakouts, and manages positions.

        Args:
            bar: The new bar data.
        """
        # Add bar to history
        self.range_bars.append(bar)

        # Track volume history
        if hasattr(bar, 'volume'):
            self.volume_history.append(float(bar.volume))

        # Keep only lookback_period bars
        if len(self.range_bars) > self.lookback_period:
            self.range_bars.pop(0)

        # Keep volume history same length
        if len(self.volume_history) > self.lookback_period:
            self.volume_history.pop(0)

        # Need enough bars to establish range
        if len(self.range_bars) < self.lookback_period:
            return

        # Update range (support and resistance)
        self._update_range()

        # Calculate average volume if needed
        if self.volume_confirm and len(self.volume_history) >= self.lookback_period:
            self.avg_volume = sum(self.volume_history) / len(self.volume_history)

        # Check for breakout signals
        price = float(bar.close)

        # If awaiting retest, check if retest occurred
        if self.awaiting_retest:
            if self._check_retest(price):
                self._execute_breakout_entry(bar, price)
            return

        # Check for new breakouts if not in position
        if not self.has_position():
            # Check for upward breakout
            if self._is_upward_breakout(bar, price):
                self._on_upward_breakout(bar, price)

            # Check for downward breakout (breakdown)
            elif self._is_downward_breakout(bar, price):
                self._on_downward_breakout(bar, price)

        # If in position, check for exit conditions
        else:
            if self.is_long():
                # Exit on breakdown below support
                if price < self.range_low:
                    self.log.info(f"💥 Breakdown detected - Exiting long | Price: {price:.2f} | Support: {self.range_low:.2f}")
                    self.close_position()

    def _update_range(self):
        """
        Update the current support and resistance range.

        Calculates highest high and lowest low over lookback period.
        """
        highs = [float(bar.high) for bar in self.range_bars]
        lows = [float(bar.low) for bar in self.range_bars]

        self.range_high = max(highs)
        self.range_low = min(lows)

    def _is_upward_breakout(self, bar: Bar, price: float) -> bool:
        """
        Check if an upward breakout occurred.

        Breakout confirmed when:
        1. Price closes above range high + threshold
        2. Optional: Volume is above average

        Args:
            bar: Current bar.
            price: Current close price.

        Returns:
            True if upward breakout detected.
        """
        if self.range_high is None:
            return False

        # Calculate breakout level
        breakout_level = self.range_high * (1 + self.breakout_threshold)

        # Check price breakout
        price_breakout = price > breakout_level

        # Check volume if required
        volume_ok = True
        if self.volume_confirm and hasattr(bar, 'volume') and self.avg_volume:
            current_volume = float(bar.volume)
            volume_ok = current_volume > (self.avg_volume * self.volume_multiplier)

        return price_breakout and volume_ok

    def _is_downward_breakout(self, bar: Bar, price: float) -> bool:
        """
        Check if a downward breakout (breakdown) occurred.

        Args:
            bar: Current bar.
            price: Current close price.

        Returns:
            True if downward breakout detected.
        """
        if self.range_low is None:
            return False

        # Calculate breakdown level
        breakdown_level = self.range_low * (1 - self.breakout_threshold)

        # Check price breakdown
        price_breakdown = price < breakdown_level

        # Check volume if required
        volume_ok = True
        if self.volume_confirm and hasattr(bar, 'volume') and self.avg_volume:
            current_volume = float(bar.volume)
            volume_ok = current_volume > (self.avg_volume * self.volume_multiplier)

        return price_breakdown and volume_ok

    def _on_upward_breakout(self, bar: Bar, price: float):
        """
        Handle upward breakout signal.

        Args:
            bar: Current bar.
            price: Current price.
        """
        range_height = self.range_high - self.range_low
        range_height_pct = (range_height / self.range_low) * 100
        breakout_strength = ((price - self.range_high) / self.range_high) * 100

        volume_info = ""
        if self.volume_confirm and hasattr(bar, 'volume'):
            volume_ratio = float(bar.volume) / self.avg_volume if self.avg_volume else 0
            volume_info = f" | Volume: {volume_ratio:.2f}x avg"

        self.log.info(
            f"🚀 UPWARD BREAKOUT | "
            f"Price: {price:.2f} | Resistance: {self.range_high:.2f} | "
            f"Range: {range_height_pct:.2f}% | Strength: {breakout_strength:.2f}%{volume_info}"
        )

        # If retest entry is enabled, wait for retest
        if self.retest_entry:
            self.breakout_level = self.range_high
            self.breakout_direction = 'up'
            self.awaiting_retest = True
            self.log.info(f"⏳ Waiting for retest of breakout level: {self.breakout_level:.2f}")
        else:
            # Enter immediately
            self._execute_breakout_entry(bar, price)

    def _on_downward_breakout(self, bar: Bar, price: float):
        """
        Handle downward breakout (breakdown) signal.

        For long-only strategy, we just log this.

        Args:
            bar: Current bar.
            price: Current price.
        """
        self.log.info(
            f"📉 DOWNWARD BREAKOUT (Breakdown) | "
            f"Price: {price:.2f} | Support: {self.range_low:.2f} "
            f"(Long-only strategy - no action)"
        )

    def _check_retest(self, price: float) -> bool:
        """
        Check if breakout level has been retested.

        Retest confirmed when price returns to within 0.5% of breakout level.

        Args:
            price: Current price.

        Returns:
            True if retest occurred.
        """
        if self.breakout_level is None:
            return False

        # For upward breakout, look for pullback to breakout level
        if self.breakout_direction == 'up':
            lower_bound = self.breakout_level * 0.995  # Within 0.5%
            upper_bound = self.breakout_level * 1.005
            return lower_bound <= price <= upper_bound

        return False

    def _execute_breakout_entry(self, bar: Bar, price: float):
        """
        Execute entry on breakout confirmation.

        Args:
            bar: Current bar.
            price: Current price.
        """
        if not self.can_enter_position():
            return

        self.log.info(f"📈 ENTERING LONG POSITION on breakout")

        # Calculate stop loss at recent low
        if self.use_stop_loss and self.range_low:
            # Override stop loss to place it at range low
            self.entry_price = Decimal(str(price))
            stop_distance_pct = ((price - self.range_low) / price)
            self.log.info(f"Stop loss set at range low: {self.range_low:.2f} ({stop_distance_pct * 100:.2f}% away)")

        self.buy()
        self.total_trades += 1

        # Reset retest state
        self.awaiting_retest = False
        self.breakout_level = None
        self.breakout_direction = None

    def on_stop(self):
        """
        Actions when strategy stops.

        Logs final range values.
        """
        if self.range_high and self.range_low:
            range_height = self.range_high - self.range_low
            range_pct = (range_height / self.range_low) * 100
            self.log.info(f"Final Range High  : {self.range_high:.2f}")
            self.log.info(f"Final Range Low   : {self.range_low:.2f}")
            self.log.info(f"Final Range Height: {range_height:.2f} ({range_pct:.2f}%)")

        super().on_stop()


# =============================================================================
# EXAMPLE CONFIGURATIONS
# =============================================================================


def create_breakout_config_standard() -> BreakoutConfig:
    """
    Create a standard breakout configuration.

    Uses 20-bar lookback with basic breakout threshold.

    Returns:
        Standard BreakoutConfig.
    """
    return BreakoutConfig(
        instrument_id="BTCUSDT.BINANCE",
        bar_type="BTCUSDT.BINANCE-1-HOUR-LAST-EXTERNAL",
        lookback_period=20,
        breakout_threshold=0.002,  # 0.2% above high
        volume_confirm=False,
        trade_size=Decimal("100"),
        use_stop_loss=True,
        stop_loss_pct=0.03,  # 3% stop loss
    )


def create_breakout_config_volume_confirmed() -> BreakoutConfig:
    """
    Create a volume-confirmed breakout configuration.

    Requires volume confirmation to filter false breakouts.

    Returns:
        Volume-confirmed BreakoutConfig.
    """
    return BreakoutConfig(
        instrument_id="BTCUSDT.BINANCE",
        bar_type="BTCUSDT.BINANCE-15-MINUTE-LAST-EXTERNAL",
        lookback_period=30,
        breakout_threshold=0.0015,  # 0.15% above high
        volume_confirm=True,
        volume_multiplier=1.5,  # 1.5x average volume required
        trade_size=Decimal("100"),
        use_stop_loss=True,
        stop_loss_pct=0.025,  # 2.5% stop loss
    )


def create_breakout_config_retest() -> BreakoutConfig:
    """
    Create a retest breakout configuration.

    Waits for price to retest breakout level before entering.

    Returns:
        Retest BreakoutConfig.
    """
    return BreakoutConfig(
        instrument_id="BTCUSDT.BINANCE",
        bar_type="BTCUSDT.BINANCE-4-HOUR-LAST-EXTERNAL",
        lookback_period=15,
        breakout_threshold=0.003,  # 0.3% above high
        retest_entry=True,  # Wait for retest
        volume_confirm=True,
        volume_multiplier=1.3,
        trade_size=Decimal("150"),
        use_stop_loss=True,
        stop_loss_pct=0.02,  # 2% stop loss
    )
