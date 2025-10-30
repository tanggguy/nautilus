"""
Mean Reversion Strategy using Bollinger Bands.

This strategy assumes that prices will revert to their mean after extreme deviations.
It uses Bollinger Bands to identify overbought and oversold conditions.

Trading signals:
- BUY when price touches or crosses below lower Bollinger Band (oversold)
- SELL when price touches or crosses above upper Bollinger Band (overbought)
- EXIT when price returns to middle band (mean)

Best suited for:
- Range-bound markets
- High volatility instruments
- Mean-reverting assets
"""

from decimal import Decimal

from nautilus_trader.indicators.bollinger_bands import BollingerBands
from nautilus_trader.model.data import Bar
from nautilus_trader.model.enums import OrderSide

from strategies.base_strategy import BaseStrategy, BaseStrategyConfig


class MeanReversionConfig(BaseStrategyConfig):
    """
    Configuration for Mean Reversion strategy.

    Attributes:
        instrument_id: The instrument to trade.
        bar_type: The bar type to subscribe to.
        bb_period: Period for Bollinger Bands calculation (default: 20).
        bb_std_dev: Number of standard deviations for bands (default: 2.0).
        oversold_threshold: Threshold below lower band to trigger buy (default: 0.0).
        overbought_threshold: Threshold above upper band to trigger sell (default: 0.0).
        exit_on_middle: Exit position when price returns to middle band (default: True).
        trade_size: Base trade size in quote currency.
        use_stop_loss: Enable stop loss (default: True).
        stop_loss_pct: Stop loss percentage (default: 0.03 = 3%).
    """

    bb_period: int = 20
    bb_std_dev: float = 2.0
    oversold_threshold: float = 0.0
    overbought_threshold: float = 0.0
    exit_on_middle: bool = True


class MeanReversionStrategy(BaseStrategy):
    """
    Mean Reversion Strategy using Bollinger Bands.

    Entry signals:
    - LONG: Price touches/crosses below lower Bollinger Band (price is oversold)
    - SHORT: Price touches/crosses above upper Bollinger Band (price is overbought)

    Exit signals:
    - Close LONG: Price returns to middle band OR hits upper band
    - Close SHORT: Price returns to middle band OR hits lower band

    Risk Management:
    - Configurable stop loss (typically wider for mean reversion)
    - Automatic exit when price reverts to mean (middle band)
    - Optional take profit at opposite band

    Example:
        Price drops below lower band -> BUY (expect reversion up)
        Price rises to middle band -> SELL (take profit on reversion)

    Best suited for:
    - Sideways/ranging markets
    - High volatility periods
    - Assets that show mean-reverting behavior

    Not suitable for:
    - Strong trending markets (will get stopped out frequently)
    - Low volatility environments (narrow bands, fewer signals)

    Example usage:
        >>> config = MeanReversionConfig(
        ...     instrument_id="BTCUSDT.BINANCE",
        ...     bar_type="BTCUSDT.BINANCE-15-MINUTE-LAST-EXTERNAL",
        ...     bb_period=20,
        ...     bb_std_dev=2.0,
        ...     trade_size=Decimal("100"),
        ...     exit_on_middle=True
        ... )
        >>> strategy = MeanReversionStrategy(config)
    """

    def __init__(self, config: MeanReversionConfig):
        """
        Initialize Mean Reversion strategy.

        Args:
            config: Strategy configuration.
        """
        super().__init__(config)

        # Strategy-specific configuration
        self.bb_period = config.bb_period
        self.bb_std_dev = config.bb_std_dev
        self.oversold_threshold = config.oversold_threshold
        self.overbought_threshold = config.overbought_threshold
        self.exit_on_middle = config.exit_on_middle

        # Initialize Bollinger Bands indicator
        self.bollinger_bands = BollingerBands(
            period=self.bb_period,
            k=self.bb_std_dev,
        )

        # State tracking
        self.last_signal = None  # Track last signal to avoid duplicate entries
        self.entry_band = None  # Track which band triggered entry

    def on_start(self):
        """
        Actions when strategy starts.

        Logs strategy-specific configuration.
        """
        super().on_start()
        self.log.info(f"BB Period         : {self.bb_period}")
        self.log.info(f"BB Std Dev        : {self.bb_std_dev}")
        self.log.info(f"Exit on Middle    : {self.exit_on_middle}")
        self.log.info(f"Oversold Thresh   : {self.oversold_threshold}")
        self.log.info(f"Overbought Thresh : {self.overbought_threshold}")
        self.log.info("-" * 80)

    def on_bar_update(self, bar: Bar):
        """
        Strategy logic for handling new bars.

        Checks for mean reversion signals using Bollinger Bands.

        Args:
            bar: The new bar data.
        """
        # Update Bollinger Bands
        self.bollinger_bands.handle_bar(bar)

        # Wait for indicator to initialize
        if not self.bollinger_bands.initialized:
            return

        # Get current values
        price = float(bar.close)
        upper_band = self.bollinger_bands.upper
        middle_band = self.bollinger_bands.middle
        lower_band = self.bollinger_bands.lower

        # Calculate distance from bands as percentage
        band_width = upper_band - lower_band
        if band_width == 0:
            return  # Avoid division by zero

        # Check for entry signals
        if not self.has_position():
            # Check for oversold condition (buy signal)
            if self._is_oversold(price, lower_band, band_width):
                self._on_oversold_signal(bar, price, lower_band, middle_band, upper_band)

            # Check for overbought condition (potential short, but we trade long only)
            elif self._is_overbought(price, upper_band, band_width):
                # Log but don't trade (long-only strategy)
                self.log.info(
                    f"🔴 Overbought | Price: {price:.2f} | Upper: {upper_band:.2f} "
                    f"(No short position opened)"
                )

        # Check for exit signals if in position
        else:
            if self.is_long():
                # Exit conditions for long position
                if self._should_exit_long(price, middle_band, upper_band):
                    self._exit_long_position(bar, price, middle_band, upper_band)

    def _is_oversold(self, price: float, lower_band: float, band_width: float) -> bool:
        """
        Check if price is oversold (below or near lower Bollinger Band).

        Args:
            price: Current price.
            lower_band: Lower Bollinger Band value.
            band_width: Width of Bollinger Bands.

        Returns:
            True if oversold condition detected.
        """
        # Check if price is below lower band + threshold
        threshold_price = lower_band + (band_width * self.oversold_threshold)
        return price <= threshold_price

    def _is_overbought(self, price: float, upper_band: float, band_width: float) -> bool:
        """
        Check if price is overbought (above or near upper Bollinger Band).

        Args:
            price: Current price.
            upper_band: Upper Bollinger Band value.
            band_width: Width of Bollinger Bands.

        Returns:
            True if overbought condition detected.
        """
        # Check if price is above upper band - threshold
        threshold_price = upper_band - (band_width * self.overbought_threshold)
        return price >= threshold_price

    def _should_exit_long(self, price: float, middle_band: float, upper_band: float) -> bool:
        """
        Determine if long position should be exited.

        Exit conditions:
        1. Price returned to middle band (if exit_on_middle=True)
        2. Price reached upper band (take profit)

        Args:
            price: Current price.
            middle_band: Middle Bollinger Band value.
            upper_band: Upper Bollinger Band value.

        Returns:
            True if position should be exited.
        """
        # Exit if price returns to middle band
        if self.exit_on_middle and price >= middle_band:
            return True

        # Exit if price reaches upper band (full reversion)
        if price >= upper_band:
            return True

        return False

    def _on_oversold_signal(
        self,
        bar: Bar,
        price: float,
        lower_band: float,
        middle_band: float,
        upper_band: float,
    ):
        """
        Handle oversold signal - enter long position.

        Args:
            bar: Current bar.
            price: Current price.
            lower_band: Lower Bollinger Band value.
            middle_band: Middle Bollinger Band value.
            upper_band: Upper Bollinger Band value.
        """
        distance_from_lower = ((price - lower_band) / lower_band) * 100
        band_width_pct = ((upper_band - lower_band) / middle_band) * 100

        self.log.info(
            f"🟢 OVERSOLD SIGNAL | "
            f"Price: {price:.2f} | Lower Band: {lower_band:.2f} | "
            f"Distance: {distance_from_lower:.2f}% | "
            f"Band Width: {band_width_pct:.2f}%"
        )

        if self.can_enter_position():
            self.log.info("📈 ENTERING LONG POSITION (Mean Reversion)")
            self.buy()
            self.entry_band = "lower"
            self.last_signal = "oversold"
            self.total_trades += 1

    def _exit_long_position(
        self,
        bar: Bar,
        price: float,
        middle_band: float,
        upper_band: float,
    ):
        """
        Exit long position on mean reversion.

        Args:
            bar: Current bar.
            price: Current price.
            middle_band: Middle Bollinger Band value.
            upper_band: Upper Bollinger Band value.
        """
        # Determine exit reason
        if price >= upper_band:
            exit_reason = "Upper Band Reached (Full Reversion)"
        elif price >= middle_band:
            exit_reason = "Middle Band Reached (Mean Reversion)"
        else:
            exit_reason = "Other"

        self.log.info(
            f"📉 EXITING LONG POSITION | "
            f"Price: {price:.2f} | Middle: {middle_band:.2f} | "
            f"Upper: {upper_band:.2f} | Reason: {exit_reason}"
        )

        self.close_position()
        self.entry_band = None
        self.last_signal = None

    def on_stop(self):
        """
        Actions when strategy stops.

        Logs final Bollinger Band values.
        """
        if self.bollinger_bands.initialized:
            self.log.info(f"Final Upper Band  : {self.bollinger_bands.upper:.2f}")
            self.log.info(f"Final Middle Band : {self.bollinger_bands.middle:.2f}")
            self.log.info(f"Final Lower Band  : {self.bollinger_bands.lower:.2f}")

        super().on_stop()


# =============================================================================
# EXAMPLE CONFIGURATIONS
# =============================================================================


def create_mean_reversion_config_standard() -> MeanReversionConfig:
    """
    Create a standard mean reversion configuration.

    Uses classic 20-period Bollinger Bands with 2 standard deviations.

    Returns:
        Standard MeanReversionConfig.
    """
    return MeanReversionConfig(
        instrument_id="BTCUSDT.BINANCE",
        bar_type="BTCUSDT.BINANCE-15-MINUTE-LAST-EXTERNAL",
        bb_period=20,
        bb_std_dev=2.0,
        trade_size=Decimal("100"),
        exit_on_middle=True,
        use_stop_loss=True,
        stop_loss_pct=0.03,  # 3% stop loss (wider for mean reversion)
    )


def create_mean_reversion_config_tight() -> MeanReversionConfig:
    """
    Create a tight mean reversion configuration.

    Uses tighter bands (1.5 std dev) for more frequent signals.

    Returns:
        Tight MeanReversionConfig.
    """
    return MeanReversionConfig(
        instrument_id="BTCUSDT.BINANCE",
        bar_type="BTCUSDT.BINANCE-5-MINUTE-LAST-EXTERNAL",
        bb_period=20,
        bb_std_dev=1.5,  # Tighter bands
        trade_size=Decimal("50"),  # Smaller size for higher frequency
        exit_on_middle=True,
        use_stop_loss=True,
        stop_loss_pct=0.02,  # 2% stop loss
    )


def create_mean_reversion_config_wide() -> MeanReversionConfig:
    """
    Create a wide mean reversion configuration.

    Uses wider bands (2.5 std dev) for fewer but stronger signals.

    Returns:
        Wide MeanReversionConfig.
    """
    return MeanReversionConfig(
        instrument_id="BTCUSDT.BINANCE",
        bar_type="BTCUSDT.BINANCE-1-HOUR-LAST-EXTERNAL",
        bb_period=20,
        bb_std_dev=2.5,  # Wider bands
        trade_size=Decimal("150"),  # Larger size for lower frequency
        exit_on_middle=False,  # Wait for full reversion to upper band
        use_stop_loss=True,
        stop_loss_pct=0.04,  # 4% stop loss (wider for stronger reversions)
    )
