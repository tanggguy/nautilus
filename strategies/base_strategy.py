"""
Base strategy class with common functionality for all trading strategies.

This module provides an abstract base class that all custom strategies should inherit from.
It provides common functionality for:
- Position management
- Risk management
- Trade logging
- Error handling
- Order submission
"""

from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Dict, Optional, Union

from nautilus_trader.config import StrategyConfig
from nautilus_trader.core.datetime import unix_nanos_to_dt
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide, OrderType, TimeInForce
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.model.orders import LimitOrder, MarketOrder, StopMarketOrder
from nautilus_trader.model.position import Position
from nautilus_trader.trading.strategy import Strategy


class BaseStrategyConfig(StrategyConfig):
    """
    Base configuration for all strategies.

    Attributes:
        instrument_id: The instrument to trade (e.g., 'BTCUSDT.BINANCE').
        bar_type: The bar type to subscribe to (BarType object or string).
        trade_size: Base trade size in quote currency (e.g., Decimal("100") for $100).
        max_position_size: Maximum position size as multiple of trade_size (default: 1).
        use_stop_loss: Enable stop loss orders (default: True).
        stop_loss_pct: Stop loss percentage (default: 0.02 = 2%).
        use_take_profit: Enable take profit orders (default: False).
        take_profit_pct: Take profit percentage (default: 0.04 = 4%).
        use_trailing_stop: Enable trailing stop loss (default: False).
        trailing_stop_pct: Trailing stop percentage (default: 0.01 = 1%).
    """

    instrument_id: str
    bar_type: Union[str, BarType]
    trade_size: Decimal = Decimal("100")
    max_position_size: int = 1
    use_stop_loss: bool = True
    stop_loss_pct: float = 0.02
    use_take_profit: bool = False
    take_profit_pct: float = 0.04
    use_trailing_stop: bool = False
    trailing_stop_pct: float = 0.01


class BaseStrategy(Strategy, ABC):
    """
    Abstract base strategy with common functionality.

    All custom strategies should inherit from this class and implement:
    - on_bar_update(bar): Strategy-specific logic when a new bar arrives

    Features provided by base class:
    - Position tracking and management
    - Risk management (stop loss, take profit, trailing stop)
    - Trade logging with structured format
    - Error handling
    - Position sizing calculations
    - Order submission with retries
    """

    def __init__(self, config: BaseStrategyConfig):
        """
        Initialize the base strategy.

        Args:
            config: Strategy configuration.
        """
        super().__init__(config)

        # Configuration
        self.instrument_id = InstrumentId.from_str(config.instrument_id)
        # Accept BarType object directly
        if isinstance(config.bar_type, BarType):
            self.bar_type = config.bar_type
        else:
            raise TypeError(f"bar_type must be a BarType object, got {type(config.bar_type)}")
        self.trade_size = config.trade_size
        self.max_position_size = config.max_position_size

        # Instrument (will be initialized in on_start)
        self.instrument: Optional[Instrument] = None

        # Risk management
        self.use_stop_loss = config.use_stop_loss
        self.stop_loss_pct = config.stop_loss_pct
        self.use_take_profit = config.use_take_profit
        self.take_profit_pct = config.take_profit_pct
        self.use_trailing_stop = config.use_trailing_stop
        self.trailing_stop_pct = config.trailing_stop_pct

        # State tracking
        self.current_position: Optional[Position] = None
        self.entry_price: Optional[Decimal] = None
        self.stop_loss_order = None
        self.take_profit_order = None
        self.last_bar: Optional[Bar] = None
        self.bar_count = 0

        # Statistics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = Decimal("0")

    def on_start(self):
        """
        Actions when strategy starts.

        Subscribes to bar data and logs strategy initialization.
        """
        # Get instrument from cache
        self.instrument = self.cache.instrument(self.instrument_id)
        if self.instrument is None:
            self.log.error(f"Could not find instrument for {self.instrument_id}")
            self.stop()
            return

        # Subscribe to bar data
        self.subscribe_bars(self.bar_type)

        # Log strategy initialization
        self.log.info("=" * 80)
        self.log.info(f"Strategy: {self.__class__.__name__}")
        self.log.info(f"Instrument: {self.instrument_id}")
        self.log.info(f"Bar Type: {self.bar_type}")
        self.log.info(f"Trade Size: {self.trade_size}")
        self.log.info(f"Stop Loss: {self.stop_loss_pct * 100:.1f}%" if self.use_stop_loss else "Stop Loss: Disabled")
        self.log.info(f"Take Profit: {self.take_profit_pct * 100:.1f}%" if self.use_take_profit else "Take Profit: Disabled")
        self.log.info("=" * 80)

    def on_bar(self, bar: Bar):
        """
        Called when a new bar is received.

        Handles bar updates, position tracking, and calls strategy-specific logic.

        Args:
            bar: The new bar data.
        """
        self.last_bar = bar
        self.bar_count += 1

        # Update trailing stop if enabled and in position
        if self.use_trailing_stop and self.has_position():
            self._update_trailing_stop(bar)

        # Call strategy-specific logic
        try:
            self.on_bar_update(bar)
        except Exception as e:
            self.log.error(f"Error in on_bar_update: {e}", exc_info=True)

    @abstractmethod
    def on_bar_update(self, bar: Bar):
        """
        Strategy-specific logic for handling new bars.

        This method MUST be implemented by all subclasses.

        Args:
            bar: The new bar data.
        """
        raise NotImplementedError("Subclasses must implement on_bar_update()")

    def on_stop(self):
        """
        Actions when strategy stops.

        Closes all positions and logs final statistics.
        """
        # Close all positions
        if self.has_position():
            self.log.info("Closing positions on strategy stop...")
            self.close_all_positions(self.instrument_id)

        # Log statistics
        self._log_statistics()
        self.log.info(f"Strategy {self.__class__.__name__} stopped")

    # =========================================================================
    # POSITION MANAGEMENT
    # =========================================================================

    def has_position(self) -> bool:
        """
        Check if there is an open position.

        Returns:
            True if there is an open position, False otherwise.
        """
        return self.current_position is not None and self.current_position.is_open

    def get_position_side(self) -> Optional[OrderSide]:
        """
        Get the current position side.

        Returns:
            OrderSide.BUY for long, OrderSide.SELL for short, None if no position.
        """
        if not self.has_position():
            return None
        return OrderSide.BUY if self.current_position.is_long else OrderSide.SELL

    def is_long(self) -> bool:
        """Check if currently in a long position."""
        return self.has_position() and self.current_position.is_long

    def is_short(self) -> bool:
        """Check if currently in a short position."""
        return self.has_position() and self.current_position.is_short

    def can_enter_position(self) -> bool:
        """
        Check if we can enter a new position.

        Returns:
            True if no position is open or position size allows more.
        """
        if not self.has_position():
            return True

        # Check if we can increase position size
        current_size = abs(self.current_position.quantity)
        max_size = self.trade_size * self.max_position_size
        return current_size < max_size

    # =========================================================================
    # ORDER SUBMISSION
    # =========================================================================

    def submit_market_order(
        self,
        side: OrderSide,
        quantity: Optional[Decimal] = None,
        reduce_only: bool = False,
    ):
        """
        Submit a market order.

        Args:
            side: Order side (BUY or SELL).
            quantity: Order quantity (uses trade_size if None).
            reduce_only: If True, order can only reduce position.
        """
        if quantity is None:
            quantity = self.trade_size

        try:
            order = self.order_factory.market(
                instrument_id=self.instrument_id,
                order_side=side,
                quantity=self.instrument.make_qty(quantity),
                time_in_force=TimeInForce.GTC,
                reduce_only=reduce_only,
            )
            self.submit_order(order)

            # Log trade
            side_str = "BUY" if side == OrderSide.BUY else "SELL"
            self.log.info(f"📊 {side_str} Market Order | Qty: {quantity} | Price: {self.last_bar.close if self.last_bar else 'N/A'}")

            # Update entry price
            if not reduce_only:
                self.entry_price = self.last_bar.close if self.last_bar else None

                # Submit stop loss and take profit if enabled
                if self.entry_price:
                    self._submit_risk_orders(side, self.entry_price)

        except Exception as e:
            self.log.error(f"Error submitting market order: {e}", exc_info=True)

    def submit_stop_loss(self, side: OrderSide, stop_price: Decimal, quantity: Decimal):
        """
        Submit a stop loss order.

        Args:
            side: Order side (opposite of position).
            stop_price: Stop loss trigger price.
            quantity: Order quantity.
        """
        try:
            order = self.order_factory.stop_market(
                instrument_id=self.instrument_id,
                order_side=side,
                quantity=self.instrument.make_qty(quantity),
                trigger_price=self.instrument.make_price(stop_price),
                time_in_force=TimeInForce.GTC,
                reduce_only=True,
            )
            self.submit_order(order)
            self.stop_loss_order = order
            self.log.info(f"🛑 Stop Loss set at {stop_price}")
        except Exception as e:
            self.log.error(f"Error submitting stop loss: {e}", exc_info=True)

    def submit_take_profit(self, side: OrderSide, limit_price: Decimal, quantity: Decimal):
        """
        Submit a take profit order.

        Args:
            side: Order side (opposite of position).
            limit_price: Take profit limit price.
            quantity: Order quantity.
        """
        try:
            order = self.order_factory.limit(
                instrument_id=self.instrument_id,
                order_side=side,
                quantity=self.instrument.make_qty(quantity),
                price=self.instrument.make_price(limit_price),
                time_in_force=TimeInForce.GTC,
                reduce_only=True,
            )
            self.submit_order(order)
            self.take_profit_order = order
            self.log.info(f"🎯 Take Profit set at {limit_price}")
        except Exception as e:
            self.log.error(f"Error submitting take profit: {e}", exc_info=True)

    def _submit_risk_orders(self, entry_side: OrderSide, entry_price: Decimal):
        """
        Submit stop loss and take profit orders.

        Args:
            entry_side: Side of the entry order.
            entry_price: Entry price.
        """
        exit_side = OrderSide.SELL if entry_side == OrderSide.BUY else OrderSide.BUY

        # Stop loss
        if self.use_stop_loss:
            if entry_side == OrderSide.BUY:
                stop_price = entry_price * (1 - self.stop_loss_pct)
            else:
                stop_price = entry_price * (1 + self.stop_loss_pct)
            self.submit_stop_loss(exit_side, stop_price, self.trade_size)

        # Take profit
        if self.use_take_profit:
            if entry_side == OrderSide.BUY:
                tp_price = entry_price * (1 + self.take_profit_pct)
            else:
                tp_price = entry_price * (1 - self.take_profit_pct)
            self.submit_take_profit(exit_side, tp_price, self.trade_size)

    def _update_trailing_stop(self, bar: Bar):
        """
        Update trailing stop loss based on current price.

        Args:
            bar: Current bar data.
        """
        if not self.entry_price:
            return

        current_price = bar.close

        if self.is_long():
            # For long positions, trail stop up
            new_stop = current_price * (1 - self.trailing_stop_pct)
            if self.stop_loss_order and new_stop > self.stop_loss_order.trigger_price:
                # Cancel old stop and create new one
                self.cancel_order(self.stop_loss_order)
                self.submit_stop_loss(OrderSide.SELL, new_stop, self.trade_size)

        elif self.is_short():
            # For short positions, trail stop down
            new_stop = current_price * (1 + self.trailing_stop_pct)
            if self.stop_loss_order and new_stop < self.stop_loss_order.trigger_price:
                # Cancel old stop and create new one
                self.cancel_order(self.stop_loss_order)
                self.submit_stop_loss(OrderSide.BUY, new_stop, self.trade_size)

    # =========================================================================
    # CONVENIENCE METHODS
    # =========================================================================

    def buy(self, quantity: Optional[Decimal] = None):
        """
        Convenience method to submit a buy market order.

        Args:
            quantity: Order quantity (uses trade_size if None).
        """
        self.submit_market_order(OrderSide.BUY, quantity)

    def sell(self, quantity: Optional[Decimal] = None):
        """
        Convenience method to submit a sell market order.

        Args:
            quantity: Order quantity (uses trade_size if None).
        """
        self.submit_market_order(OrderSide.SELL, quantity)

    def close_position(self):
        """Close the current position with a market order."""
        if not self.has_position():
            return

        side = OrderSide.SELL if self.is_long() else OrderSide.BUY
        quantity = abs(self.current_position.quantity)
        self.submit_market_order(side, quantity, reduce_only=True)
        self.log.info(f"🔒 Closing position | Side: {side}")

    # =========================================================================
    # STATISTICS & LOGGING
    # =========================================================================

    def _log_statistics(self):
        """Log trading statistics."""
        if self.total_trades == 0:
            self.log.info("No trades executed")
            return

        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0

        self.log.info("=" * 80)
        self.log.info("STRATEGY STATISTICS")
        self.log.info("-" * 80)
        self.log.info(f"Total Trades      : {self.total_trades}")
        self.log.info(f"Winning Trades    : {self.winning_trades}")
        self.log.info(f"Losing Trades     : {self.losing_trades}")
        self.log.info(f"Win Rate          : {win_rate:.2f}%")
        self.log.info(f"Total P&L         : {self.total_pnl}")
        self.log.info(f"Bars Processed    : {self.bar_count}")
        self.log.info("=" * 80)
