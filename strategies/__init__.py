"""
Strategies package for Nautilus Trading Platform.

This package provides trading strategies including:
- Base strategy with common functionality
- MACD strategy (trend following)
- EMA Cross strategy (trend following)
- Mean Reversion strategy (range trading)
- Breakout strategy (range breakout)

All strategies inherit from BaseStrategy and provide:
- Position management
- Risk management (stop loss, take profit, trailing stop)
- Unified logging
- Error handling
"""

# Base strategy
from strategies.base_strategy import BaseStrategy, BaseStrategyConfig

# Breakout strategy
from strategies.breakout import (
    BreakoutConfig,
    BreakoutStrategy,
    create_breakout_config_retest,
    create_breakout_config_standard,
    create_breakout_config_volume_confirmed,
)

# EMA Cross strategy
from strategies.ema_cross import (
    EMACrossConfig,
    EMACrossStrategy,
    create_ema_cross_config_aggressive,
    create_ema_cross_config_classic,
    create_ema_cross_config_conservative,
)

# MACD strategy
from strategies.macd_strategy import MACDStrategy, MACDStrategyConfig

# Mean Reversion strategy
from strategies.mean_reversion import (
    MeanReversionConfig,
    MeanReversionStrategy,
    create_mean_reversion_config_standard,
    create_mean_reversion_config_tight,
    create_mean_reversion_config_wide,
)

__all__ = [
    # Base
    "BaseStrategy",
    "BaseStrategyConfig",
    # MACD
    "MACDStrategy",
    "MACDStrategyConfig",
    # EMA Cross
    "EMACrossStrategy",
    "EMACrossConfig",
    "create_ema_cross_config_conservative",
    "create_ema_cross_config_aggressive",
    "create_ema_cross_config_classic",
    # Mean Reversion
    "MeanReversionStrategy",
    "MeanReversionConfig",
    "create_mean_reversion_config_standard",
    "create_mean_reversion_config_tight",
    "create_mean_reversion_config_wide",
    # Breakout
    "BreakoutStrategy",
    "BreakoutConfig",
    "create_breakout_config_standard",
    "create_breakout_config_volume_confirmed",
    "create_breakout_config_retest",
]
