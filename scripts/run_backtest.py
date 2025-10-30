#!/usr/bin/env python3
"""
Backtest execution script for Nautilus Trading Platform.

This CLI tool runs backtests with specified strategies and configurations.
Supports:
- Multiple strategies (MACD, EMA Cross, Mean Reversion, Breakout)
- Configurable parameters
- Date range selection
- Results saving and analysis
"""

import argparse
import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from nautilus_trader.model.data import BarSpecification, BarType
from nautilus_trader.model.enums import AggregationSource, BarAggregation, PriceType
from nautilus_trader.model.identifiers import InstrumentId

from configs.backtest_config import BacktestConfig, create_crypto_config
from execution.backtest_runner import BacktestRunner
from strategies import (
    BreakoutConfig,
    BreakoutStrategy,
    EMACrossConfig,
    EMACrossStrategy,
    MACDStrategy,
    MACDStrategyConfig,
    MeanReversionConfig,
    MeanReversionStrategy,
)
from utils.logging_config import get_logger, setup_logging

# Setup logging
setup_logging()
logger = get_logger(__name__)


# Strategy registry
STRATEGIES = {
    "macd": (MACDStrategy, MACDStrategyConfig),
    "ema_cross": (EMACrossStrategy, EMACrossConfig),
    "mean_reversion": (MeanReversionStrategy, MeanReversionConfig),
    "breakout": (BreakoutStrategy, BreakoutConfig),
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run backtests for trading strategies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run EMA Cross strategy on BTCUSDT
  python run_backtest.py --strategy ema_cross --symbol BTCUSDT --start 2024-01-01 --end 2024-03-01

  # Run MACD strategy with custom parameters
  python run_backtest.py --strategy macd --symbol ETHUSDT --start 2024-01-01 --end 2024-02-01 \\
    --params fast_period=10 slow_period=20

  # Run Mean Reversion with 1-hour bars
  python run_backtest.py --strategy mean_reversion --symbol BTCUSDT --interval 1h \\
    --start 2024-01-01 --end 2024-03-01

  # Run Breakout strategy with volume confirmation
  python run_backtest.py --strategy breakout --symbol BTCUSDT --interval 4h \\
    --start 2024-01-01 --end 2024-03-01 --params volume_confirm=True

Available strategies:
  - macd: MACD trend following
  - ema_cross: EMA crossover trend following
  - mean_reversion: Bollinger Bands mean reversion
  - breakout: Range breakout with volume confirmation
        """,
    )

    # Required arguments
    parser.add_argument(
        "--strategy",
        "-s",
        required=True,
        choices=list(STRATEGIES.keys()),
        help="Strategy to backtest",
    )
    parser.add_argument(
        "--symbol",
        required=True,
        help="Trading symbol (e.g., BTCUSDT, ETHUSDT)",
    )
    parser.add_argument(
        "--start",
        required=True,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        required=True,
        help="End date (YYYY-MM-DD)",
    )

    # Optional arguments
    parser.add_argument(
        "--venue",
        default="BINANCE",
        help="Trading venue (default: BINANCE)",
    )
    parser.add_argument(
        "--interval",
        "-i",
        default="1h",
        choices=["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
        help="Bar interval (default: 1h)",
    )
    parser.add_argument(
        "--capital",
        "-c",
        type=float,
        default=10000.0,
        help="Initial capital in USDT (default: 10000)",
    )
    parser.add_argument(
        "--trade-size",
        type=float,
        default=100.0,
        help="Trade size in quote currency (default: 100)",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Data directory (default: data)",
    )
    parser.add_argument(
        "--output-dir",
        default="results/backtests",
        help="Output directory (default: results/backtests)",
    )
    parser.add_argument(
        "--params",
        nargs="*",
        help="Strategy parameters as key=value pairs",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to file",
    )

    return parser.parse_args()


def parse_strategy_params(params_list):
    """
    Parse strategy parameters from command line.

    Args:
        params_list: List of "key=value" strings.

    Returns:
        Dictionary of parsed parameters.
    """
    if not params_list:
        return {}

    params = {}
    for param in params_list:
        if "=" not in param:
            logger.warning(f"Skipping invalid parameter: {param}")
            continue

        key, value = param.split("=", 1)

        # Try to parse value
        try:
            # Try int
            params[key] = int(value)
        except ValueError:
            try:
                # Try float
                params[key] = float(value)
            except ValueError:
                # Try bool
                if value.lower() in ("true", "false"):
                    params[key] = value.lower() == "true"
                else:
                    # Keep as string
                    params[key] = value

    return params


def parse_interval(interval_str):
    """
    Parse interval string to BarAggregation and step.

    Args:
        interval_str: Interval string like "1h", "5m", "1d", "30s"

    Returns:
        tuple: (step, BarAggregation)
    """
    import re

    # Parse interval string
    match = re.match(r"(\d+)([smhd])", interval_str.lower())
    if not match:
        raise ValueError(f"Invalid interval format: {interval_str}")

    step = int(match.group(1))
    unit = match.group(2)

    # Map unit to BarAggregation
    unit_map = {
        "s": BarAggregation.SECOND,
        "m": BarAggregation.MINUTE,
        "h": BarAggregation.HOUR,
        "d": BarAggregation.DAY,
    }

    aggregation = unit_map.get(unit)
    if aggregation is None:
        raise ValueError(f"Invalid time unit: {unit}")

    return step, aggregation


def create_bar_type(symbol, venue, interval):
    """
    Create a BarType object.

    Args:
        symbol: Trading symbol.
        venue: Trading venue.
        interval: Bar interval (e.g., "1h", "5m").

    Returns:
        BarType object.
    """
    # Parse interval
    step, aggregation = parse_interval(interval)

    # Create instrument ID
    instrument_id = InstrumentId.from_str(f"{symbol}.{venue}")

    # Create bar specification
    bar_spec = BarSpecification(
        step=step,
        aggregation=aggregation,
        price_type=PriceType.LAST,
    )

    # Create bar type
    bar_type = BarType(
        instrument_id=instrument_id,
        bar_spec=bar_spec,
        aggregation_source=AggregationSource.EXTERNAL,
    )

    return bar_type


def create_strategy_config(strategy_name, symbol, venue, interval, trade_size, custom_params):
    """
    Create strategy configuration.

    Args:
        strategy_name: Name of the strategy.
        symbol: Trading symbol.
        venue: Trading venue.
        interval: Bar interval.
        trade_size: Trade size.
        custom_params: Custom parameters dictionary.

    Returns:
        Strategy configuration instance.
    """
    _, config_class = STRATEGIES[strategy_name]

    # Create instrument ID and bar type
    instrument_id = f"{symbol}.{venue}"
    bar_type = create_bar_type(symbol, venue, interval)

    # Base parameters
    params = {
        "instrument_id": instrument_id,
        "bar_type": bar_type,
        "trade_size": Decimal(str(trade_size)),
    }

    # Merge custom parameters
    params.update(custom_params)

    logger.info(f"Creating {strategy_name} config with parameters:")
    for key, value in params.items():
        logger.info(f"  {key}: {value}")

    return config_class(**params)


def main():
    """Main entry point."""
    args = parse_args()

    logger.info("=" * 80)
    logger.info("NAUTILUS BACKTEST RUNNER")
    logger.info("=" * 80)

    # Parse strategy parameters
    strategy_params = parse_strategy_params(args.params)

    # Create backtest configuration
    logger.info("Creating backtest configuration")
    backtest_config = create_crypto_config(
        symbol=args.symbol,
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital,
    )
    backtest_config.data.data_path = Path(args.data_dir)
    backtest_config.output_path = Path(args.output_dir)
    backtest_config.save_results = not args.no_save

    # Create strategy configuration
    logger.info("Creating strategy configuration")
    strategy_class, _ = STRATEGIES[args.strategy]
    strategy_config = create_strategy_config(
        strategy_name=args.strategy,
        symbol=args.symbol,
        venue=args.venue,
        interval=args.interval,
        trade_size=args.trade_size,
        custom_params=strategy_params,
    )

    # Create runner
    logger.info("Initializing backtest runner")
    runner = BacktestRunner(backtest_config)

    try:
        # Initialize components
        logger.info("Initializing backtest engine")
        runner.initialize_catalog()
        runner.initialize_engine()

        # Load data
        logger.info("Loading historical data")
        runner.load_data()

        # Add strategy
        logger.info(f"Adding strategy: {strategy_class.__name__}")
        runner.add_strategy(strategy_class, strategy_config)

        # Run backtest
        logger.info("Running backtest")
        results = runner.run()

        # Save results
        if backtest_config.save_results:
            filepath = runner.save_results()
            logger.info(f"Results saved to: {filepath}")

        logger.info("=" * 80)
        logger.info("BACKTEST COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)

        sys.exit(0)

    except Exception as e:
        logger.error(f"Backtest failed: {e}", exc_info=True)
        sys.exit(1)

    finally:
        runner.cleanup()


if __name__ == "__main__":
    main()
