#!/usr/bin/env python3
"""
Complete End-to-End Backtest Example for Nautilus Trading Platform.

This example demonstrates the complete workflow:
1. Download historical data
2. Validate data quality
3. Configure and run backtest
4. Analyze results

Usage:
    python examples/complete_backtest_example.py

Or run step by step by uncommenting sections.
"""

import sys
from decimal import Decimal
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.backtest_config import (
    BacktestConfig,
    DataConfig,
    FeeConfig,
    InstrumentConfig,
    RiskConfig,
    VenueConfig,
)
from scripts.data_download import DataDownloader
from scripts.data_validation import DataValidator
from strategies import EMACrossConfig, EMACrossStrategy
from utils.logging_config import get_logger, setup_logging

# Setup logging
setup_logging()
logger = get_logger(__name__)


def main():
    """Run complete backtest example."""

    logger.info("=" * 80)
    logger.info("NAUTILUS TRADING PLATFORM - COMPLETE BACKTEST EXAMPLE")
    logger.info("=" * 80)

    # =========================================================================
    # STEP 1: DOWNLOAD HISTORICAL DATA
    # =========================================================================

    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: DOWNLOADING HISTORICAL DATA")
    logger.info("=" * 80)

    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # Initialize downloader
    downloader = DataDownloader(output_dir=data_dir)

    # Download Bitcoin data from Yahoo Finance
    symbol = "BTC-USD"
    start_date = "2024-01-01"
    end_date = "2024-03-01"
    interval = "1h"

    logger.info(f"Downloading {symbol} from {start_date} to {end_date}")
    logger.info(f"Interval: {interval}")

    # Download data
    df = downloader.download_yahoo_finance(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        interval=interval,
    )

    if df is None:
        logger.error("Failed to download data. Exiting.")
        sys.exit(1)

    logger.info(f"✅ Downloaded {len(df):,} bars")

    # Save to Parquet
    data_file = downloader.save_to_parquet(
        df=df,
        symbol=symbol,
        interval=interval,
        venue="YAHOO",
    )

    if data_file is None:
        logger.error("Failed to save data. Exiting.")
        sys.exit(1)

    logger.info(f"✅ Data saved to: {data_file}")

    # =========================================================================
    # STEP 2: VALIDATE DATA QUALITY
    # =========================================================================

    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: VALIDATING DATA QUALITY")
    logger.info("=" * 80)

    # Initialize validator
    validator = DataValidator(strict=False)

    # Validate the downloaded data
    is_valid, report = validator.validate_file(data_file)

    if not is_valid:
        logger.error("❌ Data validation failed!")
        logger.error(f"Issues found: {len(report.get('issues', []))}")
        for issue in report.get('issues', []):
            logger.error(f"  - {issue['message']}")

        # Continue anyway for demo purposes
        logger.warning("Continuing despite validation errors for demo...")
    else:
        logger.info("✅ Data validation passed!")

    # Show data info
    logger.info(f"Total rows: {report['total_rows']:,}")
    logger.info(f"Date range: {report['date_range']['start']} to {report['date_range']['end']}")
    logger.info(f"Warnings: {len(report.get('warnings', []))}")

    # =========================================================================
    # STEP 3: CONFIGURE BACKTEST
    # =========================================================================

    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: CONFIGURING BACKTEST")
    logger.info("=" * 80)

    # Create venue configuration
    venue_config = VenueConfig(
        name="BINANCE",  # We'll pretend Yahoo data is from Binance for demo
        venue_type="EXCHANGE",
        account_type="CASH",
        base_currency="USDT",
        starting_balances={"USDT": 10000.0},  # $10,000 starting capital
    )

    logger.info(f"Venue: {venue_config.name}")
    logger.info(f"Starting capital: ${venue_config.starting_balances['USDT']:,.2f}")

    # Create instrument configuration
    instrument_config = InstrumentConfig(
        symbol="BTCUSDT",
        venue="BINANCE",
        instrument_type="SPOT",
        price_precision=2,
        size_precision=8,
        min_quantity=0.00001,
    )

    logger.info(f"Instrument: {instrument_config.symbol}")

    # Create data configuration
    data_config = DataConfig(
        data_path=data_dir,
        start_date=start_date,
        end_date=end_date,
        bar_type="1-HOUR",
        data_format="parquet",
    )

    logger.info(f"Data period: {start_date} to {end_date}")

    # Create risk configuration
    risk_config = RiskConfig(
        max_order_size=0.1,  # 10% of capital per order
        max_position_size=0.2,  # 20% max position
        max_open_positions=1,
        default_stop_loss=0.02,  # 2% stop loss
        default_take_profit=0.04,  # 4% take profit
        max_daily_loss=0.05,  # 5% max daily loss
    )

    logger.info(f"Risk: {risk_config.default_stop_loss * 100:.1f}% stop loss, "
                f"{risk_config.default_take_profit * 100:.1f}% take profit")

    # Create fee configuration
    fee_config = FeeConfig(
        maker_fee=0.001,  # 0.1% maker fee
        taker_fee=0.001,  # 0.1% taker fee
    )

    logger.info(f"Fees: {fee_config.maker_fee * 100:.2f}% maker, "
                f"{fee_config.taker_fee * 100:.2f}% taker")

    # Create main backtest configuration
    backtest_config = BacktestConfig(
        name="EMA Cross BTC Example",
        description="Example backtest of EMA crossover strategy on Bitcoin",
        venue=venue_config,
        instruments=[instrument_config],
        data=data_config,
        risk=risk_config,
        fees=fee_config,
        output_path=Path("results/backtests"),
        save_results=True,
        run_analysis=True,
    )

    logger.info("✅ Backtest configuration created")

    # =========================================================================
    # STEP 4: CONFIGURE STRATEGY
    # =========================================================================

    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: CONFIGURING STRATEGY")
    logger.info("=" * 80)

    # Create EMA Cross strategy configuration
    strategy_config = EMACrossConfig(
        instrument_id="BTCUSDT.BINANCE",
        bar_type="BTCUSDT.BINANCE-1-HOUR-LAST-EXTERNAL",
        fast_period=10,  # Fast EMA period
        slow_period=30,  # Slow EMA period
        trade_size=Decimal("1000"),  # $1000 per trade
        use_stop_loss=True,
        stop_loss_pct=0.02,  # 2% stop loss
        use_take_profit=False,  # Let profits run
        use_trailing_stop=False,
    )

    logger.info("Strategy: EMA Cross")
    logger.info(f"Fast EMA: {strategy_config.fast_period} periods")
    logger.info(f"Slow EMA: {strategy_config.slow_period} periods")
    logger.info(f"Trade size: ${strategy_config.trade_size}")
    logger.info("✅ Strategy configuration created")

    # =========================================================================
    # STEP 5: RUN BACKTEST
    # =========================================================================

    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: RUNNING BACKTEST")
    logger.info("=" * 80)

    # Note: This part would require actual NautilusTrader installation
    # For demonstration, we'll show how it would be called

    logger.info("Note: Actual backtest execution requires NautilusTrader to be installed")
    logger.info("      and properly configured with historical data.")
    logger.info("")
    logger.info("To run the backtest, you would use:")
    logger.info("")
    logger.info("  from execution.backtest_runner import run_simple_backtest")
    logger.info("")
    logger.info("  results = run_simple_backtest(")
    logger.info("      backtest_config=backtest_config,")
    logger.info("      strategy_class=EMACrossStrategy,")
    logger.info("      strategy_config=strategy_config,")
    logger.info("  )")
    logger.info("")

    # Simulated results for demo
    logger.info("📊 SIMULATED RESULTS (for demonstration):")
    logger.info("-" * 80)

    simulated_results = {
        "backtest_name": "EMA Cross BTC Example",
        "start_date": start_date,
        "end_date": end_date,
        "starting_balance": 10000.0,
        "ending_balance": 11234.56,
        "total_pnl": 1234.56,
        "total_pnl_pct": 12.35,
        "total_trades": 15,
        "winning_trades": 9,
        "losing_trades": 6,
        "win_rate": 60.0,
        "bars_processed": len(df),
    }

    logger.info(f"Starting Balance  : ${simulated_results['starting_balance']:,.2f}")
    logger.info(f"Ending Balance    : ${simulated_results['ending_balance']:,.2f}")
    logger.info(f"Total P&L         : ${simulated_results['total_pnl']:,.2f} "
                f"({simulated_results['total_pnl_pct']:+.2f}%)")
    logger.info(f"Total Trades      : {simulated_results['total_trades']}")
    logger.info(f"Winning Trades    : {simulated_results['winning_trades']}")
    logger.info(f"Losing Trades     : {simulated_results['losing_trades']}")
    logger.info(f"Win Rate          : {simulated_results['win_rate']:.2f}%")
    logger.info(f"Bars Processed    : {simulated_results['bars_processed']:,}")

    # =========================================================================
    # STEP 6: ALTERNATIVE - USE CLI
    # =========================================================================

    logger.info("\n" + "=" * 80)
    logger.info("STEP 6: ALTERNATIVE - USING CLI")
    logger.info("=" * 80)

    logger.info("Instead of the Python API, you can use the CLI tool:")
    logger.info("")
    logger.info("  python scripts/run_backtest.py \\")
    logger.info("    --strategy ema_cross \\")
    logger.info("    --symbol BTCUSDT \\")
    logger.info("    --start 2024-01-01 \\")
    logger.info("    --end 2024-03-01 \\")
    logger.info("    --interval 1h \\")
    logger.info("    --capital 10000 \\")
    logger.info("    --trade-size 1000 \\")
    logger.info("    --params fast_period=10 slow_period=30")
    logger.info("")

    # =========================================================================
    # SUMMARY
    # =========================================================================

    logger.info("\n" + "=" * 80)
    logger.info("EXAMPLE COMPLETED!")
    logger.info("=" * 80)
    logger.info("")
    logger.info("What we did:")
    logger.info("  ✅ Downloaded historical Bitcoin data from Yahoo Finance")
    logger.info("  ✅ Validated data quality (OHLCV relationships, gaps, outliers)")
    logger.info("  ✅ Configured backtest with venue, instruments, risk management")
    logger.info("  ✅ Configured EMA Cross strategy with parameters")
    logger.info("  ✅ Showed how to run backtest (Python API and CLI)")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Install NautilusTrader: pip install nautilus_trader")
    logger.info("  2. Run actual backtest with your data")
    logger.info("  3. Analyze results with Phase 4 metrics (coming soon)")
    logger.info("  4. Optimize parameters with Phase 5 (Optuna)")
    logger.info("  5. Test in paper trading before going live")
    logger.info("")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
