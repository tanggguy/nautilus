"""
Backtest Runner for Nautilus Trading Platform.

This module provides the BacktestRunner class for executing backtests
with historical data using NautilusTrader's backtesting engine.
"""

import json
import pandas as pd
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional

from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.backtest.node import BacktestNode
from nautilus_trader.config import ImportableStrategyConfig
from nautilus_trader.model.currencies import USD, USDT, EUR, BTC, ETH
from nautilus_trader.model.data import Bar, BarSpecification, BarType
from nautilus_trader.model.enums import (
    AccountType,
    OmsType,
    AssetClass,
    CurrencyType,
    AggregationSource,
    BarAggregation,
    PriceType,
)
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
from nautilus_trader.model.instruments import CryptoFuture, CryptoPerpetual, Equity, CurrencyPair
from nautilus_trader.model.objects import Currency, Money, Price, Quantity
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.config import LoggingConfig
from nautilus_trader.test_kit.providers import TestInstrumentProvider

from configs.backtest_config import BacktestConfig
from utils.logging_config import get_logger, setup_logging

logger = get_logger(__name__)


class BacktestRunner:
    """
    Backtest execution runner.

    This class handles:
    - Loading historical data from Parquet/CSV
    - Initializing NautilusTrader backtest engine
    - Adding strategies with configurations
    - Running backtests with memory management
    - Exporting results to files
    - Generating performance reports

    Example:
        >>> from configs.backtest_config import create_crypto_config
        >>> from strategies import EMACrossConfig, EMACrossStrategy
        >>>
        >>> backtest_config = create_crypto_config(
        ...     symbol="BTCUSDT",
        ...     start_date="2024-01-01",
        ...     end_date="2024-03-01"
        ... )
        >>> strategy_config = EMACrossConfig(
        ...     instrument_id="BTCUSDT.BINANCE",
        ...     bar_type="BTCUSDT.BINANCE-1-HOUR-LAST-EXTERNAL"
        ... )
        >>>
        >>> runner = BacktestRunner(backtest_config)
        >>> results = runner.run(EMACrossStrategy, strategy_config)
        >>> runner.save_results(results)
    """

    def __init__(self, config: BacktestConfig):
        """
        Initialize backtest runner.

        Args:
            config: Backtest configuration.
        """
        self.config = config
        self.catalog: Optional[ParquetDataCatalog] = None
        self.engine: Optional[BacktestEngine] = None
        self.results: Dict[str, Any] = {}

        # Setup logging
        if not logger.handlers:
            setup_logging()

        logger.info("BacktestRunner initialized")
        logger.info(f"Backtest: {config.name}")
        logger.info(f"Venue: {config.venue.name}")
        logger.info(f"Instruments: {[i.symbol for i in config.instruments]}")
        logger.info(f"Period: {config.data.start_date} to {config.data.end_date}")

    def initialize_catalog(
        self, catalog_path: Optional[Path] = None
    ) -> ParquetDataCatalog:
        """
        Initialize data catalog for loading historical data.

        Args:
            catalog_path: Path to data catalog (uses config if None).

        Returns:
            Initialized ParquetDataCatalog.
        """
        if catalog_path is None:
            catalog_path = self.config.data.data_path

        logger.info(f"Initializing data catalog: {catalog_path}")

        try:
            self.catalog = ParquetDataCatalog(str(catalog_path))
            logger.info("Data catalog initialized successfully")
            return self.catalog
        except Exception as e:
            logger.error(f"Failed to initialize data catalog: {e}", exc_info=True)
            raise

    def initialize_engine(self) -> BacktestEngine:
        """
        Initialize NautilusTrader backtest engine.

        Returns:
            Configured BacktestEngine.
        """
        logger.info("Initializing backtest engine")

        try:
            # Create engine configuration
            engine_config = BacktestEngineConfig(
                trader_id=f"BACKTESTER-{self.config.name.replace(' ', '_').upper()}",
            )

            # Create engine
            self.engine = BacktestEngine(config=engine_config)

            # Add venue
            venue = Venue(self.config.venue.name)

            # Map currency strings to Currency objects
            currency_map = {
                "USD": USD,
                "USDT": USDT,
                "EUR": EUR,
                "BTC": BTC,
                "ETH": ETH,
            }

            # Create starting balances with proper Currency objects
            starting_balances = []
            for currency_str, amount in self.config.venue.starting_balances.items():
                currency_code = currency_str.upper()
                currency = currency_map.get(currency_code)
                if currency is None:
                    try:
                        currency = Currency.from_str(currency_code)
                    except (
                        ValueError
                    ) as e:  # Assuming ValueError, adjust if Currency.from_str raises something else
                        logger.error(
                            f"Invalid currency '{currency_str}' in configuration. Please check your backtest config.",
                            exc_info=True,
                        )
                        raise e
                starting_balances.append(Money(amount, currency))

            # Map account type string to enum
            account_type_map = {
                "CASH": AccountType.CASH,
                "MARGIN": AccountType.MARGIN,
                "BETTING": AccountType.BETTING,
            }
            account_type = account_type_map.get(
                self.config.venue.account_type.upper(),
                AccountType.CASH  # Default to CASH
            )

            self.engine.add_venue(
                venue=venue,
                oms_type=OmsType.NETTING,  # Use enum instead of string
                account_type=account_type,
                base_currency=None,  # Will use venue default
                starting_balances=starting_balances,
            )

            # Add instruments
            for instrument_config in self.config.instruments:
                logger.info(f"Adding instrument: {instrument_config.symbol}")
                # Instruments will be loaded from catalog automatically

            logger.info("Backtest engine initialized successfully")
            return self.engine

        except Exception as e:
            logger.error(f"Failed to initialize backtest engine: {e}", exc_info=True)
            raise

    def _create_instrument(self, instrument_config) -> CurrencyPair:
        """
        Create a NautilusTrader instrument from configuration.

        Args:
            instrument_config: Instrument configuration.

        Returns:
            CurrencyPair instrument.
        """
        # Extract base and quote currencies from symbol (e.g., BTCUSDT -> BTC, USDT)
        symbol_str = instrument_config.symbol

        # For crypto pairs, typically last 3-4 chars are quote currency
        if symbol_str.endswith("USDT"):
            base_code = symbol_str[:-4]
            quote_code = "USDT"
        elif symbol_str.endswith("USD"):
            base_code = symbol_str[:-3]
            quote_code = "USD"
        elif symbol_str.endswith("BTC"):
            base_code = symbol_str[:-3]
            quote_code = "BTC"
        elif symbol_str.endswith("ETH"):
            base_code = symbol_str[:-3]
            quote_code = "ETH"
        else:
            # Default fallback
            base_code = symbol_str[:3]
            quote_code = symbol_str[3:]

        # Create currencies
        currency_map = {
            "USD": USD,
            "USDT": USDT,
            "EUR": EUR,
            "BTC": BTC,
            "ETH": ETH,
        }

        base_currency = currency_map.get(base_code, Currency.from_str(base_code))
        quote_currency = currency_map.get(quote_code, Currency.from_str(quote_code))

        # Create instrument ID
        instrument_id = InstrumentId(
            symbol=Symbol(instrument_config.symbol),
            venue=Venue(instrument_config.venue),
        )

        # Create CurrencyPair instrument
        instrument = CurrencyPair(
            instrument_id=instrument_id,
            raw_symbol=Symbol(instrument_config.symbol),
            base_currency=base_currency,
            quote_currency=quote_currency,
            price_precision=instrument_config.price_precision,
            size_precision=instrument_config.size_precision,
            price_increment=Price(10 ** -instrument_config.price_precision, instrument_config.price_precision),
            size_increment=Quantity(10 ** -instrument_config.size_precision, instrument_config.size_precision),
            lot_size=Quantity(instrument_config.min_quantity, instrument_config.size_precision),
            max_quantity=None,
            min_quantity=Quantity(instrument_config.min_quantity, instrument_config.size_precision),
            max_price=None,
            min_price=None,
            margin_init=Decimal("0"),
            margin_maint=Decimal("0"),
            maker_fee=Decimal(str(self.config.fees.maker_fee)),
            taker_fee=Decimal(str(self.config.fees.taker_fee)),
            ts_event=0,
            ts_init=0,
        )

        return instrument

    def load_data(self) -> None:
        """
        Load historical data into the engine.

        Loads data from Parquet files and adds instruments and bars to the engine.
        """
        if self.engine is None:
            self.initialize_engine()

        logger.info("Loading historical data")

        try:
            # Load data for each instrument
            for instrument_config in self.config.instruments:
                symbol = instrument_config.symbol
                venue_name = instrument_config.venue

                logger.info(f"Loading data for {symbol}.{venue_name}")

                # Create and add instrument
                instrument = self._create_instrument(instrument_config)
                self.engine.add_instrument(instrument)
                logger.info(f"Added instrument: {instrument.id}")

                # Find and load data file
                # Expected format: yahoo_btcusd_1h.parquet or binance_btcusdt_1h.parquet
                data_path = Path(self.config.data.data_path)

                # Try different file patterns
                possible_files = [
                    data_path / f"{venue_name.lower()}_{symbol.lower()}_*.parquet",
                    data_path / f"*_{symbol.lower()}_*.parquet",
                    data_path / f"*{symbol.lower()}*.parquet",
                ]

                data_file = None
                for pattern in possible_files:
                    matches = list(data_path.glob(pattern.name))
                    if matches:
                        data_file = matches[0]
                        break

                if data_file is None:
                    logger.warning(f"No data file found for {symbol} in {data_path}")
                    logger.warning(f"Tried patterns: {[str(p) for p in possible_files]}")
                    logger.info(f"Available files: {list(data_path.glob('*.parquet'))}")
                    continue

                logger.info(f"Loading data from: {data_file}")

                # Load Parquet file
                df = pd.read_parquet(data_file)
                logger.info(f"Loaded {len(df)} rows from {data_file.name}")

                # Ensure timestamp column exists and is datetime
                if 'timestamp' not in df.columns:
                    if 'date' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['date'])
                    elif df.index.name == 'timestamp' or df.index.name == 'date':
                        df = df.reset_index()
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                    else:
                        raise ValueError(f"No timestamp column found in {data_file}")

                # Ensure timestamp is datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'])

                # Filter by date range
                start = pd.to_datetime(self.config.data.start_date)
                end = pd.to_datetime(self.config.data.end_date)
                df = df[(df['timestamp'] >= start) & (df['timestamp'] <= end)]

                logger.info(f"Filtered to {len(df)} rows between {start.date()} and {end.date()}")

                if len(df) == 0:
                    logger.warning(f"No data found for {symbol} in date range")
                    continue

                # Convert DataFrame to NautilusTrader bars
                bars = self._convert_df_to_bars(df, instrument, self.config.data.bar_type)
                logger.info(f"Converted {len(bars)} bars for {instrument.id}")

                # Add bars to engine
                self.engine.add_data(bars)
                logger.info(f"✅ Added {len(bars)} bars to engine for {symbol}")

            logger.info("Historical data loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load data: {e}", exc_info=True)
            raise

    def _convert_df_to_bars(self, df: pd.DataFrame, instrument: CurrencyPair, bar_spec_str: str) -> list[Bar]:
        """
        Convert DataFrame to NautilusTrader Bar objects.

        Args:
            df: DataFrame with OHLCV data.
            instrument: Instrument for the bars.
            bar_spec_str: Bar specification string (e.g., "1-HOUR").

        Returns:
            List of Bar objects.
        """
        # Parse bar specification
        # Expected format: "1-HOUR", "5-MINUTE", etc.
        parts = bar_spec_str.split("-")
        if len(parts) != 2:
            # Default to 1-HOUR if parsing fails
            step = 1
            aggregation = BarAggregation.HOUR
        else:
            step = int(parts[0])
            aggregation_str = parts[1].upper()
            aggregation_map = {
                "SECOND": BarAggregation.SECOND,
                "MINUTE": BarAggregation.MINUTE,
                "HOUR": BarAggregation.HOUR,
                "DAY": BarAggregation.DAY,
            }
            aggregation = aggregation_map.get(aggregation_str, BarAggregation.HOUR)

        # Create bar type
        bar_spec = BarSpecification(
            step=step,
            aggregation=aggregation,
            price_type=PriceType.LAST,
        )

        bar_type = BarType(
            instrument_id=instrument.id,
            bar_spec=bar_spec,
            aggregation_source=AggregationSource.EXTERNAL,
        )

        # Convert each row to a Bar
        bars = []
        for _, row in df.iterrows():
            # Convert timestamp to nanoseconds
            ts = int(pd.Timestamp(row['timestamp']).timestamp() * 1_000_000_000)

            bar = Bar(
                bar_type=bar_type,
                open=Price(row['open'], instrument.price_precision),
                high=Price(row['high'], instrument.price_precision),
                low=Price(row['low'], instrument.price_precision),
                close=Price(row['close'], instrument.price_precision),
                volume=Quantity(row['volume'], 0),
                ts_event=ts,
                ts_init=ts,
            )
            bars.append(bar)

        return bars

    def add_strategy(self, strategy_class, strategy_config) -> None:
        """
        Add a strategy to the backtest engine.

        Args:
            strategy_class: Strategy class to instantiate.
            strategy_config: Configuration for the strategy.
        """
        if self.engine is None:
            self.initialize_engine()

        logger.info(f"Adding strategy: {strategy_class.__name__}")

        try:
            # Add strategy to engine
            self.engine.add_strategy(strategy_class(config=strategy_config))
            logger.info(f"Strategy {strategy_class.__name__} added successfully")

        except Exception as e:
            logger.error(f"Failed to add strategy: {e}", exc_info=True)
            raise

    def run(self) -> Dict[str, Any]:
        """
        Run the backtest.

        Returns:
            Dictionary containing backtest results.
        """
        if self.engine is None:
            raise RuntimeError(
                "Engine not initialized. Call initialize_engine() first."
            )

        logger.info("=" * 80)
        logger.info("STARTING BACKTEST")
        logger.info("=" * 80)

        try:
            # Run the backtest
            start_time = datetime.now()
            self.engine.run()
            end_time = datetime.now()

            duration = (end_time - start_time).total_seconds()

            logger.info("=" * 80)
            logger.info("BACKTEST COMPLETED")
            logger.info(f"Duration: {duration:.2f} seconds")
            logger.info("=" * 80)

            # Collect results
            self.results = self._collect_results()

            if self.config.run_analysis:
                self._run_analysis()

            return self.results

        except Exception as e:
            logger.error(f"Backtest failed: {e}", exc_info=True)
            raise

    def _collect_results(self) -> Dict[str, Any]:
        """
        Collect results from the backtest engine.

        Returns:
            Dictionary with all backtest results.
        """
        logger.info("Collecting backtest results")

        results = {
            "backtest_name": self.config.name,
            "description": self.config.description,
            "start_date": self.config.data.start_date,
            "end_date": self.config.data.end_date,
            "venue": self.config.venue.name,
            "instruments": [i.symbol for i in self.config.instruments],
            "timestamp": datetime.now().isoformat(),
        }

        # Get account statistics
        try:
            accounts = self.engine.trader.accounts()
            if accounts:
                account = accounts[0]
                results["account"] = {
                    "starting_balance": str(
                        sum(self.config.venue.starting_balances.values())
                    ),
                    "ending_balance": str(account.balance_total()),
                    "currency": self.config.venue.base_currency,
                }
        except Exception as e:
            logger.warning(f"Could not collect account statistics: {e}")

        # Get orders
        try:
            results["orders"] = {
                "total": len(self.engine.trader.cache.orders()),
                "filled": len(self.engine.trader.cache.orders_filled()),
            }
        except Exception as e:
            logger.warning(f"Could not collect order statistics: {e}")

        # Get positions
        try:
            results["positions"] = {
                "total": len(self.engine.trader.cache.positions()),
                "closed": len(self.engine.trader.cache.positions_closed()),
            }
        except Exception as e:
            logger.warning(f"Could not collect position statistics: {e}")

        logger.info("Results collected successfully")
        return results

    def _run_analysis(self) -> None:
        """
        Run performance analysis on backtest results.

        This will be expanded in Phase 4 with detailed metrics.
        """
        logger.info("Running performance analysis")
        logger.info("(Detailed analysis will be implemented in Phase 4)")

        # Basic analysis
        if "account" in self.results:
            starting = float(self.results["account"]["starting_balance"])
            ending = float(self.results["account"]["ending_balance"])
            pnl = ending - starting
            pnl_pct = (pnl / starting) * 100

            logger.info(f"Starting Balance: ${starting:,.2f}")
            logger.info(f"Ending Balance  : ${ending:,.2f}")
            logger.info(f"P&L            : ${pnl:,.2f} ({pnl_pct:+.2f}%)")

            self.results["performance"] = {
                "total_pnl": pnl,
                "total_pnl_pct": pnl_pct,
            }

    def save_results(self, output_path: Optional[Path] = None) -> Path:
        """
        Save backtest results to file.

        Args:
            output_path: Path to save results (uses config if None).

        Returns:
            Path where results were saved.
        """
        if not self.results:
            raise RuntimeError("No results to save. Run backtest first.")

        if output_path is None:
            output_path = self.config.output_path

        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backtest_name = self.config.name.replace(" ", "_").lower()
        filename = f"{backtest_name}_{timestamp}.json"
        filepath = output_path / filename

        logger.info(f"Saving results to: {filepath}")

        try:
            with open(filepath, "w") as f:
                json.dump(self.results, f, indent=2, default=str)

            logger.info("Results saved successfully")
            return filepath

        except Exception as e:
            logger.error(f"Failed to save results: {e}", exc_info=True)
            raise

    def cleanup(self) -> None:
        """
        Cleanup resources and reset engine.

        Call this after backtest to free memory.
        """
        logger.info("Cleaning up backtest runner")

        if self.engine:
            try:
                self.engine.reset()
            except Exception as e:
                logger.warning(f"Error resetting engine: {e}")

        self.engine = None
        self.catalog = None
        self.results = {}

        logger.info("Cleanup complete")


def run_simple_backtest(
    backtest_config: BacktestConfig,
    strategy_class,
    strategy_config,
) -> Dict[str, Any]:
    """
    Run a simple backtest with a single strategy.

    Convenience function that handles all steps.

    Args:
        backtest_config: Backtest configuration.
        strategy_class: Strategy class to test.
        strategy_config: Strategy configuration.

    Returns:
        Backtest results dictionary.

    Example:
        >>> from configs.backtest_config import create_crypto_config
        >>> from strategies import EMACrossStrategy, EMACrossConfig
        >>>
        >>> backtest_config = create_crypto_config()
        >>> strategy_config = EMACrossConfig(
        ...     instrument_id="BTCUSDT.BINANCE",
        ...     bar_type="BTCUSDT.BINANCE-1-HOUR-LAST-EXTERNAL"
        ... )
        >>>
        >>> results = run_simple_backtest(
        ...     backtest_config,
        ...     EMACrossStrategy,
        ...     strategy_config
        ... )
    """
    logger.info("Running simple backtest")

    runner = BacktestRunner(backtest_config)

    try:
        # Initialize
        runner.initialize_catalog()
        runner.initialize_engine()

        # Load data
        runner.load_data()

        # Add strategy
        runner.add_strategy(strategy_class, strategy_config)

        # Run backtest
        results = runner.run()

        # Save results
        if backtest_config.save_results:
            runner.save_results()

        return results

    finally:
        runner.cleanup()
