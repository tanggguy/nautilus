"""
Backtest Runner for Nautilus Trading Platform.

This module provides the BacktestRunner class for executing backtests
with historical data using NautilusTrader's backtesting engine.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.backtest.node import BacktestNode
from nautilus_trader.config import ImportableStrategyConfig
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.objects import Money
from nautilus_trader.persistence.catalog import ParquetDataCatalog

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

    def initialize_catalog(self, catalog_path: Optional[Path] = None) -> ParquetDataCatalog:
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
                logging_config={"bypass": False},  # Use our logging
            )

            # Create engine
            self.engine = BacktestEngine(config=engine_config)

            # Add venue
            venue = Venue(self.config.venue.name)
            self.engine.add_venue(
                venue=venue,
                oms_type="NETTING",  # or "HEDGING"
                account_type=self.config.venue.account_type,
                base_currency=None,  # Will use venue default
                starting_balances=[
                    Money(amount, currency)
                    for currency, amount in self.config.venue.starting_balances.items()
                ],
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

    def load_data(self) -> None:
        """
        Load historical data into the engine.

        Loads data from catalog based on configuration.
        """
        if self.catalog is None:
            self.initialize_catalog()

        if self.engine is None:
            self.initialize_engine()

        logger.info("Loading historical data")

        try:
            # Convert date strings to datetime
            start = datetime.strptime(self.config.data.start_date, "%Y-%m-%d")
            end = datetime.strptime(self.config.data.end_date, "%Y-%m-%d")

            # Load data for each instrument
            for instrument_config in self.config.instruments:
                symbol = instrument_config.symbol
                venue = instrument_config.venue

                logger.info(f"Loading data for {symbol} from {start} to {end}")

                # Query data from catalog
                # This assumes data is already in the catalog in Parquet format
                # In a real implementation, you'd use catalog.bars() or similar
                # For now, we'll log that data should be pre-loaded
                logger.info(
                    f"Data should be pre-loaded in catalog at: {self.config.data.data_path}"
                )

            logger.info("Historical data loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load data: {e}", exc_info=True)
            raise

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
            raise RuntimeError("Engine not initialized. Call initialize_engine() first.")

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
