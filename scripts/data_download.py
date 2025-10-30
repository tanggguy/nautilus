#!/usr/bin/env python3
"""
Historical data download script for Nautilus Trading Platform.

This script downloads historical price data from various sources:
- Yahoo Finance (via yfinance) for stocks/crypto
- Binance (direct API) for crypto pairs
- CSV import for custom data

Data is saved in Parquet format for efficient storage and loading.
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

from utils.logging_config import get_logger, setup_logging

# Setup logging
setup_logging()
logger = get_logger(__name__)


class DataDownloader:
    """
    Download historical market data from various sources.

    Supports:
    - Yahoo Finance (stocks, crypto, ETFs, indices)
    - Binance API (crypto pairs)
    - CSV import

    Data is saved in Parquet format for efficient loading.
    """

    def __init__(self, output_dir: Path = Path("data")):
        """
        Initialize data downloader.

        Args:
            output_dir: Directory to save downloaded data.
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"DataDownloader initialized | Output: {output_dir}")

    def download_yahoo_finance(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = "1h",
    ) -> Optional[pd.DataFrame]:
        """
        Download data from Yahoo Finance.

        Args:
            symbol: Ticker symbol (e.g., 'BTC-USD', 'AAPL', 'SPY').
            start_date: Start date in YYYY-MM-DD format.
            end_date: End date in YYYY-MM-DD format.
            interval: Data interval ('1m', '5m', '15m', '1h', '1d').

        Returns:
            DataFrame with OHLCV data, or None if download fails.
        """
        logger.info(f"Downloading {symbol} from Yahoo Finance")
        logger.info(f"Period: {start_date} to {end_date} | Interval: {interval}")

        try:
            ticker = yf.Ticker(symbol)

            # Download data
            df = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=True,  # Adjust for splits/dividends
            )

            if df.empty:
                logger.error(f"No data returned for {symbol}")
                return None

            # Clean column names
            df.columns = [col.lower() for col in df.columns]

            # Ensure required columns
            required_cols = ["open", "high", "low", "close", "volume"]
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                logger.error(f"Missing columns: {missing}")
                return None

            # Reset index to make timestamp a column
            df = df.reset_index()

            # Clean column names again after reset_index (in case index name has different case)
            df.columns = [col.lower() for col in df.columns]

            # Rename columns to match Nautilus format
            if "date" in df.columns:
                df = df.rename(columns={"date": "timestamp"})
            elif "datetime" in df.columns:
                df = df.rename(columns={"datetime": "timestamp"})

            # Verify timestamp column exists
            if "timestamp" not in df.columns:
                logger.error(f"Could not create timestamp column. Available columns: {df.columns.tolist()}")
                return None

            logger.info(f"Downloaded {len(df)} bars for {symbol}")
            logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

            return df

        except Exception as e:
            logger.error(f"Failed to download {symbol}: {e}", exc_info=True)
            return None

    def download_binance(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = "1h",
    ) -> Optional[pd.DataFrame]:
        """
        Download data from Binance API.

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT', 'ETHUSDT').
            start_date: Start date in YYYY-MM-DD format.
            end_date: End date in YYYY-MM-DD format.
            interval: Kline interval ('1m', '5m', '15m', '1h', '1d').

        Returns:
            DataFrame with OHLCV data, or None if download fails.

        Note:
            This is a placeholder. Implement with python-binance library
            or direct API calls for production use.
        """
        logger.info(f"Downloading {symbol} from Binance")
        logger.info(f"Period: {start_date} to {end_date} | Interval: {interval}")

        try:
            # For now, use yfinance as fallback
            # In production, use binance.client.Client()
            yf_symbol = symbol.replace("USDT", "-USD")
            logger.info(f"Using Yahoo Finance as fallback: {yf_symbol}")
            return self.download_yahoo_finance(yf_symbol, start_date, end_date, interval)

        except Exception as e:
            logger.error(f"Failed to download {symbol} from Binance: {e}", exc_info=True)
            return None

    def save_to_parquet(
        self,
        df: pd.DataFrame,
        symbol: str,
        interval: str,
        venue: str = "YAHOO",
    ) -> Optional[Path]:
        """
        Save DataFrame to Parquet format.

        Args:
            df: DataFrame with OHLCV data.
            symbol: Symbol name.
            interval: Data interval.
            venue: Venue name (e.g., 'YAHOO', 'BINANCE').

        Returns:
            Path to saved file, or None if save fails.
        """
        # Create filename
        filename = f"{venue.lower()}_{symbol.replace('-', '').lower()}_{interval}.parquet"
        filepath = self.output_dir / filename

        logger.info(f"Saving data to: {filepath}")

        try:
            # Save to Parquet with compression
            df.to_parquet(
                filepath,
                engine="pyarrow",
                compression="snappy",
                index=False,
            )

            file_size = filepath.stat().st_size / 1024 / 1024  # MB
            logger.info(f"Saved successfully | Size: {file_size:.2f} MB | Rows: {len(df)}")

            return filepath

        except Exception as e:
            logger.error(f"Failed to save data: {e}", exc_info=True)
            return None

    def save_to_csv(
        self,
        df: pd.DataFrame,
        symbol: str,
        interval: str,
        venue: str = "YAHOO",
    ) -> Optional[Path]:
        """
        Save DataFrame to CSV format.

        Args:
            df: DataFrame with OHLCV data.
            symbol: Symbol name.
            interval: Data interval.
            venue: Venue name.

        Returns:
            Path to saved file, or None if save fails.
        """
        # Create filename
        filename = f"{venue.lower()}_{symbol.replace('-', '').lower()}_{interval}.csv"
        filepath = self.output_dir / filename

        logger.info(f"Saving data to: {filepath}")

        try:
            df.to_csv(filepath, index=False)

            file_size = filepath.stat().st_size / 1024 / 1024  # MB
            logger.info(f"Saved successfully | Size: {file_size:.2f} MB | Rows: {len(df)}")

            return filepath

        except Exception as e:
            logger.error(f"Failed to save data: {e}", exc_info=True)
            return None

    def load_from_csv(self, csv_path: Path) -> Optional[pd.DataFrame]:
        """
        Load data from CSV file.

        Args:
            csv_path: Path to CSV file.

        Returns:
            DataFrame with data, or None if load fails.
        """
        logger.info(f"Loading data from: {csv_path}")

        try:
            df = pd.read_csv(csv_path)

            # Try to parse timestamp column
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            elif "date" in df.columns:
                df["timestamp"] = pd.to_datetime(df["date"])
                df = df.drop(columns=["date"])

            logger.info(f"Loaded {len(df)} rows from CSV")
            return df

        except Exception as e:
            logger.error(f"Failed to load CSV: {e}", exc_info=True)
            return None


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Download historical market data for backtesting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download Bitcoin data from Yahoo Finance
  python data_download.py --symbol BTC-USD --start 2024-01-01 --end 2024-03-01 --interval 1h

  # Download from Binance
  python data_download.py --symbol BTCUSDT --venue binance --start 2024-01-01 --end 2024-03-01

  # Download stock data
  python data_download.py --symbol AAPL --start 2023-01-01 --end 2024-01-01 --interval 1d

  # Save as CSV instead of Parquet
  python data_download.py --symbol ETH-USD --start 2024-01-01 --end 2024-02-01 --format csv
        """,
    )

    parser.add_argument(
        "--symbol",
        "-s",
        required=True,
        help="Symbol to download (e.g., BTC-USD, AAPL, BTCUSDT)",
    )
    parser.add_argument(
        "--start",
        required=True,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        default=datetime.now().strftime("%Y-%m-%d"),
        help="End date (YYYY-MM-DD), default: today",
    )
    parser.add_argument(
        "--interval",
        "-i",
        default="1h",
        choices=["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
        help="Data interval (default: 1h)",
    )
    parser.add_argument(
        "--venue",
        "-v",
        default="yahoo",
        choices=["yahoo", "binance"],
        help="Data source (default: yahoo)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="data",
        help="Output directory (default: data)",
    )
    parser.add_argument(
        "--format",
        "-f",
        default="parquet",
        choices=["parquet", "csv"],
        help="Output format (default: parquet)",
    )

    args = parser.parse_args()

    # Initialize downloader
    downloader = DataDownloader(output_dir=Path(args.output))

    # Download data based on venue
    if args.venue == "yahoo":
        df = downloader.download_yahoo_finance(
            symbol=args.symbol,
            start_date=args.start,
            end_date=args.end,
            interval=args.interval,
        )
    elif args.venue == "binance":
        df = downloader.download_binance(
            symbol=args.symbol,
            start_date=args.start,
            end_date=args.end,
            interval=args.interval,
        )
    else:
        logger.error(f"Unknown venue: {args.venue}")
        sys.exit(1)

    if df is None:
        logger.error("Failed to download data")
        sys.exit(1)

    # Save data
    if args.format == "parquet":
        filepath = downloader.save_to_parquet(
            df=df,
            symbol=args.symbol,
            interval=args.interval,
            venue=args.venue.upper(),
        )
    else:
        filepath = downloader.save_to_csv(
            df=df,
            symbol=args.symbol,
            interval=args.interval,
            venue=args.venue.upper(),
        )

    if filepath:
        logger.info("=" * 80)
        logger.info("DOWNLOAD COMPLETE")
        logger.info(f"File: {filepath}")
        logger.info(f"Rows: {len(df):,}")
        logger.info(f"Columns: {', '.join(df.columns)}")
        logger.info("=" * 80)
    else:
        logger.error("Failed to save data")
        sys.exit(1)


if __name__ == "__main__":
    main()
