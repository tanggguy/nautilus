#!/usr/bin/env python3
"""
Data validation script for Nautilus Trading Platform.

This script validates historical market data quality:
- Checks for missing data (gaps)
- Detects outliers in prices
- Validates OHLC relationships
- Checks for duplicate timestamps
- Validates data types and formats
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from utils.logging_config import get_logger, setup_logging

# Setup logging
setup_logging()
logger = get_logger(__name__)


class DataValidator:
    """
    Validate historical market data quality.

    Performs comprehensive checks on OHLCV data:
    - Completeness (gaps, missing values)
    - Consistency (OHLC relationships)
    - Quality (outliers, duplicates)
    """

    def __init__(self, strict: bool = False):
        """
        Initialize data validator.

        Args:
            strict: If True, fail on any warnings.
        """
        self.strict = strict
        self.issues: List[Dict] = []
        self.warnings: List[Dict] = []

    def validate_file(self, filepath: Path) -> Tuple[bool, Dict]:
        """
        Validate a data file (Parquet or CSV).

        Args:
            filepath: Path to data file.

        Returns:
            Tuple of (is_valid, validation_report).
        """
        logger.info("=" * 80)
        logger.info(f"VALIDATING: {filepath.name}")
        logger.info("=" * 80)

        # Load data
        df = self._load_data(filepath)
        if df is None:
            return False, {"error": "Failed to load data"}

        # Run validations
        self.issues = []
        self.warnings = []

        self._validate_columns(df)
        self._validate_timestamps(df)
        self._validate_ohlc_relationships(df)
        self._validate_volume(df)
        self._check_for_gaps(df)
        self._detect_outliers(df)
        self._check_duplicates(df)

        # Generate report
        report = self._generate_report(df, filepath)

        # Determine if valid
        is_valid = len(self.issues) == 0
        if self.strict:
            is_valid = is_valid and len(self.warnings) == 0

        return is_valid, report

    def _load_data(self, filepath: Path) -> pd.DataFrame:
        """Load data from file."""
        try:
            if filepath.suffix == ".parquet":
                df = pd.read_parquet(filepath)
            elif filepath.suffix == ".csv":
                df = pd.read_csv(filepath)
            else:
                logger.error(f"Unsupported file format: {filepath.suffix}")
                return None

            logger.info(f"Loaded {len(df):,} rows")
            return df

        except Exception as e:
            logger.error(f"Failed to load file: {e}")
            return None

    def _validate_columns(self, df: pd.DataFrame):
        """Validate required columns exist."""
        required = ["timestamp", "open", "high", "low", "close", "volume"]
        missing = [col for col in required if col not in df.columns]

        if missing:
            self.issues.append({
                "type": "missing_columns",
                "severity": "critical",
                "message": f"Missing required columns: {missing}",
                "columns": missing,
            })
            logger.error(f"❌ Missing columns: {missing}")
        else:
            logger.info("✅ All required columns present")

    def _validate_timestamps(self, df: pd.DataFrame):
        """Validate timestamp column."""
        if "timestamp" not in df.columns:
            return

        # Check if timestamps are datetime
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            try:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                self.warnings.append({
                    "type": "timestamp_conversion",
                    "severity": "warning",
                    "message": "Timestamps converted to datetime",
                })
                logger.warning("⚠️ Timestamps converted to datetime")
            except Exception as e:
                self.issues.append({
                    "type": "invalid_timestamps",
                    "severity": "critical",
                    "message": f"Failed to parse timestamps: {e}",
                })
                logger.error(f"❌ Invalid timestamps: {e}")
                return

        # Check if timestamps are sorted
        if not df["timestamp"].is_monotonic_increasing:
            self.issues.append({
                "type": "unsorted_timestamps",
                "severity": "error",
                "message": "Timestamps are not sorted",
            })
            logger.error("❌ Timestamps not sorted")
        else:
            logger.info("✅ Timestamps sorted correctly")

    def _validate_ohlc_relationships(self, df: pd.DataFrame):
        """Validate OHLC price relationships."""
        required_cols = ["open", "high", "low", "close"]
        if not all(col in df.columns for col in required_cols):
            return

        # High should be >= all other prices
        high_violations = (
            (df["high"] < df["open"]) |
            (df["high"] < df["low"]) |
            (df["high"] < df["close"])
        ).sum()

        # Low should be <= all other prices
        low_violations = (
            (df["low"] > df["open"]) |
            (df["low"] > df["high"]) |
            (df["low"] > df["close"])
        ).sum()

        if high_violations > 0:
            self.issues.append({
                "type": "ohlc_violation",
                "severity": "error",
                "message": f"High price violations: {high_violations} rows",
                "count": int(high_violations),
            })
            logger.error(f"❌ High price violations: {high_violations} rows")

        if low_violations > 0:
            self.issues.append({
                "type": "ohlc_violation",
                "severity": "error",
                "message": f"Low price violations: {low_violations} rows",
                "count": int(low_violations),
            })
            logger.error(f"❌ Low price violations: {low_violations} rows")

        if high_violations == 0 and low_violations == 0:
            logger.info("✅ OHLC relationships valid")

    def _validate_volume(self, df: pd.DataFrame):
        """Validate volume data."""
        if "volume" not in df.columns:
            return

        # Check for negative volume
        negative_volume = (df["volume"] < 0).sum()
        if negative_volume > 0:
            self.issues.append({
                "type": "negative_volume",
                "severity": "error",
                "message": f"Negative volume: {negative_volume} rows",
                "count": int(negative_volume),
            })
            logger.error(f"❌ Negative volume: {negative_volume} rows")

        # Check for zero volume
        zero_volume = (df["volume"] == 0).sum()
        if zero_volume > 0:
            pct = (zero_volume / len(df)) * 100
            self.warnings.append({
                "type": "zero_volume",
                "severity": "warning",
                "message": f"Zero volume: {zero_volume} rows ({pct:.2f}%)",
                "count": int(zero_volume),
                "percentage": pct,
            })
            logger.warning(f"⚠️ Zero volume: {zero_volume} rows ({pct:.2f}%)")

        if negative_volume == 0:
            logger.info("✅ Volume data valid")

    def _check_for_gaps(self, df: pd.DataFrame):
        """Check for gaps in time series."""
        if "timestamp" not in df.columns:
            return

        if len(df) < 2:
            return

        # Calculate time differences
        time_diffs = df["timestamp"].diff()

        # Find the most common interval (mode)
        expected_interval = time_diffs.mode()[0]

        # Find gaps (time diff > expected * 1.5)
        gaps = time_diffs > (expected_interval * 1.5)
        gap_count = gaps.sum()

        if gap_count > 0:
            gap_pct = (gap_count / len(df)) * 100
            self.warnings.append({
                "type": "time_gaps",
                "severity": "warning",
                "message": f"Time gaps detected: {gap_count} gaps ({gap_pct:.2f}%)",
                "count": int(gap_count),
                "percentage": gap_pct,
                "expected_interval": str(expected_interval),
            })
            logger.warning(f"⚠️ Time gaps: {gap_count} gaps ({gap_pct:.2f}%)")
        else:
            logger.info("✅ No time gaps detected")

    def _detect_outliers(self, df: pd.DataFrame):
        """Detect price outliers using IQR method."""
        price_cols = ["open", "high", "low", "close"]
        if not all(col in df.columns for col in price_cols):
            return

        outliers_found = False

        for col in price_cols:
            # Calculate IQR
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            # Define outlier bounds
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR

            # Find outliers
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()

            if outliers > 0:
                outliers_found = True
                pct = (outliers / len(df)) * 100
                self.warnings.append({
                    "type": "price_outliers",
                    "severity": "warning",
                    "message": f"{col.capitalize()} outliers: {outliers} ({pct:.2f}%)",
                    "column": col,
                    "count": int(outliers),
                    "percentage": pct,
                })
                logger.warning(f"⚠️ {col.capitalize()} outliers: {outliers} ({pct:.2f}%)")

        if not outliers_found:
            logger.info("✅ No significant outliers detected")

    def _check_duplicates(self, df: pd.DataFrame):
        """Check for duplicate timestamps."""
        if "timestamp" not in df.columns:
            return

        duplicates = df["timestamp"].duplicated().sum()

        if duplicates > 0:
            self.issues.append({
                "type": "duplicate_timestamps",
                "severity": "error",
                "message": f"Duplicate timestamps: {duplicates} rows",
                "count": int(duplicates),
            })
            logger.error(f"❌ Duplicate timestamps: {duplicates} rows")
        else:
            logger.info("✅ No duplicate timestamps")

    def _generate_report(self, df: pd.DataFrame, filepath: Path) -> Dict:
        """Generate validation report."""
        logger.info("=" * 80)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 80)

        report = {
            "file": str(filepath),
            "total_rows": len(df),
            "columns": list(df.columns),
            "date_range": {
                "start": str(df["timestamp"].min()) if "timestamp" in df.columns else None,
                "end": str(df["timestamp"].max()) if "timestamp" in df.columns else None,
            },
            "issues": self.issues,
            "warnings": self.warnings,
        }

        # Count by severity
        critical_count = sum(1 for i in self.issues if i.get("severity") == "critical")
        error_count = sum(1 for i in self.issues if i.get("severity") == "error")
        warning_count = len(self.warnings)

        logger.info(f"Total Rows     : {len(df):,}")
        logger.info(f"Critical Issues: {critical_count}")
        logger.info(f"Errors         : {error_count}")
        logger.info(f"Warnings       : {warning_count}")

        if critical_count == 0 and error_count == 0:
            logger.info("✅ DATA VALIDATION PASSED")
        else:
            logger.error("❌ DATA VALIDATION FAILED")

        logger.info("=" * 80)

        return report


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Validate historical market data quality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "file",
        type=Path,
        help="Path to data file (Parquet or CSV)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on warnings (not just errors)",
    )

    args = parser.parse_args()

    # Check file exists
    if not args.file.exists():
        logger.error(f"File not found: {args.file}")
        sys.exit(1)

    # Validate
    validator = DataValidator(strict=args.strict)
    is_valid, report = validator.validate_file(args.file)

    # Exit with appropriate code
    sys.exit(0 if is_valid else 1)


if __name__ == "__main__":
    main()
