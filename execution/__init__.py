"""
Execution module for Nautilus Trading Platform.

This module provides execution runners for:
- Backtesting with historical data
- Paper trading with simulated execution
- Live trading with real money (future)

Components:
- BacktestRunner: Execute backtests with strategies and historical data
- (Future) PaperRunner: Execute paper trading with real-time data
- (Future) LiveRunner: Execute live trading with real exchange connections
"""

from execution.backtest_runner import BacktestRunner, run_simple_backtest

__all__ = [
    "BacktestRunner",
    "run_simple_backtest",
]
