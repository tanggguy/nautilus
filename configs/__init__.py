"""
Configuration module for Nautilus Trading Platform.

This module provides configuration classes for:
- Backtest configuration with Pydantic validation
- Paper trading configuration
- Live trading configuration with safety mechanisms
- Hyperparameter optimization configuration
"""

from configs.backtest_config import (
    BacktestConfig,
    DataConfig,
    FeeConfig,
    InstrumentConfig,
    RiskConfig,
    VenueConfig,
    create_crypto_config,
    create_default_config,
)
from configs.live_config import (
    CircuitBreakerConfig,
    ExchangeConfig,
    HealthCheckConfig,
    LiveRiskConfig,
    LiveTradingConfig,
    create_minimal_live_config,
)
from configs.optimization_config import (
    ObjectiveConfig,
    OptimizationConfig,
    ParallelizationConfig,
    PrunerConfig,
    SamplerConfig,
    StorageConfig,
    ValidationConfig,
    create_default_optimization_config,
    create_fast_optimization_config,
    create_thorough_optimization_config,
)
from configs.paper_config import (
    DataProviderConfig,
    ExecutionConfig,
    MonitoringConfig,
    PaperTradingConfig,
    PersistenceConfig,
    create_conservative_paper_config,
    create_default_paper_config,
)

__all__ = [
    # Backtest
    "BacktestConfig",
    "VenueConfig",
    "InstrumentConfig",
    "DataConfig",
    "RiskConfig",
    "FeeConfig",
    "create_default_config",
    "create_crypto_config",
    # Paper Trading
    "PaperTradingConfig",
    "DataProviderConfig",
    "ExecutionConfig",
    "MonitoringConfig",
    "PersistenceConfig",
    "create_default_paper_config",
    "create_conservative_paper_config",
    # Live Trading
    "LiveTradingConfig",
    "ExchangeConfig",
    "LiveRiskConfig",
    "CircuitBreakerConfig",
    "HealthCheckConfig",
    "create_minimal_live_config",
    # Optimization
    "OptimizationConfig",
    "SamplerConfig",
    "PrunerConfig",
    "ObjectiveConfig",
    "ValidationConfig",
    "StorageConfig",
    "ParallelizationConfig",
    "create_default_optimization_config",
    "create_fast_optimization_config",
    "create_thorough_optimization_config",
]
