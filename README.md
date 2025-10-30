nautilus/
├── config/
│   ├── __init__.py
│   ├── backtest_config.py
│   ├── live_config.py
│   ├── paper_config.py
│   └── optimization_config.py
│
├── strategies/
│   ├── __init__.py
│   ├── base_strategy.py
│   ├── ema_cross.py
│   ├── mean_reversion.py
│   └── breakout.py
│
├── indicators/
│   ├── __init__.py
│   └── custom_indicators.py
│
├── optimization/
│   ├── __init__.py
│   ├── objective.py              # Fonction objectif pour optuna
│   ├── hyperparameters.py        # Définition des espaces de recherche
│   ├── runner.py                 # Orchestration des optimisations
│   └── study_manager.py          # Gestion des études optuna
│
├── execution/
│   ├── __init__.py
│   ├── backtest_runner.py
│   ├── paper_runner.py
│   └── live_runner.py
│
├── analysis/
│   ├── __init__.py
│   ├── performance.py
│   ├── metrics.py
│   └── visualization.py
│
├── data/
│   ├── historical/
│   ├── catalog/
│   └── cache/
│
├── results/
│   ├── backtests/
│   ├── optimizations/
│   │   ├── studies/              # DB optuna
│   │   └── reports/
│   └── paper_trading/
│
├── notebooks/
│   ├── exploration.ipynb
│   ├── optimization_analysis.ipynb
│   └── strategy_comparison.ipynb
│
├── tests/
│   ├── __init__.py
│   ├── test_strategies.py
│   └── test_optimization.py
│
├── scripts/
│   ├── run_backtest.py
│   ├── run_optimization.py
│   └── run_paper.py
│
├── utils/
│   ├── __init__.py
│   ├── logging_config.py
│   └── helpers.py
│
├── main.py
├── requirements.txt
├── .env.example
└── README.md