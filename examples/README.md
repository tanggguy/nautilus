# Examples - Nautilus Trading Platform

This directory contains complete end-to-end examples demonstrating how to use the Nautilus Trading Platform.

## 📚 Available Examples

### 1. Complete Backtest Example (Python Script)

**File**: `complete_backtest_example.py`

A comprehensive example showing the complete workflow:
- Download historical data
- Validate data quality
- Configure backtest
- Configure strategy
- Run backtest
- Analyze results

**Run it**:
```bash
python examples/complete_backtest_example.py
```

**What it does**:
1. Downloads Bitcoin (BTC-USD) data from Yahoo Finance (Jan-Mar 2024)
2. Validates the data for quality issues
3. Configures a backtest with:
   - $10,000 starting capital
   - EMA Cross strategy (10/30 periods)
   - 2% stop loss, 4% take profit
   - 0.1% trading fees
4. Shows how to run the backtest (Python API + CLI)
5. Displays simulated results

---

## 🚀 Quick Start Guide

### Step-by-Step Tutorial

Follow these steps to run your first backtest:

#### 1. Download Data

```bash
# Download Bitcoin hourly data for 2 months
python scripts/data_download.py \
  --symbol BTC-USD \
  --start 2024-01-01 \
  --end 2024-03-01 \
  --interval 1h \
  --output data
```

**Expected output**:
- File: `data/yahoo_btcusd_1h.parquet`
- ~1400 bars (2 months × 30 days × 24 hours)

#### 2. Validate Data

```bash
# Check data quality
python scripts/data_validation.py data/yahoo_btcusd_1h.parquet
```

**What it checks**:
- ✅ Required columns (timestamp, OHLCV)
- ✅ OHLC relationships (high >= all, low <= all)
- ✅ No negative volumes
- ✅ No time gaps
- ✅ No price outliers
- ✅ No duplicate timestamps

#### 3. Run Backtest

```bash
# Run EMA Cross strategy
python scripts/run_backtest.py \
  --strategy ema_cross \
  --symbol BTCUSDT \
  --start 2024-01-01 \
  --end 2024-03-01 \
  --interval 1h \
  --capital 10000 \
  --trade-size 1000 \
  --params fast_period=10 slow_period=30
```

**Results saved to**: `results/backtests/ema_cross_btc_<timestamp>.json`

---

## 📊 Strategy Examples

### EMA Cross Strategy

**Trend following strategy using EMA crossovers**

```bash
python scripts/run_backtest.py \
  --strategy ema_cross \
  --symbol BTCUSDT \
  --start 2024-01-01 --end 2024-03-01 \
  --params fast_period=10 slow_period=30
```

**Best for**: Trending markets, medium-term timeframes

### Mean Reversion Strategy

**Range trading using Bollinger Bands**

```bash
python scripts/run_backtest.py \
  --strategy mean_reversion \
  --symbol BTCUSDT \
  --start 2024-01-01 --end 2024-03-01 \
  --interval 15m \
  --params bb_period=20 bb_std_dev=2.0 exit_on_middle=True
```

**Best for**: Sideways markets, high volatility

### Breakout Strategy

**Momentum trading on range breakouts**

```bash
python scripts/run_backtest.py \
  --strategy breakout \
  --symbol BTCUSDT \
  --start 2024-01-01 --end 2024-03-01 \
  --interval 4h \
  --params lookback_period=20 volume_confirm=True
```

**Best for**: Consolidation breakouts, volatile markets

### MACD Strategy

**Classic MACD trend following**

```bash
python scripts/run_backtest.py \
  --strategy macd \
  --symbol BTCUSDT \
  --start 2024-01-01 --end 2024-03-01 \
  --params fast_period=12 slow_period=26 signal_period=9
```

**Best for**: Trending markets, all timeframes

---

## 🔧 Configuration Options

### Common Parameters

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `--strategy` | Strategy to use | (required) | `ema_cross` |
| `--symbol` | Trading symbol | (required) | `BTCUSDT` |
| `--start` | Start date | (required) | `2024-01-01` |
| `--end` | End date | (required) | `2024-03-01` |
| `--interval` | Bar interval | `1h` | `1h`, `4h`, `1d` |
| `--capital` | Initial capital | `10000` | `5000` |
| `--trade-size` | Trade size (USDT) | `100` | `1000` |
| `--venue` | Trading venue | `BINANCE` | `BINANCE` |
| `--data-dir` | Data directory | `data` | `data/crypto` |
| `--output-dir` | Results directory | `results/backtests` | `results/my_tests` |

### Strategy-Specific Parameters

Pass custom parameters with `--params key=value`:

**EMA Cross**:
- `fast_period=10` - Fast EMA period
- `slow_period=30` - Slow EMA period

**Mean Reversion**:
- `bb_period=20` - Bollinger Bands period
- `bb_std_dev=2.0` - Standard deviations
- `exit_on_middle=True` - Exit at middle band

**Breakout**:
- `lookback_period=20` - Range lookback
- `volume_confirm=True` - Require volume
- `volume_multiplier=1.5` - Volume threshold

**MACD**:
- `fast_period=12` - Fast EMA
- `slow_period=26` - Slow EMA
- `signal_period=9` - Signal line

---

## 📈 Multiple Symbols Example

Test the same strategy on different symbols:

```bash
# Test on Bitcoin
python scripts/run_backtest.py --strategy ema_cross --symbol BTCUSDT \
  --start 2024-01-01 --end 2024-03-01

# Test on Ethereum
python scripts/run_backtest.py --strategy ema_cross --symbol ETHUSDT \
  --start 2024-01-01 --end 2024-03-01

# Test on Solana
python scripts/run_backtest.py --strategy ema_cross --symbol SOLUSDT \
  --start 2024-01-01 --end 2024-03-01
```

---

## 🧪 Parameter Optimization Example

Test different parameter combinations:

```bash
# Conservative (slower EMAs)
python scripts/run_backtest.py --strategy ema_cross --symbol BTCUSDT \
  --start 2024-01-01 --end 2024-03-01 \
  --params fast_period=20 slow_period=50

# Moderate (default)
python scripts/run_backtest.py --strategy ema_cross --symbol BTCUSDT \
  --start 2024-01-01 --end 2024-03-01 \
  --params fast_period=10 slow_period=30

# Aggressive (faster EMAs)
python scripts/run_backtest.py --strategy ema_cross --symbol BTCUSDT \
  --start 2024-01-01 --end 2024-03-01 \
  --params fast_period=5 slow_period=15
```

**Note**: Phase 5 will add automated parameter optimization using Optuna.

---

## 📁 Directory Structure After Running Examples

```
nautilus/
├── data/
│   └── yahoo_btcusd_1h.parquet        # Downloaded data
├── results/
│   └── backtests/
│       └── ema_cross_btc_20241030.json # Backtest results
├── logs/
│   ├── nautilus.log                    # Main log
│   ├── errors.log                      # Error log
│   └── nautilus_daily.log              # Daily rotating log
└── examples/
    ├── README.md                        # This file
    └── complete_backtest_example.py     # Complete example
```

---

## 🐛 Troubleshooting

### Data Download Issues

**Problem**: "No data returned for symbol"

**Solution**:
- Check symbol format (use `-USD` for Yahoo Finance, e.g., `BTC-USD`)
- Verify date range is valid
- Check internet connection
- Try a different interval

### Validation Errors

**Problem**: "OHLC violations detected"

**Solution**:
- Check data source quality
- Use `--format csv` to inspect data manually
- Try downloading data again
- Report issue if persistent

### Backtest Fails

**Problem**: "NautilusTrader not installed"

**Solution**:
```bash
pip install nautilus_trader
```

**Problem**: "Data not found in catalog"

**Solution**:
- Ensure data was downloaded to correct directory
- Check file exists: `ls data/*.parquet`
- Verify data path in config matches download path

---

## 📚 Learning Resources

### Next Steps

1. **Phase 4 (Coming Soon)**: Detailed performance analysis
   - Sharpe/Sortino ratios
   - Drawdown analysis
   - Trade-by-trade breakdown
   - Performance visualizations

2. **Phase 5 (Coming Soon)**: Parameter optimization
   - Automated grid search
   - Optuna Bayesian optimization
   - Walk-forward analysis
   - Overfitting detection

3. **Phase 6 (Coming Soon)**: Paper trading
   - Real-time simulation
   - Risk monitoring
   - Performance tracking
   - Alert system

### Strategy Development Tips

1. **Start Simple**: Test with default parameters first
2. **Multiple Timeframes**: Test on 1h, 4h, and 1d bars
3. **Multiple Symbols**: Test on BTC, ETH, and other coins
4. **Parameter Sensitivity**: Vary parameters to see impact
5. **Risk Management**: Always use stop losses
6. **Paper Trade First**: Test 3+ months before going live

---

## 🤝 Contributing Examples

Have a great example to share? Add it to this directory!

**Requirements**:
- Complete, runnable code
- Clear comments and documentation
- Realistic parameters
- Error handling

---

## 📞 Support

- **Documentation**: See `doc/` directory
- **Issues**: Report on GitHub
- **Questions**: Check existing examples first

---

**Happy Backtesting! 🚀📈**
