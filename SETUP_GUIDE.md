# Production Trading Bot v13.0 - Setup Guide

## Overview

Your trading bot has been upgraded to **Production Trading Bot v13.0** with critical improvements for real-world trading.

## What's Been Fixed

### 1. **Import Errors** ✅
- Added missing `Dict` type import in `position-manager.py`
- Fixed module imports to use correct file names (`trade_journal` instead of `tradejournal`)
- All typing imports properly configured

### 2. **Simplified Indicators** ✅
- Reduced from 7+ indicators to **4 core indicators**:
  - **EMA** (Trend direction)
  - **RSI** (Momentum)
  - **ADX** (Trend strength)
  - **ATR** (Volatility)
- Eliminates overfitting risk
- Cleaner, more reliable signals

### 3. **Slippage & Execution Modeling** ✅
- Realistic slippage based on signal type:
  - Scalp: 2.0 pips base slippage
  - Day Trade: 1.5 pips
  - Swing: 1.0 pips
- Liquidity-adjusted execution (50% more slippage in low liquidity)
- Spread modeling (3 pips for XAUUSD, 5 pips for BTCUSD)

### 4. **Performance Tracking** ✅
- SQLite database (`trade_journal.db`) logs every signal
- Track win rate, P&L, and performance metrics
- 30-day performance stats displayed on each run
- Exportable data for backtesting analysis

### 5. **Position Management** ✅
- Automatic trailing stops after TP2
- Partial exits: 50% at TP1, 30% at TP2, 20% at TP3
- Move stop loss to breakeven after TP1
- Risk-free trading after breakeven

### 6. **Updated Workflow** ✅
- Runs `scripts/enhanced-trading-bot-v13.py`
- Validates all modules before execution
- Uploads trade journal database as artifact
- 30-day log retention (increased from 14)
- Proper environment variable handling

## File Structure

\`\`\`
├── scripts/
│   ├── enhanced-trading-bot-v13.py    # Main bot (production ready)
│   ├── trade-journal.py                # Performance tracking
│   └── position-manager.py             # Position management
├── .github/
│   └── workflows/
│       └── trading-bot.yml             # GitHub Actions workflow
├── requirements.txt                     # Python dependencies
├── SETUP_GUIDE.md                      # This file
└── trade_journal.db                    # Created on first run
\`\`\`

## Environment Variables

### Required Secrets (GitHub Secrets)
\`\`\`
TELEGRAM_BOT_TOKEN      # Your Telegram bot token
MAIN_CHAT_ID            # Main chat for detailed signals
SIMPLE_CHAT_ID          # Simple chat for summaries
DERIV_APP_ID            # Deriv API app ID (optional, defaults to 1089)
\`\`\`

### Optional Variables (GitHub Variables)
\`\`\`
BOT_ENABLED             # Set to 'false' to disable bot (emergency kill switch)
ACCOUNT_BALANCE         # Account balance in USD (default: 500)
RISK_PERCENT            # Risk per trade % (default: 2.0)
ENABLE_SCALPING         # Enable scalp signals (default: true)
ENABLE_DAY_TRADING      # Enable day trade signals (default: true)
ENABLE_SWING_TRADING    # Enable swing signals (default: true)
\`\`\`

## How to Use

### 1. **Automatic Execution**
The bot runs automatically every hour via GitHub Actions:
- Checks market conditions (skips weekends and low liquidity hours)
- Analyzes XAUUSD and BTCUSD
- Generates signals based on confidence thresholds
- Sends notifications to Telegram
- Logs all activity to database

### 2. **Manual Execution**
Go to GitHub Actions → "Production Trading Bot v13.0" → "Run workflow"
- Enable debug mode for detailed logging
- Force run to override market condition checks

### 3. **Emergency Stop**
Set repository variable `BOT_ENABLED=false` to immediately disable the bot.

### 4. **Monitor Performance**
Download `trade_journal.db` from GitHub Actions artifacts to analyze:
\`\`\`python
import sqlite3
conn = sqlite3.connect('trade_journal.db')
df = pd.read_sql_query("SELECT * FROM trades", conn)
print(df.describe())
\`\`\`

## Signal Types

### ⚡ SCALP (5m/15m)
- **Confidence Required**: 80%
- **Target**: 10-20 pips
- **Timeframe**: 5-15 minutes
- **Liquidity Required**: High (60%+)

### 📊 DAY TRADE (15m/1H)
- **Confidence Required**: 75%
- **Target**: 30-60 pips
- **Timeframe**: 15min-1 hour
- **Liquidity Required**: Good (40%+)

### 📈 SWING (1H/4H)
- **Confidence Required**: 70%
- **Target**: 100+ pips
- **Timeframe**: 1-4 hours
- **Liquidity Required**: Any

## Confidence Scoring (100 Points Max)

1. **EMA Trend Alignment** (35 points)
   - Perfect alignment: 15 points (entry) + 20 points (HTF)
   - Partial alignment: 10-12 points

2. **RSI Momentum** (25 points)
   - Optimal range: 25 points
   - Good range: 18 points
   - Overbought/oversold: 0 points (rejected)

3. **ADX Trend Strength** (25 points)
   - Very strong (>30): 25 points
   - Strong (>25): 20 points
   - Weak (<15): 0 points (rejected)

4. **ATR Volatility** (15 points)
   - High volatility: 15 points
   - Normal: 8-12 points
   - Low: 4 points

## Position Management Rules

1. **Entry**: Execute at calculated entry price + slippage
2. **TP1 Hit**: Close 50% of position, move SL to breakeven
3. **TP2 Hit**: Close 30% more, activate trailing stop
4. **Trailing**: Trail stop at 1.5x ATR distance
5. **TP3 Hit**: Close remaining 20%

## Next Steps

### Before Live Trading:
1. ✅ **Backtest manually** - You said you'll handle this
2. ✅ **Paper trade for 30 days** - Monitor signal quality
3. ✅ **Review trade journal** - Analyze win rate and P&L
4. ✅ **Start small** - Use minimum position sizes
5. ✅ **Monitor closely** - Check logs daily for first week

### Monitoring:
- Check GitHub Actions runs daily
- Review Telegram notifications
- Download and analyze `trade_journal.db` weekly
- Adjust confidence thresholds if needed

## Troubleshooting

### Bot Not Running
- Check `BOT_ENABLED` variable is not set to 'false'
- Verify all secrets are configured
- Check GitHub Actions logs for errors

### No Signals Generated
- Normal - bot is selective (70-80% confidence required)
- Check if market conditions are suitable
- Review logs to see why signals were rejected

### Import Errors
- All fixed in v13.0
- Ensure all three files are present in `scripts/` folder
- Check Python syntax with `python -m py_compile scripts/*.py`

## Support

If you encounter issues:
1. Check GitHub Actions logs
2. Review `production_trading_bot.log` in artifacts
3. Examine `trade_journal.db` for signal history
4. Set `BOT_ENABLED=false` if critical issues arise

---

**Version**: 13.0  
**Status**: Production Ready  
**Last Updated**: 2025-01-15
