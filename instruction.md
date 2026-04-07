Write Python code for a trading strategy with these strict rules:

1. Goal
- Build a complete, readable, executable Python script.
- Strategy: Trend Pullback + Breakout.
- Use only standard scientific Python libraries unless told otherwise.

2. Code quality
- Keep code modular and production-like.
- Use functions with clear names.
- Add comments only where useful.
- Avoid unnecessary complexity.
- Include type hints where reasonable.
- Handle errors safely.

3. Inputs
- Accept OHLCV data in a pandas DataFrame.
- Required columns:
  ["Open", "High", "Low", "Close", "Volume"]
- Index should be datetime.
- Do not assume perfect data. Clean or validate it first.

4. Strategy logic
- Detect trend using market structure or moving averages.
- Long setup:
  - uptrend exists
  - price pulls back
  - price consolidates
  - breakout above consolidation high triggers entry
- Short setup:
  - downtrend exists
  - price pulls back
  - price consolidates
  - breakout below consolidation low triggers entry
- Make all thresholds configurable.

5. Risk management
- Stop loss:
  - below consolidation low for long
  - above consolidation high for short
- Take profit based on configurable risk-reward ratio.
- Allow optional trailing stop.
- Risk per trade should be configurable.

6. Backtest requirements
- Include a simple backtest engine in the same script.
- Avoid lookahead bias.
- Entry should occur only after signal confirmation.
- One position at a time unless explicitly enabled.
- Track:
  - total return
  - win rate
  - profit factor
  - max drawdown
  - Sharpe ratio if possible
  - number of trades

7. Output
- Return:
  - trades log as DataFrame
  - equity curve as DataFrame or Series
  - summary metrics as dict
- Print a readable summary.

8. Visualization
- Plot:
  - price
  - entry/exit markers
  - equity curve
- Use matplotlib only unless told otherwise.

9. Constraints
- Do not use future candles for current decisions.
- Do not hardcode ticker-specific values.
- Do not use external APIs.
- Do not overfit.
- Do not add machine learning unless requested.

10. Deliverables
- Provide full code in one block.
- Include a small example with synthetic or sample data.
- Explain briefly how to run it.