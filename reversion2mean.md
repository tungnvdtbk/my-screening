Act as a senior Python trading-system developer.

Write a complete Python script to scan stocks for a long-only mean-reversion swing setup in a sideways/range market.

Goal:
Find stocks that are trading in a clear horizontal range, currently near support, and showing early bounce/reversal signals for a buy-low / sell-high swing trade.

General requirements:
- Use only pandas, numpy, matplotlib unless otherwise stated.
- Code must be complete, executable, and modular.
- No placeholders.
- Include clear function separation.
- Include type hints where reasonable.
- Validate input data before processing.
- Avoid lookahead bias in all calculations.
- Accept OHLCV data for multiple tickers.
- Output a ranked scan result as a pandas DataFrame.
- Include optional chart visualization for one selected ticker.

Input assumptions:
- Data is a pandas DataFrame.
- Required columns:
  ["Ticker", "Date", "Open", "High", "Low", "Close", "Volume"]
- Date must be parsed as datetime.
- Data may contain multiple tickers.
- Sort data by Ticker, then Date.
- Handle missing or bad rows safely.

Strategy objective:
Scan for stocks suitable for long-only range swing trading:
- not strong trend-following
- not breaking down
- not near range midpoint or range top
- only stocks near support inside a stable range
- prefer signs of reversal and acceptable liquidity

Universe filters:
- Exclude stocks with insufficient history
- Exclude stocks with low liquidity
- Exclude stocks with zero or invalid OHLCV values
- Exclude stocks whose range is too narrow or too wide
- Exclude stocks in strong downtrend
- Exclude stocks in strong breakout trend

Minimum history:
- At least 80 bars per ticker

Core scan logic:

1. Define lookback windows
- range_window = 50 bars
- ema_fast = 20
- ema_slow = 50
- volume_window = 20
- atr_window = 14
- rsi_window = 3
- support_tolerance = 0.03
- bottom_zone_threshold = 0.25

2. Calculate indicators per ticker
- EMA20
- EMA50
- ATR14
- RSI3
- 20-day average volume
- 20-day average traded value = average(Volume * Close)
- 50-bar highest high
- 50-bar lowest low
- range_size_pct = (range_high - range_low) / range_low
- position_in_range = (Close - range_low) / (range_high - range_low)
- EMA slopes:
  - ema20_slope = (EMA20 today - EMA20 5 bars ago) / EMA20 5 bars ago
  - ema50_slope = (EMA50 today - EMA50 5 bars ago) / EMA50 5 bars ago

3. Detect range condition
A stock is in a valid range only if all are true:
- range_size_pct between 0.08 and 0.20
- abs(ema20_slope) <= 0.02
- abs(ema50_slope) <= 0.015
- Close is not making new 50-bar breakout highs
- Close is not making new 50-bar breakdown lows by a large margin
- At least 2 swing highs near range_high and 2 swing lows near range_low in the last 50 bars
  or, if swing detection is too complex, approximate by checking repeated touches within tolerance

4. Support location filter
The stock qualifies only if:
- position_in_range <= 0.25
- Close <= range_low * (1 + support_tolerance)
- Close is above range_low
- distance_to_support_pct = (Close - range_low) / range_low is between 0 and 0.03

5. Reversal / bounce filter
At least one of these must be true:
- today Close > yesterday High
- today Close > yesterday Close and today Low <= yesterday Low
- bullish hammer-like candle:
  - lower wick > real body
  - close in upper half of candle range
- 2 consecutive higher closes
- RSI3 < 20 and today Close > yesterday Close

6. Trend rejection filter
Reject the stock if:
- Close < EMA50 and EMA20 < EMA50 and ema50_slope < -0.02
- last 10-bar return < -0.10
- 3 consecutive large bearish candles
- support is failing with high sell volume

7. Liquidity filter
Require:
- average traded value over last 20 bars >= configurable minimum
- average volume over last 20 bars >= configurable minimum
- no abnormal illiquidity in last 5 bars

8. Optional relative strength filter
If index data is available:
- compare 10-bar return of stock vs benchmark
- keep stocks whose 10-bar return is better than benchmark
If benchmark data is not available:
- skip this filter safely

9. Scoring / ranking
Create a numeric score for ranking candidates.
Suggested scoring:
- closer to support = higher score
- lower position_in_range = higher score
- stronger bullish reversal = higher score
- higher liquidity = higher score
- less negative or better relative strength = higher score
- cleaner flat EMA structure = higher score

Example score components:
- support_score
- reversal_score
- liquidity_score
- range_quality_score
- relative_strength_score

Final score:
final_score = weighted sum of the components

10. Output columns
Return a DataFrame with at least:
- Ticker
- Date
- Close
- RangeHigh50
- RangeLow50
- RangeSizePct
- PositionInRange
- DistanceToSupportPct
- EMA20
- EMA50
- EMA20Slope
- EMA50Slope
- RSI3
- ATR14
- AvgVolume20
- AvgTradedValue20
- ReversalSignal
- IsRangeCandidate
- FinalScore

11. Output behavior
- Sort by FinalScore descending
- Return only valid candidates
- Print top N results
- Save result to CSV if requested

12. Visualization
Include an optional function to plot one ticker:
- Close price
- EMA20 and EMA50
- Range high and range low
- signal marker on scan date
- optional volume panel is not required unless easy to implement
Use matplotlib only

13. Parameters
Put key thresholds in a config dictionary or dataclass:
- range_window
- ema_fast
- ema_slow
- atr_window
- rsi_window
- support_tolerance
- bottom_zone_threshold
- min_avg_volume
- min_avg_traded_value
- min_range_pct
- max_range_pct
- max_abs_ema20_slope
- max_abs_ema50_slope
- max_10bar_drop
- top_n_results

14. Code structure
Implement at least these functions:
- validate_data(df)
- compute_indicators(df, config)
- detect_range_conditions(df, config)
- detect_reversal_signal(df, config)
- compute_scores(df, config)
- run_range_scan(df, config)
- plot_candidate(df, ticker, config)
- main()

15. Safety / correctness
- Do not use future bars
- Do not hardcode stock-specific logic
- Handle divide-by-zero safely
- Handle NaNs safely
- Ensure per-ticker rolling calculations are grouped correctly
- Keep calculations aligned to each row’s current date only

16. Example usage
At the bottom of the script:
- generate synthetic sample data for 3 to 5 tickers or load from CSV
- run the scan
- print top candidates
- optionally plot one candidate

17. Style requirements
- Keep code readable and moderately commented
- Prefer clarity over cleverness
- No unnecessary classes unless useful
- Include docstrings for main functions