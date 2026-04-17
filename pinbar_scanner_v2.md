# Pin Bar at Context Scanner — Buy-Only Price Action Rejection Setup

## Summary
Buy-only scanner detecting bullish pin bars at support (MA20, MA50, MA200, swing low).

## Supported Timeframes
- Daily
- 4H

## Key Rules
- Bullish pin bar: long lower wick >= 60%, small body <= 33%
- Must be at support context
- Minimum R:R = 2
- Requires confirmation (next candle breaks high)

## Context Levels
- MA20 (uptrend only)
- MA50
- MA200
- Swing low (20 bars)

## 4H Rule
- A 4H bullish pin bar is valid only if the higher timeframe trend is supportive
- Preferred higher timeframe filter:
  - Daily MA50 is rising, or
  - Daily close is above Daily MA50
- On 4H, the pin bar must still meet the same structure rules:
  - long lower wick >= 60%
  - small body <= 33%
  - forms at one of the defined support contexts
- 4H signals have lower priority than Daily signals if both exist on the same symbol
- If both Daily and 4H signals exist:
  - prefer Daily for primary setup
  - use 4H as earlier confirmation or refined entry