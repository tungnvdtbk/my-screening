# Swing Trading Scan Rules — Breakout & Reversal
### Khung thời gian: Daily (D1) | Kiểu giao dịch: Swing Trading

> **Tổng quan chiến lược:**
> Swing trading trên khung D1 — giữ lệnh từ **3 đến 15 ngày**, bắt các nhịp swing rõ ràng.
> Scan chạy **sau khi nến ngày đóng cửa** (thường 00:00 UTC hoặc theo giờ đóng cửa thị trường).
> Không trade intraday, không dùng nến đang hình thành để ra quyết định.

---

> **Quy ước index nến:**
> - `[-1]` = nến ngày hôm qua — nến đã đóng gần nhất, là **signal candle**
> - `[-2]` = nến ngày hôm kia
> - `[-N:-2]` = chuỗi N nến, **không bao gồm signal candle [-1]**
> - `[0]` = nến ngày hôm nay đang hình thành — **KHÔNG dùng để trigger scan**
> - ATR(10) luôn tính trên `[-11:-2]` — loại trừ signal candle tránh self-inflate

---

## Phương Án 1 — BREAKOUT MOMENTUM

### Mục tiêu
Tìm nến ngày phá vỡ đỉnh 20 ngày gần nhất, momentum mạnh, trong uptrend rõ ràng.
Phù hợp thị trường đang trending — cổ phiếu, crypto, forex pair trending.

### Biến cần tính trước

```
ATR10        = ATR(10)        tính trên nến [-11] đến [-2]   // 10 nến, không gồm signal
AVG_VOL20    = avg(volume,20) tính trên nến [-21] đến [-2]   // baseline volume 20 ngày
AVG_VOL_PRE5 = avg(volume,5)  tính trên nến [-6]  đến [-2]   // volume 5 ngày trước signal
HIGH10       = max(high)      tính trên nến [-11] đến [-2]   // đỉnh cao nhất 10 ngày qua
HIGH20       = max(high)      tính trên nến [-21] đến [-2]   // đỉnh cao nhất 20 ngày qua
MA50_NOW     = MA(50) tại nến [-1]
MA50_PREV5   = MA(50) tại nến [-6]                           // MA50 của 5 ngày trước
```

### Điều kiện bắt buộc (tất cả phải đúng)

```
[1] TREND        close[-1] > MA50_NOW
                 AND MA50_NOW > MA50_PREV5
                 // Giá trên MA50 ngày VÀ MA50 đang dốc lên trong 5 ngày

[2] BULL         close[-1] > open[-1]
                 // Nến ngày hôm qua là nến tăng

[3] CLOSE_HIGH   close[-1] >= high[-1] * 0.998
                 // Đóng cửa tại đỉnh hoặc trong khoảng 0.2% dưới đỉnh nến
                 // Trên D1: 0.2% chấp nhận được do spread/slippage cuối ngày

[4] SIZE         (high[-1] - low[-1]) > 1.5 * ATR10
                 // Biên độ ngày hôm qua lớn hơn 1.5 lần ATR trung bình
                 // Dùng high-low (full candle), không dùng close-open (body)

[4b] BODY        (close[-1] - open[-1]) >= 0.6 * (high[-1] - low[-1])
                 // Thân nến chiếm ít nhất 60% toàn bộ biên độ nến
                 // Loại nến có bóng trên dài — giá bị từ chối gần đỉnh
                 // Xác nhận buyers kiểm soát toàn bộ nhịp tăng, không chỉ close

[5] BREAKOUT     high[-1] > HIGH20                → Signal=BREAKOUT_STRONG
                 HOẶC high[-1] > HIGH10
                      AND high[-1] <= HIGH20       → Signal=BREAKOUT_EARLY
                 // STRONG: phá đỉnh 20 ngày — tín hiệu chất lượng cao nhất
                 // EARLY:  phá đỉnh 10 ngày nhưng chưa phá đỉnh 20 ngày
                 //         bắt sớm trước khi đám đông nhận ra; ưu tiên thấp hơn
```

### Điều kiện volume

```
[6] VOL_SPIKE    volume[-1] > 2.0 * AVG_VOL20
                 // Volume ngày hôm qua ít nhất 2x mức trung bình 20 ngày
                 // Nâng ngưỡng từ 1.5x → 2.0x: chỉ giữ breakout có lực thực sự
                 // Breakout không đủ volume = tín hiệu yếu, hạ xuống TIER 3

[7] VOL_CONTRACT AVG_VOL_PRE5 < AVG_VOL20 * 0.75
                 // Volume trung bình 5 ngày trước signal thấp hơn 75% baseline
                 // Nâng ngưỡng từ 0.8 → 0.75: co cụm volume thực sự đáng kể
                 // Pattern: sellers cạn kiệt → bùng nổ mua = classic VCP signature

Phân loại ưu tiên signal theo volume:
  TIER 1 = [6] + [7] cả hai đạt    → chất lượng cao nhất, ưu tiên trade
  TIER 2 = [6] đạt, [7] không đạt  → tốt, xem xét trade
  TIER 3 = [6] không đạt           → tín hiệu yếu, chỉ theo dõi
```

### Lọc tránh tín hiệu rác

```
[X1] close[-1] <= MA50_NOW * 1.08
     // Loại nếu giá đã cao hơn MA50 trên 8% — quá xa, rủi ro revert về mean

[X2] Không có kháng cự lớn trong vòng 1*ATR10 phía trên high[-1]
     // Kiểm tra thủ công hoặc dùng pivot point / previous swing high
     // Nếu có kháng cự gần = breakout dễ bị chặn ngay
```

### Entry / SL / TP — Swing D1

```
Entry A (aggressive) = open[0] ngày hôm nay — vào lệnh đầu phiên ngay sau scan
Entry B (safe)       = chờ retest vùng HIGH20 trong 1-3 ngày tiếp theo
                       // Entry B có tỷ lệ thắng cao hơn nhưng bỏ lỡ nếu giá chạy thẳng

SL      = low[-1]
          // Dưới đáy nến signal — nếu giá về đây là setup thất bại
          // Trên D1 SL thường 2-5% — chấp nhận được cho swing

TP      = entry + 2.0 * ATR10    // Target tối thiểu R:R = 2.0 cho swing
Trailing= sau mỗi 1*ATR10 giá đi đúng hướng, kéo SL lên theo
          // Không chốt sớm — mục tiêu giữ lệnh 5-15 ngày
```

### Output scan

```
Symbol | Date=[-1] | Signal=BREAKOUT_STRONG | BREAKOUT_EARLY | Close=X |
HIGH10=X | HIGH20=X | ATR10=X | SL=X | TP=X | RR=X |
Volume=TIER1 | TIER2 | TIER3
```

---

## Phương Án 2 — REVERSAL HUNTER

### Mục tiêu
Tìm nến ngày đảo chiều mạnh tại vùng hỗ trợ sau đợt giảm rõ ràng.
Phù hợp cổ phiếu/crypto đang trong pullback của uptrend lớn, hoặc đảo chiều tại vùng giá trị.

### Biến cần tính trước

```
ATR10        = ATR(10)        tính trên nến [-11] đến [-2]
AVG_VOL20    = avg(volume,20) tính trên nến [-21] đến [-2]
AVG_VOL_PRE5 = avg(volume,5)  tính trên nến [-6]  đến [-2]
MA50_NOW     = MA(50) tại nến [-1]
MA200_NOW    = MA(200) tại nến [-1]
```

### Điều kiện bắt buộc (tất cả phải đúng)

```
[1] DOWNTREND    close[-1] < MA50_NOW
                 // Giá dưới MA50 ngày — đang trong nhịp giảm

[2] SUPPORT      abs(close[-1] - MA200_NOW) <= 1.0 * ATR10
                 HOẶC gần structure low quan trọng (pivot / thủ công)
                 // Không bắt đảo chiều khi không có vùng hỗ trợ kỹ thuật
                 // MA200 ngày là vùng hỗ trợ mạnh nhất cho swing trader

[3] REJECTION    low[-1] < low[-2]
                 // Nến hôm qua đã test vùng thấp hơn hôm kia nhưng bị từ chối
                 // Thể hiện áp lực bán đã không thể đẩy giá xuống thêm
                 // Thường tạo bóng nến dưới dài

[4] BULL         close[-1] > open[-1]

[5] CLOSE_HIGH   close[-1] >= high[-1] * 0.998

[6] SIZE         (high[-1] - low[-1]) > 1.5 * ATR10

[6b] BODY        (close[-1] - open[-1]) >= 0.6 * (high[-1] - low[-1])
                 // Thân nến chiếm ít nhất 60% toàn bộ biên độ nến
                 // Xác nhận buyers đẩy giá mạnh từ vùng support, không chỉ bounce yếu
```

### Điều kiện volume

```
[7] VOL_SPIKE    volume[-1] > 2.0 * AVG_VOL20
                 // Volume bùng nổ = smart money hấp thụ hàng tại vùng support
                 // Nâng từ 1.5x → 2.0x: cần lực mua thực sự để đảo chiều thuyết phục
                 // Là dấu hiệu quan trọng nhất xác nhận đảo chiều thực sự

[8] VOL_CONTRACT AVG_VOL_PRE5 < AVG_VOL20 * 0.75
                 // Volume 5 ngày trước signal giảm dần = sellers đang kiệt sức
                 // Nâng từ 0.8 → 0.75: co cụm rõ hơn trước khi bùng nổ
                 // [7]+[8]: volume kiệt → bùng nổ = classic reversal signature

Phân loại ưu tiên signal theo volume:
  TIER 1 = [7] + [8] cả hai đạt    → đảo chiều chất lượng cao nhất
  TIER 2 = [7] đạt, [8] không đạt  → tốt, xem xét trade
  TIER 3 = [7] không đạt           → tín hiệu yếu, chờ confirm chắc chắn hơn
```

### Điều kiện confirm (bắt buộc với reversal)

```
[9] CONFIRM      close[0] > high[-1]
                 // Nến ngày hôm nay vượt đỉnh signal candle
                 // Nếu chưa đạt → trạng thái PENDING
                 // Nếu đạt → trạng thái CONFIRMED → mới vào lệnh
                 // Không vào lệnh khi chỉ có signal candle mà chưa có confirm
```

### Lọc tránh tín hiệu rác

```
[X1] close[-1] >= MA200_NOW * 0.92
     // Loại nếu giá đã rơi hơn 8% dưới MA200 — downtrend quá mạnh, dễ dead cat
```

### Entry / SL / TP — Swing D1

```
// PHẢI chờ confirm [9] — không entry chỉ dựa vào signal candle

Entry   = open[0] khi nến [0] mở cửa cao hơn high[-1]
          HOẶC close[0] nếu nến hôm nay đóng xác nhận trên high[-1]

SL      = low[-1]   // dưới đáy signal candle

TP1     = MA50_NOW              // Chốt 50% tại MA50 — mục tiêu gần
TP2     = entry + 2.0 * ATR10   // Chốt 50% còn lại
```

### Output scan

```
Symbol | Date=[-1] | Signal=REVERSAL | Status=PENDING/CONFIRMED |
Close=X | MA200=X | ATR10=X | SL=X | TP1=X | TP2=X |
Volume=SPIKE+CONTRACT | SPIKE_ONLY | NO_VOL_DATA
```

---

## Bảng So Sánh

|  | Breakout | Reversal |
|---|---|---|
| Signal candle | `[-1]` đã đóng | `[-1]` đã đóng |
| Entry | Sáng hôm sau open[0] | Chờ confirm close[0] > high[-1] |
| Trend context | Uptrend — giá trên MA50 | Downtrend — giá dưới MA50 |
| Vùng hỗ trợ | Không yêu cầu | Bắt buộc — gần MA200 hoặc swing low |
| Breakout window | HIGH10 (EARLY) hoặc HIGH20 (STRONG) | N/A |
| Body filter | body ≥ 60% range | body ≥ 60% range |
| Volume contract | TIER 1 nếu < 75% baseline | TIER 1 nếu < 75% baseline |
| Volume spike | TIER 1/2 nếu ≥ 2x baseline | TIER 1/2 nếu ≥ 2x baseline |
| TIER 3 (no spike) | Chỉ theo dõi | Chờ confirm chắc chắn hơn |
| Hold time kỳ vọng | 5–15 ngày | 3–10 ngày |
| R:R mục tiêu | 2.0 | 2.0 |
| Timeframe | **Daily (D1)** | **Daily (D1)** |

---

## Hướng Dẫn Backtest

### Nguyên tắc chung

Backtest trên D1 cần **tối thiểu 3 năm dữ liệu** (khoảng 750 nến ngày).
Chia dữ liệu làm 2 phần: **70% in-sample** để develop rule, **30% out-of-sample** để validate.
Không được điều chỉnh tham số sau khi đã nhìn thấy kết quả out-of-sample — đó là overfit.

---

### Bước 1 — Chuẩn bị dữ liệu

```
Nguồn dữ liệu gợi ý:
- yfinance (Python)     → cổ phiếu, ETF, forex
- Binance API           → crypto
- IBKR / Alpaca API     → cổ phiếu US chuyên nghiệp

Cột bắt buộc: date | open | high | low | close | volume

Làm sạch trước khi chạy:
- Loại bỏ ngày không có giao dịch (volume = 0)
- Dùng adjusted close nếu là cổ phiếu (tránh sai lệch do cổ tức, split)
- Kiểm tra gap dữ liệu — nếu thiếu > 3 ngày liên tiếp thì xem lại nguồn
- Đảm bảo sort theo date tăng dần trước khi tính chỉ báo
```

---

### Bước 2 — Tính toán chỉ báo

```python
import pandas as pd
import numpy as np

# ATR(10) tính từ [-11] đến [-2] — loại trừ signal candle [-1]
# shift(2) để bắt đầu từ [-2], rolling(10) lấy 10 nến
def compute_atr(df, period=10):
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low']  - df['close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.shift(2).rolling(period).mean()   # shift(2) loại nến [-1]

df['atr10']        = compute_atr(df, 10)
df['avg_vol20']    = df['volume'].shift(2).rolling(20).mean()  # [-21] đến [-2]
df['avg_vol_pre5'] = df['volume'].shift(2).rolling(5).mean()   # [-6]  đến [-2]
df['high10']       = df['high'].shift(2).rolling(10).max()     # [-11] đến [-2]
df['high20']       = df['high'].shift(2).rolling(20).max()     # [-21] đến [-2]
df['ma50']         = df['close'].rolling(50).mean()
df['ma200']        = df['close'].rolling(200).mean()
df['ma50_prev5']   = df['ma50'].shift(5)
```

---

### Bước 3 — Hàm scan tín hiệu

```python
def scan_breakout(df):
    d = df
    body      = d['close'] - d['open']
    range_    = d['high'] - d['low']
    body_ratio = body / range_.replace(0, float('nan'))

    base_cond = (
        (d['close'] > d['ma50']) &
        (d['ma50'] > d['ma50_prev5']) &
        (d['close'] > d['open']) &
        (d['close'] >= d['high'] * 0.998) &
        (range_ > 1.5 * d['atr10']) &
        (body_ratio >= 0.6) &                         # [4b] body filter
        (d['close'] <= d['ma50'] * 1.08)
    )

    strong_cond = base_cond & (d['high'] > d['high20'])
    early_cond  = base_cond & (d['high'] > d['high10']) & (d['high'] <= d['high20'])

    strong = df[strong_cond].copy()
    strong['signal_type'] = 'BREAKOUT_STRONG'

    early = df[early_cond].copy()
    early['signal_type'] = 'BREAKOUT_EARLY'

    signals = pd.concat([strong, early]).sort_index()

    # Phân loại volume theo tier
    def vol_tier(row):
        if pd.isna(row['avg_vol20']): return 'NO_VOL_DATA'
        spike    = row['volume'] > 2.0 * row['avg_vol20']     # nâng từ 1.5x → 2.0x
        contract = row['avg_vol_pre5'] < row['avg_vol20'] * 0.75  # nâng từ 0.8 → 0.75
        if spike and contract: return 'TIER1'
        if spike:              return 'TIER2'
        return 'TIER3'

    signals['vol_tier'] = signals.apply(vol_tier, axis=1)
    return signals


def scan_reversal(df):
    results = []
    for i in range(9, len(df)):
        row  = df.iloc[i]
        prev = df.iloc[i-1]
        body_ratio = (row['close'] - row['open']) / (row['high'] - row['low']) if (row['high'] - row['low']) != 0 else 0

        cond = (
            row['close'] < row['ma50'] and
            abs(row['close'] - row['ma200']) <= 1.0 * row['atr10'] and
            row['low'] < prev['low'] and
            row['close'] > row['open'] and
            row['close'] >= row['high'] * 0.998 and
            (row['high'] - row['low']) > 1.5 * row['atr10'] and
            body_ratio >= 0.6 and                             # [6b] body filter
            row['close'] >= row['ma200'] * 0.92
        )
        if cond:
            results.append(i)

    signals = df.iloc[results].copy()

    def vol_tier(row):
        if pd.isna(row['avg_vol20']): return 'NO_VOL_DATA'
        spike    = row['volume'] > 2.0 * row['avg_vol20']
        contract = row['avg_vol_pre5'] < row['avg_vol20'] * 0.75
        if spike and contract: return 'TIER1'
        if spike:              return 'TIER2'
        return 'TIER3'

    signals['vol_tier'] = signals.apply(vol_tier, axis=1)
    return signals
```

---

### Bước 4 — Mô phỏng lệnh

```python
def simulate_trade(df, signal_idx, tp_multiplier=2.0, max_hold=15):
    """
    signal_idx : index của signal candle [-1] trong df
    Entry      : open của nến kế tiếp (signal_idx + 1)
    SL         : low của signal candle
    TP         : entry + tp_multiplier * atr10
    """
    if signal_idx + 1 >= len(df):
        return None

    entry  = df.iloc[signal_idx + 1]['open']
    sl     = df.iloc[signal_idx]['low']
    atr10  = df.iloc[signal_idx]['atr10']
    tp     = entry + tp_multiplier * atr10
    rr     = (tp - entry) / (entry - sl) if entry > sl else 0

    # Trừ slippage 0.1% cho thực tế
    entry  = entry * 1.001

    for j in range(signal_idx + 1, min(signal_idx + 1 + max_hold, len(df))):
        candle = df.iloc[j]
        # Quy ước bảo thủ: kiểm tra SL trước TP trong cùng một ngày
        if candle['low'] <= sl:
            return {'result': 'LOSS', 'days': j - signal_idx, 'pnl_pct': (sl - entry) / entry * 100}
        if candle['high'] >= tp:
            return {'result': 'WIN',  'days': j - signal_idx, 'pnl_pct': (tp - entry) / entry * 100}

    # Timeout — thoát theo close ngày cuối
    exit_price = df.iloc[min(signal_idx + max_hold, len(df)-1)]['close']
    return {'result': 'TIMEOUT', 'days': max_hold, 'pnl_pct': (exit_price - entry) / entry * 100}
```

---

### Bước 5 — Tính metrics kết quả

```python
def calc_metrics(trades):
    df_t = pd.DataFrame([t for t in trades if t is not None])
    
    wins     = df_t[df_t['result'] == 'WIN']
    losses   = df_t[df_t['result'] == 'LOSS']
    
    win_rate      = len(wins) / len(df_t) * 100
    avg_win       = wins['pnl_pct'].mean()
    avg_loss      = losses['pnl_pct'].mean()
    profit_factor = wins['pnl_pct'].sum() / abs(losses['pnl_pct'].sum())
    avg_hold      = df_t['days'].mean()
    
    # Max consecutive loss
    results_list = df_t['result'].tolist()
    max_consec_loss = 0
    cur = 0
    for r in results_list:
        cur = cur + 1 if r == 'LOSS' else 0
        max_consec_loss = max(max_consec_loss, cur)
    
    print(f"Tổng tín hiệu    : {len(df_t)}")
    print(f"Win rate         : {win_rate:.1f}%")
    print(f"Avg win          : +{avg_win:.2f}%")
    print(f"Avg loss         : {avg_loss:.2f}%")
    print(f"Profit factor    : {profit_factor:.2f}")
    print(f"Avg hold (ngày)  : {avg_hold:.1f}")
    print(f"Max consec loss  : {max_consec_loss}")
```

---

### Bước 6 — Phân tích theo nhóm volume

```python
# Sau khi có kết quả, tách ra theo vol_tag để xem volume có cải thiện kết quả không

for tier in ['TIER1', 'TIER2', 'TIER3', 'NO_VOL_DATA']:
    subset = [t for t, s in zip(trades, signals) if s.get('vol_tier') == tier]
    print(f"\n--- {tier} ---")
    calc_metrics(subset)

# Mục tiêu: TIER1 phải có win rate và profit factor tốt hơn TIER2, TIER2 tốt hơn TIER3
# Nếu không → điều kiện volume không giúp ích, cân nhắc hạ ngưỡng hoặc bỏ
# TIER3 nên cân nhắc loại khỏi trade list thực tế
```

---

### Bước 7 — Ngưỡng chấp nhận kết quả

```
Để rule được coi là đủ tốt để trade thực:

Số tín hiệu         >= 50        ít hơn thì kết quả không có ý nghĩa thống kê
Win rate            >= 40%       swing trading thắng ít nhưng lãi phải lớn hơn lỗ
Profit factor       >= 1.5       tổng lãi / tổng lỗ
Max consecutive loss <= 6        tâm lý chịu được chuỗi thua liên tiếp
Out-of-sample PF    >= in-sample PF * 0.7     không tệ hơn 30% là chấp nhận được

Nếu out-of-sample tệ hơn 30% → rule đang bị overfit → không dùng thực tế
```

---

### Những Lỗi Phổ Biến Khi Backtest

```
[LỖI 1] Look-ahead bias
Dùng dữ liệu tương lai khi tính chỉ báo — lỗi nghiêm trọng nhất.
→ Luôn dùng shift() khi tính ATR, avg_vol, high20 để đảm bảo
  chỉ dùng dữ liệu đã có tại thời điểm signal.

[LỖI 2] Entry tại close[-1]
Thực tế không thể mua đúng giá close[-1] vì scan chạy sau khi nến đóng.
→ Luôn dùng open[0] (ngày hôm sau) làm entry price.

[LỖI 3] Bỏ qua slippage và spread
→ Trừ 0.1% vào entry cho cổ phiếu/crypto lớn.
  Trừ 0.3–0.5% cho cổ phiếu nhỏ, altcoin.

[LỖI 4] Overfit tham số
Thay đổi 1.5 ATR, 20 nến, 0.8 volume cho đến khi backtest đẹp.
→ Chỉ điều chỉnh trên in-sample. Kết quả out-of-sample là final.

[LỖI 5] Không kiểm tra phân phối theo thời gian
Rule có thể chỉ hoạt động trong bull market, thất bại trong bear market.
→ Chia kết quả theo năm để kiểm tra tính nhất quán qua các chu kỳ.

[LỖI 6] Không tính max drawdown
Win rate cao nhưng drawdown 40% thì vẫn không thể trade thực tế được.
→ Luôn tính equity curve và max drawdown trước khi kết luận.
```