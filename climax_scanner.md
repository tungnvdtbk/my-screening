# Rule filter: Sell Climax -> False Break Support -> Reversal Buy

## 1. Ý tưởng setup
Tìm các cổ phiếu bị bán mạnh kiểu **sell climax**, có thể bị đạp thủng support ngắn hạn, sau đó xuất hiện **nến đảo chiều mạnh** như:

- Bullish marubozu
- Bullish pin bar / hammer

Mục tiêu là lọc các mã có khả năng bị washout xong rồi kéo ngược mạnh.

---

## 2. Điều kiện filter tổng quát

### A. Bối cảnh giảm trước đó
Phải có pha giảm đủ mạnh trước khi xét đảo chiều.

**Rule:**
- `Close < MA20`
- `MA20 dốc xuống`
- Trong `5-10` nến gần nhất:
  - số nến đỏ >= `60%`
  - giá giảm từ swing high gần nhất >= `6%` đến `12%` với daily
- Tuỳ chọn:
  - `RSI(14) < 35`

---

### B. Sell climax
Phải có dấu hiệu bán tháo mạnh.

**Rule:**
Trong `2-4` nến gần nhất có ít nhất:
- `2` nến đỏ mạnh  
hoặc
- `1` nến bearish marubozu + `1` nến đỏ mạnh tiếp diễn

**Định nghĩa bearish wide-range candle:**
- `range = high - low`
- `range > 1.5 * ATR(14)`
- `close < open`
- `body / range >= 0.65`

**Định nghĩa bearish marubozu:**
- `close < open`
- `body / range >= 0.8`
- `upper_wick <= 0.1 * range`
- `lower_wick <= 0.1 * range`

Trong đó:
- `body = abs(close - open)`
- `upper_wick = high - max(open, close)`
- `lower_wick = min(open, close) - low`

---

### C. Thủng support giả
Giá bị đạp xuống dưới support rồi kéo ngược lên.

**Rule:**
- Xác định `support` là:
  - low thấp nhất của `10-20` nến trước  
  hoặc
  - pivot low gần nhất
- Nến climax hoặc nến đảo chiều có:
  - `low < support * (1 - buffer)`
- Với:
  - `buffer = 0.2% đến 1%`
- Điều kiện mạnh hơn:
  - `close >= support`

---

### D. Nến đảo chiều mạnh

#### 1. Bullish marubozu
**Rule:**
- `close > open`
- `body / range >= 0.8`
- `upper_wick <= 0.1 * range`
- `lower_wick <= 0.15 * range`
- `close` nằm gần đỉnh nến

#### 2. Bullish pin bar / hammer
**Rule:**
- `lower_wick >= 2 * body`
- `upper_wick <= 0.3 * body`
- `close >= low + 0.66 * range`
- Ưu tiên:
  - `close > open`
  - xuất hiện ngay tại vùng support hoặc dưới support rồi đóng lại phía trên

---

### E. Xác nhận buy ngược
Không mua chỉ vì có một nến đẹp.

**Rule xác nhận:**
- Nến sau đó phải thỏa một trong các điều kiện:
  - `close > high` của nến đảo chiều
  - hoặc intraday break high nến đảo chiều
- Volume:
  - `volume_reversal >= 1.5 * SMA(volume,20)`
  - hoặc `volume_confirm >= 1.2 * SMA(volume,20)`

---

## 3. Rule filter phiên bản chặt

```text
1. Close < MA20
2. MA20 dốc xuống
3. Giá giảm >= 8% trong 10 nến gần nhất
4. Trong 3 nến gần nhất có ít nhất 2 nến đỏ mạnh
5. Có ít nhất 1 nến có range > 1.5 * ATR14
6. Low hiện tại < support cũ ít nhất 0.5%
7. Close hiện tại >= support cũ
8. Xuất hiện bullish marubozu hoặc bullish pin bar
9. Nếu là pin bar:
   - lower_wick >= 2 * body
10. Nếu là marubozu:
   - body / range >= 0.8
11. Nến sau đóng cửa vượt high nến đảo chiều
12. Volume nến đảo chiều hoặc nến xác nhận > 1.5 * VolMA20
```

---

## 4. Volume trên nến Sell Climax

Sell climax thực sự phải có volume bất thường — dấu hiệu capitulation:

**Rule:**
- Volume nến climax (bearish wide-range) >= `1.3 * SMA(volume, 20)`
- Ưu tiên: volume nến climax >= `2.0 * SMA(volume, 20)` (panic selling)

---

## 5. SL / TP / R:R

### Stop Loss
- `SL = low` của nến đảo chiều (reversal candle)
- Nếu nến đảo chiều là pin bar dài: `SL = low * 0.998` (buffer nhỏ)

### Take Profit
- `TP1 = MA20` tại thời điểm entry (mean reversion target)
- `TP2 = entry + 2 × (entry - SL)` (R:R = 2)
- Chọn TP = `max(TP1, TP2)` để đảm bảo R:R >= 2

### R:R filter
- Chỉ lấy signal nếu `R:R >= 2.0`
- `R:R = (TP - entry) / (entry - SL)`

---

## 6. Status: PENDING / CONFIRMED

- Khi phát hiện sell climax + false break + nến đảo chiều → status = `PENDING`
- Khi nến tiếp theo đóng cửa > high nến đảo chiều → status = `CONFIRMED`
- Chỉ entry khi `CONFIRMED`
- Nếu nến tiếp theo đóng cửa < low nến đảo chiều → huỷ signal

---

## 7. Quality Tier (A / B) — Hard Gate

Chỉ giữ Tier A và Tier B. Tín hiệu không đạt sẽ bị loại.

### Tier A (target ~100% WR)
```
decline_pct >= 8%
AND risk_pct < 2%
AND reversal_type in (HAMMER, MARUBOZU)
AND R:R >= 2.0
```

### Tier B (target >50% WR, R:R >= 2:1)
```
RSI < 35
AND climax_vol_ok (volume climax xác nhận)
AND reversal_type in (HAMMER, MARUBOZU, ENGULFING)
AND R:R >= 2.0
```

### Rejection
```
if not (Tier A or Tier B):
    reject signal
```