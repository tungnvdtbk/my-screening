# SPEC: Rule quét cổ phiếu Việt Nam theo setup Pullback về MA trong xu hướng tăng

## Mục đích

Tài liệu này mô tả đầy đủ rule để một bot code có thể:
- quét danh sách cổ phiếu Việt Nam
- xác định cổ phiếu đang mạnh hơn thị trường
- phát hiện nhịp nghỉ ngắn trong xu hướng tăng
- tìm điểm kích hoạt mua dạng continuation entry
- loại bỏ các setup nhiễu thường gặp trên thị trường Việt Nam
- chấm điểm và xếp hạng setup theo score

Bot không dùng rule này để bắt đáy.  
Bot chỉ dùng rule này để tìm điểm mua tiếp diễn xu hướng.

---

# 1. Triết lý setup

Setup cần thỏa chuỗi logic sau:

1. Cổ phiếu đang trong xu hướng tăng rõ ràng
2. Cổ phiếu mạnh hơn benchmark thị trường
3. Sau nhịp tăng, giá có một đoạn nghỉ ngắn hoặc co hẹp
4. Giá giữ được vùng MA ngắn hạn
5. Có tín hiệu bật tăng trở lại với lực mua xác nhận
6. Điểm vào nằm đủ gần vùng hỗ trợ để risk/reward hợp lý

Tên setup:
- Pullback về MA
- Continuation setup
- Mini-base breakout
- Co hẹp rồi bật tiếp trong uptrend

---

# 2. Đầu vào dữ liệu

Bot cần có dữ liệu OHLCV tối thiểu:

- open
- high
- low
- close
- volume

Ngoài ra cần:
- dữ liệu benchmark: `VNINDEX`
- dữ liệu tối thiểu 120 phiên gần nhất để tính các biến trung hạn
- tốt hơn nếu có 250 phiên để mở rộng thêm xếp hạng sức mạnh

Khung thời gian mặc định:
- Daily

Universe mặc định:
- cổ phiếu niêm yết HOSE, HNX, UPCOM
- có thể giới hạn theo thanh khoản nếu cần

---

# 3. Định nghĩa các chỉ báo cơ bản

## 3.1 Moving Average

Dùng SMA hoặc EMA đều được, nhưng phải thống nhất trong toàn hệ thống.

Khuyến nghị:
- `MA10 = SMA(close, 10)`
- `MA20 = SMA(close, 20)`
- `MA50 = SMA(close, 50)`

Nếu muốn phản ứng nhanh hơn:
- có thể thay bằng EMA
- nhưng toàn bộ rule và backtest phải dùng cùng một chuẩn

---

## 3.2 Độ dốc MA20

Mục tiêu là xác định MA20 đang đi lên.

Cách đơn giản:

```text
slopeMA20 = MA20 - MA20[3]
```

Điều kiện MA20 dốc lên:

```text
slopeMA20 > 0
```

Có thể thay bằng:

```text
MA20 > MA20[1]
```

hoặc mạnh hơn:

```text
MA20 > MA20[3]
```

Khuyến nghị dùng:

```text
MA20 > MA20[3]
```

vì bớt nhiễu hơn.

---

## 3.3 Relative Strength so với VNINDEX

### RS 20 phiên

```text
RS_20 = (close / close[20] - 1) - (VNINDEX_close / VNINDEX_close[20] - 1)
```

### RS 55 phiên

```text
RS_55 = (close / close[55] - 1) - (VNINDEX_close / VNINDEX_close[55] - 1)
```

Ý nghĩa:
- `RS > 0`: cổ phiếu đang outperform benchmark
- `RS < 0`: cổ phiếu yếu hơn benchmark

Khuyến nghị:
- dùng đồng thời `RS_20` và `RS_55`
- `RS_20` để bắt độ mạnh ngắn hạn
- `RS_55` để tránh các pha hồi kỹ thuật ngắn

---

# 4. Điều kiện xu hướng nền

Bot chỉ xét setup nếu cổ phiếu đang ở trong xu hướng tăng đủ tốt.

## 4.1 Điều kiện bắt buộc

```text
close > MA20
MA20 > MA50
close > MA50
MA20 > MA20[3]
```

## 4.2 Ý nghĩa

- `close > MA20`: giá vẫn nằm phía trên hỗ trợ ngắn hạn
- `MA20 > MA50`: cấu trúc trung hạn là tăng
- `close > MA50`: loại các mã hồi kỹ thuật dưới trend chính
- `MA20 > MA20[3]`: MA20 thực sự đang đi lên

## 4.3 Ghi chú triển khai

Không dùng setup này trên mã:
- đang dưới MA50
- MA20 đang phẳng hoặc chúc xuống
- vừa có một nhịp breakdown rồi bật lại yếu

---

# 5. Điều kiện sức mạnh tương đối

## 5.1 Điều kiện chuẩn

```text
RS_20 > 0
RS_55 > 0
```

## 5.2 Điều kiện chặt hơn

Nếu muốn lọc hàng leader rõ hơn:

```text
RS_20 > 0.03
RS_55 > 0.03
```

hoặc:

```text
RS_20 > 0.05
RS_55 > 0.05
```

## 5.3 Khuyến nghị

Mặc định cho bot:

```text
RS_20 > 0
RS_55 > 0
```

Nếu số lượng mã ra quá nhiều:
- tăng ngưỡng lên `0.02` hoặc `0.03`

---

# 6. Điều kiện nhịp nghỉ / co hẹp

Đây là phần quan trọng nhất của setup.

Mục tiêu:
- xác định giá đang nghỉ sau nhịp tăng
- nhưng chưa gãy trend
- và đang nén lại để có thể chạy tiếp

## 6.1 Thời lượng nhịp nghỉ

Khuyến nghị:
- nhịp nghỉ kéo dài từ 3 đến 8 nến gần nhất

Bot có thể xét trong cửa sổ:
- 5 nến
- hoặc 8 nến

Khuyến nghị mặc định:
- cửa sổ 5 nến để đơn giản
- nếu muốn mềm hơn thì mở 8 nến

---

## 6.2 Biên độ co hẹp

Cách đo đơn giản:

```text
range_5 = highest(high, 5) / lowest(low, 5)
```

Điều kiện co hẹp:

```text
range_5 < 1.06
```

Nghĩa là biên độ toàn cụm 5 nến nhỏ hơn 6%.

Điều kiện chặt hơn:

```text
range_5 < 1.05
```

Điều kiện mềm hơn:

```text
range_5 < 1.08
```

Khuyến nghị mặc định:

```text
range_5 < 1.06
```

---

## 6.3 Không phá cấu trúc tăng

Mục tiêu là tránh những nhịp giảm sâu trá hình.

Khuyến nghị mặc định:

```text
lowest(low, 5) >= MA20 * 0.97
```

Ý nghĩa:
- toàn bộ nhịp nghỉ vẫn bám tương đối gần MA20
- không có nến thủng sâu bất thường

---

## 6.4 Giá chạm vùng MA ngắn

Bot cần ưu tiên các setup nghỉ gần MA10/MA20.

Điều kiện:

```text
low <= MA10 * 1.01
close >= MA10
```

Ý nghĩa:
- trong phiên có test vùng MA10
- cuối phiên vẫn lấy lại được MA10

Khuyến nghị:
- ưu tiên MA10 cho điểm vào đẹp hơn
- dùng MA20 làm mức kiểm tra phụ

---

## 6.5 Thanh khoản co lại trong nhịp nghỉ

Mục tiêu:
- nhịp nghỉ phải là nghỉ thật
- không phải bị phân phối

Khuyến nghị mặc định:

```text
SMA(volume, 3) < SMA(volume, 20)
```

Nếu code đơn giản hơn thì dùng:

```text
volume <= SMA(volume, 20)
```

---

# 7. Điều kiện kích hoạt mua

Bot chỉ sinh tín hiệu khi có xác nhận bật tăng trở lại.

## 7.1 Trigger chuẩn

```text
close > close[1]
close > high[1]
close >= high * 0.98
volume > volume[1]
```

## 7.2 Ý nghĩa

- `close > close[1]`: đóng cửa mạnh hơn phiên trước
- `close > high[1]`: vượt đỉnh ngắn hạn gần nhất
- `close >= high * 0.98`: đóng cửa sát đỉnh phiên, chứng tỏ lực mua giữ được đến cuối phiên
- `volume > volume[1]`: lực mua tốt hơn ít nhất so với phiên ngay trước

---

# 8. Bộ rule chuẩn để scan

## 8.1 Rule mặc định

```text
close > MA20
MA20 > MA50
close > MA50
MA20 > MA20[3]

RS_20 > 0
RS_55 > 0

highest(high, 5) / lowest(low, 5) < 1.06
lowest(low, 5) >= MA20 * 0.97

low <= MA10 * 1.01
close >= MA10

SMA(volume, 3) < SMA(volume, 20)

close > close[1]
close > high[1]
close >= high * 0.98
volume > volume[1]
```

---

# 9. Bộ lọc loại nhiễu

## 9.1 Loại mã kéo quá xa khỏi MA20

```text
(close / MA20) - 1 <= 0.05
```

## 9.2 Loại nến breakout quá dài

```text
(high - low) / close <= 0.05
```

## 9.3 Loại nến rút đầu mạnh

```text
close >= high * 0.98
```

## 9.4 Loại mã đầu cơ quá yếu

Có thể thêm:

```text
close > 10
```

hoặc:

```text
avgTradingValue20 = SMA(close * volume, 20)
avgTradingValue20 > ngưỡng tối thiểu
```

---

# 10. Score và xếp hạng tín hiệu

Bot không chỉ trả về pass/fail, mà phải tính `score` để xếp hạng các setup.

## 10.1 Định nghĩa score

`score` là tổng điểm của nhiều cụm điều kiện:
- xu hướng
- sức mạnh tương đối
- độ đẹp của nhịp nghỉ
- chất lượng trigger
- mức nhiễu/rủi ro

Score càng cao thì setup càng đẹp.

---

## 10.2 Khung điểm chuẩn 100

### A. Điểm xu hướng: tối đa 25

```text
+10 nếu close > MA20
+10 nếu MA20 > MA50
+5  nếu MA20 > MA20[3]
```

### B. Điểm sức mạnh tương đối: tối đa 20

```text
+10 nếu RS_20 > 0
+10 nếu RS_55 > 0
```

Có thể thưởng thêm:
```text
+5 nếu RS_20 > 0.03
+5 nếu RS_55 > 0.03
```

Nếu dùng thưởng thêm thì phải chuẩn hóa lại về 100 hoặc giới hạn trần.

### C. Điểm nhịp nghỉ / co hẹp: tối đa 20

```text
+8 nếu highest(high, 5) / lowest(low, 5) < 1.06
+6 nếu lowest(low, 5) >= MA20 * 0.97
+6 nếu low <= MA10 * 1.01 và close >= MA10
```

### D. Điểm volume: tối đa 10

```text
+5 nếu SMA(volume, 3) < SMA(volume, 20)
+5 nếu volume > volume[1]
```

### E. Điểm trigger: tối đa 15

```text
+5 nếu close > close[1]
+5 nếu close > high[1]
+5 nếu close >= high * 0.98
```

### F. Điểm lọc nhiễu: tối đa 10

```text
+5 nếu (close / MA20) - 1 <= 0.05
+5 nếu (high - low) / close <= 0.05
```

Tổng:
```text
25 + 20 + 20 + 10 + 15 + 10 = 100
```

---

## 10.3 Công thức score mặc định

```text
score = 0

if close > MA20: score += 10
if MA20 > MA50: score += 10
if MA20 > MA20[3]: score += 5

if RS_20 > 0: score += 10
if RS_55 > 0: score += 10

if highest(high, 5) / lowest(low, 5) < 1.06: score += 8
if lowest(low, 5) >= MA20 * 0.97: score += 6
if low <= MA10 * 1.01 and close >= MA10: score += 6

if SMA(volume, 3) < SMA(volume, 20): score += 5
if volume > volume[1]: score += 5

if close > close[1]: score += 5
if close > high[1]: score += 5
if close >= high * 0.98: score += 5

if (close / MA20) - 1 <= 0.05: score += 5
if (high - low) / close <= 0.05: score += 5
```

---

## 10.4 Xếp hạng theo score

```text
score >= 85      => A: setup rất đẹp
score 70-84      => B: setup tốt
score 55-69      => C: theo dõi thêm
score < 55       => D: bỏ qua
```

Khuyến nghị cho bot:
- chỉ alert khi `score >= 70`
- ưu tiên top các mã có `score` cao nhất
- nếu nhiều mã bằng điểm, ưu tiên mã có `RS_20` và `RS_55` cao hơn

---

# 11. Logic signal

`signal` là biến boolean cuối cùng để bot xác định mã có setup hợp lệ hay không.

```text
trend_ok =
    close > MA20
    and MA20 > MA50
    and close > MA50
    and MA20 > MA20[3]

strength_ok =
    RS_20 > 0
    and RS_55 > 0

pullback_ok =
    highest(high, 5) / lowest(low, 5) < 1.06
    and lowest(low, 5) >= MA20 * 0.97
    and low <= MA10 * 1.01
    and close >= MA10
    and SMA(volume, 3) < SMA(volume, 20)

trigger_ok =
    close > close[1]
    and close > high[1]
    and close >= high * 0.98
    and volume > volume[1]

filter_ok =
    ((close / MA20) - 1) <= 0.05
    and ((high - low) / close) <= 0.05

signal =
    trend_ok
    and strength_ok
    and pullback_ok
    and trigger_ok
    and filter_ok
```

Ý nghĩa:
- `signal = true`: cổ phiếu thỏa điều kiện setup
- `signal = false`: bỏ qua

Khuyến nghị:
- bot nên xuất cả `signal` và `score`
- `signal` dùng để lọc
- `score` dùng để xếp hạng

---

# 12. Logic vào lệnh

Khuyến nghị mặc định:
- dùng `Standard Entry`

## 12.1 Entry mặc định

```text
entryPrice = close
```

hoặc bảo thủ hơn:

```text
entryPrice = high[1] + minimum_tick
```

Nếu hệ thống chạy cuối ngày:
- dùng `close` của ngày signal
- hoặc lên kế hoạch mua phiên sau nếu giá vượt `high[1]`

---

# 13. Stop loss

## 13.1 Stop mặc định

```text
stopLoss = lowest(low, 5)
```

hoặc an toàn hơn:

```text
stopLoss = lowest(low, 5) * 0.995
```

Khuyến nghị mặc định:
- stop dưới đáy cụm co hẹp gần nhất

---

# 14. Risk filter

```text
riskPct = (entryPrice - stopLoss) / entryPrice
```

Loại nếu:

```text
riskPct > 0.04
```

Khuyến nghị:
- chỉ nhận setup có `riskPct <= 0.04`

---

# 15. Pseudocode triển khai

```text
MA10 = SMA(close, 10)
MA20 = SMA(close, 20)
MA50 = SMA(close, 50)

RS_20 = (close / close[20] - 1) - (VNINDEX_close / VNINDEX_close[20] - 1)
RS_55 = (close / close[55] - 1) - (VNINDEX_close / VNINDEX_close[55] - 1)

trend_ok =
    close > MA20
    and MA20 > MA50
    and close > MA50
    and MA20 > MA20[3]

strength_ok =
    RS_20 > 0
    and RS_55 > 0

pullback_ok =
    highest(high, 5) / lowest(low, 5) < 1.06
    and lowest(low, 5) >= MA20 * 0.97
    and low <= MA10 * 1.01
    and close >= MA10
    and SMA(volume, 3) < SMA(volume, 20)

trigger_ok =
    close > close[1]
    and close > high[1]
    and close >= high * 0.98
    and volume > volume[1]

filter_ok =
    ((close / MA20) - 1) <= 0.05
    and ((high - low) / close) <= 0.05

signal =
    trend_ok
    and strength_ok
    and pullback_ok
    and trigger_ok
    and filter_ok

score = 0
if close > MA20: score += 10
if MA20 > MA50: score += 10
if MA20 > MA20[3]: score += 5
if RS_20 > 0: score += 10
if RS_55 > 0: score += 10
if highest(high, 5) / lowest(low, 5) < 1.06: score += 8
if lowest(low, 5) >= MA20 * 0.97: score += 6
if low <= MA10 * 1.01 and close >= MA10: score += 6
if SMA(volume, 3) < SMA(volume, 20): score += 5
if volume > volume[1]: score += 5
if close > close[1]: score += 5
if close > high[1]: score += 5
if close >= high * 0.98: score += 5
if (close / MA20) - 1 <= 0.05: score += 5
if (high - low) / close <= 0.05: score += 5

entryPrice = close
stopLoss = lowest(low, 5)
riskPct = (entryPrice - stopLoss) / entryPrice
```

---

# 16. Output mong muốn của bot

Mỗi mã pass rule cần trả ra:

- symbol
- signalDate
- signal
- score
- grade
- entryPrice
- stopLoss
- riskPct
- RS_20
- RS_55
- MA10
- MA20
- MA50
- volume
- avgVol20

Ví dụ:

```json
{
  "symbol": "ABC",
  "signalDate": "2025-04-18",
  "signal": true,
  "score": 82,
  "grade": "B",
  "entryPrice": 27.4,
  "stopLoss": 26.2,
  "riskPct": 0.0438,
  "RS_20": 0.06,
  "RS_55": 0.08,
  "MA10": 26.9,
  "MA20": 26.4,
  "MA50": 24.8
}
```

---

# 17. Quy tắc vận hành khuyến nghị

## 17.1 Cho scanner

- quét cuối ngày
- chỉ giữ các mã có `signal = true`
- sắp xếp giảm dần theo `score`
- nếu cùng score, sắp theo `RS_20` rồi `RS_55`

## 17.2 Cho alert

Chỉ gửi alert nếu:
- `signal = true`
- `score >= 70`
- `riskPct <= 0.04`

## 17.3 Cho dashboard

Hiển thị:
- top 10 mã score cao nhất
- màu phân cấp theo grade
- cờ cảnh báo nếu `riskPct` quá cao

---

# 18. Kết luận

Rule này được thiết kế cho thị trường Việt Nam với mục tiêu:
- không bắt đáy
- không đuổi breakout quá muộn
- chỉ săn continuation setup đẹp trong xu hướng mạnh

Trọng tâm triển khai cho bot là:
1. dùng `signal` để xác định setup hợp lệ
2. dùng `score` để xếp hạng setup
3. dùng `riskPct` để loại các điểm vào quá xấu

Công thức tối ưu nhất để bot vận hành:
- `signal` để lọc
- `score` để ưu tiên
- `riskPct` để kiểm soát chất lượng điểm vào
