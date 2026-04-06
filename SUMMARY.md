# VN HOSE Stock Scanner — System Summary

Streamlit app that scans toàn sàn HOSE theo hệ thống **Trend Following** (từ `he-thong-trading-vn.html`).

---

## Tính năng chính

### 🚦 Tín hiệu thị trường (Traffic Light)
Đánh giá sức khoẻ thị trường chung mỗi lần scan:

| Đèn | Điều kiện | Hành động |
|-----|-----------|-----------|
| 🟢 Xanh | VN-Index > MA50 **và** MA20 > MA50 **và** Breadth ≥ 55% | Giao dịch đầy đủ |
| 🟡 Vàng | Điều kiện trung gian (Breadth 40–55%) | Giảm exposure 50%, chỉ RS ≥ 85 |
| 🔴 Đỏ | VN-Index < MA50 **hoặc** MA20 < MA50 **hoặc** Breadth < 40% | Không mở vị thế mới |

- **Market Breadth** = % cổ phiếu HOSE đang trên MA50 của chúng (tính từ dữ liệu scan)
- **VN-Index** lấy từ vnstock3 (VCI/TCBS) hoặc fallback yfinance

---

### 📋 Trend Template — 10 tiêu chí bắt buộc
Mỗi cổ phiếu phải qua đủ 10/10 để vào watchlist:

| # | Tiêu chí |
|---|----------|
| 1 | Giá đóng cửa > MA50 daily |
| 2 | Giá đóng cửa > MA150 daily |
| 3 | MA50 > MA150 |
| 4 | MA150 tăng liên tục ≥ 4 tuần |
| 5 | Giá ≥ 25% trên đáy 52 tuần |
| 6 | Giá trong phạm vi 30% so với đỉnh 52 tuần |
| 7 | KLGD trung bình 20 phiên ≥ 200,000 cổ |
| 8 | Giá ≥ 15,000 VNĐ |
| 9 | Relative Strength ≥ 80 percentile (so với toàn sàn HOSE) |
| 10 | Không có dấu hiệu phân phối (3 phiên liên tiếp: giá giảm + khối lượng tăng) |

---

### 💪 Relative Strength (RS)
Xếp hạng sức mạnh tương đối so với **toàn sàn HOSE** (không phải chỉ VN30):

```
RS_raw = Perf_1M × 0.25 + Perf_3M × 0.35 + Perf_6M × 0.25 + Perf_12M × 0.15
```

- Percentile ranking toàn sàn → RS 90+ = ★★★ (top 10%), RS 80–89 = ★★, RS 70–79 = ★
- Đèn vàng: tự động nâng ngưỡng lên RS ≥ 85

---

### 🏆 Top 5 cổ phiếu
Hiển thị nổi bật 5 mã thoả mãn đủ 10 tiêu chí + RS cao nhất, kèm:
- Giá hiện tại, RS score + sao
- MA50, MA150, khối lượng trung bình
- **Stop Loss**: giá + % (mặc định −5%)
- **Target**: giá + % (mặc định +10%)

---

### 🛑 Stop Loss & 🎯 Target
Tính từ giá hiện tại, hiển thị cả giá tuyệt đối và phần trăm:

| | Công thức | Mặc định | Hiển thị ví dụ |
|--|-----------|----------|----------------|
| Stop Loss | `price × (1 − sl%)` | 5% | `47.5 (−5%)` |
| Target    | `price × (1 + tgt%)` | 10% | `55.0 (+10%)` |

Sidebar cho phép điều chỉnh: **Stop Loss 3–15%**, **Target 5–50%**.

---

### 💾 Cache dữ liệu giá (Incremental)
- Mỗi mã lưu vào `./data/cache/<SYMBOL>.parquet` (fallback CSV nếu thiếu pyarrow)
- Lần sau chỉ tải dữ liệu từ ngày cuối cùng trở đi → **scan nhanh hơn đáng kể**
- Docker volume mount `./data:/app/data` đảm bảo cache không mất khi restart

---

### 📊 Bảng kết quả đầy đủ
Cột hiển thị: Mã, Giá, **Stoploss**, **Target**, MA50, MA150, Đỉnh/Đáy 52T, Vol(20), Score(/9), RS%, Trạng thái, và 9 cột tiêu chí riêng lẻ (✓/✗).

Bộ lọc:
- **Hiển thị**: Chỉ Pass / Gần pass (score ≥ 7) / Tất cả
- **Score tối thiểu**: slider 0–9
- **Export Excel** (.xlsx)

---

### 📈 Chart Viewer
- Chọn bất kỳ mã trong kết quả để xem biểu đồ giá + MA20 + MA50 + MA150 (252 phiên gần nhất)
- Expander **"Chi tiết 10 tiêu chí"**: hiển thị ✅/❌ cho từng tiêu chí của mã đang xem

---

## Tuỳ chọn & cấu hình

| Tham số | Vị trí | Mặc định |
|---------|--------|----------|
| Danh sách mã | Sidebar radio | HOSE (toàn sàn ~400 mã) hoặc VN30 |
| Dùng cache giá | Sidebar checkbox | Bật |
| Stop Loss % | Sidebar slider | 5% |
| Target % | Sidebar slider | 10% |
| Hiển thị kết quả | Dropdown | Chỉ Pass |
| Score tối thiểu | Slider | 0 |

---

## Luồng hoạt động

```
[Scan Now]
    │
    ├─ Lấy danh sách HOSE (vnstock3 → file cache → VN30 fallback)
    ├─ Lấy VN-Index (vnstock3 VCI/TCBS → yfinance fallback)
    │
    ├─ For each symbol [progress bar]:
    │       ├─ Load giá (parquet cache, append mới nếu cần)
    │       ├─ Trend Template 9 tiêu chí kỹ thuật
    │       └─ Tính RS_raw
    │
    ├─ Tính Market Breadth → Traffic Light 🟢/🟡/🔴
    ├─ Rank RS percentile toàn sàn
    │
    ├─ Top 5 (Trend Template pass + RS ≥ 80)
    ├─ Bảng đầy đủ (filter + export)
    └─ Chart viewer
```

---

## Triển khai

```bash
# Docker (khuyến nghị)
docker compose up --build
# → http://localhost:8000

# Local
pip install -r requirements.txt
streamlit run app.py
```

Cache dữ liệu lưu tại `./data/cache/` — được mount vào container qua `docker-compose.yml`.

---

## Tests

```bash
docker exec ai_trading_project_v2-app-1 python test_app.py
# Ran 59 tests in 0.3s — OK
```

59 unit tests bao phủ: RS stars, RS ranking, RS raw formula, market filter, Trend Template (9 criteria + stoploss/target), cache helpers.
