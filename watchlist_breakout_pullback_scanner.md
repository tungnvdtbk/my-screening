Watchlist Breakout Pullback Scanner — chọn top 20 mã uptrend đang có cụm bull bar
mạnh để đưa vào watchlist theo dõi pullback / test. Một mã chỉ cần thỏa MỘT trong
hai filter dưới đây là vào watchlist.

Điều kiện chung (áp cho cả 2 filter):
- Close hiện tại > MA200.
- MA200 hiện tại dốc lên, ưu tiên: MA200[t] > MA200[t-5]

Filter A — 2 bull bar liên tiếp + big body + volume (Tier A/B):
- Tồn tại 2 bull bar liên tiếp (d1, d2) trong 25 phiên gần nhất (0 <= t - d2 <= 24).
  Window 25 phiên cho phép bắt cụm bull đã hình thành rồi pullback vài phiên — đúng
  ngữ cảnh watchlist test/pullback.
- Hai bull bar liên tiếp phải thỏa:
  - Close[d-1] > Open[d-1]
  - Close[d]   > Open[d]
- Ít nhất 1 trong 2 bull bar là big body bar:
  - RealBody = abs(Close - Open)
  - RealBody > 1.5 * MA10(RealBody trước đó)
- Bar big body đó phải có volume tương xứng:
  - Volume[d]   >= 1.5 * MA10(Volume trước đó)
  - hoặc nếu big body nằm ở d-1 thì: Volume[d-1] >= 1.5 * MA10(Volume trước đó tại d-1)
- Tier:
  - A: cả d1 và d2 đều big body + đủ volume.
  - B: chỉ 1 trong 2 bar đạt.

Filter C — 3 bull bar liên tiếp, bar cuối trên MA20 (Tier C):
- Tồn tại 3 bull bar liên tiếp (d1, d2, d3) trong 25 phiên gần nhất (0 <= t - d3 <= 24).
- Cả 3 bar đều bull:
  - Close[d1] > Open[d1]
  - Close[d2] > Open[d2]
  - Close[d3] > Open[d3]
- Bar cuối đóng cửa trên MA20 hiện tại của nó:
  - Close[d3] > MA20[d3]
- Filter C KHÔNG yêu cầu big body / volume bùng nổ — đây là filter "soft" để mở
  rộng watchlist với những cụm bull bền nhưng nhẹ hơn Filter A.

Output:
- Trả về top 20 mã đạt điều kiện cùng metadata cụm bull bar (filter type & tier)
  để dùng cho bước theo dõi pullback / test.
- Ưu tiên xếp hạng: Tier A → B → C, sau đó gap_t (mới hơn lên trước), rồi mức độ
  mạnh: body ratio (cho A/B) hoặc khoảng cách Close vs MA20 (cho C).
