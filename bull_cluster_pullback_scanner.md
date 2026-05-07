Bull Cluster Pullback Scanner (BCP) — danh sách top 15 mã đã hình thành cụm
3 bull bar liên tiếp (giống Filter C của BPE) NHƯNG hiện tại đã pullback xuống
dưới đỉnh cao nhất của cụm và vẫn đang nằm trên đáy thấp nhất của cụm. Đây là
setup pullback đã thực sự xảy ra → actionable hơn BPE.

Mức ưu tiên: BCP > BPE (Watchlist Breakout Pullback). Trong UI, BCP hiển thị
TRƯỚC BPE; trong quy trình quyết định, BCP candidate được ưu tiên xem xét trước.

Điều kiện trigger (cụm bull, kế thừa Filter C):
- Close[t] > MA200[t]
- MA200 dốc lên: MA200[t] > MA200[t-5]
- Tồn tại 3 bull bar liên tiếp (d1, d2, d3) trong 25 phiên gần nhất.
  - Cả 3 bar đều bull: Close[d] > Open[d] với d ∈ {d1, d2, d3}
- Bar cuối của cụm đóng cửa trên MA20 của nó: Close[d3] > MA20[d3]

Điều kiện pullback (mới so với Filter C):
- t > d3 — đã có ít nhất 1 phiên sau khi cụm hoàn thành.
- Close[t] < max(High[d1], High[d2], High[d3]) — giá hiện tại đã pullback xuống
  dưới đỉnh cao nhất của cụm.
- Close[t] > min(Low[d1], Low[d2], Low[d3]) — giá vẫn nằm trên đáy thấp nhất
  của cụm; nếu thủng đáy này thì coi như cụm đã hỏng và mã bị loại.

Output:
- Trả về top 15 mã đạt điều kiện.
- Sắp xếp ưu tiên:
  1. gap_t = t - d3 DESC — pullback càng lâu càng được ưu tiên (đã reset lâu rồi,
     gần thời điểm bounce có thể xảy ra).
  2. depth_in_zone DESC — close càng gần cluster_high càng "khỏe" (pullback
     nông hơn = ít tổn thương hơn). depth_in_zone = (close - cluster_low) /
     (cluster_high - cluster_low).
  3. Symbol ASC — tie-break ổn định.
- Metadata: cluster_high, cluster_low, depth_in_zone (%), pullback_pct
  (= (close - cluster_high) / cluster_high × 100, luôn âm), MA20, MA200,
  SL = cluster_low × 0.99, TP = entry + 2R, R:R.
