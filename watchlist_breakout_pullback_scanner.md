Điều kiện lọc:
Close hiện tại > MA200.
MA200 hiện tại dốc lên, ưu tiên:
MA200[t] > MA200[t-5]
Tồn tại 2 bull bar liên tiếp trong 10 phiên gần nhất.
Hai bull bar liên tiếp phải thỏa:
Close[d-1] > Open[d-1]
Close[d] > Open[d]
Ít nhất 1 trong 2 bull bar là big body bar:
RealBody = abs(Close - Open)
RealBody > 1.5 * MA10(RealBody trước đó)
Bar big body đó phải có volume tương xứng:
Volume[d] >= 1.5 * MA10(Volume trước đó)
hoặc nếu big body nằm ở d-1 thì:
Volume[d-1] >= 1.5 * MA10(Volume trước đó tại d-1)
Output trả về danh sách mã đạt điều kiện cùng metadata của cặp bull bar để dùng cho bước theo dõi pullback/test.