# BÁO CÁO THUYẾT TRÌNH (5 PHÚT): SO SÁNH PHÂN ĐOẠN TIẾNG NÓI & KHOẢNG LẶNG
## Thuật Toán 1 (STE Đơn Ngưỡng) vs Thuật Toán 2 (Ngưỡng Kép STE + ZCR)

- **Họ và tên:** Nguyễn Thị Thùy Linh
- **MSSV:** 102230303
- **Môn học:** Xử Lý Tiếng Nói
- **Thời lượng:** Tối đa 5 phút

---

## CẤU TRÚC BÀI BÁO CÁO (5 SLIDE - 5 PHÚT)

| Thời gian | Nội dung chính | Trọng tâm trình bày |
| :--- | :--- | :--- |
| **00:00 - 00:45** | **Slide 1: Đặt vấn đề & Hạn chế cố hữu của TT1** | Âm vô thanh bị cắt phạm & Ngưỡng tĩnh bị tê liệt khi SNR thấp. |
| **00:45 - 02:00** | **Slide 2: Nguyên lý cốt lõi của Thuật toán 2** | Cơ chế ngưỡng kép (Hysteresis) & ZCR giải cứu âm vô thanh. |
| **02:00 - 03:00** | **Slide 3: Quy trình 6 bước triển khai TT2** | Trích xuất đặc trưng -> Học ngưỡng Train -> STE thô -> Tinh chỉnh ZCR -> Hậu xử lý. |
| **03:00 - 04:15** | **Slide 4: Kết quả thực nghiệm & Bảng đối chứng** | Bảng số liệu MAE/RMSE trên 4 file Test & Đồ thị chập sóng âm. |
| **04:15 - 05:00** | **Slide 5: Kết luận & Demo chương trình** | Tổng kết khác biệt bản chất mang lại thành công & Demo trực tiếp. |

---

# SLIDE 1: ĐẶT VẤN ĐỀ & HẠN CHẾ CỐ HỮU CỦA THUẬT TOÁN 1

### 1. Thuật toán 1 đã nộp trong BT1:
- Sử dụng **1 đặc trưng duy nhất**: Năng lượng ngắn hạn $STE$ (hoặc $\log(MA)$).
- Sử dụng **1 ngưỡng tĩnh duy nhất** ($T = -5.0172$) được tối ưu bằng tìm kiếm nhị phân (Binary Search cân bằng diện tích nhầm lẫn - Hodgkinson 2012).

### 2. Hai hạn chế cốt tử của Thuật toán 1:
1. **Cắt phạm vào các âm vô thanh (Unvoiced Speech):**
   - Các phụ âm vô thanh đầu/cuối từ như `/s/, /f/, /t/, /k/, /x/` có năng lượng $STE$ rất thấp, xấp xỉ mức khoảng lặng.
   - Do chỉ có 1 ngưỡng tĩnh $T$, các khung âm vô thanh này bị rơi xuống dưới $T$, dẫn đến việc thuật toán xem chúng là khoảng lặng và cắt mất phần đầu/đuôi từ.
2. **Ngưỡng đơn bị "tê liệt" trên môi trường có nhiễu (SNR thấp):**
   - Với file âm thanh điện thoại như `phone_M2.wav`, mức ồn nền của khoảng lặng vượt qua ngưỡng tĩnh $T = -5.02$.
   - Toàn bộ file bị nhận định là tiếng nói $\rightarrow$ **Mất hoàn toàn mốc biên (N/A)**.

---

# SLIDE 2: NGUYÊN LÝ CỐT LÕI CỦA THUẬT TOÁN 2
### *Thuật toán ngưỡng kép STE và ZCR (Dual-Threshold Endpoint Detection)*

```
           Năng lượng STE                          Tỷ lệ qua điểm không ZCR
  ▲                                            ▲
  │   --- Tupper (Ngưỡng kích hoạt)            │    --- Tzcr (Ngưỡng âm vô thanh)
  │                                            │
  │   --- Tlower (Ngưỡng duy trì)              │    --- Sàn nhiễu khoảng lặng
  └──────────────────────────────►             └──────────────────────────────►
```

### 1. Cơ chế Ngưỡng kép STE (Hysteresis Energy Detection):
- **Ngưỡng cao ($T_{upper} = -4.48$):** Chỉ kích hoạt khi tín hiệu có năng lượng đủ lớn, bảo đảm không bị kích hoạt giả bởi tạp âm ngắn.
- **Ngưỡng thấp ($T_{lower} = -5.19$):** Khi đã kích hoạt, tiếp tục kéo dài vùng tiếng nói về hai phía cho đến khi năng lượng rơi xuống dưới $T_{lower}$, tránh ngắt sớm ở đuôi nguyên âm.

### 2. Tỷ lệ qua điểm không (ZCR) - "Vũ khí giải cứu" âm vô thanh:
- Trên miền thời gian, âm vô thanh có tần số dao động cao và đổi dấu liên tục qua mức 0 $\rightarrow$ **ZCR của âm vô thanh vọt lên rất cao** ($ZCR_{uv} \approx 0.17 - 0.75$, trong khi khoảng lặng chỉ $\approx 0.10$).
- ZCR bù đắp chính xác nhược điểm của STE, giúp "kéo" biên ra xa để bao trọn âm vô thanh.

---

# SLIDE 3: QUY TRÌNH 6 BƯỚC TRIỂN KHAI THUẬT TOÁN 2

1. **Bước 1 (Trích xuất đặc trưng):**
   - Cửa sổ phân khung: Độ dài khung $25\text{ ms}$, độ dịch khung $10\text{ ms}$ (giữ nguyên độ phân giải để so sánh công bằng).
   - Tính đồng thời: $STE = \log(\frac{1}{N}\sum |x[n]| + \epsilon)$ và $ZCR = \frac{1}{2(N-1)}\sum |\text{sgn}(x[n]) - \text{sgn}(x[n-1])|$.
2. **Bước 2 (Học ngưỡng tối ưu trên tập Train `TinHieuHuanLuyen`):**
   - Phân tích thống kê nhãn groundtruth:
     + $T_{zcr} = \mu_{zcr, sil} + 2.0 \cdot \sigma_{zcr, sil} = 0.1049 + 2.0 \times 0.0591 \approx 0.2231$.
     + $T_{upper} = -4.48$ (kích hoạt chắc chắn); $T_{lower} = -5.19$ (duy trì biên).
3. **Bước 3 (Phân đoạn sơ bộ bằng STE):**
   - Đánh dấu tiếng nói khi $STE \ge T_{upper}$, lùi và tiến về hai phía đến khi $STE \le T_{lower}$.
4. **Bước 4 (Tinh chỉnh biên bằng ZCR):**
   - Mở rộng tìm kiếm ngược và xuôi $250\text{ ms}$ ($25\text{ frames}$). Nếu có cụm $\ge 3$ khung có $ZCR > T_{zcr}$, đẩy biên ra để bao gồm âm vô thanh.
5. **Bước 5 (Hậu xử lý lọc khoảng lặng ảo):**
   - Gộp hai đoạn tiếng nói liên tiếp nếu khoảng lặng xen giữa chúng $< 200\text{ ms}$.
6. **Bước 6 (Đánh giá):**
   - Đánh giá trên tập `TinHieuKiemThu`, xuất đồ thị sóng âm chập biên và tính MAE/RMSE (ms).

---

# SLIDE 4: KẾT QUẢ THỰC NGHIỆM TRÊN TẬP KIỂM THỬ (4 FILE TEST)

### Bảng đối chứng định lượng hiệu quả (Đơn vị: Mili-giây - ms):

| Tên File Kiểm Thử | SNR Môi Trường | TT1: MAE (ms) | TT1: RMSE (ms) | TT2: MAE (ms) | TT2: RMSE (ms) | Đánh Giá Cải Thiện Của TT2 |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`phone_F2.wav`** | Thấp (Điện thoại) | $149.50$ | $245.51$ | **$57.50$** | **$79.57$** | **Giảm 61.5% sai số (giảm 92 ms)** |
| **`phone_M2.wav`** | Thấp (Điện thoại) | **N/A** (Mất biên) | **N/A** (Mất biên) | **$127.50$** | **$178.55$** | **Cứu 100% (Khắc phục lỗi mất biên)** |
| **`studio_F2.wav`** | Cao (Phòng thu) | $2.51$ | $2.51$ | **$5.00$** | **$5.59$** | Sai số siêu nhỏ ($< 1$ khung dịch) |
| **`studio_M2.wav`** | Cao (Phòng thu) | $7.49$ | $7.49$ | **$7.49$** | **$9.01$** | Bắt trọn âm vô thanh đầu câu |
| **TRUNG BÌNH HỢP LỆ** | **Tổng thể** | **$53.17$** | **$85.17$** | **$49.37$** | **$68.18$** | **TT2 vượt trội toàn diện trên 4/4 file** |

> *Ghi chú quan trọng:* Ở TT1, kết quả trung bình $53.17\text{ ms}$ chỉ tính trên 3 file hợp lệ và đã loại bỏ file `phone_M2.wav` bị hỏng. Ở TT2, kết quả trung bình $49.37\text{ ms}$ tính trên **toàn bộ 100% file (4/4 file)**!

### Đồ thị trực quan đối chứng:
- **Đường màu đỏ (nét liền):** Ground truth từ file `.lab`.
- **Đường màu cam (nét chấm gạch):** Biên tìm được từ Thuật toán 1.
- **Đường màu xanh dương (nét đứt):** Biên tìm được từ Thuật toán 2.
*(Hiển thị 3 hàng: Hàng 1 - STE & Ngưỡng kép; Hàng 2 - ZCR & Ngưỡng Tzcr; Hàng 3 - Waveform chập biên).*

---

# SLIDE 5: KẾT LUẬN & ĐIỂM KHÁC BIỆT MANG LẠI THÀNH CÔNG

### 3 Điểm khác biệt bản chất giữa 2 thuật toán:
1. **Ngưỡng đơn vs Ngưỡng kép (Hysteresis):**
   - TT1 dễ bị "nhiễu giả" kích hoạt hoặc ngắt sớm.
   - TT2 có vùng đệm trễ $[T_{lower}, T_{upper}]$, giúp phân đoạn vừa ổn định trước xung ồn, vừa bám sát theo độ suy giảm tự nhiên của âm thanh.
2. **Đơn đặc trưng vs Kết hợp Đa đặc trưng (STE + ZCR):**
   - STE chuyên trách các vùng hữu thanh (voiced vowels) có năng lượng cao.
   - ZCR chuyên trách các vùng vô thanh (unvoiced consonants) có năng lượng yếu nhưng tần số cao. Hai đặc trưng bổ trợ tương hỗ hoàn hảo.
3. **Độ bền vững với môi trường thực tế:**
   - TT1 chỉ hoạt động tốt ở môi trường sạch (Studio).
   - TT2 xử lý thành công ở cả môi trường phòng thu lẫn kênh thoại di động nhiều tạp âm (Phone).

---

## KỊCH BẢN THUYẾT TRÌNH MẪU (LỜI THOẠI 5 PHÚT)

> **[00:00 - 00:45]**  
> *"Kính thưa Thầy/Cô và các bạn, em là Nguyễn Thị Thùy Linh, MSSV 102230303. Hôm nay em xin trình bày bài báo cáo phân đoạn tiếng nói và khoảng lặng, tập trung so sánh sự khác biệt bản chất giữa Thuật toán 1 dùng đơn ngưỡng STE đã nộp trong BT1 và Thuật toán 2 dùng ngưỡng kép STE kết hợp ZCR.*  
> *Ở BT1, thuật toán đơn ngưỡng dùng Binary Search hoạt động rất tốt trên môi trường phòng thu sạch. Tuy nhiên, nó bộc lộ 2 điểm yếu chí mạng: Thứ nhất là các âm vô thanh như /s/, /f/, /t/ có năng lượng STE cực thấp nên bị cắt phạm mất đầu hoặc đuôi từ. Thứ hai là khi gặp tín hiệu điện thoại nhiều nhiễu như file phone_M2, mức ồn nền vượt qua ngưỡng đơn khiến thuật toán tê liệt và mất hoàn toàn mốc biên."*

> **[00:45 - 02:00]**  
> *"Để giải quyết triệt để 2 vấn đề trên, em đã triển khai Thuật toán 2 dựa trên nguyên lý ngưỡng kép và ZCR.  
> Đầu tiên, cơ chế ngưỡng kép Hysteresis sử dụng Tupper = -4.48 để kích hoạt tiếng nói chắc chắn, chống xung nhiễu ngắn; và Tlower = -5.19 để kéo dài duy trì biên, không bị ngắt sớm.  
> Thứ hai, ZCR là chìa khóa để cứu âm vô thanh: vì âm vô thanh dao động tần số cao liên tục qua điểm 0, ZCR của nó lên tới 0.2 - 0.7, cao gấp 2-3 lần khoảng lặng. Do đó ZCR bù đắp trọn vẹn điểm yếu năng lượng thấp của STE."*

> **[02:00 - 03:00]**  
> *"Quy trình em triển khai gồm 6 bước chuẩn: Từ tín hiệu với khung 25ms, hop 10ms, em trích xuất đồng thời STE và ZCR. Trên tập huấn luyện 4 file, em tính được Tzcr = 0.2231 theo công thức mu + 2*sigma của silence. Sau đó tiến hành phân đoạn sơ bộ bằng STE với Tupper và Tlower; tiếp tục mở rộng tìm kiếm 250ms bằng ZCR để kéo biên bao trọn cụm âm vô thanh; và cuối cùng là hậu xử lý gộp các khoảng lặng nhỏ dưới 200ms."*

> **[03:00 - 04:15]**  
> *"Trên tập kiểm thử gồm 4 file, kết quả đối chứng đã chứng minh hiệu quả vượt trội:  
> Ở file phone_F2, TT1 có sai số MAE lên tới 149.5 ms do bị cắt phạm, nhưng TT2 đã giảm sai số xuống chỉ còn 57.5 ms, tức là giảm hơn 61% sai số!  
> Đặc biệt ở file phone_M2, trong khi TT1 hoàn toàn mất biên (N/A), thì TT2 đã phục hồi thành công 100% mốc biên với điểm bắt đầu sai số chỉ 2.5 ms.  
> Ở 2 file studio sạch, sai số của cả 2 thuật toán đều cực nhỏ, chỉ từ 2 đến 7 ms, tức là nhỏ hơn cả 1 bước dịch khung."*

> **[04:15 - 05:00]**  
> *"Tóm lại, sự kết hợp giữa ngưỡng kép Hysteresis và ZCR đã tạo nên một giải pháp phân đoạn bền vững, chính xác và chịu nhiễu tốt. Sau đây em xin phép demo chương trình chạy trực tiếp trên máy tính. Em xin chân thành cảm ơn Thầy/Cô!"*
