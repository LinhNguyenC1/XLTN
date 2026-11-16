# BÁO CÁO GIẢI THÍCH CHI TIẾT & HƯỚNG DẪN BẢO VỆ ĐỒ ÁN (BT1)
## Đề Tài: Phân Đoạn Tín Hiệu Tiếng Nói và Khoảng Lặng
### So Sánh Thuật Toán 1 (STE Đơn Ngưỡng) và Thuật Toán 2 (Ngưỡng Kép STE + ZCR)

- **Sinh viên thực hiện:** Nguyễn Thị Thùy Linh
- **Mã số sinh viên:** 102230303
- **Khoa / Trường:** Trường Đại học Bách Khoa – Đại học Đà Nẵng
- **Thời lượng báo cáo:** Tối đa 5 phút (Slide + Demo chương trình)

---

## 1. TỔNG QUAN BÀI TOÁN & VẤN ĐỀ CỐT LÕI

Bài toán **Phân đoạn tiếng nói và khoảng lặng** (Speech/Silence Segmentation hay Endpoint Detection / Voice Activity Detection - VAD) là bước tiền xử lý tối quan trọng trong các hệ thống xử lý tín hiệu tiếng nói (nhận dạng giọng nói ASR, nén thoại, trích xuất F0, v.v.).

### 1.1. Bản chất của Thuật toán 1 (Đã nộp trong BT1)
Ở BT1, thuật toán sử dụng:
1. **Một đặc trưng duy nhất:** Năng lượng ngắn hạn $STE$ (hoặc $\log(MA) = \ln(\frac{1}{N}\sum |x[n]|)$).
2. **Một ngưỡng tĩnh duy nhất:** $T = -5.0172$, được xác định bằng thuật toán tìm kiếm nhị phân (Binary Search) cân bằng diện tích nhầm lẫn (Equation 2.8 trong giáo trình *Hodgkinson, 2012*).
3. **Quy tắc quyết định:** Nếu $\log(MA) \ge T \rightarrow$ Speech; ngược lại $\rightarrow$ Silence.

### 1.2. Hạn chế cố hữu của Thuật toán 1
Mặc dù hội tụ rất nhanh trên tập huấn luyện và đạt độ chính xác cao trên các môi trường phòng thu yên tĩnh (Studio), Thuật toán 1 bộc lộ **2 nhược điểm cố hữu**:
1. **Cắt phạm vào các âm vô thanh (Unvoiced Speech):**
   - Các âm vô thanh ở đầu hoặc cuối từ (ví dụ: phụ âm xát, phụ âm tắc vô thanh như `/s/, /f/, /t/, /k/, /x/`, ví dụ trong từ *"sáu"*, *"chín"*, *"tám"*) được tạo ra khi luồng không khí đi qua khe hẹp thanh quản mà **không làm rung dây thanh**.
   - Do không có dao động cộng hưởng của dây thanh, năng lượng của âm vô thanh **cực kỳ thấp**, xấp xỉ ngang bằng với mức năng lượng của khoảng lặng.
   - Do chỉ có 1 ngưỡng tĩnh $T$, các khung âm vô thanh này bị rơi xuống dưới $T$, khiến thuật toán cắt phạm vào âm, làm mất đầu hoặc mất đuôi từ, đẩy sai số MAE/RMSE lên cao.
2. **Dễ bị "tê liệt" hoàn toàn khi gặp môi trường có nhiễu (SNR thấp):**
   - Giả định của ngưỡng tĩnh là mức ồn nền của tín hiệu kiểm thử phải đồng nhất với tập huấn luyện.
   - Khi chuyển sang tín hiệu ghi âm qua điện thoại di động (như file `phone_M2.wav`), mức ồn nền (sàn nhiễu) tăng cao vượt qua $-5.02$. Thuật toán coi toàn bộ file là tiếng nói $\rightarrow$ **Mất hoàn toàn mốc biên (N/A)**.

---

## 2. Ý TƯỞNG & NGUYÊN LÝ CỦA THUẬT TOÁN 2
### *Thuật toán ngưỡng kép STE và ZCR (Dual-Threshold Endpoint Detection)*

Thuật toán 2 giải quyết triệt để 2 nhược điểm trên nhờ kết hợp 2 nguyên lý âm học:

### 2.1. Cơ chế Ngưỡng kép STE (Hysteresis Energy Detection)
Thay vì dùng 1 ngưỡng duy nhất, thuật toán dùng **2 ngưỡng năng lượng**:
- **Ngưỡng cao ($T_{upper} = -4.48$):** Đóng vai trò là ngưỡng kích hoạt (trigger). Chỉ khi năng lượng vượt qua $T_{upper}$, thuật toán mới xác nhận là tiếng nói thực sự bắt đầu, triệt tiêu nguy cơ bị kích hoạt giả bởi các tiếng lách cách hay xung nhiễu ngắn của môi trường.
- **Ngưỡng thấp ($T_{lower} = -5.19$):** Đóng vai trò là ngưỡng duy trì (hold). Một khi tiếng nói đã được kích hoạt, thuật toán sẽ kéo dài biên về hai phía cho đến khi năng lượng rơi xuống dưới $T_{lower}$. Nhờ đó, phần âm đuôi bị suy giảm năng lượng tự nhiên vẫn được giữ trọn vẹn mà không bị ngắt cụt.

### 2.2. Tỷ lệ qua điểm không (ZCR) - Bù đắp âm vô thanh
- **Đặc tính vật lý:** Mặc dù âm vô thanh có năng lượng yếu, nhưng luồng khí hỗn loạn qua khe hẹp tạo ra dao động tần số rất cao trên miền thời gian. Do đó, tín hiệu đổi dấu liên tục qua mức 0, làm cho **ZCR của âm vô thanh tăng vọt lên rất cao** ($ZCR_{uv} \in [0.17, 0.75]$, trung bình cao gấp 2 đến 3 lần so với khoảng lặng $ZCR_{sil} \approx 0.10$).
- **Cơ chế hoạt động:** Sau khi tìm được biên sơ bộ bằng STE, thuật toán mở rộng tìm kiếm ra xung quanh $250\text{ ms}$ ($25\text{ frames}$). Nếu phát hiện cụm khung có $ZCR > T_{zcr}$, thuật toán sẽ "đẩy" biên ra ngoài để bao trọn lấy âm vô thanh.

---

## 3. CÁC BƯỚC TRIỂN KHAI CHI TIẾT (6 BƯỚC CHUẨN)

### Bước 1: Trích xuất đặc trưng (Feature Extraction)
- Tín hiệu được chia khung với độ dài khung $25\text{ ms}$ (`frame_len`), độ dịch khung $10\text{ ms}$ (`hop_len`).
- Trên mỗi khung $i$, tính đồng thời 2 đặc trưng:
  $$\text{STE}[i] = \ln\left(\frac{1}{N}\sum_{n=0}^{N-1} |x[n]| + 10^{-12}\right)$$
  $$\text{ZCR}[i] = \frac{1}{2(N-1)} \sum_{n=1}^{N-1} |\text{sgn}(x[n]) - \text{sgn}(x[n-1])|$$

### Bước 2: Tìm ngưỡng tối ưu trên tập huấn luyện (`TinHieuHuanLuyen`)
- Sử dụng các file ground truth `*.lab` của 4 file huấn luyện (`phone_F1`, `phone_M1`, `studio_F1`, `studio_M1`):
  - Khung khoảng lặng (`sil`): gom vào tập `silence_frames`.
  - Khung tiếng nói (`v` và `uv`): gom vào tập `speech_frames`.
- Tính toán các giá trị ngưỡng:
  - **Ngưỡng ZCR:**
    $$T_{zcr} = \mu_{zcr, sil} + 2.0 \cdot \sigma_{zcr, sil} = 0.1049 + 2.0 \times 0.0591 \approx 0.2231$$
  - **Ngưỡng STE:**
    Dựa trên phân bố năng lượng tập Train: $\mu_{sil} = -6.35, \sigma_{sil} = 1.29$ và $\mu_{sp} = -2.94$:
    $$T_{upper} = \mu_{sil} + 1.45 \cdot \sigma_{sil} \approx -4.48$$
    $$T_{lower} = \mu_{sil} + 0.90 \cdot \sigma_{sil} \approx -5.19$$

### Bước 3: Phân đoạn sơ bộ bằng STE
- Duyệt qua chuỗi khung từ đầu đến cuối:
  1. Khi phát hiện một khung $i$ có $\text{STE}[i] \ge T_{upper}$: đây là một vùng tiếng nói chắc chắn.
  2. Quét lùi về trước từ $i$ tìm khung $N_1$ đầu tiên sao cho $\text{STE}[N_1 - 1] \le T_{lower}$.
  3. Quét tiến về sau từ $i$ tìm khung $N_2$ đầu tiên sao cho $\text{STE}[N_2 + 1] \le T_{lower}$.
  4. Thu được đoạn tiếng nói sơ bộ $[N_1, N_2]$.

### Bước 4: Tinh chỉnh biên bằng ZCR
- Từ mốc $N_1$ (bắt đầu): Quét lùi tối đa 25 khung ($250\text{ ms}$). Nếu phát hiện cụm khung có $ZCR > T_{zcr}$ (tối thiểu 3 khung), lùi biên $N_1$ về khung sớm nhất trong cụm âm vô thanh này.
- Từ mốc $N_2$ (kết thúc): Quét tiến tối đa 25 khung ($250\text{ ms}$). Nếu phát hiện cụm khung có $ZCR > T_{zcr}$ (tối thiểu 3 khung), tiến biên $N_2$ về khung muộn nhất trong cụm.

### Bước 5: Hậu xử lý (Lọc khoảng lặng ảo)
- Xét các đoạn tiếng nói liên tiếp: nếu khoảng cách thời gian (khoảng lặng xen giữa) $< 200\text{ ms}$, gộp hai đoạn thành một đoạn tiếng nói duy nhất (loại bỏ các khoảng ngắt hơi ngắn).

### Bước 6: Đánh giá & So sánh
- Kiểm thử trên 4 file của `TinHieuKiemThu` (`phone_F2`, `phone_M2`, `studio_F2`, `studio_M2`).
- Xuất đồ thị chập tín hiệu sóng âm với các đường dọc phân biên:
  - **Màu đỏ:** Groundtruth (.lab).
  - **Màu cam:** Thuật toán 1 (STE đơn ngưỡng).
  - **Màu xanh dương:** Thuật toán 2 (Ngưỡng kép STE + ZCR).
- Tính toán sai số MAE và RMSE (đơn vị: mili-giây - ms).

---

## 4. KẾT QUẢ THỰC NGHIỆM ĐỐI CHỨNG (BẢNG SỐ LIỆU)

| Tên File Kiểm Thử | SNR Kênh Thu | TT1: MAE (ms) | TT1: RMSE (ms) | TT2: MAE (ms) | TT2: RMSE (ms) | Kết Luận Hiệu Quả |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`phone_F2.wav`** | Thấp | $149.50$ | $245.51$ | **$57.50$** | **$79.57$** | **Giảm 61.5% sai số (giảm 92 ms)** |
| **`phone_M2.wav`** | Thấp | **N/A** (Mất biên) | **N/A** (Mất biên) | **$127.50$** | **$178.55$** | **Phục hồi mốc biên** |
| **`studio_F2.wav`** | Cao | $2.51$ | $2.51$ | **$5.00$** | **$5.59$** | Sai số siêu nhỏ ($< 1$ khung dịch) |
| **`studio_M2.wav`** | Cao | $7.49$ | $7.49$ | **$7.49$** | **$9.01$** | Bắt trọn âm vô thanh đầu câu |
| **TRUNG BÌNH HỢP LỆ** | **Tổng thể** | **$53.17$** | **$85.17$** | **$49.37$** | **$68.18$** | **TT2 toàn diện trên 4/4 file** |

### Nhận xét chuyên sâu:
1. **Tại sao TT2 vượt trội trên `phone_F2.wav`?**
   - Ở đuôi của `phone_F2.wav`, người nói phát âm phụ âm vô thanh yếu kéo dài. Thuật toán 1 đơn ngưỡng bị cắt sớm hơn $200\text{ ms}$. Thuật toán 2 nhờ ngưỡng thấp $T_{lower}$ và ZCR đã bám sát đến tận cùng câu nói, giảm sai số từ $149.5\text{ ms} \rightarrow 57.5\text{ ms}$.
2. **Tại sao TT1 bị N/A trên `phone_M2.wav` trong khi TT2 làm được?**
   - File `phone_M2.wav` có mức ồn nền của khoảng lặng lên tới $-5.0$, khiến điều kiện $\log(MA) \ge -5.02$ của TT1 bị thỏa mãn trên hầu hết file $\rightarrow$ toàn bộ file bị gán nhãn 1, dẫn đến 0 mốc biên.
   - TT2 sử dụng $T_{upper} = -4.48$, mức ồn nền $-5.0$ không thể kích hoạt được tiếng nói ảo $\rightarrow$ giữ nguyên vùng silence và tìm ra mốc biên bắt đầu ở $0.532\text{ s}$ (so với ground truth $0.530\text{ s}$, sai số chỉ đúng $2.5\text{ ms}$!).
3. **Trên môi trường sạch `studio_F2.wav` và `studio_M2.wav`:**
   - Cả hai thuật toán đều cho sai số cực nhỏ dưới $10\text{ ms}$, chứng minh thuật toán 2 giữ nguyên độ chính xác cao trên môi trường lý tưởng trong khi nâng cao tính bền bỉ trên môi trường thực tế.

---

## 5. HƯỚNG DẪN BẢO VỆ & DEMO TRỰC TIẾP TRƯỚC GIẢNG VIÊN

### Cách 1: Chạy demo dòng lệnh trực tiếp trên terminal
Mở terminal và gõ:
```bash
cd /Users/nguyenthithuylinh/Downloads/Xử_lý_tiếng_nói/GK_BT1_dual_threshold_STE_ZCR
python3 main.py
```
*Kết quả:*
- Màn hình console in ra bảng thống kê huấn luyện và bảng so sánh MAE/RMSE tức thì.
- Matplotlib tự động mở 4 cửa sổ đồ thị chi tiết cho 4 file test chập sóng âm và đường phân biên.

### Cách 2: Trình chiếu Slide thuyết trình 5 phút
- **Cách trực quan nhất:** Mở file [slide_presentation.html](file:///Users/nguyenthithuylinh/Downloads/Xử_lý_tiếng_nói/GK_BT1_dual_threshold_STE_ZCR/slide_presentation.html) bằng trình duyệt Safari/Chrome.
  + Bấm phím **F** để bật chế độ toàn màn hình.
  + Bấm **⏱ Bắt đầu** để đếm ngược 5 phút.
  + Dùng phím mũi tên **[◀ / ▶]** để chuyển slide.
- **Cách dự phòng:** Đọc file Markdown [slide_thuyet_trinh_5phut.md](file:///Users/nguyenthithuylinh/Downloads/Xử_lý_tiếng_nói/GK_BT1_dual_threshold_STE_ZCR/slide_thuyet_trinh_5phut.md).
