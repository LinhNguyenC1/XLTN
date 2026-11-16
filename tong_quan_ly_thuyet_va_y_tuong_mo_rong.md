# TỔNG QUAN KIẾN THỨC NỀN TẢNG, ĐIỂM THÚ VỊ & HƯỚNG PHÁT TRIỂN
## Đề Tài: Phân Đoạn Tiếng Nói & Khoảng Lặng (Speech/Silence Segmentation & Endpoint Detection)
**Thư mục dự án:** `GK_BT1_binary_search_v2`  
**Sinh viên:** Nguyễn Thị Thùy Linh  
**MSSV:** 102230303  

---

## MỤC LỤC
1. [Bản đồ Kiến thức Nền tảng & Lý thuyết Cốt lõi](#1-bản-đồ-kiến-thức-nền-tảng--lý-thuyết-cốt-lõi)
   - 1.1. Bản chất Vật lý & Âm học của Tiếng nói (Speech Acoustics)
   - 1.2. Phân tích Tín hiệu Miền thời gian Ngắn hạn (Short-Time Analysis)
   - 1.3. Cơ sở Toán học & Tối ưu hóa trong Phân ngưỡng
   - 1.4. Phương pháp Đánh giá Định lượng (Evaluation Metrics)
2. [Những Điểm "Đắt Giá" & Thú Vị của Bài Tập Này](#2-những-điểm-đắt-giá--thú-vị-của-bài-tập-này)
3. [Làm Gì Để Bài Tập Này Trở Nên Thú Vị & "Đẳng Cấp" Hơn?](#3-làm-gì-để-bài-tập-này-trở-nên-thú-vị--đẳng-cấp-hơn)
   - Hướng 1: Thích ứng Nhiễu Động theo Thời gian thực (Adaptive Noise Tracker)
   - Hướng 2: Bổ sung Đặc trưng Miền Tần số (Spectral Entropy & Flatness)
   - Hướng 3: Xây dựng Dashboard Web Demo Tương tác (Streamlit / Gradio)
   - Hướng 4: Đối chiếu với Công nghệ Công nghiệp (WebRTC VAD & Silero VAD)
   - Hướng 5: Gamification - Điều khiển Ứng dụng / Game bằng Giọng nói

---

## 1. BẢN ĐỒ KIẾN THỨC NỀN TẢNG & LÝ THUYẾT CỐT LÕI

Để làm chủ trọn vẹn bài tập này (cả khi bảo vệ đồ án lẫn áp dụng vào các hệ thống AI sau này), bạn cần nắm vững 4 mảng kiến thức trụ cột sau:

```
                            BẢN ĐỒ KIẾN THỨC BÀI TẬP 1
  ┌─────────────────────────┬─────────────────────────┬─────────────────────────┐
  │   1. ÂM HỌC TIẾNG NÓI   │  2. PHÂN TÍCH NGẮN HẠN  │   3. TOÁN TỐI ƯU HÓA    │
  ├─────────────────────────┼─────────────────────────┼─────────────────────────┤
  │ • Voiced vs. Unvoiced   │ • Chia khung & Dịch bƣớc│ • Binary Search cân bằng│
  │ • Cơ chế rung dây thanh │ • Phân bố năng lượng STE│   diện tích (Hodgkinson)│
  │ • Sàn nhiễu & Tỉ số SNR │ • Tần số qua zero (ZCR) │ • Ngưỡng kép Hysteresis │
  └─────────────────────────┴─────────────────────────┴─────────────────────────┘
                                       │
                                       ▼
                      ┌─────────────────────────────────┐
                      │  4. ĐÁNH GIÁ ĐỊNH LƯỢNG MÔ HÌNH │
                      │  • Sai số mốc biên: MAE & RMSE  │
                      │  • Over-segment vs Under-segment│
                      └─────────────────────────────────┘
```

---

### 1.1. Bản chất Vật lý & Âm học của Tiếng nói (Speech Acoustics)

Trong âm học và xử lý tiếng nói, tín hiệu âm thanh thu được không phải là một sóng đồng nhất, mà là sự pha trộn của 3 trạng thái vật lý hoàn toàn khác biệt:

| Trạng thái | Nguồn gốc cơ học trong vòm họng | Đặc tính miền thời gian | Năng lượng (STE) | Tỷ lệ qua 0 (ZCR) | Ví dụ điển hình |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **Âm hữu thanh** *(Voiced)* | Luồng khí làm **rung cơ khép dây thanh** (Vocal Cords). Tạo dao động tuần hoàn với tần số cơ bản $F_0$. | Sóng có biên độ lớn, dạng tuần hoàn giả (quasi-periodic). | **Rất cao** ($\mu \approx -2.9$) | **Rất thấp** ($0.03 - 0.12$) | Các nguyên âm: */a/, /e/, /u/, /i/*, phụ âm vang */m/, /n/, /l/*. |
| **Âm vô thanh** *(Unvoiced)* | Luồng khí đi qua khe hẹp thanh quản mà **không làm rung dây thanh**, tạo nhiễu ma sát hỗn loạn (Turbulent noise). | Dao động ngẫu nhiên, chu kỳ siêu ngắn, tần số cao. | **Rất thấp** (xấp xỉ khoảng lặng) | **Rất cao** ($0.18 - 0.75$) | Các phụ âm xát, tắc vô thanh: */s/, /f/, /t/, /k/, /x/* (*sáu, chín, tám*). |
| **Khoảng lặng** *(Silence / Noise)* | Không có phát âm. Micro chỉ thu tiếng ồn môi trường, tiếng quạt gió hoặc tạp âm điện từ. | Biên độ nhỏ, biến động quanh 0 ngẫu nhiên. | **Cực thấp** ở phòng thu; **Trung bình** ở điện thoại. | **Trung bình - thấp** ($0.08 - 0.14$) | Khoảng nghỉ thở giữa các từ hoặc đoạn đầu/cuối file. |

> **Điểm mấu chốt cần ghi nhớ:**
> - Nếu chỉ dùng năng lượng, bạn **chắc chắn sẽ nhầm âm vô thanh thành khoảng lặng**.
> - Nếu chỉ dùng ZCR, bạn **chắc chắn sẽ nhầm tiếng ồn trắng (white noise) thành âm vô thanh**.
> - **Sự kết hợp giữa Năng lượng + ZCR** là chiếc chìa khóa hóa giải mâu thuẫn này.

---

### 1.2. Phân tích Tín hiệu Miền thời gian Ngắn hạn (Short-Time Analysis)

#### a. Tại sao phải chia khung (Framing)?
Tín hiệu tiếng nói là **tín hiệu bất dừng (non-stationary)**: tần số, biên độ và nội dung thay đổi liên tục theo thời gian. Ta không thể áp dụng các biến đổi toàn cục (như Fourier toàn phần) cho cả câu nói dài $5\text{ s}$.  
Tuy nhiên, bộ máy phát âm của con người (dây thanh, cơ môi, lưỡi) không thể thay đổi vị trí tức thì. Trong một khoảng thời gian cực ngắn ($10 - 30\text{ ms}$), cấu hình vòm họng được coi là đứng yên. Do đó, tín hiệu được coi là **chuẩn dừng (quasi-stationary)** trên từng khung thời gian này.

- **Độ dài khung (`frame_len`):** Chọn $25\text{ ms}$.
  $$\text{Số mẫu mỗi khung } N = \frac{f_s \times 25}{1000}$$
  *(Với $f_s = 16000\text{ Hz} \rightarrow N = 400\text{ mẫu}$)*.
- **Độ dịch khung (`hop_len` / `step`):** Chọn $10\text{ ms}$ (độ chồng lấp $60\%$) để đảm bảo các đặc trưng thay đổi mượt mà, không bị gián đoạn hay mất thông tin ở biên khung.

#### b. Năng lượng ngắn hạn (Short-Time Energy - STE / Log MA)
Đại diện cho cường độ âm thanh trong từng khung thời gian:
$$\text{MA}[i] = \frac{1}{N} \sum_{n=0}^{N-1} |x_i[n]| \quad \Longrightarrow \quad \log(\text{MA})[i] = \ln(\text{MA}[i] + \epsilon)$$
- $\epsilon = 10^{-12}$: Đại lượng chống tràn số âm vô cực khi khung chứa toàn số 0 tuyệt đối.
- Lấy $\ln$ giúp co thang đo biên độ lớn và kéo giãn sự khác biệt ở vùng biên độ nhỏ, tạo phân bố dạng hai đỉnh (Bimodal distribution) rõ nét.

#### c. Tỷ lệ qua điểm không (Zero-Crossing Rate - ZCR)
Đo số lần tín hiệu đổi dấu giữa hai mẫu liên tiếp trong một khung:
$$\text{ZCR}[i] = \frac{1}{2(N-1)} \sum_{n=1}^{N-1} |\text{sgn}(x[n]) - \text{sgn}(x[n-1])|$$
- Giá trị chuẩn hóa nằm trong đoạn $[0, 1]$.
- ZCR tỉ lệ thuận với tần số chiếm ưu thế của tín hiệu: tín hiệu có nhiều thành phần tần số cao sẽ dao động qua mốc $0$ liên tục $\rightarrow$ ZCR cao.

---

### 1.3. Cơ sở Toán học & Tối ưu hóa trong Phân ngưỡng

#### a. Thuật toán Tìm kiếm Nhị phân cân bằng diện tích (Hodgkinson, 2012)
Gọi $f$ là phân bố của khoảng lặng (Silence) và $g$ là phân bố của tiếng nói (Speech). Giữa $f$ và $g$ có một vùng chồng lấn (Area of Confusion).

```
   Mật độ
     ▲
     │       Phân bố Silence f(x)             Phân bố Speech g(x)
     │            ╭─────╮                            ╭─────╮
     │           │       │                          │       │
     │          │         │    Vùng chồng lấn      │         │
     │         │           │         ▼            │           │
     │        │             ╲       ╱╲           ╱             │
     │       │               ╲     ╱  ╲         ╱               │
     │      │                 ╲   ╱    ╲       ╱                 │
     │    ──┴───────────────────╳───────╳─────┴───────────────────► log(MA)
                               ▲    ▲    ▲
                            T_min   T   T_max
```

Mục tiêu là tìm ngưỡng $T$ sao cho:
$$\text{Area}_f(T) = \text{Area}_g(T)$$
$$\frac{1}{N_f} \sum_{i: f_i > T} (f_i - T) = \frac{1}{N_g} \sum_{j: g_j < T} (T - g_j)$$

- **Vế trái ($\text{Area}_f$):** Tổng phần năng lượng của Silence bị vượt quá $T$ (Nhận nhầm khoảng lặng thành tiếng nói - False Alarm / Type I Error).
- **Vế phải ($\text{Area}_g$):** Tổng phần năng lượng của Speech bị tụt dưới $T$ (Bỏ sót tiếng nói - Missed Detection / Type II Error).
- **Tính chất hội tụ:** Hàm hiệu $\Delta(T) = \text{Area}_f(T) - \text{Area}_g(T)$ là một hàm **nghịch biến đơn điệu nghiêm ngặt** theo $T$:
  - Khi $T$ tăng: $\text{Area}_f(T)$ giảm dần về 0, $\text{Area}_g(T)$ tăng dần $\rightarrow \Delta(T)$ giảm.
  - Nhờ tính đơn điệu này, **Binary Search được chứng minh bằng toán học là luôn luôn hội tụ tới nghiệm duy nhất** sau khoảng $15 - 20$ vòng lặp!

#### b. Nguyên lý Hysteresis (Ngưỡng kép)
Tại sao 1 ngưỡng tĩnh luôn gặp thất bại ở vùng biên âm?
- Nếu ngưỡng $T$ quá cao: Không bị nhiễu kích hoạt nhầm, nhưng sẽ ngắt cụt đuôi âm thanh khi năng lượng suy yếu.
- Nếu ngưỡng $T$ quá thấp: Bắt trọn âm đuôi, nhưng sẽ bị kích hoạt giả bởi tạp âm môi trường.

**Giải pháp Hysteresis (Mượn từ Trigger Schmitt trong điện tử):**
- **$T_{upper}$ (Ngưỡng kích hoạt):** Đặt cao. Tín hiệu phải đủ mạnh mới được coi là bắt đầu nói.
- **$T_{lower}$ (Ngưỡng duy trì):** Đặt thấp. Một khi đã bắt đầu nói, trạng thái nói được duy trì liên tục cho đến khi năng lượng rơi xuống dưới $T_{lower}$.

---

### 1.4. Phương pháp Đánh giá Định lượng (Evaluation Metrics)

Thay vì tính Accuracy đơn thuần (vốn bị thiên lệch do khoảng lặng chiếm đa số thời lượng), bài toán VAD đo lường trực tiếp độ lệch giữa các mốc thời gian biên chuẩn (Ground Truth từ file `.lab`) và biên dự đoán (Predicted Boundaries):

1. **Sai số Tuyệt đối Trung bình (MAE - Mean Absolute Error):**
   $$\text{MAE} = \frac{1}{K} \sum_{k=1}^K |t_{\text{pred}, k} - t_{\text{ref}, k}| \quad (\text{ms})$$
   *Đo lường mức độ lệch trung bình giữa hai đường biên.*

2. **Căn bậc hai Sai số Toàn phương Trung bình (RMSE - Root Mean Squared Error):**
   $$\text{RMSE} = \sqrt{\frac{1}{K} \sum_{k=1}^K (t_{\text{pred}, k} - t_{\text{ref}, k})^2} \quad (\text{ms})$$
   *Phạt rất nặng các trường hợp bị lệch quá xa hoặc bắt nhầm đoạn nhiễu lớn.*

---

## 2. NHỮNG ĐIỂM "ĐẮT GIÁ" & THÚ VỊ CỦA BÀI TẬP NÀY

Nếu nhìn bề ngoài, đây có vẻ chỉ là một bài tập phân ngưỡng đơn giản. Nhưng đi sâu vào bản chất kỹ thuật, bài tập này chứa đựng những bài học thực chiến vô cùng quý giá:

### 🌟 Điểm 1: Bài học kinh điển về "Mô hình phòng thí nghiệm vs Thực tế trần trụi"
- **Trên môi trường Studio (`studio_M2`, `studio_F2`):** Thuật toán 1 (Binary Search đơn ngưỡng) chạy cực kỳ mỹ mãn, cho sai số siêu nhỏ chỉ **$2.51\text{ ms} - 7.49\text{ ms}$** (nhỏ hơn cả 1 bước nhảy khung $10\text{ ms}$!). Bạn rất dễ ngộ nhận rằng thuật toán này đã hoàn hảo.
- **Nhưng khi chuyển sang môi trường thực tế (`phone_M2`):** Âm thanh thu qua điện thoại di động có sàn nhiễu dâng cao. Ngưỡng tĩnh $-5.02$ bị chìm nghỉm dưới mức nhiễu $\rightarrow$ Thuật toán tê liệt $100\%$ (Mất toàn bộ biên, kết quả N/A).
- 👉 *Đây là bài học sâu sắc nhất trong kỹ thuật: Một mô hình thanh lịch về mặt toán học nhưng dựa trên giả định môi trường tĩnh sẽ lập tức sụp đổ khi bước ra thế giới thực.*

### 🌟 Điểm 2: Sự bù trừ âm dương tuyệt mỹ giữa STE và ZCR
- **STE** đại diện cho "sức mạnh" (năng lượng cơ học của luồng hơi).
- **ZCR** đại diện cho "tốc độ" (tần số biến thiên của luồng khí hỗn loạn).
- Hai đặc trưng này bổ khuyết cho nhau như hình với bóng: nơi nào STE "mù" (các phụ âm vô thanh $/s/, /t/, /k/$ năng lượng yếu ớt), ZCR lập tức tỏa sáng rực rỡ để cứu lấy âm đầu và âm đuôi câu nói.

### 🌟 Điểm 3: Vẻ đẹp của thuật toán Hysteresis & State Machine
- Thuật toán 2 thực chất là một **máy trạng thái hữu hạn (Finite State Machine - FSM)** với hiện tượng trễ:
  ```
  [Trạng thái SILENCE] ──(STE > Tupper)──► [Trạng thái SPEECH]
           ▲                                       │
           └────────────(STE < Tlower)─────────────┘
  ```
- Ý tưởng này không chỉ giới hạn trong xử lý tiếng nói, mà xuất hiện khắp nơi trong công nghệ: từ bộ điều nhiệt máy lạnh (bật lúc $28^\circ\text{C}$, chỉ tắt khi hạ xuống $24^\circ\text{C}$ để tránh bật/tắt liên tục làm cháy máy nén), thuật toán dò biên Canny trong xử lý ảnh, cho tới các cảm biến tự hành của xe Tesla.

---

## 3. LÀM GÌ ĐỂ BÀI TẬP NÀY TRỞ NÊN THÚ VỊ & "ĐẲNG CẤP" HƠN?

Để nâng tầm bài tập này từ một bài nộp điểm 10 thông thường thành một **project nghiên cứu ấn tượng để đưa vào CV hoặc phỏng vấn tuyển dụng**, bạn có thể phát triển theo 5 hướng cụ thể sau:

---

### 💡 Hướng 1: Thích ứng Nhiễu Động theo Thời gian thực (Adaptive Noise Tracker)
*Khắc phục triệt để nhược điểm của việc học ngưỡng cố định trên tập Train.*

- **Vấn đề hiện tại:** Ngưỡng $T_{upper}, T_{lower}, T_{zcr}$ vẫn đang được "học chết" từ tập dữ liệu huấn luyện. Nếu đem sang một file có tiếng còi xe hoặc tiếng mưa rào, ngưỡng này lại có thể sai lệch.
- **Ý tưởng cải tiến:**
  1. Giả định rằng **$100 - 200\text{ ms}$ đầu tiên** của bất kỳ file ghi âm nào luôn là khoảng lặng (tiếng ồn nền).
  2. Đo đạc $\mu_{\text{noise}}$ và $\sigma_{\text{noise}}$ ngay trên các khung đầu tiên này.
  3. Áp dụng bộ lọc trung bình trượt có trọng số (Recursive Average) để cập nhật động ngưỡng theo thời gian:
     $$\mu_{\text{noise}}[t] = \alpha \cdot \mu_{\text{noise}}[t-1] + (1 - \alpha) \cdot \text{STE}[t] \quad (\text{chỉ cập nhật khi khung } t \text{ là silence})$$
  4. Ngưỡng tự động "nổi lên" khi môi trường ồn và "hạ xuống" khi môi trường yên tĩnh.

---

### 💡 Hướng 2: Bổ sung Đặc trưng Miền Tần số (Spectral Features)
*Nâng cấp từ VAD miền thời gian lên VAD miền phổ tần.*

- **Hạn chế của ZCR:** ZCR rất dễ bị đánh lừa bởi tiếng ồn trắng băng rộng (White noise) hoặc tiếng quạt gió rít tần số cao.
- **Vũ khí mới:**
  1. **Spectral Entropy (Độ hỗn loạn phổ):**
     - Khoảng lặng và tiếng ồn trắng có phân bố năng lượng đều trên toàn dải tần $\rightarrow$ Entropy cực cao.
     - Tiếng nói (đặc biệt là âm hữu thanh) có năng lượng tập trung tại các tần số Formant ($F_1, F_2, F_3$) $\rightarrow$ Phổ có đỉnh nhọn rõ rệt $\rightarrow$ Entropy giảm mạnh.
  2. **Spectral Centroid (Trọng tâm phổ):** Phân biệt cực kỳ hiệu quả giữa âm vô thanh tần số cao và tiếng ồn nền trầm tần số thấp.

---

### 💡 Hướng 3: Xây dựng Dashboard Web Demo Tương tác (Streamlit / Gradio)
*Biến mã nguồn thành sản phẩm trực quan sinh động có thể tương tác bằng chuột.*

Thay vì chỉ chạy file python và xem ảnh tĩnh:
- Dùng thư viện **Streamlit** hoặc **Gradio** tạo một ứng dụng Web cục bộ chỉ với $\approx 50$ dòng code:
  - Cho phép người dùng kéo-thả file âm thanh bất kỳ.
  - Có các thanh trượt (Sliders) điều chỉnh $T_{upper}, T_{lower}, T_{zcr}$ trực tiếp.
  - Hiển thị đồ thị dạng sóng tương tác (Plotly / Wavesurfer.js) với các vạch cắt biên dịch chuyển real-time theo thanh trượt.
  - Có nút **Play Audio** phát đoạn tiếng nói đã cắt để nghe thử xem có bị mất âm đầu/đuôi hay không!

---

### 💡 Hướng 4: Đối chiếu với Công nghệ Công nghiệp (Industry Benchmarking)
*Đưa bài tập lên tầm báo cáo khoa học bằng cách so sánh với các thư viện chuẩn quốc tế.*

Hãy đưa vào bảng so sánh đối chứng thêm 2 đối thủ "hàng hiệu":
1. **Google WebRTC VAD (`webrtcvad`):** Chuẩn công nghiệp do Google phát triển bằng C++, dùng mô hình GMM phân loại theo 6 băng tần con, cực nhanh và nhẹ.
2. **Silero VAD (Deep Learning):** Mô hình Deep Learning hiện đại dựa trên kiến trúc nén ONNX siêu nhẹ, có khả năng phân đoạn chuẩn xác ngay cả trong môi trường karaoke hoặc quán cafe ồn ào.
- 👉 *Khi bạn đặt thuật toán STE+ZCR của mình lên bàn cân cùng Google WebRTC VAD, giảng viên sẽ cực kỳ ấn tượng vì bạn có tư duy đối chuẩn công nghệ thực tế!*

---

### 💡 Hướng 5: Gamification - Điều khiển Ứng dụng / Game bằng Giọng nói
*Ứng dụng VAD làm bộ kích hoạt (Voice Trigger) cho một minigame vui nhộn.*

- Tạo một trò chơi mini (ví dụ: điều khiển chú khủng long nhảy qua xương rồng giống Google Chrome khi mất mạng):
  - **Silence:** Khủng long chạy bình thường.
  - **Voiced Sound (Hét "Aaaa"):** Khủng long nhảy cao.
  - **Unvoiced Sound (Xì "Ssss" hoặc tặc lưỡi "Ttt"):** Khủng long cúi rạp người xuống tránh chướng ngại vật trên không.
- 👉 *Đây là minh chứng trực quan sinh động nhất cho việc phân biệt thành công giữa Voiced và Unvoiced dựa trên STE và ZCR!*

---

## 4. TỔNG KẾT BỎ TÚI KHI TRẢ LỜI VẤN ĐÁP GIẢNG VIÊN

| Câu hỏi giảng viên có thể hỏi | Câu trả lời mấu chốt (Ghi điểm tuyệt đối) |
| :--- | :--- |
| **"Tại sao thuật toán 1 lại bị lỗi N/A trên file `phone_M2`?"** | *"Dạ thưa Thầy/Cô, vì file phone có sàn nhiễu nền rất cao (khoảng $-5.0$), vượt qua ngưỡng tĩnh duy nhất $-5.02$ học từ tập Train. Hệ thống coi toàn bộ file là tiếng nói nên không tìm thấy bất kỳ điểm chuyển đổi trạng thái nào, dẫn đến mất biên hoàn toàn."* |
| **"Tại sao không tăng ngưỡng $T$ của thuật toán 1 lên cao hơn?"** | *"Dạ nếu tăng ngưỡng $T$ lên để tránh nhiễu, thuật toán 1 sẽ cắt cụt các âm vô thanh như /s/, /t/ vì năng lượng của chúng rất thấp, khiến sai số MAE ở các file khác tăng vọt."* |
| **"Bản chất vật lý nào giúp ZCR nhận diện được âm vô thanh?"** | *"Dạ âm vô thanh được sinh ra do luồng khí ma sát qua khe hẹp mà không rung dây thanh, tạo ra thành phần tần số rất cao trên miền thời gian. Do đó tín hiệu đổi dấu qua điểm 0 liên tục, làm giá trị ZCR vọt lên cao gấp 2 đến 3 lần so với khoảng lặng."* |
| **"Nguyên lý Hysteresis mang lại lợi ích gì trong phân đoạn?"** | *"Dạ cơ chế 2 ngưỡng giúp chống kích hoạt giả bởi các xung nhiễu ngắn nhờ $T_{upper}$, đồng thời giữ trọn vẹn phần đuôi âm thanh bị suy hao tự nhiên nhờ $T_{lower}$ trước khi rơi về trạng thái silence."* |

