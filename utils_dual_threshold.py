"""
utils_dual_threshold.py
-----------------------
Các hàm hỗ trợ phân đoạn tiếng nói bằng ngưỡng kép STE + ZCR.

Pipeline:
1. Đọc WAV và chuẩn hóa an toàn.
2. Đọc nhãn thời gian từ file .lab.ß
3. Trích xuất Short-Time Energy (STE) và Zero-Crossing Rate (ZCR).
4. Ước lượng T_lower, T_upper và T_zcr từ dữ liệu huấn luyện.
5. Phân đoạn sơ bộ bằng STE (hysteresis), tinh chỉnh biên bằng ZCR,
   gộp các đoạn cách nhau dưới 200 ms và loại bỏ đoạn quá ngắn.
6. Đánh giá sai số biên bằng ghép cặp một-một.

Lưu ý:
- STE dùng dạng năng lượng trung bình: sum(x**2) / frame_len.
- T_upper là heuristic khởi tạo; cần kiểm tra trên tập validation nếu có.
- Các hàm giữ tương thích với main_dual_threshold.py hiện tại.
"""

import warnings
import numpy as np
from scipy.io import wavfile
from scipy.optimize import linear_sum_assignment

warnings.filterwarnings("ignore", category=wavfile.WavFileWarning)


# Các nhãn thường được dùng để chỉ khoảng lặng / pause trong file .lab.
_SILENCE_LABELS = {
    "sil", "silence", "silent", "pau", "pause", "sp", "short_pause"
}


def load_wav(filepath):
    """
    Đọc WAV, lấy kênh đầu tiên nếu stereo và chuẩn hóa biên độ.

    Trả về:
        sr: sample rate (Hz)
        signal: ndarray float64, thường nằm trong [-1, 1].

    Chuyển sang float TRƯỚC khi lấy trị tuyệt đối để tránh overflow
    với mẫu int16 có giá trị -32768.
    """
    sr, data = wavfile.read(filepath)

    if data.ndim > 1:
        data = data[:, 0]

    signal = np.asarray(data, dtype=np.float64)

    if signal.size == 0:
        raise ValueError(f"File WAV rỗng: {filepath}")

    if np.issubdtype(data.dtype, np.integer):
        # Chuẩn hóa theo biên độ biểu diễn của kiểu dữ liệu gốc.
        info = np.iinfo(data.dtype)
        scale = float(max(abs(info.min), abs(info.max)))
        if scale > 0:
            signal /= scale
    else:
        # Với WAV float, không tự động co giãn nếu dữ liệu đã là float.
        # Chỉ chuẩn hóa nếu biên độ vượt quá 1.
        peak = float(np.max(np.abs(signal)))
        if peak > 1.0:
            signal /= peak

    return int(sr), signal


def _normalize_label(label):
    """Chuẩn hóa nhãn .lab thành 'sil' hoặc 'speech'."""
    label = label.strip().lower()
    return "sil" if label in _SILENCE_LABELS else "speech"


def read_lab_boundaries(lab_path):
    """
    Đọc file .lab.

    Mỗi dòng hợp lệ có dạng:
        start_time end_time label

    Trả về:
        boundaries: các mốc chuyển nhãn giữa các đoạn liên tiếp (giây).
                    Bao gồm cả speech->sil và sil->speech.
        merged:     danh sách (start, end, 'sil'/'speech') sau khi gộp
                    các đoạn cùng nhãn kề nhau.

    Chú ý: đây là biên CHUYỂN NHÃN. Nếu bài yêu cầu chỉ lấy onset/offset
    của tiếng nói, hãy lọc các biên theo nhãn hai bên khi tính metric.
    """
    segments = []

    with open(lab_path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.upper().startswith("F0"):
                continue

            parts = line.split()
            if len(parts) < 3:
                continue

            try:
                start = float(parts[0])
                end = float(parts[1])
            except ValueError:
                continue

            if not np.isfinite(start) or not np.isfinite(end) or end <= start:
                continue

            label = _normalize_label(parts[2])
            segments.append((max(0.0, start), max(0.0, end), label))

    if not segments:
        return np.array([], dtype=float), []

    # Sắp xếp để tránh file .lab không đúng thứ tự.
    segments.sort(key=lambda item: (item[0], item[1]))

    merged = []
    for start, end, label in segments:
        if not merged:
            merged.append([start, end, label])
            continue

        prev = merged[-1]
        # Chỉ gộp cùng nhãn khi hai đoạn chạm nhau hoặc có sai số làm tròn nhỏ.
        if label == prev[2] and start <= prev[1] + 1e-6:
            prev[1] = max(prev[1], end)
        else:
            merged.append([start, end, label])

    merged = [(s, e, label) for s, e, label in merged]
    boundaries = np.asarray(
        [merged[i][1] for i in range(len(merged) - 1)],
        dtype=float
    )
    return boundaries, merged


def extract_ste_zcr(signal, sr, frame_ms=25, hop_ms=10):
    """
    Trích xuất STE, ZCR và thời điểm tâm khung.

    Args:
        signal: tín hiệu mono.
        sr: sample rate (Hz).
        frame_ms: độ dài khung, mặc định 25 ms.
        hop_ms: bước dịch khung, mặc định 10 ms.

    Returns:
        ste, zcr, frame_times: các mảng 1-D có cùng độ dài.
    """
    signal = np.asarray(signal, dtype=np.float64).reshape(-1)

    if sr <= 0:
        raise ValueError("sr phải lớn hơn 0.")
    if frame_ms <= 0 or hop_ms <= 0:
        raise ValueError("frame_ms và hop_ms phải lớn hơn 0.")
    if signal.size == 0:
        empty = np.array([], dtype=float)
        return empty.copy(), empty.copy(), empty.copy()

    frame_len = max(1, int(round(sr * frame_ms / 1000.0)))
    hop_len = max(1, int(round(sr * hop_ms / 1000.0)))

    # Đệm zero nếu file ngắn hơn một khung để vẫn tạo được một frame.
    if signal.size < frame_len:
        padded = np.zeros(frame_len, dtype=float)
        padded[:signal.size] = signal
        signal = padded

    num_frames = 1 + (len(signal) - frame_len) // hop_len
    ste = np.empty(num_frames, dtype=float)
    zcr = np.empty(num_frames, dtype=float)
    frame_times = np.empty(num_frames, dtype=float)

    for i in range(num_frames):
        start = i * hop_len
        frame = signal[start:start + frame_len]

        # Short-Time Energy dạng trung bình bình phương.
        ste[i] = float(np.mean(frame * frame))

        # ZCR chuẩn hóa về khoảng [0, 1].
        if frame_len > 1:
            signs = np.sign(frame)
            # Quy ước mẫu bằng 0 cùng phía dương để tránh đếm thêm crossing.
            signs[signs == 0] = 1
            zcr[i] = float(np.count_nonzero(signs[1:] != signs[:-1]) /
                            (frame_len - 1))
        else:
            zcr[i] = 0.0

        frame_times[i] = (start + frame_len / 2.0) / sr

    return ste, zcr, frame_times


def learn_dual_thresholds(all_sil_ste, all_sp_ste, all_sil_zcr):
    """
    Ước lượng ba ngưỡng từ đặc trưng của tập huấn luyện.

    T_lower = mean(STE_silence) + 2*std(STE_silence)
    T_upper = chặn heuristic dựa trên T_lower và percentile 15 của STE speech
    T_zcr   = mean(ZCR_silence) + 2*std(ZCR_silence)

    T_upper không phải nghiệm tối ưu được đảm bảo về mặt toán học; đây là
    quy tắc khởi tạo để sau đó đánh giá/tinh chỉnh trên validation.
    """
    silence_ste = np.asarray(all_sil_ste, dtype=float).ravel()
    speech_ste = np.asarray(all_sp_ste, dtype=float).ravel()
    silence_zcr = np.asarray(all_sil_zcr, dtype=float).ravel()

    silence_ste = silence_ste[np.isfinite(silence_ste)]
    speech_ste = speech_ste[np.isfinite(speech_ste)]
    silence_zcr = silence_zcr[np.isfinite(silence_zcr)]

    if silence_ste.size == 0:
        raise ValueError(
            "Không có frame khoảng lặng để học T_lower. "
            "Kiểm tra file .lab và cách gán nhãn frame."
        )
    if speech_ste.size == 0:
        raise ValueError(
            "Không có frame tiếng nói để học T_upper. "
            "Kiểm tra file .lab và cách gán nhãn frame."
        )
    if silence_zcr.size == 0:
        raise ValueError(
            "Không có frame khoảng lặng để học T_zcr."
        )

    mean_sil_ste = float(np.mean(silence_ste))
    std_sil_ste = float(np.std(silence_ste))
    t_lower = max(0.0, mean_sil_ste + 2.0 * std_sil_ste)

    sp_p15 = float(np.percentile(speech_ste, 15))
    # Giữ đúng ý tưởng heuristic ban đầu, nhưng bảo đảm T_upper > T_lower.
    t_upper = max(t_lower * 2.5, min(t_lower * 4.0, sp_p15))
    if t_upper <= t_lower:
        t_upper = t_lower + max(1e-12, abs(t_lower) * 1e-6)

    mean_sil_zcr = float(np.mean(silence_zcr))
    std_sil_zcr = float(np.std(silence_zcr))
    t_zcr = float(np.clip(mean_sil_zcr + 2.0 * std_sil_zcr, 0.0, 1.0))

    return float(t_lower), float(t_upper), float(t_zcr)


def _merge_segments(segments):
    """Gộp các đoạn frame chồng lấn hoặc chạm nhau; đầu/cuối là inclusive."""
    if not segments:
        return []

    ordered = sorted(([int(s), int(e)] for s, e in segments), key=lambda x: x[0])
    merged = [ordered[0]]

    for start, end in ordered[1:]:
        if start <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    return merged


def _zcr_window_is_active(zcr, start, end, t_zcr, min_ratio=0.40,
                          min_high_frames=2):
    """Kiểm tra một cửa sổ ZCR có đủ mật độ frame vượt T_zcr hay không."""
    if end <= start:
        return False

    window = zcr[start:end]
    if window.size < min_high_frames:
        return False

    high_count = int(np.count_nonzero(window > t_zcr))
    return high_count >= min_high_frames and (high_count / window.size) >= min_ratio


def _refine_start_with_zcr(start, zcr, t_zcr, search_frames,
                           window_frames=5, min_ratio=0.40):
    """
    Mở rộng biên đầu về trước nếu các cửa sổ ZCR liên tiếp có mật độ cao.

    Mỗi cửa sổ được kiểm tra theo tỉ lệ frame có ZCR > T_zcr. Dừng khi gặp
    cửa sổ không đạt, tránh bị kéo xa bởi một vài frame ZCR cao rời rạc.
    """
    boundary = int(start)
    lower_limit = max(0, boundary - search_frames)

    while boundary > lower_limit:
        win_start = max(lower_limit, boundary - window_frames)
        if _zcr_window_is_active(
            zcr, win_start, boundary, t_zcr, min_ratio=min_ratio
        ):
            active = np.flatnonzero(zcr[win_start:boundary] > t_zcr)
            if active.size == 0:
                break
            new_boundary = win_start + int(active[0])
            if new_boundary >= boundary:
                break
            boundary = new_boundary
        else:
            break

    return boundary


def _refine_end_with_zcr(end, zcr, t_zcr, search_frames,
                         window_frames=5, min_ratio=0.40):
    """Mở rộng biên cuối theo tiêu chí mật độ ZCR, đối xứng với biên đầu."""
    boundary = int(end)
    upper_limit = min(len(zcr) - 1, boundary + search_frames)

    while boundary < upper_limit:
        win_end = min(upper_limit + 1, boundary + 1 + window_frames)
        if _zcr_window_is_active(
            zcr, boundary + 1, win_end, t_zcr, min_ratio=min_ratio
        ):
            active = np.flatnonzero(zcr[boundary + 1:win_end] > t_zcr)
            if active.size == 0:
                break
            new_boundary = boundary + 1 + int(active[-1])
            if new_boundary <= boundary:
                break
            boundary = new_boundary
        else:
            break

    return boundary


def segment_dual_threshold(
    ste, zcr, frame_times, t_lower, t_upper, t_zcr,
    search_back_ms=250, min_silence_ms=250, hop_ms=10,
    min_segment_ms=50, zcr_window_ms=50, zcr_min_ratio=0.40
):
    """
    Bước 3-5: STE hysteresis -> tinh chỉnh ZCR -> hậu xử lý.

    STE:
      - Bắt đầu một vùng khi STE > T_upper.
      - Mở rộng hai phía đến khi STE <= T_lower.
    ZCR:
      - Tìm trong phạm vi search_back_ms ở mỗi biên.
      - Chỉ mở rộng khi các cửa sổ ZCR đạt tỉ lệ frame > T_zcr tối thiểu.
    Hậu xử lý:
      - Gộp hai đoạn nếu khoảng cách giữa chúng < min_silence_ms.
      - Loại bỏ đoạn ngắn hơn min_segment_ms.

    Returns:
      labels: nhãn frame (1=speech, 0=silence).
      pred_boundaries: các thời điểm chuyển nhãn (giây), nằm giữa hai tâm frame.
    """
    ste = np.asarray(ste, dtype=float).ravel()
    zcr = np.asarray(zcr, dtype=float).ravel()
    frame_times = np.asarray(frame_times, dtype=float).ravel()

    if not (len(ste) == len(zcr) == len(frame_times)):
        raise ValueError("ste, zcr và frame_times phải có cùng độ dài.")
    if search_back_ms < 0 or min_silence_ms < 0 or hop_ms <= 0:
        raise ValueError("Các tham số thời gian không hợp lệ.")

    n_frames = len(ste)
    if n_frames == 0:
        return np.array([], dtype=int), np.array([], dtype=float)
    if not np.all(np.isfinite(ste)) or not np.all(np.isfinite(zcr)):
        raise ValueError("ste/zcr chứa NaN hoặc giá trị vô hạn.")

    # Đảm bảo ngưỡng khởi động cao hơn ngưỡng duy trì.
    t_lower = float(t_lower)
    t_upper = max(float(t_upper), t_lower + 1e-12)
    t_zcr = float(t_zcr)

    search_frames = int(round(search_back_ms / hop_ms))
    merge_gap_frames = int(round(min_silence_ms / hop_ms))
    min_segment_frames = max(1, int(np.ceil(min_segment_ms / hop_ms)))
    zcr_window_frames = max(1, int(round(zcr_window_ms / hop_ms)))

    # BƯỚC 3: phân đoạn sơ bộ bằng ngưỡng kép/hysteresis.
    segments = []
    i = 0

    while i < n_frames:
        if ste[i] > t_upper:
            start = i
            end = i

            while start > 0 and ste[start - 1] > t_lower:
                start -= 1
            while end < n_frames - 1 and ste[end + 1] > t_lower:
                end += 1

            segments.append([start, end])
            i = end + 1
        else:
            i += 1

    if not segments:
        return np.zeros(n_frames, dtype=int), np.array([], dtype=float)

    segments = _merge_segments(segments)

    # BƯỚC 4: tinh chỉnh mỗi biên bằng mật độ ZCR trong vùng tìm kiếm.
    refined = []
    for start, end in segments:
        new_start = _refine_start_with_zcr(
            start, zcr, t_zcr, search_frames,
            window_frames=zcr_window_frames,
            min_ratio=zcr_min_ratio
        )
        new_end = _refine_end_with_zcr(
            end, zcr, t_zcr, search_frames,
            window_frames=zcr_window_frames,
            min_ratio=zcr_min_ratio
        )
        refined.append([new_start, new_end])

    refined = _merge_segments(refined)

    # BƯỚC 5: gộp các đoạn có khoảng lặng ngắn hơn ngưỡng.
    final_segments = [refined[0]]
    for start, end in refined[1:]:
        gap_frames = start - final_segments[-1][1] - 1
        # "Dưới 200 ms" là <, không phải <=.
        if gap_frames < merge_gap_frames:
            final_segments[-1][1] = max(final_segments[-1][1], end)
        else:
            final_segments.append([start, end])

    labels = np.zeros(n_frames, dtype=int)
    for start, end in final_segments:
        if end - start + 1 >= min_segment_frames:
            labels[start:end + 1] = 1

    # Biên dự đoán là trung điểm giữa hai tâm frame có nhãn khác nhau.
    change_indices = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    pred_boundaries = np.asarray(
        [(frame_times[k - 1] + frame_times[k]) / 2.0
         for k in change_indices],
        dtype=float
    )

    return labels, pred_boundaries


def evaluate_metrics_detailed(pred_b, gt_b, max_match_ms=200.0):
    """
    Đánh giá biên bằng ghép cặp một-một, tránh nhiều biên dự đoán cùng khớp
    với một biên ground truth.

    Chỉ cặp có sai số <= max_match_ms mới được xem là khớp. MAE/RMSE được
    tính trên các cặp khớp; precision/recall/F1 và số biên bỏ sót/thừa giúp
    diễn giải metric công bằng hơn.

    Returns dict gồm mae_ms, rmse_ms, precision, recall, f1, matched,
    unmatched_pred, unmatched_gt, errors_ms.
    """
    pred = np.asarray(pred_b, dtype=float).ravel()
    gt = np.asarray(gt_b, dtype=float).ravel()

    pred = pred[np.isfinite(pred)]
    gt = gt[np.isfinite(gt)]

    pred.sort()
    gt.sort()

    if max_match_ms < 0:
        raise ValueError("max_match_ms không được âm.")

    n_pred, n_gt = len(pred), len(gt)
    if n_pred == 0 or n_gt == 0:
        matched = 0
        errors_ms = np.array([], dtype=float)
        unmatched_pred = n_pred
        unmatched_gt = n_gt
    else:
        # Ma trận sai số theo ms và ghép cặp tối ưu toàn cục.
        cost_ms = np.abs(pred[:, None] - gt[None, :]) * 1000.0
        rows, cols = linear_sum_assignment(cost_ms)
        valid = cost_ms[rows, cols] <= max_match_ms

        matched_errors = cost_ms[rows[valid], cols[valid]]
        errors_ms = np.asarray(matched_errors, dtype=float)
        matched = int(valid.sum())
        unmatched_pred = n_pred - matched
        unmatched_gt = n_gt - matched

    if errors_ms.size:
        mae = float(np.mean(errors_ms))
        rmse = float(np.sqrt(np.mean(errors_ms ** 2)))
    else:
        mae = None
        rmse = None

    precision = matched / n_pred if n_pred else 0.0
    recall = matched / n_gt if n_gt else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0 else 0.0
    )

    return {
        "mae_ms": mae,
        "rmse_ms": rmse,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "matched": matched,
        "unmatched_pred": int(unmatched_pred),
        "unmatched_gt": int(unmatched_gt),
        "errors_ms": errors_ms,
    }


def evaluate_metrics(pred_b, gt_b, max_match_ms=200.0):
    """
    Tương thích với main hiện tại: trả về (MAE_ms, RMSE_ms).

    Metric tính trên các cặp biên được ghép một-một trong phạm vi
    max_match_ms. Để biết số biên thừa/bỏ sót, dùng evaluate_metrics_detailed().
    """
    result = evaluate_metrics_detailed(
        pred_b, gt_b, max_match_ms=max_match_ms
    )
    return result["mae_ms"], result["rmse_ms"]
