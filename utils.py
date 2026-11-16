"""
Module: utils.py
Chứa các hàm xử lý tín hiệu, thuật toán Binary Search và đánh giá sai số.
"""

import os
import warnings
import numpy as np
from scipy.io import wavfile

# Tắt cảnh báo metadata thừa của file WAV
warnings.filterwarnings("ignore", category=wavfile.WavFileWarning)


def load_wav(filepath):
    """Đọc file WAV, chuẩn hóa biên độ về [-1, 1]."""
    sr, data = wavfile.read(filepath)
    if data.ndim > 1:
        data = data[:, 0]
    max_val = np.max(np.abs(data))
    return sr, (data.astype(np.float64) / max_val) if max_val > 0 else data.astype(np.float64)


def read_lab_boundaries(lab_path):
    """Đọc các mốc biên thời gian chuẩn (chuyển giữa sil và speech) từ file .lab."""
    segments = []
    with open(lab_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('F0'):
                continue
            parts = line.split()
            if len(parts) >= 3:
                try:
                    s, e, lbl = float(parts[0]), float(parts[1]), parts[2].lower()
                    segments.append((s, e, 'sil' if lbl == 'sil' else 'speech'))
                except ValueError:
                    continue

    if not segments:
        return np.array([]), []

    # Gộp các đoạn liên tiếp cùng trạng thái
    merged = [segments[0]]
    for s, e, lbl in segments[1:]:
        if lbl == merged[-1][2]:
            merged[-1] = (merged[-1][0], e, lbl)
        else:
            merged.append((s, e, lbl))

    boundaries = np.array([m[1] for m in merged[:-1] if m[1] > 0])
    return boundaries, merged


def extract_log_ma(signal, sr, frame_ms=25, hop_ms=10):
    """
    Chia khung và tính đặc trưng log(MA) theo công thức Hodgkinson (2012).
    Cửa sổ 25ms, bước nhảy 10ms.
    """
    frame_len = int(sr * frame_ms / 1000)
    hop_len = int(sr * hop_ms / 1000)
    num_frames = int(np.floor((len(signal) - frame_len) / hop_len)) + 1

    log_ma = np.zeros(num_frames)
    frame_times = np.zeros(num_frames)

    for i in range(num_frames):
        st = i * hop_len
        frame = signal[st : st + frame_len]
        ma = np.sum(np.abs(frame)) / frame_len
        log_ma[i] = np.log(ma + 1e-12)
        frame_times[i] = (st + frame_len / 2.0) / sr

    return log_ma, frame_times


def binary_search_threshold(f_silence, g_speech, verbose=True, max_iters=200):
    """
    Thuật toán tìm kiếm nhị phân cân bằng diện tích nhầm lẫn (Equation 2.8 & Section 2.1.3).
    """
    overlap_min = np.min(g_speech)
    overlap_max = np.max(f_silence)

    if overlap_min >= overlap_max:
        T = (overlap_min + overlap_max) / 2.0
        if verbose:
            print("\n[Binary Search] Không có vùng chồng lấn.")
            print(f"T = {T:.6f}")
        return T, 0, f_silence, g_speech

    # 1. Trích xuất dữ liệu trong vùng chồng lấn
    f = f_silence[(f_silence >= overlap_min) & (f_silence <= overlap_max)]
    g = g_speech[(g_speech >= overlap_min) & (g_speech <= overlap_max)]

    Nf, Ng = len(f), len(g)

    # 2. Thiết lập T_min, T_max
    T_min = overlap_min
    T_max = overlap_max

    # 3. Khởi tạo T ban đầu
    T = 0.5 * (T_min + T_max)

    # 4. Đếm số lượng mẫu ban đầu hai phía T
    i = int(np.sum(f < T))
    p = int(np.sum(g > T))

    # 5. Khởi tạo j, q
    j, q = -1, -1

    iterations = 0

    if verbose:
        print("\n==================================================")
        print("        QUÁ TRÌNH TÌM KIẾM NGƯỠNG T")
        print("==================================================")
        print(f"T_min ban đầu = {T_min:.6f} | T_max ban đầu = {T_max:.6f}")
        print(f"T khởi tạo    = {T:.6f}")
        print("--------------------------------------------------")

    # 6. Lặp chừng nào (i != j hoặc p != q)
    while (i != j or p != q) and (iterations < max_iters):
        iterations += 1

        # 7. Đánh giá vế trái Equation 2.8
        area_f = np.sum(np.maximum(f - T, 0.0)) / Nf
        area_g = np.sum(np.maximum(T - g, 0.0)) / Ng
        diff = area_f - area_g

        if verbose:
            print(
                f"Vòng {iterations:3d}: T = {T:.6f} | "
                f"Area_f = {area_f:.6f} | Area_g = {area_g:.6f} | Diff = {diff:+.6f}"
            )

        if diff > 0:
            T_min = T
        else:
            T_max = T

        # 8. Cập nhật T mới
        T = 0.5 * (T_min + T_max)

        # 9. Lưu trạng thái
        j, q = i, p

        # 10. Đếm lại số lượng mẫu với T mới
        i = int(np.sum(f < T))
        p = int(np.sum(g > T))

    if verbose:
        print("--------------------------------------------------")
        print(f"[DỪNG] Hội tụ sau {iterations} vòng lặp.")
        print(f"T tối ưu = {T:.6f}")
        print("==================================================\n")

    return T, iterations, f, g


def segment_and_clean(log_ma, frame_times, threshold, min_silence_ms=250):
    """
    Phân đoạn theo ngưỡng T và triệt tiêu khoảng lặng ảo < 300ms.
    """
    raw_labels = (log_ma >= threshold).astype(int)
    n_frames = len(raw_labels)

    cleaned_labels = raw_labels.copy()
    idx = 0
    while idx < n_frames:
        if cleaned_labels[idx] == 0:
            start_idx = idx
            while idx < n_frames and cleaned_labels[idx] == 0:
                idx += 1
            end_idx = idx - 1
            dur_ms = (frame_times[end_idx] - frame_times[start_idx]) * 1000.0
            if dur_ms < min_silence_ms:
                cleaned_labels[start_idx : end_idx + 1] = 1
        else:
            idx += 1

    pred_boundaries = []
    for k in range(1, n_frames):
        if cleaned_labels[k] != cleaned_labels[k - 1]:
            pred_boundaries.append((frame_times[k] + frame_times[k - 1]) / 2.0)

    return cleaned_labels, np.array(pred_boundaries)


def evaluate_metrics(pred_b, gt_b):
    """
    Tính MAE và RMSE (ms). Trả về None nếu không tìm thấy biên.
    """
    if len(pred_b) == 0 or len(gt_b) == 0:
        return None, None

    errs = []
    for g in gt_b:
        errs.append(np.min(np.abs(pred_b - g)))
    for p in pred_b:
        errs.append(np.min(np.abs(gt_b - p)))

    errs = np.array(errs) * 1000.0
    return np.mean(errs), np.sqrt(np.mean(errs ** 2))