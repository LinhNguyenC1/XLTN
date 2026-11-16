"""
File khởi chạy chính: main.py
"""

import os
import glob
import sys
import numpy as np
import matplotlib.pyplot as plt

from utils import (
    load_wav,
    read_lab_boundaries,
    extract_log_ma,
    binary_search_threshold,
    segment_and_clean,
    evaluate_metrics
)


def find_valid_folder(folder_name):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(base_dir)

    for path in [
        os.path.join(base_dir, folder_name),
        os.path.join(parent_dir, folder_name),
        os.path.join(os.getcwd(), folder_name)
    ]:
        if os.path.exists(path):
            return path
    return os.path.join(base_dir, folder_name)


def train_threshold(train_dir):
    wav_files = sorted(glob.glob(os.path.join(train_dir, "*.wav")))
    if len(wav_files) == 0:
        print(f"[LỖI] Không tìm thấy file .wav nào trong: {train_dir}")
        sys.exit(1)

    all_silence, all_speech = [], []
    for wav_p in wav_files:
        lab_p = os.path.splitext(wav_p)[0] + ".lab"
        if not os.path.exists(lab_p):
            continue

        sr, signal = load_wav(wav_p)
        log_ma, f_times = extract_log_ma(signal, sr)
        _, segments = read_lab_boundaries(lab_p)

        for idx_f, t in enumerate(f_times):
            for s, e, lbl in segments:
                if s <= t <= e:
                    if lbl == 'sil':
                        all_silence.append(log_ma[idx_f])
                    else:
                        all_speech.append(log_ma[idx_f])
                    break

    all_sil = np.array(all_silence)
    all_sp = np.array(all_speech)
    T, iters, f_overlap, g_overlap = binary_search_threshold(all_sil, all_sp, verbose=True)
    return T, iters, all_sil, all_sp, f_overlap, g_overlap

def plot_training_analysis(all_silence, all_speech, f, g, T, iters):
    # ==========================================
    # HÌNH 1: PHÂN BỐ TẦN SUẤT NĂNG LƯỢNG (KDE)
    # ==========================================
    plt.figure( num=10,figsize=(8, 4.5))
    plt.title("Hình 1: Phân bố tần suất log(MA)")
    bins = np.linspace(-10.0, -2.0, 100)
    
    # Histogram chuẩn hóa mật độ (density=True) tạo đường phân bố mượt mà
    plt.hist(all_silence, bins=bins, density=True, alpha=0.45, color='dodgerblue', label='Silence (f)')
    plt.hist(all_speech, bins=bins, density=True, alpha=0.45, color='darkorange', label='Speech (g)')
    
    plt.axvline(T, color='red', linestyle='--', linewidth=1.8, label=f'T tối ưu = {T:.4f} ({iters} vòng lặp)')
    plt.title(f"Phân bố năng lượng Speech vs Silence trên tập Train (Hội tụ sau {iters} vòng)", fontsize=11)
    plt.xlabel("log(MA)")
    plt.ylabel("Mật độ phân bố (Density)")
    plt.xlim(-10.0, -2.0)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

# ==========================================
    # HÌNH 2: DIỆN TÍCH NHẦM LẪN (FIGURE 2.4 HODGKINSON)
    # ==========================================
    # Khởi tạo 2 subplot độc lập, không dùng plt.title() ở cấp Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), num="Hinh 2 - Dien tich nham lan (Figure 2.4)")
    fig.suptitle("Hình 2: Diện tích nhầm lẫn (Area of Confusion)", fontsize=12, fontweight='bold', y=0.98)

    # Sắp xếp mảng f tăng dần và g giảm dần theo Hodgkinson (2012)
    f_sorted = np.sort(f)
    g_sorted = np.sort(g)[::-1]
    
    x_f = np.linspace(0, 1, len(f_sorted))
    x_g = np.linspace(0, 1, len(g_sorted))

    # Cận Y chung để hai biểu đồ đồng nhất thang đo
    y_min = min(np.min(f_sorted), np.min(g_sorted)) - 0.2
    y_max = max(np.max(f_sorted), np.max(g_sorted)) + 0.2

    # --- Subplot trái: Silence ---
    ax1.plot(x_f, f_sorted, color='dodgerblue', lw=1.6, label=r'$f(n / N_f)$')
    ax1.axhline(T, color='gray', linestyle='--', lw=1.2, label=f'$T = {T:.2f}$')
    ax1.fill_between(x_f, T, f_sorted, where=(f_sorted > T), color='gray', alpha=0.6, label='Area of Confusion')
    ax1.set_title("Silence (f)", fontsize=11)
    ax1.set_xlabel(r"$n / N_f$", fontsize=10)
    ax1.set_ylabel("log(MA)", fontsize=10)
    ax1.set_ylim(y_min, y_max)
    ax1.set_xlim(0, 1)
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # --- Subplot phải: Speech ---
    ax2.plot(x_g, g_sorted, color='darkorange', lw=1.6, label=r'$g(n / N_g)$')
    ax2.axhline(T, color='gray', linestyle='--', lw=1.2, label=f'$T = {T:.2f}$')
    ax2.fill_between(x_g, g_sorted, T, where=(g_sorted < T), color='gray', alpha=0.6, label='Area of Confusion')
    ax2.set_title("Speech (g)", fontsize=11)
    ax2.set_xlabel(r"$n / N_g$", fontsize=10)
    ax2.set_ylabel("log(MA)", fontsize=10)
    ax2.set_ylim(y_min, y_max)
    ax2.set_xlim(0, 1)
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Căn chỉnh lề thủ công chuẩn xác: wspace=0.32 tạo khoảng cách đủ rộng giữa 2 đồ thị
    fig.subplots_adjust(top=0.88, bottom=0.14, left=0.08, right=0.96, wspace=0.32)

def main():
    train_dir = find_valid_folder("TinHieuHuanLuyen")
    test_dir = find_valid_folder("TinHieuKiemThu")
    active_test_dir = test_dir if os.path.exists(test_dir) else train_dir

    print("==================================================")
    print("      HUẤN LUYỆN TÌM NGƯỠNG DÙNG CHUNG (T)")
    print(f">> Thư mục Train: {train_dir}")
    print(f">> Thư mục Test:  {active_test_dir}")
    print("==================================================")

    T_common, iters, all_sil, all_sp, f_overlap, g_overlap = train_threshold(train_dir)
    print(f">> Ngưỡng tối ưu T xác định được: {T_common:.4f}")
    print(f">> Thuật toán hội tụ sau: {iters} vòng lặp\n")

    # Hiển thị 2 hình ảnh phân tích tập Train
    plot_training_analysis(all_sil, all_sp, f_overlap, g_overlap, T_common, iters)

    test_wavs = sorted(glob.glob(os.path.join(active_test_dir, "*.wav")))[:4]
    screen_corners = ["+0+0", "+750+0", "+0+450", "+750+450"]

    print("==================================================")
    print(f"{'Tên File':<20} | {'MAE (ms)':<10} | {'RMSE (ms)':<10}")
    print("--------------------------------------------------")

    valid_mae, valid_rmse = [], []

    for idx, wav_p in enumerate(test_wavs):
        fname = os.path.basename(wav_p)
        lab_p = os.path.splitext(wav_p)[0] + ".lab"

        sr, signal = load_wav(wav_p)
        log_ma, f_times = extract_log_ma(signal, sr)

        cleaned_labels, pred_b = segment_and_clean(log_ma, f_times, T_common, min_silence_ms=200)
        gt_b, _ = read_lab_boundaries(lab_p) if os.path.exists(lab_p) else (np.array([]), [])
        
        mae, rmse = evaluate_metrics(pred_b, gt_b)

        if mae is not None:
            valid_mae.append(mae)
            valid_rmse.append(rmse)
            mae_str = f"{mae:.2f}"
            rmse_str = f"{rmse:.2f}"
            title_metric = f"MAE = {mae:.1f} ms, RMSE = {rmse:.1f} ms"
        else:
            mae_str = "N/A"
            rmse_str = "N/A"
            title_metric = "Không phát hiện biên (Do nhiễu nền SNR thấp)"

        print(f"{fname:<20} | {mae_str:<10} | {rmse_str:<10}")

        # Vẽ cửa sổ đồ thị
        fig = plt.figure(idx + 1, figsize=(7.5, 4.2))
        t_axis = np.arange(len(signal)) / sr

        # Subplot 1: log(MA) + Ngưỡng T
        ax1 = plt.subplot(2, 1, 1)
        ax1.plot(f_times, log_ma, color='darkorange', lw=1.2, label='log(MA)')
        ax1.axhline(T_common, color='black', ls='--', lw=1.2, label=f'T = {T_common:.2f}')
        ax1.set_title(f"[{fname}] log(MA) & Ngưỡng T", fontsize=10)
        ax1.set_ylabel("log(MA)")
        ax1.legend(loc='upper right', fontsize=8)
        ax1.grid(True, alpha=0.3)

        # Subplot 2: Tín hiệu sóng âm + Biên đỏ/xanh
        ax2 = plt.subplot(2, 1, 2)
        ax2.plot(t_axis, signal, color='gray', alpha=0.5, lw=0.6, label='Waveform')

        if len(gt_b) > 0:
            for gb in gt_b:
                ax2.axvline(gb, color='red', lw=1.6, ls='-', label='Biên chuẩn (.lab)' if gb == gt_b[0] else "")
        if len(pred_b) > 0:
            for pb in pred_b:
                ax2.axvline(pb, color='blue', lw=1.4, ls='--', label='Biên tìm được' if pb == pred_b[0] else "")

        ax2.set_title(f"Phân đoạn: {title_metric}", fontsize=10)
        ax2.set_xlabel("Thời gian (giây)")
        ax2.set_ylabel("Biên độ")
        ax2.legend(loc='upper right', fontsize=8)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        try:
            fig.canvas.manager.window.wm_geometry(screen_corners[idx])
        except Exception:
            pass

    print("--------------------------------------------------")
    if valid_mae:
        print(f"{'TRUNG BÌNH (hợp lệ)':<20} | {np.mean(valid_mae):<10.2f} | {np.mean(valid_rmse):<10.2f}")
    else:
        print(f"{'TRUNG BÌNH':<20} | {'N/A':<10} | {'N/A':<10}")
    print("==================================================")
    plt.show()


if __name__ == "__main__":
    main()