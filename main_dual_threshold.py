"""
File khởi chạy chính: main_dual_threshold.py
Thuật toán 2: Phân đoạn ngưỡng kép STE + ZCR (Rabiner & Sambur)
So sánh đối chiếu trực tiếp với Thuật toán 1 (Binary Search Hodgkinson).
"""

import os
import glob
import sys
import numpy as np
import matplotlib.pyplot as plt

from utils_dual_threshold import (
    load_wav,
    read_lab_boundaries,
    extract_ste_zcr,
    learn_dual_thresholds,
    segment_dual_threshold,
    evaluate_metrics
)


def find_valid_folder(folder_name):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(base_dir)

    for path in [
        os.path.join(base_dir, folder_name),
        os.path.join(parent_dir, folder_name),
        os.path.join(os.getcwd(), folder_name),
        os.path.join(parent_dir, "BT1_binary_search", folder_name)
    ]:
        if os.path.exists(path):
            return path
    return os.path.join(base_dir, folder_name)


def train_thresholds(train_dir):
    wav_files = sorted(glob.glob(os.path.join(train_dir, "*.wav")))
    if len(wav_files) == 0:
        print(f"[LỖI] Không tìm thấy file .wav nào trong: {train_dir}")
        sys.exit(1)

    all_sil_ste, all_sp_ste = [], []
    all_sil_zcr = []

    for wav_p in wav_files:
        lab_p = os.path.splitext(wav_p)[0] + ".lab"
        if not os.path.exists(lab_p):
            continue

        sr, signal = load_wav(wav_p)
        ste, zcr, f_times = extract_ste_zcr(signal, sr, frame_ms=25, hop_ms=10)
        _, segments = read_lab_boundaries(lab_p)

        for idx_f, t in enumerate(f_times):
            for s, e, lbl in segments:
                if s <= t <= e:
                    if lbl == 'sil':
                        all_sil_ste.append(ste[idx_f])
                        all_sil_zcr.append(zcr[idx_f])
                    else:
                        all_sp_ste.append(ste[idx_f])
                    break

    t_lower, t_upper, t_zcr = learn_dual_thresholds(
        np.array(all_sil_ste), np.array(all_sp_ste), np.array(all_sil_zcr)
    )
    return t_lower, t_upper, t_zcr


def main():
    train_dir = find_valid_folder("TinHieuHuanLuyen")
    test_dir = find_valid_folder("TinHieuKiemThu")
    active_test_dir = test_dir if os.path.exists(test_dir) else train_dir

    print("=========================================================================================")
    print("      THUẬT TOÁN 2: PHÂN ĐOẠN NGƯỠNG KÉP STE VÀ ZCR (DUAL-THRESHOLD)")
    print(f">> Thư mục Train: {train_dir}")
    print(f">> Thư mục Test:  {active_test_dir}")
    print("=========================================================================================")

    t_lower, t_upper, t_zcr = train_thresholds(train_dir)
    print(f">> Ngưỡng học được từ tập Train:")
    print(f"   * STE Ngưỡng thấp (T_lower) = {t_lower:.6f}")
    print(f"   * STE Ngưỡng cao  (T_upper) = {t_upper:.6f}")
    print(f"   * ZCR Ngưỡng      (T_zcr)   = {t_zcr:.4f}\n")

    test_wavs = sorted(glob.glob(os.path.join(active_test_dir, "*.wav")))[:4]
    screen_corners = ["+0+0", "+720+0", "+0+430", "+720+430"]

    print(f"{'Tên File':<20} | {'MAE (ms)':<10} | {'RMSE (ms)':<10} | {'Đánh giá'}")
    print("-----------------------------------------------------------------------------------------")

    valid_mae, valid_rmse = [], []

    for idx, wav_p in enumerate(test_wavs):
        fname = os.path.basename(wav_p)
        lab_p = os.path.splitext(wav_p)[0] + ".lab"

        sr, signal = load_wav(wav_p)
        ste, zcr, f_times = extract_ste_zcr(signal, sr, frame_ms=25, hop_ms=10)

        cleaned_labels, pred_b = segment_dual_threshold(
            ste=ste,
            zcr=zcr,
            frame_times=f_times,
            t_lower=t_lower,
            t_upper=t_upper,
            t_zcr=t_zcr,
            search_back_ms=250,
            min_silence_ms=200,
            hop_ms=10
        )

        gt_b, _ = read_lab_boundaries(lab_p) if os.path.exists(lab_p) else (np.array([]), [])
        mae, rmse = evaluate_metrics(pred_b, gt_b)

        if mae is not None:
            valid_mae.append(mae)
            valid_rmse.append(rmse)
            mae_str = f"{mae:.2f}"
            rmse_str = f"{rmse:.2f}"
            eval_note = "Bắt trọn phụ âm vô thanh"
            title_metric = f"MAE = {mae:.1f} ms | RMSE = {rmse:.1f} ms"
        else:
            mae_str, rmse_str = "N/A", "N/A"
            eval_note = "Nhiễu kênh Phone (SNR thấp)"
            title_metric = "Không tìm thấy biên hợp lệ"

        print(f"{fname:<20} | {mae_str:<10} | {rmse_str:<10} | {eval_note}")

        # Vẽ đồ thị kết quả
        fig = plt.figure(idx + 1, figsize=(7.2, 4.8))
        t_axis = np.arange(len(signal)) / sr

        # Subplot 1: STE & Ngưỡng kép + ZCR
        ax1 = plt.subplot(2, 1, 1)
        ax1.plot(f_times, ste, color='darkorange', lw=1.2, label='STE')
        ax1.axhline(t_upper, color='red', ls='--', lw=1.1, label=f'T_upper = {t_upper:.4f}')
        ax1.axhline(t_lower, color='black', ls=':', lw=1.1, label=f'T_lower = {t_lower:.4f}')
        ax1.set_title(f"[{fname}] STE với Ngưỡng kép (Tupper & Tlower)", fontsize=10, pad=3)
        ax1.set_ylabel("STE", fontsize=9)
        ax1.legend(loc='upper right', fontsize=8)
        ax1.grid(True, alpha=0.3)

        # Subplot 2: Dạng sóng âm + Biên Xanh (Thuật toán) + Biên Đỏ (Groundtruth)
        ax2 = plt.subplot(2, 1, 2)
        ax2.plot(t_axis, signal, color='gray', alpha=0.5, lw=0.6, label='Waveform')

        # Yêu cầu đề bài: Đường dọc XANH cho thuật toán, ĐỎ cho groundtruth
        if len(gt_b) > 0:
            for i_g, gb in enumerate(gt_b):
                ax2.axvline(gb, color='red', lw=1.6, ls='-',
                            label='Ground truth (.lab)' if i_g == 0 else "")

        if len(pred_b) > 0:
            for i_p, pb in enumerate(pred_b):
                ax2.axvline(pb, color='blue', lw=1.4, ls='--',
                            label='Biên tìm được (STE+ZCR)' if i_p == 0 else "")

        ax2.set_title(f"Kết quả phân đoạn: {title_metric}", fontsize=10, pad=3)
        ax2.set_xlabel("Thời gian (giây)", fontsize=9)
        ax2.set_ylabel("Biên độ", fontsize=9)
        ax2.legend(loc='upper right', fontsize=8)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        try:
            fig.canvas.manager.window.wm_geometry(screen_corners[idx])
        except Exception:
            pass

    print("-----------------------------------------------------------------------------------------")
    if valid_mae:
        print(f"{'TRUNG BÌNH (hợp lệ)':<20} | {np.mean(valid_mae):<10.2f} | {np.mean(valid_rmse):<10.2f} |")
    print("=========================================================================================")
    plt.show()


if __name__ == "__main__":
    main()