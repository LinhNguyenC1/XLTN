"""Minh bach hoa viec chon heuristic T_upper bang leave-one-recording-out.

Chay tu thu muc goc repository:
    python tune_heuristic.py

Quy trinh:
- Chi doc TinHieuHuanLuyen; moi luot giu lai mot recording de validation.
- Uoc luong T_lower va T_zcr, cung P_p(STE_speech), tren cac recording con lai.
- Duyet luoi T_upper=max(a*T_lower, min(b*T_lower, P_p)).
- Chay nguyen pipeline segment_dual_threshold va ham danh gia mot-mot hien co.
- Chon tham so theo macro F1 (moi recording trong so nhu nhau), tie-break bang
  tong so bien khong khop it hon, roi macro MAE thap hon.
- So sanh truc tiep voi heuristic goc (a=2.5, b=4.0, p=15).

Khong doc tap test de chon tham so. Ket qua LOO tren it recording chi la
bang chung thuc nghiem noi bo, khong chung minh tinh toi uu pho quat.

Dau ra trong heuristic_results/:
- heuristic_validation_folds.csv: tung cau hinh x tung fold, co nguong va metric.
- heuristic_validation_summary.csv: tong hop macro/pooled cho moi cau hinh.
- heuristic_best_parameters.json: cau hinh duoc chon va baseline.
- heuristic_experiment_report.md: giao thuc, tieu chi, ket qua va gioi han.
- heuristic_f1_heatmap.png, heuristic_mae_heatmap.png.
- heuristic_percentile_sensitivity.png.
- validation_fold_*.png: STE, ZCR, waveform, nguong va bien cho tung fold.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils_dual_threshold import (
    evaluate_metrics_detailed,
    extract_ste_zcr,
    load_wav,
    read_lab_boundaries,
    segment_dual_threshold,
)

# Candidate grid. Includes the existing heuristic (2.5, 4.0, P15).
A_VALUES = (1.5, 2.0, 2.5, 3.0)
B_VALUES = (3.0, 3.5, 4.0, 4.5, 5.0)
PERCENTILES = (5, 10, 15, 20, 25)
BASELINE = (2.5, 4.0, 15)
MATCH_TOLERANCE_MS = 200.0


@dataclass
class Recording:
    name: str
    wav_path: str
    sr: int
    signal: np.ndarray
    ste: np.ndarray
    zcr: np.ndarray
    frame_times: np.ndarray
    boundaries: np.ndarray
    silence_ste: np.ndarray
    speech_ste: np.ndarray
    silence_zcr: np.ndarray


def find_data_dir(name: str, script_dir: str) -> str:
    candidates = (
        os.path.join(script_dir, name),
        os.path.join(os.path.dirname(script_dir), name),
        os.path.join(os.getcwd(), name),
    )
    for path in candidates:
        if os.path.isdir(path):
            return path
    raise FileNotFoundError(
        f"Khong tim thay thu muc {name!r}; da kiem tra: {', '.join(candidates)}"
    )


def load_recording(wav_path: str) -> Recording | None:
    lab_path = os.path.splitext(wav_path)[0] + ".lab"
    if not os.path.isfile(lab_path):
        print(f"[SKIP] Thieu file nhan .lab: {wav_path}")
        return None

    sr, signal = load_wav(wav_path)
    ste, zcr, frame_times = extract_ste_zcr(
        signal, sr, frame_ms=25, hop_ms=10
    )
    boundaries, segments = read_lab_boundaries(lab_path)
    silence_ste: List[float] = []
    speech_ste: List[float] = []
    silence_zcr: List[float] = []

    # Giong main_dual_threshold.py: gan nhan theo tam frame va .lab.
    for index, timestamp in enumerate(frame_times):
        for start, end, label in segments:
            if start <= timestamp <= end:
                if label == "sil":
                    silence_ste.append(float(ste[index]))
                    silence_zcr.append(float(zcr[index]))
                else:  # gom ca voiced va unvoiced vao speech
                    speech_ste.append(float(ste[index]))
                break

    return Recording(
        name=os.path.basename(wav_path),
        wav_path=wav_path,
        sr=sr,
        signal=signal,
        ste=ste,
        zcr=zcr,
        frame_times=frame_times,
        boundaries=boundaries,
        silence_ste=np.asarray(silence_ste, dtype=float),
        speech_ste=np.asarray(speech_ste, dtype=float),
        silence_zcr=np.asarray(silence_zcr, dtype=float),
    )


def concatenate(records: Sequence[Recording], field: str) -> np.ndarray:
    arrays = [getattr(record, field) for record in records]
    arrays = [array for array in arrays if array.size]
    return np.concatenate(arrays) if arrays else np.asarray([], dtype=float)


def estimate_thresholds(
    training_records: Sequence[Recording], a: float, b: float, percentile: float
) -> Tuple[float, float, float, float, str]:
    """Use only training-fold features; return thresholds and active Tupper rule."""
    sil_ste = concatenate(training_records, "silence_ste")
    speech_ste = concatenate(training_records, "speech_ste")
    sil_zcr = concatenate(training_records, "silence_zcr")
    if not sil_ste.size or not speech_ste.size or not sil_zcr.size:
        raise ValueError("Fold train khong du frame silence/speech co nhan.")

    # Keep T_lower/T_zcr exactly aligned with learn_dual_thresholds().
    t_lower = max(0.0, float(np.mean(sil_ste) + 2.0 * np.std(sil_ste)))
    p_speech = float(np.percentile(speech_ste, percentile))
    lower_bound = a * t_lower
    upper_bound = b * t_lower
    t_upper = max(lower_bound, min(upper_bound, p_speech))
    if p_speech < lower_bound:
        active_rule = "lower_floor_a*T_lower"
    elif p_speech > upper_bound:
        active_rule = "upper_cap_b*T_lower"
    else:
        active_rule = "speech_percentile"
    if t_upper <= t_lower:
        t_upper = t_lower + max(1e-12, abs(t_lower) * 1e-6)
        active_rule += "+strict_order_guard"

    t_zcr = float(np.clip(
        np.mean(sil_zcr) + 2.0 * np.std(sil_zcr), 0.0, 1.0
    ))
    return t_lower, t_upper, t_zcr, p_speech, active_rule


def evaluate_recording(
    recording: Recording, t_lower: float, t_upper: float, t_zcr: float
) -> Tuple[dict, np.ndarray]:
    _, predicted = segment_dual_threshold(
        ste=recording.ste,
        zcr=recording.zcr,
        frame_times=recording.frame_times,
        t_lower=t_lower,
        t_upper=t_upper,
        t_zcr=t_zcr,
        search_back_ms=250,
        min_silence_ms=200,
        hop_ms=10,
    )
    metrics = evaluate_metrics_detailed(
        predicted, recording.boundaries, max_match_ms=MATCH_TOLERANCE_MS
    )
    return metrics, predicted


def metric_row(
    metrics: dict,
    recording: Recording,
    a: float,
    b: float,
    percentile: int,
    fold_index: int,
    t_lower: float,
    t_upper: float,
    t_zcr: float,
    p_speech: float,
    active_rule: str,
) -> dict:
    return {
        "a_lower_multiplier": a,
        "b_upper_multiplier": b,
        "speech_percentile": percentile,
        "fold": fold_index,
        "held_out_recording": recording.name,
        "t_lower": t_lower,
        "t_upper": t_upper,
        "t_zcr": t_zcr,
        "speech_ste_percentile_value": p_speech,
        "active_t_upper_rule": active_rule,
        "f1": metrics["f1"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "mae_ms": metrics["mae_ms"] if metrics["mae_ms"] is not None else "",
        "rmse_ms": metrics["rmse_ms"] if metrics["rmse_ms"] is not None else "",
        "matched_boundaries": metrics["matched"],
        "predicted_boundaries": metrics["matched"] + metrics["unmatched_pred"],
        "ground_truth_boundaries": metrics["matched"] + metrics["unmatched_gt"],
        "unmatched_predicted": metrics["unmatched_pred"],
        "missed_ground_truth": metrics["unmatched_gt"],
        "match_tolerance_ms": MATCH_TOLERANCE_MS,
    }


def summarize_combo(fold_rows: Sequence[dict]) -> dict:
    """Return equal-recording macro metrics plus pooled boundary counts/errors."""
    f1s = np.asarray([float(row["f1"]) for row in fold_rows], dtype=float)
    precisions = np.asarray([float(row["precision"]) for row in fold_rows], dtype=float)
    recalls = np.asarray([float(row["recall"]) for row in fold_rows], dtype=float)
    macro_f1 = float(np.mean(f1s))
    macro_f1_std = float(np.std(f1s))
    macro_precision = float(np.mean(precisions))
    macro_recall = float(np.mean(recalls))

    matched = sum(int(row["matched_boundaries"]) for row in fold_rows)
    predicted = sum(int(row["predicted_boundaries"]) for row in fold_rows)
    ground_truth = sum(int(row["ground_truth_boundaries"]) for row in fold_rows)
    pooled_precision = matched / predicted if predicted else 0.0
    pooled_recall = matched / ground_truth if ground_truth else 0.0
    pooled_f1 = (2 * pooled_precision * pooled_recall /
                 (pooled_precision + pooled_recall)
                 if pooled_precision + pooled_recall else 0.0)

    fold_maes = [float(row["mae_ms"]) for row in fold_rows if row["mae_ms"] != ""]
    errors = []
    # MAE/RMSE pooled boundary-level are recovered from per-fold MAE/RMSE and
    # matched counts, preserving the original evaluator's one-to-one matching.
    squared_error_sum = 0.0
    absolute_error_sum = 0.0
    for row in fold_rows:
        count = int(row["matched_boundaries"])
        if not count or row["mae_ms"] == "":
            continue
        mae = float(row["mae_ms"])
        rmse = float(row["rmse_ms"])
        absolute_error_sum += mae * count
        squared_error_sum += (rmse ** 2) * count
    matched_for_error = sum(
        int(row["matched_boundaries"])
        for row in fold_rows if row["mae_ms"] != ""
    )
    pooled_mae = absolute_error_sum / matched_for_error if matched_for_error else float("nan")
    pooled_rmse = float(np.sqrt(squared_error_sum / matched_for_error)) if matched_for_error else float("nan")

    def mean_std(values: Sequence[float]) -> Tuple[float, float]:
        if not values:
            return float("nan"), float("nan")
        return float(np.mean(values)), float(np.std(values))

    macro_mae, macro_mae_std = mean_std(fold_maes)
    rules: Dict[str, int] = {}
    for row in fold_rows:
        rule = str(row["active_t_upper_rule"])
        rules[rule] = rules.get(rule, 0) + 1
    threshold_keys = ("t_lower", "t_upper", "t_zcr", "speech_ste_percentile_value")
    threshold_means = {
        "mean_" + key: float(np.mean([float(row[key]) for row in fold_rows]))
        for key in threshold_keys
    }
    return {
        "a_lower_multiplier": fold_rows[0]["a_lower_multiplier"],
        "b_upper_multiplier": fold_rows[0]["b_upper_multiplier"],
        "speech_percentile": fold_rows[0]["speech_percentile"],
        "folds": len(fold_rows),
        "macro_f1": macro_f1,
        "macro_f1_std": macro_f1_std,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "pooled_f1": pooled_f1,
        "pooled_precision": pooled_precision,
        "pooled_recall": pooled_recall,
        "macro_mae_ms": macro_mae,
        "macro_mae_std_ms": macro_mae_std,
        "pooled_mae_ms": pooled_mae,
        "pooled_rmse_ms": pooled_rmse,
        "matched_boundaries": matched,
        "predicted_boundaries": predicted,
        "ground_truth_boundaries": ground_truth,
        "unmatched_predicted": predicted - matched,
        "missed_ground_truth": ground_truth - matched,
        "lower_floor_count": rules.get("lower_floor_a*T_lower", 0),
        "percentile_active_count": rules.get("speech_percentile", 0),
        "upper_cap_count": rules.get("upper_cap_b*T_lower", 0),
        "strict_order_guard_count": sum(v for k, v in rules.items() if "strict_order_guard" in k),
        **threshold_means,
    }


def run_grid(records: Sequence[Recording]) -> Tuple[List[dict], List[dict], Dict[Tuple[float, float, int], dict]]:
    """Run each candidate on each held-out recording, estimating on the rest."""
    fold_rows: List[dict] = []
    fold_artifacts: Dict[Tuple[float, float, int], dict] = {}
    combinations = [
        (a, b, p)
        for a in A_VALUES
        for b in B_VALUES if b > a
        for p in PERCENTILES
    ]
    for combo_index, (a, b, percentile) in enumerate(combinations, start=1):
        artifacts = {}
        current_rows = []
        for fold_index, held_out in enumerate(records, start=1):
            fold_train = [record for record in records if record is not held_out]
            t_lower, t_upper, t_zcr, p_speech, active_rule = estimate_thresholds(
                fold_train, a, b, percentile
            )
            metrics, predicted = evaluate_recording(
                held_out, t_lower, t_upper, t_zcr
            )
            row = metric_row(
                metrics, held_out, a, b, percentile, fold_index,
                t_lower, t_upper, t_zcr, p_speech, active_rule,
            )
            current_rows.append(row)
            artifacts[held_out.name] = {
                "recording": held_out,
                "predicted": predicted,
                "thresholds": (t_lower, t_upper, t_zcr),
                "metrics": metrics,
                "active_rule": active_rule,
            }
        fold_rows.extend(current_rows)
        fold_artifacts[(a, b, percentile)] = {
            "fold_rows": current_rows,
            "artifacts": artifacts,
            "summary": summarize_combo(current_rows),
        }
        print(f"[{combo_index}/{len(combinations)}] a={a:g}, b={b:g}, P{percentile} done")

    summary_rows = [entry["summary"] for entry in fold_artifacts.values()]
    return fold_rows, summary_rows, fold_artifacts


def choose_best(summary_rows: Sequence[dict]) -> dict:
    """Predeclared selection: macro F1, fewer unmatched, then macro MAE."""
    return max(
        summary_rows,
        key=lambda row: (
            float(row["macro_f1"]),
            -int(row["unmatched_predicted"] + row["missed_ground_truth"]),
            -float(row["macro_mae_ms"]) if np.isfinite(row["macro_mae_ms"]) else -np.inf,
        ),
    )


def write_csv(rows: Sequence[dict], path: str) -> None:
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_heatmaps(summary_rows: Sequence[dict], output_dir: str) -> None:
    a_values = sorted({float(row["a_lower_multiplier"]) for row in summary_rows})
    b_values = sorted({float(row["b_upper_multiplier"]) for row in summary_rows})
    percentiles = sorted({int(row["speech_percentile"]) for row in summary_rows})
    for metric, filename, title, cmap in (
        ("macro_f1", "heuristic_f1_heatmap.png", "Macro F1 (higher is better)", "viridis"),
        ("macro_mae_ms", "heuristic_mae_heatmap.png", "Macro boundary MAE in ms (lower is better)", "magma_r"),
    ):
        finite = [float(row[metric]) for row in summary_rows if np.isfinite(float(row[metric]))]
        if not finite:
            continue
        fig, axes = plt.subplots(
            1, len(percentiles), figsize=(3.6 * len(percentiles), 4.0),
            squeeze=False, constrained_layout=True
        )
        image = None
        vmin, vmax = min(finite), max(finite)
        for ax, percentile in zip(axes[0], percentiles):
            matrix = np.full((len(a_values), len(b_values)), np.nan)
            for row in summary_rows:
                if int(row["speech_percentile"]) != percentile:
                    continue
                i = a_values.index(float(row["a_lower_multiplier"]))
                j = b_values.index(float(row["b_upper_multiplier"]))
                matrix[i, j] = float(row[metric])
            image = ax.imshow(
                np.ma.masked_invalid(matrix), origin="lower", aspect="auto",
                cmap=cmap, vmin=vmin, vmax=vmax,
            )
            ax.set_xticks(range(len(b_values)), [f"{v:g}" for v in b_values])
            ax.set_yticks(range(len(a_values)), [f"{v:g}" for v in a_values])
            ax.set_xlabel("b: upper cap multiplier")
            ax.set_ylabel("a: lower floor multiplier")
            ax.set_title(f"P{percentile}")
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    if np.isfinite(matrix[i, j]):
                        normalized = (matrix[i, j] - vmin) / (vmax - vmin + 1e-12)
                        color = "white" if normalized < 0.48 else "black"
                        ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=7, color=color)
        fig.suptitle(f"Leave-one-recording-out validation: {title}")
        if image is not None:
            fig.colorbar(image, ax=list(axes[0]), shrink=0.82)
        fig.savefig(os.path.join(output_dir, filename), dpi=180, bbox_inches="tight")
        plt.close(fig)


def save_percentile_plot(summary_rows: Sequence[dict], output_path: str) -> None:
    """For each percentile, plot the best a/b result, showing sensitivity."""
    percentiles = sorted({int(row["speech_percentile"]) for row in summary_rows})
    selected = []
    for percentile in percentiles:
        options = [row for row in summary_rows if int(row["speech_percentile"]) == percentile]
        selected.append(max(options, key=lambda row: (float(row["macro_f1"]), -int(row["unmatched_predicted"] + row["missed_ground_truth"]))))
    f1 = [float(row["macro_f1"]) for row in selected]
    mae = [float(row["macro_mae_ms"]) for row in selected]
    fig, ax1 = plt.subplots(figsize=(8, 4.8))
    ax1.plot(percentiles, f1, marker="o", color="tab:blue", label="Best macro F1 for percentile")
    ax1.set_xlabel("Speech STE percentile p")
    ax1.set_ylabel("Macro F1", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_xticks(percentiles)
    ax1.grid(alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(percentiles, mae, marker="s", color="tab:orange", label="Macro boundary MAE")
    ax2.set_ylabel("Macro boundary MAE (ms)", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    for p, row, score in zip(percentiles, selected, f1):
        ax1.annotate(f"a={row['a_lower_multiplier']:g}, b={row['b_upper_multiplier']:g}",
                     (p, score), xytext=(4, 8), textcoords="offset points", fontsize=8)
    fig.suptitle("Percentile sensitivity (best a,b within each percentile)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def safe_stem(filename: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", os.path.splitext(filename)[0])


def save_fold_figure(artifact: dict, selected: dict, output_path: str) -> None:
    rec: Recording = artifact["recording"]
    predicted = artifact["predicted"]
    t_lower, t_upper, t_zcr = artifact["thresholds"]
    metrics = artifact["metrics"]
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True,
                             gridspec_kw={"height_ratios": [1.2, 0.8, 1.2]})
    energy_ax, zcr_ax, wave_ax = axes
    energy_ax.plot(rec.frame_times, rec.ste, color="darkorange", lw=1.0, label="STE")
    energy_ax.axhline(t_upper, color="crimson", ls="--", label=f"T_upper={t_upper:.4g}")
    energy_ax.axhline(t_lower, color="black", ls=":", label=f"T_lower={t_lower:.4g}")
    energy_ax.set_ylabel("STE (mean square)")
    energy_ax.legend(loc="upper right", fontsize=8)
    energy_ax.grid(alpha=0.25)

    zcr_ax.plot(rec.frame_times, rec.zcr, color="teal", lw=0.9, label="ZCR")
    zcr_ax.axhline(t_zcr, color="purple", ls="--", label=f"T_zcr={t_zcr:.4g}")
    zcr_ax.set_ylabel("ZCR")
    zcr_ax.legend(loc="upper right", fontsize=8)
    zcr_ax.grid(alpha=0.25)

    time_axis = np.arange(rec.signal.size) / rec.sr
    wave_ax.plot(time_axis, rec.signal, color="gray", alpha=0.65, lw=0.65, label="Waveform")
    for i, boundary in enumerate(rec.boundaries):
        wave_ax.axvline(boundary, color="red", lw=1.25,
                        label="Ground truth (.lab)" if i == 0 else None)
    for i, boundary in enumerate(predicted):
        wave_ax.axvline(boundary, color="blue", ls="--", lw=1.15,
                        label="Predicted boundary" if i == 0 else None)
    wave_ax.set_xlabel("Time (s)")
    wave_ax.set_ylabel("Amplitude")
    wave_ax.legend(loc="upper right", fontsize=8)
    wave_ax.grid(alpha=0.25)

    mae_text = "N/A" if metrics["mae_ms"] is None else f"{metrics['mae_ms']:.1f} ms"
    fig.suptitle(
        f"LOO validation: {rec.name} | chosen a={selected['a_lower_multiplier']:g}, "
        f"b={selected['b_upper_multiplier']:g}, P{selected['speech_percentile']} | "
        f"fold F1={metrics['f1']:.3f}, MAE={mae_text} | "
        f"T_upper rule: {artifact['active_rule']}"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def make_report(
    path: str,
    recordings: Sequence[Recording],
    selected: dict,
    baseline: dict,
    fold_rows: Sequence[dict],
) -> None:
    improvement = float(selected["macro_f1"]) - float(baseline["macro_f1"])
    lines = [
        "# Bao cao thuc nghiem chon heuristic T_upper",
        "",
        "## Muc tieu va cong thuc duoc kiem tra",
        "",
        "Voi tung fold, T_lower va T_zcr duoc uoc luong tren cac file train cua fold theo dung code cu:",
        "",
        "- `T_lower = max(0, mean(STE_sil) + 2 * std(STE_sil))`.",
        "- `T_zcr = clip(mean(ZCR_sil) + 2 * std(ZCR_sil), 0, 1)`.",
        "- `P_p = percentile(STE_speech, p)`.",
        "- `T_upper = max(a * T_lower, min(b * T_lower, P_p))`.",
        "",
        "Parametres a, b, p duoc quet tren luoi ghi trong `tune_heuristic.py`. Heuristic cu `(2.5, 4.0, 15)` duoc giu lam baseline de so sanh truc tiep.",
        "",
        "## Giao thuc danh gia",
        "",
        f"- So recording co ca WAV va LAB: **{len(recordings)}**.",
        "- Validation: leave-one-recording-out; moi lan mot recording bi giu lai, khong dung de tinh nguong.",
        "- Tap `TinHieuKiemThu` khong duoc doc trong qua trinh tuning.",
        f"- Bien du doan duoc ghep mot-mot bang ham danh gia hien co, dung sai toi da **{MATCH_TOLERANCE_MS:g} ms**.",
        "- Chon tham so theo **macro F1 cao nhat** (moi recording co trong so bang nhau); neu hoa, chon it bien thua/bo sot hon, sau do macro MAE thap hon.",
        "- Cac tham so sau khi chon van can duoc dong bang truoc khi danh gia test doc lap.",
        "",
        "## Ket qua chinh",
        "",
        "| Cau hinh | a | b | p | Macro F1 (mean +/- SD) | Pooled F1 | Macro MAE (ms) | Bien bo sot | Bien du thua |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, row in (("Duoc chon tren LOO", selected), ("Baseline cu", baseline)):
        macro_mae = row["macro_mae_ms"]
        mae_str = "N/A" if not np.isfinite(float(macro_mae)) else f"{float(macro_mae):.2f}"
        lines.append(
            f"| {label} | {row['a_lower_multiplier']:g} | {row['b_upper_multiplier']:g} | {row['speech_percentile']} | "
            f"{row['macro_f1']:.4f} +/- {row['macro_f1_std']:.4f} | {row['pooled_f1']:.4f} | {mae_str} | "
            f"{row['missed_ground_truth']} | {row['unmatched_predicted']} |"
        )
    lines += [
        "",
        f"Chenhlech macro F1 (chosen - baseline): **{improvement:+.4f}**.",
        "",
        "### Ket qua tung fold voi cau hinh duoc chon",
        "",
        "| File validation | F1 | Precision | Recall | MAE (ms) | Bien khop | Bo sot | Du thua | T_lower | T_upper | T_zcr | Quy tac T_upper |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in fold_rows:
        mae = "N/A" if row["mae_ms"] == "" else f"{float(row['mae_ms']):.2f}"
        lines.append(
            f"| {row['held_out_recording']} | {float(row['f1']):.3f} | {float(row['precision']):.3f} | {float(row['recall']):.3f} | {mae} | "
            f"{row['matched_boundaries']} | {row['missed_ground_truth']} | {row['unmatched_predicted']} | "
            f"{float(row['t_lower']):.6g} | {float(row['t_upper']):.6g} | {float(row['t_zcr']):.5f} | {row['active_t_upper_rule']} |"
        )
    lines += [
        "",
        "## Hinh va file ket qua",
        "",
        "- `heuristic_f1_heatmap.png`: macro F1 cho moi a,b va percentile.",
        "- `heuristic_mae_heatmap.png`: macro MAE cho moi a,b va percentile.",
        "- `heuristic_percentile_sensitivity.png`: do nhay theo percentile; moi diem chon a,b tot nhat trong percentile do.",
        "- `validation_fold_*.png`: STE/ZCR, nguong va bien chuan/du doan cho tung fold cua cau hinh duoc chon.",
        "- `heuristic_validation_folds.csv`: tat ca cau hinh va tung fold; `heuristic_validation_summary.csv`: tong hop.",
        "",
        "## Cach dien giai trung thuc",
        "",
        "Ket qua nay cho biet cau hinh nao co ket qua tot hon trong phep LOO tren cac recording hien co; no khong chung minh cac he so la hang so khoa hoc pho quat. Neu baseline cu thang hoac ket qua giua cac fold bien dong lon, khong nen doi ten heuristic thanh cong thuc toi uu. Can bao cao ca ket qua khong cai thien/khong on dinh va mo rong tap du lieu neu muon ket luan manh hon.",
        "",
        "Luu y: metric MAE chi tinh tren cac cap bien ghep duoc; vi vay can doc cung F1, so bien bo sot va so bien du thua. Tap test chi duoc dung sau khi chot tham so.",
    ]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", default=None, help="Thu muc WAV/LAB huan luyen")
    parser.add_argument("--output-dir", default="heuristic_results", help="Thu muc luu CSV, PNG, JSON, MD")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_dir = args.train_dir or find_data_dir("TinHieuHuanLuyen", script_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    wav_files = sorted(glob.glob(os.path.join(train_dir, "*.wav")))
    recordings = [record for path in wav_files if (record := load_recording(path)) is not None]
    if len(recordings) < 2:
        raise RuntimeError("LOO can it nhat 2 WAV co file LAB tuong ung.")

    print(f"LOO validation tren {len(recordings)} recording co nhan.")
    print("Chi doc thu muc train; khong dung tap test de chon tham so.")
    fold_rows, summary_rows, artifacts = run_grid(recordings)
    selected = choose_best(summary_rows)
    baseline = next(
        row for row in summary_rows
        if float(row["a_lower_multiplier"]) == BASELINE[0]
        and float(row["b_upper_multiplier"]) == BASELINE[1]
        and int(row["speech_percentile"]) == BASELINE[2]
    )

    # CSVs are complete audit trails, not just the winning row.
    write_csv(fold_rows, os.path.join(args.output_dir, "heuristic_validation_folds.csv"))
    write_csv(summary_rows, os.path.join(args.output_dir, "heuristic_validation_summary.csv"))
    save_heatmaps(summary_rows, args.output_dir)
    save_percentile_plot(
        summary_rows,
        os.path.join(args.output_dir, "heuristic_percentile_sensitivity.png"),
    )

    chosen_key = (
        float(selected["a_lower_multiplier"]),
        float(selected["b_upper_multiplier"]),
        int(selected["speech_percentile"]),
    )
    chosen_artifacts = artifacts[chosen_key]["artifacts"]
    selected_fold_rows = artifacts[chosen_key]["fold_rows"]
    for recording in recordings:
        artifact = chosen_artifacts[recording.name]
        figure_name = "validation_fold_" + safe_stem(recording.name) + ".png"
        save_fold_figure(artifact, selected, os.path.join(args.output_dir, figure_name))

    result_json = {
        "selection_protocol": "leave-one-recording-out on TinHieuHuanLuyen",
        "selection_criterion": "highest macro F1; tie-break fewer unmatched boundaries; then lower macro MAE",
        "test_set_used_for_tuning": False,
        "match_tolerance_ms": MATCH_TOLERANCE_MS,
        "selected": selected,
        "baseline_original": baseline,
    }
    with open(os.path.join(args.output_dir, "heuristic_best_parameters.json"), "w", encoding="utf-8") as handle:
        json.dump(result_json, handle, ensure_ascii=False, indent=2, allow_nan=True)
    make_report(
        os.path.join(args.output_dir, "heuristic_experiment_report.md"),
        recordings, selected, baseline, selected_fold_rows,
    )

    print("\nTham so chon tren validation:")
    print(f"  a={chosen_key[0]:g}, b={chosen_key[1]:g}, p={chosen_key[2]}")
    print(f"  Macro F1={selected['macro_f1']:.4f} +/- {selected['macro_f1_std']:.4f}")
    print(f"  Pooled F1={selected['pooled_f1']:.4f}; macro MAE={selected['macro_mae_ms']:.2f} ms")
    print(f"Baseline (2.5, 4.0, P15): macro F1={baseline['macro_f1']:.4f}, macro MAE={baseline['macro_mae_ms']:.2f} ms")
    print(f"Da luu CSV, bao cao va cac PNG vao: {os.path.abspath(args.output_dir)}")
    print("Day la ket qua LOO tren tap train; hay giu test rieng cho danh gia cuoi.")


if __name__ == "__main__":
    main()
