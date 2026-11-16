"""Tune the STE hysteresis heuristic with leave-one-recording-out validation.

Run from the repository root:
    python tune_heuristic.py

This script keeps the held-out training recording out of threshold estimation,
then evaluates candidate (a, b, percentile) settings on that recording. The
separate TinHieuKiemThu folder is never used for parameter selection.

Outputs (under heuristic_results/ by default):
    heuristic_validation_results.csv
    heuristic_f1_heatmap.png
    heuristic_mae_heatmap.png
    heuristic_best_validation_example.png
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")  # Export figures reliably in terminals and CI.
import matplotlib.pyplot as plt
import numpy as np

from utils_dual_threshold import (
    evaluate_metrics_detailed,
    extract_ste_zcr,
    load_wav,
    read_lab_boundaries,
    segment_dual_threshold,
)


A_VALUES = (1.5, 2.0, 2.5, 3.0)
B_VALUES = (3.0, 3.5, 4.0, 4.5, 5.0)
PERCENTILES = (5, 10, 15, 20, 25)
MATCH_TOLERANCE_MS = 200.0


@dataclass
class Recording:
    name: str
    wav_path: str
    lab_path: str
    sr: int
    signal: np.ndarray
    ste: np.ndarray
    zcr: np.ndarray
    frame_times: np.ndarray
    boundaries: np.ndarray
    segments: list
    silence_ste: np.ndarray
    speech_ste: np.ndarray
    silence_zcr: np.ndarray


def find_data_dir(name: str, script_dir: str) -> str:
    """Find a data folder without ever falling back from test to train."""
    candidates = (
        os.path.join(script_dir, name),
        os.path.join(os.path.dirname(script_dir), name),
        os.path.join(os.getcwd(), name),
    )
    for path in candidates:
        if os.path.isdir(path):
            return path
    raise FileNotFoundError(
        f"Cannot find {name!r}. Looked in: " + ", ".join(candidates)
    )


def load_recording(wav_path: str) -> Recording | None:
    lab_path = os.path.splitext(wav_path)[0] + ".lab"
    if not os.path.isfile(lab_path):
        print(f"[SKIP] No matching .lab file: {wav_path}")
        return None

    sr, signal = load_wav(wav_path)
    ste, zcr, frame_times = extract_ste_zcr(
        signal, sr, frame_ms=25, hop_ms=10
    )
    boundaries, segments = read_lab_boundaries(lab_path)
    sil_ste: List[float] = []
    speech_ste: List[float] = []
    sil_zcr: List[float] = []

    # Match the production training logic: assign each frame by its center
    # time to the first ground-truth interval that contains it.
    for index, timestamp in enumerate(frame_times):
        for start, end, label in segments:
            if start <= timestamp <= end:
                if label == "sil":
                    sil_ste.append(float(ste[index]))
                    sil_zcr.append(float(zcr[index]))
                else:
                    speech_ste.append(float(ste[index]))
                break

    return Recording(
        name=os.path.basename(wav_path),
        wav_path=wav_path,
        lab_path=lab_path,
        sr=sr,
        signal=signal,
        ste=ste,
        zcr=zcr,
        frame_times=frame_times,
        boundaries=boundaries,
        segments=segments,
        silence_ste=np.asarray(sil_ste, dtype=float),
        speech_ste=np.asarray(speech_ste, dtype=float),
        silence_zcr=np.asarray(sil_zcr, dtype=float),
    )


def concatenate(records: Sequence[Recording], attr: str) -> np.ndarray:
    arrays = [getattr(record, attr) for record in records]
    arrays = [array for array in arrays if array.size]
    if not arrays:
        return np.asarray([], dtype=float)
    return np.concatenate(arrays)


def estimate_thresholds(
    training_records: Sequence[Recording],
    a: float,
    b: float,
    percentile: float,
) -> Tuple[float, float, float]:
    """Estimate T_lower/T_upper/T_zcr using only the fold's training files."""
    silence_ste = concatenate(training_records, "silence_ste")
    speech_ste = concatenate(training_records, "speech_ste")
    silence_zcr = concatenate(training_records, "silence_zcr")
    if not silence_ste.size or not speech_ste.size or not silence_zcr.size:
        raise ValueError("Training fold lacks labeled silence/speech features.")

    t_lower = max(0.0, float(np.mean(silence_ste) + 2.0 * np.std(silence_ste)))
    speech_percentile = float(np.percentile(speech_ste, percentile))
    t_upper = max(a * t_lower, min(b * t_lower, speech_percentile))
    if t_upper <= t_lower:
        t_upper = t_lower + max(1e-12, abs(t_lower) * 1e-6)

    t_zcr = float(np.clip(
        np.mean(silence_zcr) + 2.0 * np.std(silence_zcr), 0.0, 1.0
    ))
    return t_lower, t_upper, t_zcr


def evaluate_recording(
    recording: Recording,
    t_lower: float,
    t_upper: float,
    t_zcr: float,
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


def aggregate_metrics(metrics_list: Sequence[dict]) -> dict:
    matched = sum(item["matched"] for item in metrics_list)
    predicted = sum(item["matched"] + item["unmatched_pred"] for item in metrics_list)
    ground_truth = sum(item["matched"] + item["unmatched_gt"] for item in metrics_list)
    precision = matched / predicted if predicted else 0.0
    recall = matched / ground_truth if ground_truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    errors = [item["errors_ms"] for item in metrics_list if item["errors_ms"].size]
    errors = np.concatenate(errors) if errors else np.asarray([], dtype=float)
    mae = float(np.mean(errors)) if errors.size else float("nan")
    rmse = float(np.sqrt(np.mean(errors ** 2))) if errors.size else float("nan")
    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "mae_ms": mae,
        "rmse_ms": rmse,
        "matched": matched,
        "predicted_boundaries": predicted,
        "ground_truth_boundaries": ground_truth,
        "unmatched_pred": predicted - matched,
        "unmatched_gt": ground_truth - matched,
    }


def run_grid(records: Sequence[Recording]) -> Tuple[List[dict], dict]:
    """Evaluate every candidate with leave-one-recording-out folds."""
    rows: List[dict] = []
    best_example: dict | None = None

    for a in A_VALUES:
        for b in B_VALUES:
            if b <= a:
                continue
            for percentile in PERCENTILES:
                fold_metrics = []
                fold_thresholds = []
                example = None
                for held_out in records:
                    training = [item for item in records if item is not held_out]
                    t_lower, t_upper, t_zcr = estimate_thresholds(
                        training, a, b, percentile
                    )
                    metrics, predicted = evaluate_recording(
                        held_out, t_lower, t_upper, t_zcr
                    )
                    fold_metrics.append(metrics)
                    fold_thresholds.append((t_lower, t_upper, t_zcr))
                    if example is None:
                        example = {
                            "recording": held_out,
                            "predicted": predicted,
                            "thresholds": (t_lower, t_upper, t_zcr),
                            "metrics": metrics,
                        }

                aggregate = aggregate_metrics(fold_metrics)
                threshold_array = np.asarray(fold_thresholds, dtype=float)
                row = {
                    "a_lower_multiplier": a,
                    "b_upper_multiplier": b,
                    "speech_percentile": percentile,
                    **aggregate,
                    "folds": len(records),
                    "mean_t_lower": float(np.mean(threshold_array[:, 0])),
                    "mean_t_upper": float(np.mean(threshold_array[:, 1])),
                    "mean_t_zcr": float(np.mean(threshold_array[:, 2])),
                }
                rows.append(row)
                # Choose by F1 first, then lower boundary MAE. F1 includes
                # unmatched boundaries, so a missed boundary is not hidden.
                rank = (aggregate["f1"], -aggregate["mae_ms"] if np.isfinite(aggregate["mae_ms"]) else -np.inf)
                if best_example is None or rank > best_example["rank"]:
                    best_example = {
                        "rank": rank,
                        "row": row,
                        "example": example,
                        "a": a,
                        "b": b,
                        "percentile": percentile,
                    }

    if not rows or best_example is None:
        raise RuntimeError("No valid parameter combinations were evaluated.")
    return rows, best_example


def save_csv(rows: Sequence[dict], path: str) -> None:
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_heatmaps(rows: Sequence[dict], output_dir: str) -> None:
    """Export percentile-faceted F1 and MAE heatmaps as PNG files."""
    a_values = sorted({row["a_lower_multiplier"] for row in rows})
    b_values = sorted({row["b_upper_multiplier"] for row in rows})
    percentiles = sorted({row["speech_percentile"] for row in rows})

    for metric, filename, title, cmap, reverse in (
        ("f1", "heuristic_f1_heatmap.png", "Validation F1", "viridis", False),
        ("mae_ms", "heuristic_mae_heatmap.png", "Boundary MAE (ms)", "magma_r", True),
    ):
        fig, axes = plt.subplots(
            1, len(percentiles), figsize=(4.0 * len(percentiles), 4.2),
            constrained_layout=True, squeeze=False
        )
        all_values = [row[metric] for row in rows if np.isfinite(row[metric])]
        if not all_values:
            plt.close(fig)
            continue
        vmin, vmax = min(all_values), max(all_values)
        image = None
        for axis, percentile in zip(axes[0], percentiles):
            matrix = np.full((len(a_values), len(b_values)), np.nan)
            for row in rows:
                if row["speech_percentile"] == percentile:
                    ai = a_values.index(row["a_lower_multiplier"])
                    bi = b_values.index(row["b_upper_multiplier"])
                    matrix[ai, bi] = row[metric]
            masked = np.ma.masked_invalid(matrix)
            image = axis.imshow(
                masked, aspect="auto", origin="lower", cmap=cmap,
                vmin=vmin, vmax=vmax
            )
            axis.set_xticks(range(len(b_values)), [f"{v:g}" for v in b_values])
            axis.set_yticks(range(len(a_values)), [f"{v:g}" for v in a_values])
            axis.set_xlabel("b: upper multiplier")
            axis.set_ylabel("a: lower multiplier")
            axis.set_title(f"Speech percentile = {percentile}")
            for ai in range(len(a_values)):
                for bi in range(len(b_values)):
                    if np.isfinite(matrix[ai, bi]):
                        axis.text(bi, ai, f"{matrix[ai, bi]:.3f}",
                                  ha="center", va="center", fontsize=7,
                                  color="white" if (matrix[ai, bi] - vmin) / (vmax - vmin + 1e-12) < 0.5 else "black")
        fig.suptitle(f"Leave-one-recording-out validation: {title}")
        if image is not None:
            fig.colorbar(image, ax=list(axes[0]), shrink=0.82, label=title)
        fig.savefig(os.path.join(output_dir, filename), dpi=180, bbox_inches="tight")
        plt.close(fig)


def save_best_example(best: dict, output_path: str) -> None:
    example = best["example"]
    recording: Recording = example["recording"]
    predicted = example["predicted"]
    t_lower, t_upper, t_zcr = example["thresholds"]
    metrics = example["metrics"]

    fig, (energy_ax, wave_ax) = plt.subplots(
        2, 1, figsize=(12, 7), sharex=True,
        gridspec_kw={"height_ratios": [1, 1.25]}
    )
    energy_ax.plot(recording.frame_times, recording.ste, color="darkorange", lw=1.1, label="STE")
    energy_ax.axhline(t_upper, color="crimson", ls="--", label=f"T_upper={t_upper:.3g}")
    energy_ax.axhline(t_lower, color="black", ls=":", label=f"T_lower={t_lower:.3g}")
    energy_ax.set_ylabel("STE (mean square)")
    energy_ax.set_title(
        f"Best validation example: {recording.name} | "
        f"a={best['a']:g}, b={best['b']:g}, P{best['percentile']} | "
        f"F1={best['row']['f1']:.3f}, MAE={best['row']['mae_ms']:.1f} ms"
    )
    energy_ax.grid(alpha=0.25)
    energy_ax.legend(loc="upper right")

    time_axis = np.arange(recording.signal.size) / recording.sr
    wave_ax.plot(time_axis, recording.signal, color="gray", alpha=0.65, lw=0.7, label="Waveform")
    for i, boundary in enumerate(recording.boundaries):
        wave_ax.axvline(boundary, color="red", lw=1.4,
                        label="Ground truth (.lab)" if i == 0 else None)
    for i, boundary in enumerate(predicted):
        wave_ax.axvline(boundary, color="blue", ls="--", lw=1.2,
                        label="Predicted boundary" if i == 0 else None)
    wave_ax.set_xlabel("Time (s)")
    wave_ax.set_ylabel("Amplitude")
    wave_ax.grid(alpha=0.25)
    wave_ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", default=None, help="Training recordings folder")
    parser.add_argument("--output-dir", default="heuristic_results", help="Output folder for CSV and PNG files")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_dir = args.train_dir or find_data_dir("TinHieuHuanLuyen", script_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    wav_files = sorted(glob.glob(os.path.join(train_dir, "*.wav")))
    recordings = [item for path in wav_files if (item := load_recording(path)) is not None]
    if len(recordings) < 2:
        raise RuntimeError("LOO validation requires at least two labeled training WAV files.")

    print(f"LOO validation: {len(recordings)} labeled training recordings")
    print("The separate test directory is not read during tuning.")
    rows, best = run_grid(recordings)
    csv_path = os.path.join(args.output_dir, "heuristic_validation_results.csv")
    save_csv(rows, csv_path)
    save_heatmaps(rows, args.output_dir)
    example_path = os.path.join(args.output_dir, "heuristic_best_validation_example.png")
    save_best_example(best, example_path)

    chosen = best["row"]
    print("\nBest validation parameters (F1 first, then lower MAE):")
    print(f"  a={best['a']:g}, b={best['b']:g}, speech percentile=P{best['percentile']}")
    print(f"  F1={chosen['f1']:.4f}, precision={chosen['precision']:.4f}, recall={chosen['recall']:.4f}")
    print(f"  Boundary MAE={chosen['mae_ms']:.2f} ms, RMSE={chosen['rmse_ms']:.2f} ms")
    print(f"  CSV: {csv_path}")
    print(f"  PNGs: {os.path.abspath(args.output_dir)}")
    print("Apply the selected settings only after inspecting validation results; do not tune on test labels.")


if __name__ == "__main__":
    main()
