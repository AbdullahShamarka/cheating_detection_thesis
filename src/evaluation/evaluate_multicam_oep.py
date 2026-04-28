from pathlib import Path

from src.config import AppConfig
from src.multicam.dual_pipeline import DualCameraFusionPipeline
from src.evaluation.gt_parser import parse_gt_file
from src.evaluation.fused_prediction_logger import FusedPredictionLogger
from src.evaluation.interval_metrics import (
    load_prediction_intervals,
    attach_interval_reasons,
    evaluate_intervals,
    evaluate_intervals_many_to_one,
    explain_interval_matches,
)


def find_webcam_video(subject_dir: Path):
    avi_files = sorted(subject_dir.glob("*1.avi"))
    if not avi_files:
        raise FileNotFoundError(f"No webcam video (*1.avi) found in {subject_dir}")
    return avi_files[0]


def find_glasses_video(subject_dir: Path):
    avi_files = sorted(subject_dir.glob("*2.avi"))
    if not avi_files:
        raise FileNotFoundError(f"No glasses video (*2.avi) found in {subject_dir}")
    return avi_files[0]


def merge_gt_intervals(gt_intervals, max_gap_sec=8.0):
    if not gt_intervals:
        return []

    gt_intervals = sorted(gt_intervals, key=lambda x: x["start"])
    merged = [gt_intervals[0].copy()]

    for nxt in gt_intervals[1:]:
        current = merged[-1]

        if nxt["start"] - current["end"] <= max_gap_sec:
            current["end"] = max(current["end"], nxt["end"])

            current_type = str(current.get("raw_type", ""))
            next_type = str(nxt.get("raw_type", ""))

            if next_type and next_type not in current_type.split("+"):
                current["raw_type"] = f"{current_type}+{next_type}" if current_type else next_type
        else:
            merged.append(nxt.copy())

    return merged


def evaluate_subject_multicam(subject_dir: str):
    subject_path = Path(subject_dir)
    gt_path = subject_path / "gt7.txt"

    webcam_video_path = find_webcam_video(subject_path)
    glasses_video_path = find_glasses_video(subject_path)

    output_csv = Path("outputs/alerts") / f"{subject_path.name}_multicam_predictions.csv"

    config = AppConfig()
    config.video.input_path = str(webcam_video_path)
    config.video.use_webcam = False
    config.video.show_window = False

    config.multicam.enabled = True
    config.multicam.glasses_input_path = str(glasses_video_path)

    config.glasses.enabled = True
    config.glasses.confirmation_window_size = 5
    config.glasses.min_confirmed_frames = 2

    logger = FusedPredictionLogger(str(output_csv))
    pipeline = DualCameraFusionPipeline(config, prediction_logger=logger)
    pipeline.run()

    gt_intervals = parse_gt_file(str(gt_path))
    gt_intervals = merge_gt_intervals(gt_intervals, max_gap_sec=8.0)

    pred_intervals = load_prediction_intervals(str(output_csv))
    pred_intervals = attach_interval_reasons(pred_intervals, str(output_csv), top_k=3)

    strict_metrics = evaluate_intervals(gt_intervals, pred_intervals)
    coverage_metrics = evaluate_intervals_many_to_one(gt_intervals, pred_intervals)
    debug_info = explain_interval_matches(gt_intervals, pred_intervals)

    print(f"\nSubject: {subject_path.name}")
    print(f"Webcam video: {webcam_video_path.name}")
    print(f"Glasses video: {glasses_video_path.name}")
    print(f"GT intervals: {len(gt_intervals)}")
    print(f"Pred intervals: {len(pred_intervals)}")

    print("\nStrict metrics (one-to-one):")
    print(strict_metrics)

    print("\nCoverage metrics (many-to-one):")
    print(coverage_metrics)

    print("\nGT intervals:")
    for x in gt_intervals:
        print(x)

    print("\nPredicted intervals:")
    for x in pred_intervals:
        print({
            "start": x["start"],
            "end": x["end"],
            "label": x["label"],
            "reasons": x["reasons"],
        })

    print("\nUnmatched predicted intervals under strict matching (counted as FP):")
    for item in debug_info["unmatched_predictions"]:
        pred = item["pred"]
        print({
            "pred_index": item["pred_index"],
            "start": pred["start"],
            "end": pred["end"],
            "label": pred["label"],
            "reasons": pred.get("reasons", []),
        })

    print("\nUnmatched GT intervals under strict matching (counted as FN):")
    for item in debug_info["unmatched_ground_truth"]:
        gt = item["gt"]
        print({
            "gt_index": item["gt_index"],
            "start": gt["start"],
            "end": gt["end"],
            "label": gt["label"],
            "raw_type": gt.get("raw_type", ""),
        })

    return {
        "strict_metrics": strict_metrics,
        "coverage_metrics": coverage_metrics,
        "debug_info": debug_info,
    }


if __name__ == "__main__":
    evaluate_subject_multicam("data/raw/subject7")