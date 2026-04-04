from pathlib import Path

from src.config import AppConfig
from src.pipeline import CheatingDetectionPipeline
from src.evaluation.gt_parser import parse_gt_file
from src.evaluation.prediction_logger import PredictionLogger
from src.evaluation.interval_metrics import (
    load_prediction_intervals,
    attach_interval_reasons,
    evaluate_intervals,
)


def find_webcam_video(subject_dir: Path):
    avi_files = sorted(subject_dir.glob("*1.avi"))
    if not avi_files:
        raise FileNotFoundError(f"No webcam video (*1.avi) found in {subject_dir}")
    return avi_files[0]


def merge_gt_intervals(gt_intervals, max_gap_sec=8.0):
    """
    Merge adjacent/nearby ground-truth intervals into broader cheating episodes.
    This is useful when the dataset annotates multiple short sub-events inside
    one larger cheating episode.
    """
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


def evaluate_subject(subject_dir: str):
    subject_path = Path(subject_dir)
    gt_path = subject_path / "gt5.txt"
    video_path = find_webcam_video(subject_path)

    output_csv = Path("outputs/alerts") / f"{subject_path.name}_predictions.csv"

    config = AppConfig()
    config.video.input_path = str(video_path)
    config.video.use_webcam = False
    config.video.show_window = False

    logger = PredictionLogger(str(output_csv))
    pipeline = CheatingDetectionPipeline(config, prediction_logger=logger)
    pipeline.run()

    gt_intervals = parse_gt_file(str(gt_path))
    gt_intervals = merge_gt_intervals(gt_intervals, max_gap_sec=8.0)

    pred_intervals = load_prediction_intervals(str(output_csv))
    pred_intervals = attach_interval_reasons(pred_intervals, str(output_csv), top_k=3)

    metrics = evaluate_intervals(gt_intervals, pred_intervals)

    print(f"\nSubject: {subject_path.name}")
    print(f"Video: {video_path.name}")
    print(f"GT intervals: {len(gt_intervals)}")
    print(f"Pred intervals: {len(pred_intervals)}")
    print(metrics)

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

    return metrics


if __name__ == "__main__":
    evaluate_subject("data/raw/subject5")