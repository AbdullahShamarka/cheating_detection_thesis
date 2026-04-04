import csv
from pathlib import Path


class PredictionLogger:
    def __init__(self, output_csv: str):
        self.output_csv = Path(output_csv)
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)

        with open(self.output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "frame_index",
                "timestamp_sec",
                "status",
                "reasons",
                "gaze_direction",
                "gaze_horizontal_ratio",
                "yaw",
                "mouth_open_ratio",
            ])

    def log(self, frame_index: int, timestamp_sec: float, decision: dict, features=None):
        status = decision.get("status", "normal")
        reasons = "|".join(decision.get("reasons", []))

        gaze_direction = ""
        gaze_horizontal_ratio = ""
        yaw = ""
        mouth_open_ratio = ""

        if features:
            gaze = features.get("gaze")
            head = features.get("head_pose")
            mouth = features.get("mouth")

            if gaze:
                gaze_direction = gaze.get("direction", "")
                gaze_horizontal_ratio = gaze.get("horizontal_ratio", "")

            if head:
                yaw = head.get("yaw", "")

            if mouth:
                mouth_open_ratio = mouth.get("mouth_open_ratio", "")

        with open(self.output_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                frame_index,
                f"{timestamp_sec:.3f}",
                status,
                reasons,
                gaze_direction,
                gaze_horizontal_ratio,
                yaw,
                mouth_open_ratio,
            ])