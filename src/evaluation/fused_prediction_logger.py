import csv
from pathlib import Path


class FusedPredictionLogger:
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
                "webcam_status",
                "webcam_reasons",
                "glasses_status",
                "glasses_reasons",
                "gaze_direction",
                "gaze_horizontal_ratio",
                "yaw",
                "mouth_open_ratio",
                "webcam_detection_labels",
                "glasses_detection_labels",
            ])

    def log(
        self,
        frame_index: int,
        timestamp_sec: float,
        final_decision: dict,
        webcam_decision: dict,
        glasses_decision: dict,
        webcam_features=None,
        webcam_detections=None,
        glasses_detections=None,
    ):
        final_status = final_decision.get("status", "normal")
        final_reasons = "|".join(final_decision.get("reasons", []))

        webcam_status = webcam_decision.get("status", "normal")
        webcam_reasons = "|".join(webcam_decision.get("reasons", []))

        glasses_status = glasses_decision.get("status", "normal")
        glasses_reasons = "|".join(glasses_decision.get("reasons", []))

        gaze_direction = ""
        gaze_horizontal_ratio = ""
        yaw = ""
        mouth_open_ratio = ""

        if webcam_features:
            gaze = webcam_features.get("gaze")
            head = webcam_features.get("head_pose")
            mouth = webcam_features.get("mouth")

            if gaze:
                gaze_direction = gaze.get("direction", "")
                gaze_horizontal_ratio = gaze.get("horizontal_ratio", "")

            if head:
                yaw = head.get("yaw", "")

            if mouth:
                mouth_open_ratio = mouth.get("mouth_open_ratio", "")

        webcam_detection_labels = self._join_detection_labels(webcam_detections)
        glasses_detection_labels = self._join_detection_labels(glasses_detections)

        with open(self.output_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                frame_index,
                f"{timestamp_sec:.3f}",
                final_status,
                final_reasons,
                webcam_status,
                webcam_reasons,
                glasses_status,
                glasses_reasons,
                gaze_direction,
                gaze_horizontal_ratio,
                yaw,
                mouth_open_ratio,
                webcam_detection_labels,
                glasses_detection_labels,
            ])

    @staticmethod
    def _join_detection_labels(detections):
        if not detections:
            return ""

        labels = [det.get("label", "") for det in detections if det.get("label")]
        return "|".join(labels)