import numpy as np


class BaselineEstimator:
    def __init__(self, required_frames=30):
        self.required_frames = required_frames
        self.samples = []
        self.is_ready = False
        self.baseline = {}

    def update(self, features):
        if self.is_ready:
            return

        if features["head_pose"] is None:
            return

        self.samples.append(features)

        if len(self.samples) >= self.required_frames:
            self._compute_baseline()
            self.is_ready = True
            print("Baseline calibration complete")

    def _compute_baseline(self):
        yaw = []
        pitch = []
        gaze = []
        lean = []

        for f in self.samples:
            if f["head_pose"]:
                yaw.append(f["head_pose"]["yaw"])
                pitch.append(f["head_pose"]["pitch"])

            if f["gaze"]:
                gaze.append(f["gaze"]["horizontal_ratio"])

            if f["posture"]:
                lean.append(f["posture"]["lean_score"])

        self.baseline = {
            "yaw": float(np.mean(yaw)) if yaw else 0.0,
            "pitch": float(np.mean(pitch)) if pitch else 0.0,
            "gaze": float(np.mean(gaze)) if gaze else 0.5,
            "lean": float(np.mean(lean)) if lean else 0.0,
        }

    def get(self):
        return self.baseline