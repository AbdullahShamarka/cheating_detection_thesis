from dataclasses import dataclass, field
from typing import List


@dataclass
class VideoConfig:
    input_path: str = "data/samples/cheating_test.mp4"
    use_webcam: bool = False
    webcam_index: int = 0
    resize_width: int = 960
    resize_height: int = 540
    sample_every_n_frames: int = 3
    show_window: bool = True


@dataclass
class YOLOConfig:
    model_path: str = "yolov8n.pt"
    conf_threshold: float = 0.40
    forbidden_classes: List[str] = field(default_factory=lambda: [
        "cell phone",
        "book",
        "person",
    ])


@dataclass
class RuleConfig:
    buffer_size: int = 30

    head_yaw_threshold: float = 25.0
    head_pitch_down_threshold: float = 18.0

    gaze_away_min_frames: int = 8
    head_turn_min_frames: int = 8
    mouth_activity_min_frames: int = 8
    body_missing_min_frames: int = 5
    leaning_min_frames: int = 8

    mouth_open_threshold: float = 0.08
    lean_threshold: float = 0.18

    suspicious_score_threshold: int = 1


@dataclass
class AppConfig:
    video: VideoConfig = field(default_factory=VideoConfig)
    yolo: YOLOConfig = field(default_factory=YOLOConfig)
    rules: RuleConfig = field(default_factory=RuleConfig)