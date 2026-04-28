from dataclasses import dataclass, field
from typing import List


@dataclass
class VideoConfig:
    input_path: str = "data/samples/cheating_test.mp4"
    use_webcam: bool = False
    webcam_index: int = 0
    resize_width: int = 960
    resize_height: int = 540
    sample_every_n_frames: int = 1
    show_window: bool = True


@dataclass
class YOLOConfig:
    model_path: str = "yolov8n.pt"
    conf_threshold: float = 0.55

    # Default phone threshold (single-camera / generic use)
    phone_conf_threshold: float = 0.80

    # Stricter threshold for webcam branch in dual-camera mode
    multicam_webcam_phone_conf_threshold: float = 0.93

    # More permissive threshold for glasses branch
    glasses_phone_conf_threshold: float = 0.72

    forbidden_classes: List[str] = field(default_factory=lambda: [
        "cell phone",
        "book",
        "person",
    ])


@dataclass
class RuleConfig:
    buffer_size: int = 30
    use_baseline: bool = True

    head_yaw_threshold: float = 25.0
    head_pitch_down_threshold: float = 16.0

    gaze_away_min_frames: int = 10
    downward_attention_min_frames: int = 6
    head_turn_min_frames: int = 12
    mouth_activity_min_frames: int = 14
    body_missing_min_frames: int = 15
    leaning_min_frames: int = 8
    short_glance_min_events: int = 4

    mouth_open_threshold: float = 0.11
    lean_threshold: float = 0.18


@dataclass
class MultiCameraConfig:
    enabled: bool = False
    glasses_input_path: str = ""


@dataclass
class GlassesConfig:
    enabled: bool = False
    confirmation_window_size: int = 5
    min_confirmed_frames: int = 2


@dataclass
class FusionConfig:
    webcam_suspicious_weight: int = 1
    webcam_cheating_weight: int = 3

    glasses_phone_weight: int = 3
    glasses_book_weight: int = 3
    glasses_extra_person_weight: int = 3

    suspicious_threshold: int = 1
    cheating_threshold: int = 3


@dataclass
class AppConfig:
    video: VideoConfig = field(default_factory=VideoConfig)
    yolo: YOLOConfig = field(default_factory=YOLOConfig)
    rules: RuleConfig = field(default_factory=RuleConfig)
    multicam: MultiCameraConfig = field(default_factory=MultiCameraConfig)
    glasses: GlassesConfig = field(default_factory=GlassesConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)