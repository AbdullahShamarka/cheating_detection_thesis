from collections import deque
from typing import Dict, List, Tuple

from src.video.video_reader import VideoReader
from src.video.frame_sampler import should_process_frame
from src.video.preprocessing import preprocess_frame
from src.detection.yolo_detector import YOLODetector


class GlassesFrameProcessor:
    """
    Lightweight object-only processor for glasses-mounted camera input.

    This pipeline intentionally avoids:
    - face mesh
    - pose estimation
    - gaze estimation
    - behavioral rules

    It only uses YOLO detections and a small temporal confirmation buffer
    to stabilize object detections.
    """

    def __init__(self, config):
        self.config = config
        self.yolo = YOLODetector(
            config.yolo,
            use_glasses_phone_threshold=True
        )
        self.phone_conf_threshold = config.yolo.glasses_phone_conf_threshold
        self.window_size = config.glasses.confirmation_window_size
        self.min_confirmed_frames = config.glasses.min_confirmed_frames

        self.history = deque(maxlen=self.window_size)

    def process_frame(self, frame) -> Tuple[Dict, List[Dict]]:
        detections = self.yolo.detect(frame)
        frame_flags, frame_reasons = self._extract_detection_flags(detections)

        self.history.append(frame_flags)
        confirmed_objects = self._get_confirmed_objects()

        reasons = []
        if confirmed_objects["cell phone"]:
            reasons.append("forbidden_object:cell phone")
        if confirmed_objects["book"]:
            reasons.append("forbidden_object:book")
        if confirmed_objects["extra_person"]:
            reasons.append("forbidden_object:extra_person")

        status = "cheating" if reasons else "normal"

        decision = {
            "status": status,
            "reasons": reasons,
            "detected_objects": confirmed_objects,
            "raw_frame_reasons": frame_reasons,
        }
        return decision, detections

    def _extract_detection_flags(self, detections: List[Dict]) -> Tuple[Dict[str, bool], List[str]]:
        person_count = 0
        phone_found = False
        book_found = False
        reasons = []

        for det in detections:
            label = det["label"]
            conf = det["confidence"]

            if label == "person":
                person_count += 1
                continue

            if label == "cell phone":
                if conf >= self.phone_conf_threshold:
                    phone_found = True
                    reasons.append("forbidden_object:cell phone")
                continue

            if label == "book":
                book_found = True
                reasons.append("forbidden_object:book")

        extra_person = person_count > 1
        if extra_person:
            reasons.append("forbidden_object:extra_person")

        flags = {
            "cell phone": phone_found,
            "book": book_found,
            "extra_person": extra_person,
        }
        return flags, reasons

    def _get_confirmed_objects(self) -> Dict[str, bool]:
        counts = {
            "cell phone": 0,
            "book": 0,
            "extra_person": 0,
        }

        for item in self.history:
            for key in counts:
                if item.get(key, False):
                    counts[key] += 1

        return {
            key: (counts[key] >= self.min_confirmed_frames)
            for key in counts
        }


class GlassesDetectionPipeline:
    """
    Offline glasses-video pipeline for prerecorded evaluation.
    """

    def __init__(self, config, prediction_logger=None):
        self.config = config
        self.prediction_logger = prediction_logger
        self.video_reader = VideoReader(config.video)
        self.processor = GlassesFrameProcessor(config)

    def run(self):
        for frame_index, timestamp_sec, frame in self.video_reader.frames():
            if not should_process_frame(
                frame_index,
                self.config.video.sample_every_n_frames
            ):
                continue

            frame = preprocess_frame(
                frame,
                self.config.video.resize_width,
                self.config.video.resize_height
            )

            decision, detections = self.processor.process_frame(frame)

            if self.prediction_logger is not None:
                self.prediction_logger.log(
                    frame_index=frame_index,
                    timestamp_sec=timestamp_sec,
                    decision=decision,
                    detections=detections,
                )