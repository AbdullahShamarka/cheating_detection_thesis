from typing import Dict, Tuple

from src.video.frame_sampler import should_process_frame
from src.video.preprocessing import preprocess_frame
from src.detection.yolo_detector import YOLODetector
from src.landmarks.face_mesh_detector import FaceMeshDetector
from src.landmarks.pose_detector import PoseDetector
from src.behavior.head_pose import HeadPoseEstimator
from src.behavior.eye_gaze import EyeGazeEstimator
from src.behavior.mouth_activity import MouthActivityEstimator
from src.behavior.body_posture import BodyPostureEstimator
from src.behavior.baseline import BaselineEstimator
from src.rules.temporal_buffer import TemporalBuffer
from src.rules.rule_engine import RuleEngine
from src.rules.event_smoother import EventSmoother

from src.pipeline_glasses import GlassesFrameProcessor
from src.fusion.late_fusion_engine import LateFusionEngine
from src.multicam.dual_video_reader import DualVideoReader


class WebcamFrameProcessor:
    """
    Frame-oriented version of the existing webcam pipeline logic.

    This mirrors the current CheatingDetectionPipeline behavior so it can be
    used in offline dual-camera fusion without changing the original live
    single-camera pipeline.
    """

    def __init__(self, config):
        self.config = config

        self.yolo = YOLODetector(
            config.yolo,
            use_multicam_webcam_phone_threshold=True
        )

        self.face_detector = FaceMeshDetector()
        self.pose_detector = PoseDetector()

        self.head_pose_estimator = HeadPoseEstimator()
        self.eye_gaze_estimator = EyeGazeEstimator()
        self.mouth_estimator = MouthActivityEstimator()
        self.body_posture_estimator = BodyPostureEstimator()

        self.baseline_estimator = BaselineEstimator(required_frames=30)
        self.temporal_buffer = TemporalBuffer(maxlen=config.rules.buffer_size)
        self.rule_engine = RuleEngine(config.rules)
        self.event_smoother = EventSmoother(start_threshold=3, stop_threshold=4)

    def process_frame(self, frame) -> Tuple[Dict, Dict, list]:
        detections = self.yolo.detect(frame)
        cheating_found, cheating_reasons = self.yolo.has_forbidden_object(detections)

        features = {}

        if cheating_found:
            raw_decision = {
                "status": "cheating",
                "reasons": cheating_reasons,
            }
        else:
            face_landmarks = self.face_detector.detect(frame)
            pose_landmarks = self.pose_detector.detect(frame)

            features = {
                "head_pose": self.head_pose_estimator.estimate(face_landmarks, frame.shape) if face_landmarks else None,
                "gaze": self.eye_gaze_estimator.estimate(face_landmarks) if face_landmarks else None,
                "mouth": self.mouth_estimator.estimate(face_landmarks) if face_landmarks else None,
                "posture": self.body_posture_estimator.estimate(pose_landmarks) if pose_landmarks else None,
                "face_present": face_landmarks is not None,
                "body_present": pose_landmarks is not None,
            }

            self.temporal_buffer.update(features)

            if self.config.rules.use_baseline:
                self.baseline_estimator.update(features)

                if not self.baseline_estimator.is_ready:
                    raw_decision = {
                        "status": "calibrating",
                        "reasons": ["collecting_baseline"],
                    }
                else:
                    baseline = self.baseline_estimator.get()
                    raw_decision = self.rule_engine.evaluate(self.temporal_buffer, baseline)
            else:
                baseline = {
                    "yaw": 0.0,
                    "pitch": 0.0,
                    "gaze": 0.5,
                    "lean": 0.0,
                }
                raw_decision = self.rule_engine.evaluate(self.temporal_buffer, baseline)

        if raw_decision["status"] == "calibrating":
            final_decision = raw_decision
        elif raw_decision["status"] == "cheating":
            self.event_smoother.update("cheating")
            final_decision = raw_decision
        else:
            smoothed_status = self.event_smoother.update(raw_decision["status"])

            if smoothed_status == "suspicious":
                final_decision = {
                    "status": "suspicious",
                    "reasons": raw_decision.get("reasons", []),
                }
            else:
                final_decision = {
                    "status": "normal",
                    "reasons": [],
                }

        return final_decision, features, detections


class DualCameraFusionPipeline:
    """
    Offline dual-camera evaluation pipeline:
    - webcam stream: full behavioral analysis
    - glasses stream: YOLO-only object analysis
    - final decision: deterministic late fusion
    """

    def __init__(self, config, prediction_logger=None):
        if not config.multicam.glasses_input_path:
            raise ValueError("config.multicam.glasses_input_path must be set for dual-camera mode")

        self.config = config
        self.prediction_logger = prediction_logger

        self.reader = DualVideoReader(
            webcam_path=config.video.input_path,
            glasses_path=config.multicam.glasses_input_path,
        )

        self.webcam_processor = WebcamFrameProcessor(config)
        self.glasses_processor = GlassesFrameProcessor(config)
        self.fusion_engine = LateFusionEngine(config.fusion)

    def run(self):
        for frame_index, timestamp_sec, webcam_frame, glasses_frame in self.reader.frames():
            if not should_process_frame(
                frame_index,
                self.config.video.sample_every_n_frames
            ):
                continue

            webcam_frame = preprocess_frame(
                webcam_frame,
                self.config.video.resize_width,
                self.config.video.resize_height
            )
            glasses_frame = preprocess_frame(
                glasses_frame,
                self.config.video.resize_width,
                self.config.video.resize_height
            )

            webcam_decision, webcam_features, webcam_detections = self.webcam_processor.process_frame(webcam_frame)
            glasses_decision, glasses_detections = self.glasses_processor.process_frame(glasses_frame)

            fused_decision = self._fuse_decisions_safely(
                webcam_decision=webcam_decision,
                glasses_decision=glasses_decision,
            )

            if self.prediction_logger is not None:
                self.prediction_logger.log(
                    frame_index=frame_index,
                    timestamp_sec=timestamp_sec,
                    final_decision=fused_decision,
                    webcam_decision=webcam_decision,
                    glasses_decision=glasses_decision,
                    webcam_features=webcam_features,
                    webcam_detections=webcam_detections,
                    glasses_detections=glasses_detections,
                )

    def _fuse_decisions_safely(self, webcam_decision: Dict, glasses_decision: Dict) -> Dict:
        """
        Preserve calibration behavior:
        - if webcam is still calibrating and glasses has no cheating evidence,
          final output stays calibrating
        - if glasses has hard cheating evidence, it overrides calibration
        """
        webcam_status = webcam_decision.get("status", "normal")
        glasses_status = glasses_decision.get("status", "normal")

        if webcam_status == "calibrating" and glasses_status != "cheating":
            return {
                "status": "calibrating",
                "reasons": ["collecting_baseline"],
                "source_statuses": {
                    "webcam": webcam_status,
                    "glasses": glasses_status,
                },
            }

        return self.fusion_engine.fuse(webcam_decision, glasses_decision)