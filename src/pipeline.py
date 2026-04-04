import cv2

from src.video.video_reader import VideoReader
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
from src.utils.drawing import draw_status


class CheatingDetectionPipeline:
    def __init__(self, config, prediction_logger=None):
        self.config = config
        self.prediction_logger = prediction_logger

        self.video_reader = VideoReader(config.video)
        self.yolo = YOLODetector(config.yolo)

        self.face_detector = FaceMeshDetector()
        self.pose_detector = PoseDetector()

        self.head_pose_estimator = HeadPoseEstimator()
        self.eye_gaze_estimator = EyeGazeEstimator()
        self.mouth_estimator = MouthActivityEstimator()
        self.body_posture_estimator = BodyPostureEstimator()

        self.baseline_estimator = BaselineEstimator(required_frames=30)
        self.temporal_buffer = TemporalBuffer(maxlen=config.rules.buffer_size)
        self.rule_engine = RuleEngine(config.rules)

        # Event smoother for more stable final outputs
        self.event_smoother = EventSmoother(start_threshold=3, stop_threshold=4)

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

            # Apply event smoothing only after calibration
            if raw_decision["status"] == "calibrating":
                final_decision = raw_decision
            elif raw_decision["status"] == "cheating":
                # Keep explicit cheating alerts strong and immediate
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

            if self.prediction_logger is not None:
                self.prediction_logger.log(frame_index, timestamp_sec, final_decision, features)

            output_frame = draw_status(frame.copy(), detections, final_decision, features)

            if self.config.video.show_window:
                cv2.imshow("Cheating Detection Pipeline", output_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

        cv2.destroyAllWindows()