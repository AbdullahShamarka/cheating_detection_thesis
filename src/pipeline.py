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
from src.utils.drawing import draw_status


class CheatingDetectionPipeline:
    def __init__(self, config):
        self.config = config

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

    def run(self):
        frame_index = 0

        for frame in self.video_reader.frames():
            frame_index += 1

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
                decision = {
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

                self.baseline_estimator.update(features)

                if not self.baseline_estimator.is_ready:
                    decision = {
                        "status": "calibrating",
                        "reasons": ["collecting_baseline"],
                    }
                else:
                    baseline = self.baseline_estimator.get()
                    # if features["head_pose"] is not None:
                    #     print(
                    #             f"Baseline pitch={baseline['pitch']:.2f}, "
                    #             f"Current pitch={features['head_pose']['pitch']:.2f}, "
                    #             f"Diff={features['head_pose']['pitch'] - baseline['pitch']:.2f}"
                    #     )
                    self.temporal_buffer.update(features)
                    decision = self.rule_engine.evaluate(self.temporal_buffer, baseline)

            output_frame = draw_status(frame.copy(), detections, decision, features)

            if self.config.video.show_window:
                cv2.imshow("Cheating Detection Pipeline", output_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

        cv2.destroyAllWindows()