import cv2


class DualVideoReader:
    """
    Simple paired video reader for offline dual-camera evaluation.

    Current policy:
    - frames are paired by index
    - timestamps are derived from each stream separately
    - yielded timestamp is min(webcam_ts, glasses_ts) for conservative alignment
    - reading stops when either stream ends
    """

    def __init__(self, webcam_path: str, glasses_path: str):
        self.webcam_path = webcam_path
        self.glasses_path = glasses_path

    def frames(self):
        webcam_cap = cv2.VideoCapture(self.webcam_path)
        glasses_cap = cv2.VideoCapture(self.glasses_path)

        if not webcam_cap.isOpened():
            raise RuntimeError(f"Could not open webcam video: {self.webcam_path}")

        if not glasses_cap.isOpened():
            webcam_cap.release()
            raise RuntimeError(f"Could not open glasses video: {self.glasses_path}")

        webcam_fps = webcam_cap.get(cv2.CAP_PROP_FPS)
        glasses_fps = glasses_cap.get(cv2.CAP_PROP_FPS)

        if webcam_fps is None or webcam_fps <= 0:
            webcam_fps = 30.0

        if glasses_fps is None or glasses_fps <= 0:
            glasses_fps = 30.0

        frame_index = 0

        try:
            while True:
                ret_webcam, webcam_frame = webcam_cap.read()
                ret_glasses, glasses_frame = glasses_cap.read()

                if not ret_webcam or not ret_glasses:
                    break

                webcam_ts = frame_index / webcam_fps
                glasses_ts = frame_index / glasses_fps
                fused_ts = min(webcam_ts, glasses_ts)

                yield frame_index, fused_ts, webcam_frame, glasses_frame
                frame_index += 1

        finally:
            webcam_cap.release()
            glasses_cap.release()