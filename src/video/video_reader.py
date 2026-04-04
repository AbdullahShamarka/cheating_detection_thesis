import cv2


class VideoReader:
    def __init__(self, video_config):
        self.video_config = video_config
        self.source = (
            video_config.webcam_index
            if video_config.use_webcam
            else video_config.input_path
        )

    def frames(self):
        cap = cv2.VideoCapture(self.source)

        if not cap.isOpened():
            raise RuntimeError(f"Could not open video source: {self.source}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 0:
            fps = 30.0

        try:
            frame_index = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                timestamp_sec = frame_index / fps
                yield frame_index, timestamp_sec, frame
                frame_index += 1
        finally:
            cap.release()