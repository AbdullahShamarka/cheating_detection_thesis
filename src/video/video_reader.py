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

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                yield frame
        finally:
            cap.release()