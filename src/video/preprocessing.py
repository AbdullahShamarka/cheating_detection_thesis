import cv2


def preprocess_frame(frame, width: int, height: int):
    return cv2.resize(frame, (width, height))