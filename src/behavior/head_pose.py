import cv2
import numpy as np


class HeadPoseEstimator:
    # Approximate MediaPipe landmark indices
    NOSE_TIP = 1
    CHIN = 152
    LEFT_EYE_OUTER = 33
    RIGHT_EYE_OUTER = 263
    LEFT_MOUTH = 61
    RIGHT_MOUTH = 291

    def estimate(self, face_landmarks, frame_shape):
        if face_landmarks is None:
            return None

        h, w = frame_shape[:2]

        image_points = np.array([
            self._to_xy(face_landmarks[self.NOSE_TIP], w, h),
            self._to_xy(face_landmarks[self.CHIN], w, h),
            self._to_xy(face_landmarks[self.LEFT_EYE_OUTER], w, h),
            self._to_xy(face_landmarks[self.RIGHT_EYE_OUTER], w, h),
            self._to_xy(face_landmarks[self.LEFT_MOUTH], w, h),
            self._to_xy(face_landmarks[self.RIGHT_MOUTH], w, h),
        ], dtype="double")

        # Generic 3D face model points
        model_points = np.array([
            (0.0, 0.0, 0.0),
            (0.0, -63.6, -12.5),
            (-43.3, 32.7, -26.0),
            (43.3, 32.7, -26.0),
            (-28.9, -28.9, -24.1),
            (28.9, -28.9, -24.1),
        ])

        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")

        dist_coeffs = np.zeros((4, 1))

        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return None

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        proj_matrix = np.hstack((rotation_matrix, translation_vector))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)

        pitch = float(euler_angles[0, 0])
        yaw = float(euler_angles[1, 0])
        roll = float(euler_angles[2, 0])

        return {
            "yaw": yaw,
            "pitch": pitch,
            "roll": roll,
        }

    @staticmethod
    def _to_xy(landmark, width, height):
        return [landmark.x * width, landmark.y * height]