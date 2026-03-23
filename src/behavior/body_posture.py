class BodyPostureEstimator:
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    NOSE = 0

    def estimate(self, pose_landmarks):
        if pose_landmarks is None:
            return None

        left_shoulder = pose_landmarks[self.LEFT_SHOULDER]
        right_shoulder = pose_landmarks[self.RIGHT_SHOULDER]
        nose = pose_landmarks[self.NOSE]

        shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2.0
        lean_score = abs(nose.x - shoulder_center_x)

        body_in_frame = all(
            0.0 <= lm.x <= 1.0 and 0.0 <= lm.y <= 1.0
            for lm in [left_shoulder, right_shoulder, nose]
        )

        return {
            "lean_score": lean_score,
            "body_in_frame": body_in_frame,
        }