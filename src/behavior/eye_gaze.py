class EyeGazeEstimator:
    # Simple coarse gaze estimation using eye corners and iris center
    LEFT_EYE_OUTER = 33
    LEFT_EYE_INNER = 133
    RIGHT_EYE_INNER = 362
    RIGHT_EYE_OUTER = 263

    LEFT_IRIS = [468, 469, 470, 471, 472]
    RIGHT_IRIS = [473, 474, 475, 476, 477]

    def estimate(self, face_landmarks):
        if face_landmarks is None:
            return None

        try:
            left_ratio = self._eye_horizontal_ratio(
                face_landmarks,
                self.LEFT_EYE_OUTER,
                self.LEFT_EYE_INNER,
                self.LEFT_IRIS,
            )
            right_ratio = self._eye_horizontal_ratio(
                face_landmarks,
                self.RIGHT_EYE_OUTER,
                self.RIGHT_EYE_INNER,
                self.RIGHT_IRIS,
            )

            avg_ratio = (left_ratio + right_ratio) / 2.0

            # Coarse thresholds
            if avg_ratio < 0.35:
                direction = "left"
            elif avg_ratio > 0.65:
                direction = "right"
            else:
                direction = "center"

            return {
                "direction": direction,
                "horizontal_ratio": avg_ratio,
            }
        except Exception:
            return {
                "direction": "center",
                "horizontal_ratio": 0.5,
            }

    def _eye_horizontal_ratio(self, face_landmarks, outer_idx, inner_idx, iris_indices):
        outer = face_landmarks[outer_idx]
        inner = face_landmarks[inner_idx]

        iris_x = sum(face_landmarks[i].x for i in iris_indices) / len(iris_indices)

        eye_min = min(outer.x, inner.x)
        eye_max = max(outer.x, inner.x)
        eye_width = max(eye_max - eye_min, 1e-6)

        return (iris_x - eye_min) / eye_width