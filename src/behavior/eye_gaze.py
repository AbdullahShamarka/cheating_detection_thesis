class EyeGazeEstimator:
    LEFT_EYE_OUTER = 33
    LEFT_EYE_INNER = 133
    RIGHT_EYE_INNER = 362
    RIGHT_EYE_OUTER = 263

    LEFT_EYE_UPPER = 159
    LEFT_EYE_LOWER = 145
    RIGHT_EYE_UPPER = 386
    RIGHT_EYE_LOWER = 374

    LEFT_IRIS = [468, 469, 470, 471, 472]
    RIGHT_IRIS = [473, 474, 475, 476, 477]

    def estimate(self, face_landmarks):
        if face_landmarks is None:
            return None

        try:
            left_h = self._eye_horizontal_ratio(
                face_landmarks,
                self.LEFT_EYE_OUTER,
                self.LEFT_EYE_INNER,
                self.LEFT_IRIS,
            )
            right_h = self._eye_horizontal_ratio(
                face_landmarks,
                self.RIGHT_EYE_OUTER,
                self.RIGHT_EYE_INNER,
                self.RIGHT_IRIS,
            )

            left_v = self._eye_vertical_ratio(
                face_landmarks,
                self.LEFT_EYE_UPPER,
                self.LEFT_EYE_LOWER,
                self.LEFT_IRIS,
            )
            right_v = self._eye_vertical_ratio(
                face_landmarks,
                self.RIGHT_EYE_UPPER,
                self.RIGHT_EYE_LOWER,
                self.RIGHT_IRIS,
            )

            avg_h = (left_h + right_h) / 2.0
            avg_v = (left_v + right_v) / 2.0

            direction = self._classify_direction(avg_h, avg_v)

            return {
                "direction": direction,
                "horizontal_ratio": avg_h,
                "vertical_ratio": avg_v,
            }

        except Exception:
            return {
                "direction": "center",
                "horizontal_ratio": 0.5,
                "vertical_ratio": 0.5,
            }

    def _eye_horizontal_ratio(self, face_landmarks, outer_idx, inner_idx, iris_indices):
        outer = face_landmarks[outer_idx]
        inner = face_landmarks[inner_idx]

        iris_x = sum(face_landmarks[i].x for i in iris_indices) / len(iris_indices)

        eye_min = min(outer.x, inner.x)
        eye_max = max(outer.x, inner.x)
        eye_width = max(eye_max - eye_min, 1e-6)

        return (iris_x - eye_min) / eye_width

    def _eye_vertical_ratio(self, face_landmarks, upper_idx, lower_idx, iris_indices):
        upper = face_landmarks[upper_idx]
        lower = face_landmarks[lower_idx]

        iris_y = sum(face_landmarks[i].y for i in iris_indices) / len(iris_indices)

        eye_min = min(upper.y, lower.y)
        eye_max = max(upper.y, lower.y)
        eye_height = max(eye_max - eye_min, 1e-6)

        return (iris_y - eye_min) / eye_height

    def _classify_direction(self, horizontal_ratio, vertical_ratio):
        # Horizontal first
        if horizontal_ratio < 0.40:
            return "left"
        if horizontal_ratio > 0.60:
            return "right"

        # Vertical
        # Smaller ratio means iris is closer to upper eyelid -> looking up
        if vertical_ratio < 0.35:
            return "up"
        if vertical_ratio > 0.70:
            return "down"

        return "center"