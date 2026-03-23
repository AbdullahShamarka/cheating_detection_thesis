import math


class MouthActivityEstimator:
    UPPER_LIP = 13
    LOWER_LIP = 14
    LEFT_MOUTH = 61
    RIGHT_MOUTH = 291

    def estimate(self, face_landmarks):
        if face_landmarks is None:
            return None

        upper = face_landmarks[self.UPPER_LIP]
        lower = face_landmarks[self.LOWER_LIP]
        left = face_landmarks[self.LEFT_MOUTH]
        right = face_landmarks[self.RIGHT_MOUTH]

        vertical = self._distance(upper.x, upper.y, lower.x, lower.y)
        horizontal = self._distance(left.x, left.y, right.x, right.y)

        if horizontal < 1e-6:
            ratio = 0.0
        else:
            ratio = vertical / horizontal

        return {
            "mouth_open_ratio": ratio
        }

    @staticmethod
    def _distance(x1, y1, x2, y2):
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)