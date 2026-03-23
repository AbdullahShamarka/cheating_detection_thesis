class RuleEngine:
    def __init__(self, rule_config):
        self.cfg = rule_config

    def evaluate(self, temporal_buffer):
        history = temporal_buffer.get_recent()

        if not history:
            return {"status": "normal", "reasons": []}

        gaze_away_count = 0
        head_turn_count = 0
        mouth_active_count = 0
        body_missing_count = 0
        leaning_count = 0

        for item in history:
            gaze = item.get("gaze")
            head_pose = item.get("head_pose")
            mouth = item.get("mouth")
            posture = item.get("posture")
            face_present = item.get("face_present", False)
            body_present = item.get("body_present", False)

            if gaze and gaze.get("direction") in ["left", "right"]:
                gaze_away_count += 1

            if head_pose and abs(head_pose.get("yaw", 0.0)) > self.cfg.head_yaw_threshold:
                head_turn_count += 1

            if head_pose and head_pose.get("pitch", 0.0) > self.cfg.head_pitch_down_threshold:
                head_turn_count += 1

            if mouth and mouth.get("mouth_open_ratio", 0.0) > self.cfg.mouth_open_threshold:
                mouth_active_count += 1

            if (not face_present) or (not body_present):
                body_missing_count += 1

            if posture and posture.get("lean_score", 0.0) > self.cfg.lean_threshold:
                leaning_count += 1

        reasons = []

        if gaze_away_count >= self.cfg.gaze_away_min_frames:
            reasons.append("prolonged_gaze_away")

        if head_turn_count >= self.cfg.head_turn_min_frames:
            reasons.append("repeated_head_turn")

        if mouth_active_count >= self.cfg.mouth_activity_min_frames:
            reasons.append("sustained_mouth_activity")

        if body_missing_count >= self.cfg.body_missing_min_frames:
            reasons.append("body_or_face_missing")

        if leaning_count >= self.cfg.leaning_min_frames:
            reasons.append("strong_body_lean")

        if reasons:
            return {
                "status": "suspicious",
                "reasons": reasons,
            }

        return {
            "status": "normal",
            "reasons": [],
        }