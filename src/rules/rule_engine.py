class RuleEngine:
    def __init__(self, rule_config):
        self.cfg = rule_config

    def evaluate(self, temporal_buffer, baseline):
        history = temporal_buffer.get_recent()

        if not history:
            return {"status": "normal", "reasons": []}

        gaze_away = 0.0
        head_turn = 0
        downward_attention = 0.0
        mouth_active = 0
        body_missing = 0
        leaning = 0

        for item in history:
            gaze = item.get("gaze")
            head = item.get("head_pose")
            mouth = item.get("mouth")
            posture = item.get("posture")

            face_present = item.get("face_present", False)
            body_present = item.get("body_present", False)

            if head:
                yaw_diff = abs(head["yaw"] - baseline["yaw"])
                pitch_diff = head["pitch"] - baseline["pitch"]

                if yaw_diff > self.cfg.head_yaw_threshold:
                    head_turn += 1

                # Accumulate downward attention with decay instead of strict continuity
                if pitch_diff > self.cfg.head_pitch_down_threshold:
                    downward_attention += 1.0
                else:
                    downward_attention -= 0.5

                downward_attention = max(0.0, downward_attention)

            # Accumulate left/right gaze-away with decay
            if gaze and gaze.get("direction") in ["left", "right"]:
                gaze_away += 1.0
            else:
                gaze_away -= 0.5

            gaze_away = max(0.0, gaze_away)

            if mouth and mouth.get("mouth_open_ratio", 0.0) > self.cfg.mouth_open_threshold:
                mouth_active += 1

            if (not face_present) and (not body_present):
                body_missing += 1

            if posture:
                lean_diff = abs(posture["lean_score"] - baseline["lean"])
                if lean_diff > self.cfg.lean_threshold:
                    leaning += 1

        immediate_reasons = []
        support_reasons = []

        if gaze_away >= self.cfg.gaze_away_min_frames:
            immediate_reasons.append("prolonged_gaze_away")

        if head_turn >= self.cfg.head_turn_min_frames:
            immediate_reasons.append("repeated_head_turn")

        if downward_attention >= self.cfg.downward_attention_min_frames:
            immediate_reasons.append("offscreen_attention_down")

        if mouth_active >= int(self.cfg.mouth_activity_min_frames * 1.5):
            immediate_reasons.append("sustained_mouth_activity")

        if body_missing >= self.cfg.body_missing_min_frames:
            support_reasons.append("body_or_face_missing")

        if leaning >= self.cfg.leaning_min_frames:
            support_reasons.append("strong_body_lean")

        reasons = immediate_reasons + support_reasons

        if immediate_reasons:
            return {
                "status": "suspicious",
                "reasons": reasons,
            }

        return {
            "status": "normal",
            "reasons": [],
        }