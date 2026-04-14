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
        mouth_active = 0.0
        body_missing = 0
        leaning = 0

        # NEW: count short glance events
        short_glance_events = 0
        prev_gaze_dir = None

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

                # Slightly stronger downward accumulation
                if pitch_diff > self.cfg.head_pitch_down_threshold:
                    downward_attention += 1.0
                else:
                    downward_attention -= 0.3

                downward_attention = max(0.0, downward_attention)

            # Left/right sustained gaze-away
            if gaze and gaze.get("direction") in ["left", "right"]:
                gaze_away += 1.0
            else:
                gaze_away -= 0.4

            gaze_away = max(0.0, gaze_away)

            # NEW: short glance event counter
            if gaze and gaze.get("direction") in ["left", "right", "up", "down"]:
                current_dir = gaze.get("direction")
                if prev_gaze_dir is not None and current_dir != prev_gaze_dir:
                    short_glance_events += 1
                prev_gaze_dir = current_dir

            # Mouth with strict decay
            if mouth and mouth.get("mouth_open_ratio", 0.0) > self.cfg.mouth_open_threshold:
                mouth_active += 1.0
            else:
                mouth_active -= 2.0

            mouth_active = max(0.0, mouth_active)

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

        # Slightly easier downward trigger
        if downward_attention >= self.cfg.downward_attention_min_frames:
            immediate_reasons.append("offscreen_attention_down")

        # Keep mouth strict
        if mouth_active >= self.cfg.mouth_activity_min_frames:
            immediate_reasons.append("sustained_mouth_activity")

        # NEW: short glance pattern is support only
        if short_glance_events >= self.cfg.short_glance_min_events:
            support_reasons.append("short_glance_pattern")

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