from typing import Dict, List


class LateFusionEngine:
    """
    Deterministic late-fusion engine for combining webcam and glasses decisions.
    """

    def __init__(self, fusion_config):
        self.cfg = fusion_config

    def fuse(self, webcam_decision: Dict, glasses_decision: Dict) -> Dict:
        webcam_decision = webcam_decision or {"status": "normal", "reasons": []}
        glasses_decision = glasses_decision or {
            "status": "normal",
            "reasons": [],
            "detected_objects": {},
        }

        score = 0
        fused_reasons: List[str] = []

        webcam_status = webcam_decision.get("status", "normal")
        detected_objects = glasses_decision.get("detected_objects", {})

        webcam_reasons = [
            f"webcam:{reason}"
            for reason in webcam_decision.get("reasons", [])
        ]
        glasses_reasons = [
            f"glasses:{reason}"
            for reason in glasses_decision.get("reasons", [])
        ]

        if webcam_status == "suspicious":
            score += self.cfg.webcam_suspicious_weight
            fused_reasons.extend(webcam_reasons)

        elif webcam_status == "cheating":
            score += self.cfg.webcam_cheating_weight
            fused_reasons.extend(webcam_reasons)

        if detected_objects.get("cell phone", False):
            score += self.cfg.glasses_phone_weight

        if detected_objects.get("book", False):
            score += self.cfg.glasses_book_weight

        if detected_objects.get("extra_person", False):
            score += self.cfg.glasses_extra_person_weight

        fused_reasons.extend(glasses_reasons)
        fused_reasons = self._deduplicate_preserve_order(fused_reasons)

        if score >= self.cfg.cheating_threshold:
            final_status = "cheating"
        elif score >= self.cfg.suspicious_threshold:
            final_status = "suspicious"
        else:
            final_status = "normal"

        if final_status == "normal":
            fused_reasons = []

        return {
            "status": final_status,
            "reasons": fused_reasons,
            "source_statuses": {
                "webcam": webcam_status,
                "glasses": glasses_decision.get("status", "normal"),
            },
            "scores": {
                "total": score,
            },
        }

    @staticmethod
    def _deduplicate_preserve_order(values: List[str]) -> List[str]:
        seen = set()
        result = []

        for value in values:
            if value not in seen:
                seen.add(value)
                result.append(value)

        return result