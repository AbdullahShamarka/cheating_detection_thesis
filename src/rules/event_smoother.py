class EventSmoother:
    def __init__(self, start_threshold=3, stop_threshold=4):
        self.start_threshold = start_threshold
        self.stop_threshold = stop_threshold
        self.active = False
        self.positive_count = 0
        self.negative_count = 0

    def update(self, decision_status: str) -> str:
        is_positive = decision_status in {"suspicious", "cheating"}

        if is_positive:
            self.positive_count += 1
            self.negative_count = 0
        else:
            self.negative_count += 1
            self.positive_count = 0

        if not self.active and self.positive_count >= self.start_threshold:
            self.active = True

        if self.active and self.negative_count >= self.stop_threshold:
            self.active = False

        return "suspicious" if self.active else "normal"