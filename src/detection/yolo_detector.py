from ultralytics import YOLO


class YOLODetector:
    def __init__(self, yolo_config):
        self.model = YOLO(yolo_config.model_path)
        self.conf_threshold = yolo_config.conf_threshold
        self.forbidden_classes = set(yolo_config.forbidden_classes)

    def detect(self, frame):
        results = self.model.predict(frame, conf=self.conf_threshold, verbose=False)
        detections = []

        for result in results:
            names = result.names
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                detections.append({
                    "class_id": class_id,
                    "label": names[class_id],
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2],
                })

        return detections

    def has_forbidden_object(self, detections):
        person_count = 0

        for det in detections:
            label = det["label"]

            if label == "person":
                person_count += 1
            elif label in self.forbidden_classes:
                return True, [f"forbidden_object:{label}"]

        # extra person rule
        if person_count > 1:
            return True, ["forbidden_object:extra_person"]

        return False, []