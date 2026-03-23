import cv2


def draw_status(frame, detections, decision, features=None):
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        label = f'{det["label"]} {det["confidence"]:.2f}'
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            label,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
        )

    status = decision.get("status", "normal")
    reasons = ", ".join(decision.get("reasons", []))

    color = (0, 255, 0)
    if status == "suspicious":
        color = (0, 255, 255)
    elif status == "cheating":
        color = (0, 0, 255)

    cv2.putText(
        frame,
        f"Status: {status}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
    )

    cv2.putText(
        frame,
        f"Reasons: {reasons}",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )

    if features:
        head_pose = features.get("head_pose")
        gaze = features.get("gaze")
        mouth = features.get("mouth")
        posture = features.get("posture")

        y = 90

        if head_pose:
            cv2.putText(
                frame,
                f"Yaw:{head_pose['yaw']:.1f} Pitch:{head_pose['pitch']:.1f} Roll:{head_pose['roll']:.1f}",
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            y += 25

        if gaze:
            cv2.putText(
                frame,
                f"Gaze: {gaze['direction']}",
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            y += 25

        if mouth:
            cv2.putText(
                frame,
                f"Mouth ratio: {mouth['mouth_open_ratio']:.3f}",
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            y += 25

        if posture:
            cv2.putText(
                frame,
                f"Lean score: {posture['lean_score']:.3f}",
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

    return frame