from pathlib import Path


def mmss_to_seconds(value: str) -> int:
    value = value.strip()

    if len(value) != 4 or not value.isdigit():
        raise ValueError(f"Invalid MMSS time format: {value}")

    minutes = int(value[:2])
    seconds = int(value[2:])
    return minutes * 60 + seconds


def parse_gt_file(gt_path: str):
    intervals = []
    gt_file = Path(gt_path)

    if not gt_file.exists():
        raise FileNotFoundError(f"Ground truth file not found: {gt_path}")

    with open(gt_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    for line in lines:
        parts = line.split()

        if len(parts) < 3:
            continue

        start_raw = parts[0]
        end_raw = parts[1]
        cheat_type = parts[2]

        try:
            start_time = mmss_to_seconds(start_raw)
            end_time = mmss_to_seconds(end_raw)

            if end_time < start_time:
                continue

            intervals.append({
                "start": start_time,
                "end": end_time,
                "label": "cheating",
                "raw_type": cheat_type,
            })

        except ValueError:
            continue

    intervals.sort(key=lambda x: x["start"])
    return intervals