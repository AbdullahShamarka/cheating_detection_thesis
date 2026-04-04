import csv
from collections import Counter


def merge_intervals(intervals, max_gap_sec=8.0):
    """
    Merge intervals that overlap or are separated by at most max_gap_sec.
    """
    if not intervals:
        return []

    intervals = sorted(intervals, key=lambda x: x["start"])
    merged = [intervals[0].copy()]

    for nxt in intervals[1:]:
        current = merged[-1]

        if nxt["start"] - current["end"] <= max_gap_sec:
            current["end"] = max(current["end"], nxt["end"])
        else:
            merged.append(nxt.copy())

    return merged


def load_prediction_rows(csv_path: str):
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            reasons = row["reasons"].split("|") if row["reasons"] else []
            reasons = [r for r in reasons if r]

            rows.append({
                "timestamp_sec": float(row["timestamp_sec"]),
                "status": row["status"],
                "reasons": reasons,
            })
    return rows


def load_prediction_intervals(
    csv_path: str,
    positive_statuses=None,
    max_gap_sec=8.0,
    pre_pad_sec=2.0,
    post_pad_sec=2.0,
    min_duration_sec=1.5,
):
    if positive_statuses is None:
        positive_statuses = {"suspicious", "cheating"}

    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "timestamp_sec": float(row["timestamp_sec"]),
                "status": row["status"],
            })

    positive_times = [
        r["timestamp_sec"]
        for r in rows
        if r["status"] in positive_statuses
    ]

    if not positive_times:
        return []

    raw_intervals = []
    start = positive_times[0]
    prev = positive_times[0]

    for t in positive_times[1:]:
        if t - prev <= max_gap_sec:
            prev = t
        else:
            raw_intervals.append({
                "start": start,
                "end": prev,
                "label": "positive",
            })
            start = t
            prev = t

    raw_intervals.append({
        "start": start,
        "end": prev,
        "label": "positive",
    })

    padded_intervals = []
    for interval in raw_intervals:
        padded_intervals.append({
            "start": max(0.0, interval["start"] - pre_pad_sec),
            "end": interval["end"] + post_pad_sec,
            "label": "positive",
        })

    merged_intervals = merge_intervals(padded_intervals, max_gap_sec=max_gap_sec)

    final_intervals = []
    for interval in merged_intervals:
        duration = interval["end"] - interval["start"]
        if duration >= min_duration_sec:
            final_intervals.append(interval)

    return final_intervals


def attach_interval_reasons(pred_intervals, csv_path: str, positive_statuses=None, top_k=3):
    """
    Attach the dominant reasons to each already-created predicted interval,
    without changing the evaluation logic.
    """
    if positive_statuses is None:
        positive_statuses = {"suspicious", "cheating"}

    rows = load_prediction_rows(csv_path)
    enriched = []

    for interval in pred_intervals:
        counter = Counter()

        for row in rows:
            t = row["timestamp_sec"]
            if interval["start"] <= t <= interval["end"] and row["status"] in positive_statuses:
                counter.update(row["reasons"])

        enriched.append({
            **interval,
            "reasons": [reason for reason, _ in counter.most_common(top_k)],
            "reason_counts": dict(counter),
        })

    return enriched


def interval_overlap(a_start, a_end, b_start, b_end, tolerance_sec=2.0):
    """
    Overlap with boundary tolerance.
    Expands both intervals slightly to avoid unfair misses due to small timing shifts.
    """
    a_start -= tolerance_sec
    a_end += tolerance_sec
    b_start -= tolerance_sec
    b_end += tolerance_sec

    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def evaluate_intervals(gt_intervals, pred_intervals, tolerance_sec=2.0):
    """
    One-to-one interval matching with boundary tolerance:
    - each GT interval can match at most one predicted interval
    - each predicted interval can match at most one GT interval
    """
    matched_gt = set()
    matched_pred = set()

    for gi, gt in enumerate(gt_intervals):
        for pi, pred in enumerate(pred_intervals):
            if pi in matched_pred:
                continue

            overlap = interval_overlap(
                gt["start"], gt["end"],
                pred["start"], pred["end"],
                tolerance_sec=tolerance_sec,
            )

            if overlap > 0:
                matched_gt.add(gi)
                matched_pred.add(pi)
                break

    tp = len(matched_gt)
    fp = len(pred_intervals) - len(matched_pred)
    fn = len(gt_intervals) - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }