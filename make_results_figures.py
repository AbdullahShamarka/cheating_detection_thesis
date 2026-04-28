from pathlib import Path
import math
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# Configuration
# ============================================================

OUTPUT_DIR = Path("Figures/results_generated")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Helper functions
# ============================================================

def sorted_subjects(data_dict):
    def key_fn(name):
        # subject1, subject2, ...
        digits = "".join(ch for ch in name if ch.isdigit())
        return int(digits) if digits else 999
    return sorted(data_dict.keys(), key=key_fn)


def compute_averages(metrics_by_subject):
    vals = {
        "precision": [],
        "recall": [],
        "f1": [],
    }
    for subj, m in metrics_by_subject.items():
        if m is None:
            continue
        vals["precision"].append(m["precision"])
        vals["recall"].append(m["recall"])
        vals["f1"].append(m["f1"])

    avg = {}
    for k, arr in vals.items():
        avg[k] = sum(arr) / len(arr) if arr else None
    return avg


def compute_total_counts(metrics_by_subject):
    total_tp = 0
    total_fp = 0
    total_fn = 0
    for subj, m in metrics_by_subject.items():
        if m is None:
            continue
        total_tp += m["tp"]
        total_fp += m["fp"]
        total_fn += m["fn"]
    return {"tp": total_tp, "fp": total_fp, "fn": total_fn}


def save_figure(fig, filename):
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def filter_available_subjects(metric_dict):
    return {k: v for k, v in metric_dict.items() if v is not None}


# ============================================================
# Data
# ============================================================

# ------------------------------------------------------------
# 1) Dual-camera AFTER GT refinement
# Filled from your latest message.
#
# NOTE:
# - subject2 metrics in your pasted text appear misformatted.
# - I left subject2 as None. Fill it once you confirm the exact values.
# ------------------------------------------------------------

dual_after_strict = {
    "subject1": {"tp": 9, "fp": 2, "fn": 2, "precision": 0.8181818181818182, "recall": 0.8181818181818182, "f1": 0.8181818181818182},
    "subject2": {"tp": 5, "fp": 2, "fn": 2, "precision": 0.7142857142857143, "recall": 0.7142857142857143, "f1": 0.7142857142857143},
    "subject3": {"tp": 9, "fp": 5, "fn": 2, "precision": 0.6428571428571429, "recall": 0.8181818181818182, "f1": 0.72},
    "subject4": {"tp": 8, "fp": 2, "fn": 1, "precision": 0.8, "recall": 0.8888888888888888, "f1": 0.8421052631578948},
    "subject5": {"tp": 13, "fp": 3, "fn": 3, "precision": 0.8125, "recall": 0.8125, "f1": 0.8125},
    "subject6": {"tp": 12, "fp": 5, "fn": 4, "precision": 0.7058823529411765, "recall": 0.75, "f1": 0.7272727272727272},
    "subject7": {"tp": 13, "fp": 2, "fn": 5, "precision": 0.8666666666666667, "recall": 0.7222222222222222, "f1": 0.7878787878787877},
}

dual_after_coverage = {
    "subject1": {"tp": 10, "fp": 1, "fn": 1, "precision": 0.9090909090909091, "recall": 0.9090909090909091, "f1": 0.9090909090909091},
    "subject2": {"tp": 5, "fp": 1, "fn": 2, "precision": 0.8333333333333334, "recall": 0.7142857142857143, "f1": 0.7692307692307692},
    "subject3": {"tp": 9, "fp": 2, "fn": 2, "precision": 0.8181818181818182, "recall": 0.8181818181818182, "f1": 0.8181818181818182},
    "subject4": {"tp": 8, "fp": 2, "fn": 1, "precision": 0.8, "recall": 0.8888888888888888, "f1": 0.8421052631578948},
    "subject5": {"tp": 13, "fp": 1, "fn": 3, "precision": 0.9285714285714286, "recall": 0.8125, "f1": 0.8666666666666666},
    "subject6": {"tp": 13, "fp": 5, "fn": 3, "precision": 0.7222222222222222, "recall": 0.8125, "f1": 0.7647058823529411},
    "subject7": {"tp": 17, "fp": 1, "fn": 1, "precision": 0.9444444444444444, "recall": 0.9444444444444444, "f1": 0.9444444444444444},
}

# ------------------------------------------------------------
# 2) Dual-camera BEFORE GT refinement
# Fill these from the earlier numbers you already have.
# ------------------------------------------------------------

dual_before_strict = {
    "subject1": {"tp": 5, "fp": 6, "fn": 2, "precision": 0.45454545454545453, "recall": 0.7142857142857143, "f1": 0.5555555555555556},
    "subject2": {"tp": 3, "fp": 4, "fn": 2, "precision": 0.42857142857142855, "recall": 0.6, "f1": 0.5},
    "subject3": {"tp": 8, "fp": 6, "fn": 2, "precision": 0.5714285714285714, "recall": 0.8, "f1": 0.6666666666666666},
    "subject4": {"tp": 4, "fp": 6, "fn": 1, "precision": 0.4, "recall": 0.8, "f1": 0.5333333333333333},
    "subject5": {"tp": 10, "fp": 6, "fn": 4, "precision": 0.625, "recall": 0.7142857142857143, "f1": 0.6666666666666666},
    "subject6": {"tp": 13, "fp": 8, "fn": 2, "precision": 0.6190476190476191, "recall": 0.8666666666666667, "f1": 0.7222222222222222},
    "subject7": {"tp": 9, "fp": 6, "fn": 7, "precision": 0.6, "recall": 0.5625, "f1": 0.5806451612903225},
}

dual_before_coverage = {
    "subject1": {"tp": 6, "fp": 6, "fn": 1, "precision": 0.5, "recall": 0.8571428571428571, "f1": 0.631578947368421},
    "subject2": {"tp": 3, "fp": 3, "fn": 2, "precision": 0.5, "recall": 0.6, "f1": 0.5454545454545454},
    "subject3": {"tp": 8, "fp": 3, "fn": 2, "precision": 0.7272727272727273, "recall": 0.8, "f1": 0.761904761904762},  # reconstructed
    "subject4": {"tp": 4, "fp": 6, "fn": 1, "precision": 0.4, "recall": 0.8, "f1": 0.5333333333333333},
    "subject5": {"tp": 10, "fp": 5, "fn": 4, "precision": 0.6666666666666666, "recall": 0.7142857142857143, "f1": 0.689655172413793},
    "subject6": {"tp": 13, "fp": 8, "fn": 2, "precision": 0.6190476190476191, "recall": 0.8666666666666667, "f1": 0.7222222222222222},
    "subject7": {"tp": 13, "fp": 5, "fn": 3, "precision": 0.7222222222222222, "recall": 0.8125, "f1": 0.7647058823529411},
}

# ------------------------------------------------------------
# 3) Webcam-only BEFORE GT refinement
# Fill from your screenshot / existing single-camera results.
# ------------------------------------------------------------

webcam_before_strict = {
    "subject1": {"tp": 7, "fp": 7, "fn": 0, "precision": 0.500, "recall": 1.000, "f1": 0.667},
    "subject2": {"tp": 3, "fp": 5, "fn": 2, "precision": 0.375, "recall": 0.600, "f1": 0.462},
    "subject3": {"tp": 8, "fp": 10, "fn": 2, "precision": 0.444, "recall": 0.800, "f1": 0.571},
    "subject4": None,  # TODO: fill when available
    "subject5": {"tp": 11, "fp": 7, "fn": 3, "precision": 0.611, "recall": 0.786, "f1": 0.688},
    "subject6": {"tp": 11, "fp": 8, "fn": 4, "precision": 0.579, "recall": 0.733, "f1": 0.647},
    "subject7": {"tp": 11, "fp": 6, "fn": 5, "precision": 0.647, "recall": 0.688, "f1": 0.667},
}

webcam_before_coverage = {
    "subject1": None,
    "subject2": None,
    "subject3": None,
    "subject4": None,
    "subject5": None,
    "subject6": None,
    "subject7": None,
}

# ------------------------------------------------------------
# 4) Webcam-only AFTER GT refinement
# You said you will add these later.
# ------------------------------------------------------------

webcam_after_strict = {
    "subject1": None,
    "subject2": None,
    "subject3": None,
    "subject4": None,
    "subject5": None,
    "subject6": None,
    "subject7": None,
}

webcam_after_coverage = {
    "subject1": None,
    "subject2": None,
    "subject3": None,
    "subject4": None,
    "subject5": None,
    "subject6": None,
    "subject7": None,
}


# ============================================================
# Plotting functions
# ============================================================

def plot_metric_bars_per_subject(metrics_by_subject, title, filename):
    data = filter_available_subjects(metrics_by_subject)
    subjects = sorted_subjects(data)

    precision = [data[s]["precision"] for s in subjects]
    recall = [data[s]["recall"] for s in subjects]
    f1 = [data[s]["f1"] for s in subjects]

    x = np.arange(len(subjects))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, precision, width, label="Precision")
    ax.bar(x, recall, width, label="Recall")
    ax.bar(x + width, f1, width, label="F1-score")

    ax.set_title(title)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(subjects, rotation=45)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    save_figure(fig, filename)


def plot_strict_vs_coverage_f1(strict_metrics, coverage_metrics, title, filename):
    subjects = sorted_subjects({k: v for k, v in strict_metrics.items() if v is not None and coverage_metrics.get(k) is not None})
    strict_f1 = [strict_metrics[s]["f1"] for s in subjects]
    coverage_f1 = [coverage_metrics[s]["f1"] for s in subjects]

    x = np.arange(len(subjects))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width / 2, strict_f1, width, label="Strict F1")
    ax.bar(x + width / 2, coverage_f1, width, label="Coverage F1")

    ax.set_title(title)
    ax.set_ylabel("F1-score")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(subjects, rotation=45)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    save_figure(fig, filename)


def plot_counts_per_subject(metrics_by_subject, title, filename):
    data = filter_available_subjects(metrics_by_subject)
    subjects = sorted_subjects(data)

    tp = [data[s]["tp"] for s in subjects]
    fp = [data[s]["fp"] for s in subjects]
    fn = [data[s]["fn"] for s in subjects]

    x = np.arange(len(subjects))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, tp, width, label="TP")
    ax.bar(x, fp, width, label="FP")
    ax.bar(x + width, fn, width, label="FN")

    ax.set_title(title)
    ax.set_ylabel("Count")
    ax.set_xticks(x)
    ax.set_xticklabels(subjects, rotation=45)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    save_figure(fig, filename)


def plot_average_metric_comparison(configs, metric_type, title, filename):
    labels = []
    values = []

    for label, metrics_by_subject in configs:
        avg = compute_averages(filter_available_subjects(metrics_by_subject))
        if avg[metric_type] is not None:
            labels.append(label)
            values.append(avg[metric_type])

    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x, values)

    ax.set_title(title)
    ax.set_ylabel(metric_type.capitalize())
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20)
    ax.grid(axis="y", alpha=0.3)

    for i, v in enumerate(values):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center")

    save_figure(fig, filename)


def plot_error_matrix_heatmap(metrics_by_subject, title, filename):
    totals = compute_total_counts(filter_available_subjects(metrics_by_subject))
    matrix = np.array([[totals["tp"], totals["fp"], totals["fn"]]])

    fig, ax = plt.subplots(figsize=(8, 2.8))
    im = ax.imshow(matrix, aspect="auto")

    ax.set_title(title)
    ax.set_yticks([0])
    ax.set_yticklabels(["Counts"])
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["TP", "FP", "FN"])

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax, shrink=0.8)
    save_figure(fig, filename)


def plot_improvement_before_after(before_metrics, after_metrics, metric_type, title, filename):
    subjects = sorted_subjects({
        k: v for k, v in before_metrics.items()
        if v is not None and after_metrics.get(k) is not None
    })

    before_vals = [before_metrics[s][metric_type] for s in subjects]
    after_vals = [after_metrics[s][metric_type] for s in subjects]

    x = np.arange(len(subjects))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width / 2, before_vals, width, label="Before refinement")
    ax.bar(x + width / 2, after_vals, width, label="After refinement")

    ax.set_title(title)
    ax.set_ylabel(metric_type.capitalize())
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(subjects, rotation=45)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    save_figure(fig, filename)


def plot_subjectwise_lines(metrics_by_subject, title, filename):
    data = filter_available_subjects(metrics_by_subject)
    subjects = sorted_subjects(data)

    precision = [data[s]["precision"] for s in subjects]
    recall = [data[s]["recall"] for s in subjects]
    f1 = [data[s]["f1"] for s in subjects]

    x = np.arange(len(subjects))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x, precision, marker="o", label="Precision")
    ax.plot(x, recall, marker="o", label="Recall")
    ax.plot(x, f1, marker="o", label="F1-score")

    ax.set_title(title)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(subjects, rotation=45)
    ax.legend()
    ax.grid(alpha=0.3)

    save_figure(fig, filename)


# ============================================================
# Generate figures for dual-camera AFTER refinement
# ============================================================

plot_metric_bars_per_subject(
    dual_after_strict,
    "Dual-camera after GT refinement: strict precision, recall, and F1 by subject",
    "dual_after_strict_metrics_bar.png"
)

plot_metric_bars_per_subject(
    dual_after_coverage,
    "Dual-camera after GT refinement: coverage precision, recall, and F1 by subject",
    "dual_after_coverage_metrics_bar.png"
)

plot_subjectwise_lines(
    dual_after_strict,
    "Dual-camera after GT refinement: strict metrics across subjects",
    "dual_after_strict_metrics_line.png"
)

plot_strict_vs_coverage_f1(
    dual_after_strict,
    dual_after_coverage,
    "Dual-camera after GT refinement: strict vs coverage F1 by subject",
    "dual_after_strict_vs_coverage_f1.png"
)

plot_counts_per_subject(
    dual_after_strict,
    "Dual-camera after GT refinement: TP, FP, and FN by subject (strict)",
    "dual_after_strict_counts_bar.png"
)

plot_counts_per_subject(
    dual_after_coverage,
    "Dual-camera after GT refinement: TP, FP, and FN by subject (coverage)",
    "dual_after_coverage_counts_bar.png"
)

plot_error_matrix_heatmap(
    dual_after_strict,
    "Dual-camera after GT refinement: total strict TP / FP / FN",
    "dual_after_strict_error_matrix.png"
)

plot_error_matrix_heatmap(
    dual_after_coverage,
    "Dual-camera after GT refinement: total coverage TP / FP / FN",
    "dual_after_coverage_error_matrix.png"
)

# ============================================================
# Comparison figures across configurations
# These will auto-skip empty configurations.
# ============================================================

strict_config_list = [
    ("Webcam before", webcam_before_strict),
    ("Dual before", dual_before_strict),
    ("Dual after", dual_after_strict),
    ("Webcam after", webcam_after_strict),
]

coverage_config_list = [
    ("Webcam before", webcam_before_coverage),
    ("Dual before", dual_before_coverage),
    ("Dual after", dual_after_coverage),
    ("Webcam after", webcam_after_coverage),
]

plot_average_metric_comparison(
    strict_config_list,
    "precision",
    "Average strict precision across configurations",
    "average_strict_precision_comparison.png"
)

plot_average_metric_comparison(
    strict_config_list,
    "recall",
    "Average strict recall across configurations",
    "average_strict_recall_comparison.png"
)

plot_average_metric_comparison(
    strict_config_list,
    "f1",
    "Average strict F1-score across configurations",
    "average_strict_f1_comparison.png"
)

plot_average_metric_comparison(
    coverage_config_list,
    "f1",
    "Average coverage F1-score across configurations",
    "average_coverage_f1_comparison.png"
)

# ============================================================
# Before vs after refinement (dual-camera)
# These will work once you fill dual_before_*.
# ============================================================

plot_improvement_before_after(
    dual_before_strict,
    dual_after_strict,
    "f1",
    "Dual-camera strict F1 before vs after GT refinement",
    "dual_strict_f1_before_vs_after.png"
)

plot_improvement_before_after(
    dual_before_coverage,
    dual_after_coverage,
    "f1",
    "Dual-camera coverage F1 before vs after GT refinement",
    "dual_coverage_f1_before_vs_after.png"
)

print(f"All figures saved to: {OUTPUT_DIR.resolve()}")