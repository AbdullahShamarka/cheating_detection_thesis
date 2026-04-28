from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# Output folder
# ============================================================

OUTPUT_DIR = Path("Figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Dual-camera BEFORE refinement (STRICT)
# ============================================================

dual_before_strict = {
    "subject1": {"tp": 5, "fp": 6, "fn": 2, "precision": 0.45454545454545453, "recall": 0.7142857142857143, "f1": 0.5555555555555556},
    "subject2": {"tp": 3, "fp": 4, "fn": 2, "precision": 0.42857142857142855, "recall": 0.6, "f1": 0.5},
    "subject3": {"tp": 8, "fp": 6, "fn": 2, "precision": 0.5714285714285714, "recall": 0.8, "f1": 0.6666666666666666},
    "subject4": {"tp": 4, "fp": 6, "fn": 1, "precision": 0.4, "recall": 0.8, "f1": 0.5333333333333333},
    "subject5": {"tp": 10, "fp": 6, "fn": 4, "precision": 0.625, "recall": 0.7142857142857143, "f1": 0.6666666666666666},
    "subject6": {"tp": 13, "fp": 8, "fn": 2, "precision": 0.6190476190476191, "recall": 0.8666666666666667, "f1": 0.7222222222222222},
    "subject7": {"tp": 9, "fp": 6, "fn": 7, "precision": 0.6, "recall": 0.5625, "f1": 0.5806451612903225},
}

# ============================================================
# Dual-camera AFTER refinement (STRICT)
# ============================================================

dual_after_strict = {
    "subject1": {"tp": 9, "fp": 2, "fn": 2, "precision": 0.8181818181818182, "recall": 0.8181818181818182, "f1": 0.8181818181818182},
    "subject2": {"tp": 5, "fp": 2, "fn": 2, "precision": 0.7142857142857143, "recall": 0.7142857142857143, "f1": 0.7142857142857143},
    "subject3": {"tp": 9, "fp": 5, "fn": 2, "precision": 0.6428571428571429, "recall": 0.8181818181818182, "f1": 0.72},
    "subject4": {"tp": 8, "fp": 2, "fn": 1, "precision": 0.8, "recall": 0.8888888888888888, "f1": 0.8421052631578948},
    "subject5": {"tp": 13, "fp": 3, "fn": 3, "precision": 0.8125, "recall": 0.8125, "f1": 0.8125},
    "subject6": {"tp": 12, "fp": 5, "fn": 4, "precision": 0.7058823529411765, "recall": 0.75, "f1": 0.7272727272727272},
    "subject7": {"tp": 13, "fp": 2, "fn": 5, "precision": 0.8666666666666667, "recall": 0.7222222222222222, "f1": 0.7878787878787877},
}

# ============================================================
# Helpers
# ============================================================

subjects = ["subject1", "subject2", "subject3", "subject4", "subject5", "subject6", "subject7"]
subject_labels = ["S1", "S2", "S3", "S4", "S5", "S6", "S7"]

def average_metric(data, key):
    values = [data[s][key] for s in subjects]
    return sum(values) / len(values)

# ============================================================
# 1. Average metrics before vs after
# ============================================================

before_avg = [
    average_metric(dual_before_strict, "precision"),
    average_metric(dual_before_strict, "recall"),
    average_metric(dual_before_strict, "f1"),
]

after_avg = [
    average_metric(dual_after_strict, "precision"),
    average_metric(dual_after_strict, "recall"),
    average_metric(dual_after_strict, "f1"),
]

metric_labels = ["Precision", "Recall", "F1-score"]
x = np.arange(len(metric_labels))
width = 0.35

plt.figure(figsize=(8, 5))
plt.bar(x - width/2, before_avg, width, label="Before refinement")
plt.bar(x + width/2, after_avg, width, label="After refinement")
plt.xticks(x, metric_labels)
plt.ylim(0, 1.0)
plt.ylabel("Score")
plt.title("Dual-Camera Performance Before vs After GT Refinement")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "dual_before_vs_after_average.png", dpi=300)
plt.close()

# ============================================================
# 2. Per-subject F1 before vs after
# ============================================================

before_f1 = [dual_before_strict[s]["f1"] for s in subjects]
after_f1 = [dual_after_strict[s]["f1"] for s in subjects]

x = np.arange(len(subjects))
width = 0.35

plt.figure(figsize=(10, 5))
plt.bar(x - width/2, before_f1, width, label="Before refinement")
plt.bar(x + width/2, after_f1, width, label="After refinement")
plt.xticks(x, subject_labels)
plt.ylim(0, 1.0)
plt.ylabel("F1-score")
plt.title("Dual-Camera F1-score Per Subject: Before vs After Refinement")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "dual_before_vs_after_f1_per_subject.png", dpi=300)
plt.close()

# ============================================================
# 3. Per-subject precision before vs after
# ============================================================

before_precision = [dual_before_strict[s]["precision"] for s in subjects]
after_precision = [dual_after_strict[s]["precision"] for s in subjects]

plt.figure(figsize=(10, 5))
plt.bar(x - width/2, before_precision, width, label="Before refinement")
plt.bar(x + width/2, after_precision, width, label="After refinement")
plt.xticks(x, subject_labels)
plt.ylim(0, 1.0)
plt.ylabel("Precision")
plt.title("Dual-Camera Precision Per Subject: Before vs After Refinement")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "dual_before_vs_after_precision_per_subject.png", dpi=300)
plt.close()

print("Saved:")
print(OUTPUT_DIR / "dual_before_vs_after_average.png")
print(OUTPUT_DIR / "dual_before_vs_after_f1_per_subject.png")
print(OUTPUT_DIR / "dual_before_vs_after_precision_per_subject.png")