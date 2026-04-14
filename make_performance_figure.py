import matplotlib.pyplot as plt
import numpy as np

subjects = ["Subject 1", "Subject 2", "Subject 3", "Subject 4", "Subject 5", "Subject 6", "Subject 7"]

precision = [0.50, 0.375, 0.444, 0.44, 0.611, 0.579, 0.647]
recall = [1.00, 0.60, 0.80, 0.80, 0.786, 0.733, 0.688]
f1_score = [0.667, 0.462, 0.571, 0.57, 0.688, 0.647, 0.667]

x = np.arange(len(subjects))
width = 0.25

plt.figure(figsize=(12, 6))

plt.bar(x - width, precision, width, label="Precision")
plt.bar(x, recall, width, label="Recall")
plt.bar(x + width, f1_score, width, label="F1-score")

plt.xlabel("Subjects")
plt.ylabel("Score")
plt.title("Performance Across Subjects")
plt.xticks(x, subjects, rotation=20)
plt.ylim(0, 1.05)
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("figures/performance_across_subjects.png", dpi=300, bbox_inches="tight")
plt.show()