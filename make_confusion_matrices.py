import matplotlib.pyplot as plt
import numpy as np

# ========================
# DATA (STRICT METRICS)
# ========================

subjects = ["S1","S2","S3","S4","S5","S6","S7"]

# Webcam BEFORE refinement
webcam = {
    "tp": [7,3,8, None,11,11,11],
    "fp": [7,5,10, None,7,8,6],
    "fn": [0,2,2, None,3,4,5]
}

# Dual AFTER refinement (STRICT)
dual = {
    "tp": [9,5,9,8,13,12,13],
    "fp": [2,2,5,2,3,5,2],
    "fn": [2,2,2,1,3,4,5]
}

# ========================
# HELPER
# ========================

def build_conf_matrix(tp, fp, fn):
    tn = 0  # not used in interval eval
    return np.array([[tp, fn],
                     [fp, tn]])

def plot_conf_matrix(matrix, title, filename):
    fig, ax = plt.subplots()
    im = ax.imshow(matrix)

    ax.set_xticks([0,1])
    ax.set_yticks([0,1])
    ax.set_xticklabels(["GT Positive","GT Negative"])
    ax.set_yticklabels(["Pred Positive","Pred Negative"])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, matrix[i, j],
                    ha="center", va="center")

    ax.set_title(title)
    plt.savefig(filename)
    plt.close()


# ========================
# GLOBAL CONFUSION MATRICES
# ========================

def aggregate(data):
    tp = sum([x for x in data["tp"] if x is not None])
    fp = sum([x for x in data["fp"] if x is not None])
    fn = sum([x for x in data["fn"] if x is not None])
    return tp, fp, fn

tp_w, fp_w, fn_w = aggregate(webcam)
tp_d, fp_d, fn_d = aggregate(dual)

plot_conf_matrix(
    build_conf_matrix(tp_w, fp_w, fn_w),
    "Webcam Only (Before Refinement)",
    "figures/conf_webcam.png"
)

plot_conf_matrix(
    build_conf_matrix(tp_d, fp_d, fn_d),
    "Dual Camera (After Refinement)",
    "figures/conf_dual.png"
)

# ========================
# PRECISION / RECALL BAR CHART
# ========================

def compute_metrics(tp, fp, fn):
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1 = 2*precision*recall/(precision+recall)
    return precision, recall, f1

prec_w, rec_w, f1_w = compute_metrics(tp_w, fp_w, fn_w)
prec_d, rec_d, f1_d = compute_metrics(tp_d, fp_d, fn_d)

labels = ["Precision","Recall","F1"]

webcam_vals = [prec_w, rec_w, f1_w]
dual_vals = [prec_d, rec_d, f1_d]

x = np.arange(len(labels))
width = 0.35

plt.figure()
plt.bar(x - width/2, webcam_vals, width, label='Webcam')
plt.bar(x + width/2, dual_vals, width, label='Dual')

plt.xticks(x, labels)
plt.ylabel("Score")
plt.title("Webcam vs Dual Camera Performance")
plt.legend()

plt.savefig("figures/comparison_bar.png")
plt.close()

# ========================
# PER SUBJECT F1
# ========================

f1_webcam = []
f1_dual = []

for i in range(len(subjects)):
    if webcam["tp"][i] is None:
        continue

    p,r,f = compute_metrics(
        webcam["tp"][i],
        webcam["fp"][i],
        webcam["fn"][i]
    )
    f1_webcam.append(f)

    p,r,f = compute_metrics(
        dual["tp"][i],
        dual["fp"][i],
        dual["fn"][i]
    )
    f1_dual.append(f)

plt.figure()
plt.plot(subjects[:len(f1_webcam)], f1_webcam, marker='o', label="Webcam")
plt.plot(subjects[:len(f1_dual)], f1_dual, marker='o', label="Dual")

plt.title("F1 Score Per Subject")
plt.xlabel("Subjects")
plt.ylabel("F1 Score")
plt.legend()

plt.savefig("figures/f1_per_subject.png")
plt.close()