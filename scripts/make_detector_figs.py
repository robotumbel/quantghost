"""Render the detector ROC curves and the adaptive arms-race sweep to PDF for the paper."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = np.load("results/DET_roc.npz")
S = np.load("results/DET_sweep.npz")

fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9))

ax[0].plot([0, 1], [0, 1], "--", color="0.6", lw=1, label="chance (0.500)")
ax[0].plot(R["basic_fpr"], R["basic_tpr"], "-", color="#1f77b4", lw=2, label="basic attack (AUC 1.000)")
ax[0].plot(R["adaptive_fpr"], R["adaptive_tpr"], "-", color="#d62728", lw=2, label="adaptive attack (AUC 0.75)")
ax[0].set_xlabel("false positive rate"); ax[0].set_ylabel("true positive rate")
ax[0].set_title("(a) Detector ROC"); ax[0].legend(loc="lower right", fontsize=7.5); ax[0].set_aspect("equal")
ax[0].set_xlim(0, 1); ax[0].set_ylim(0, 1.02)

ax[1].plot(S["delta"], S["auc"], "o-", color="#2ca02c", lw=2, ms=5)
ax[1].axhline(0.5, ls="--", color="0.6", lw=1)
ax[1].set_xlabel(r"attacker sub-quantum offset width $\delta$"); ax[1].set_ylabel("detector ROC-AUC")
ax[1].set_title("(b) Arms-race sweep"); ax[1].set_ylim(0.45, 1.03)
ax[1].annotate("no $\\delta$ evades:\nnarrow offset = sharp cutoff", xy=(0.30, 1.0), xytext=(0.13, 0.62),
               fontsize=7, arrowprops=dict(arrowstyle="->", color="0.4"))

plt.tight_layout()
plt.savefig("paper/fig_detector_eval.pdf", bbox_inches="tight")
plt.savefig("paper/fig_detector_eval.png", dpi=150, bbox_inches="tight")
print("wrote paper/fig_detector_eval.pdf / .png")
