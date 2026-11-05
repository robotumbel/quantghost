"""Render the survival-vs-quantization-bit-width analysis figure for the paper."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

S = np.load("results/ANL_survbits.npz", allow_pickle=True)
labels = [str(x) for x in S["labels"]]
x = np.arange(len(labels))

fig, ax = plt.subplots(figsize=(3.5, 2.7))
ax.plot(x, S["ours"], "o-", color="#1f77b4", lw=2, ms=6, label="QuantGhost (embed at grid)")
ax.plot(x, S["naive"], "s--", color="#d62728", lw=2, ms=5, label="naive X-LSB")
ax.axhline(0.5, ls=":", color="0.6", lw=1)
ax.text(len(labels) - 1, 0.52, "chance", color="0.5", fontsize=7, ha="right")
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_xlabel("quantization bit-width"); ax.set_ylabel("payload recovery")
ax.set_ylim(0.4, 1.05); ax.legend(loc="center right", fontsize=8, framealpha=0.9)
ax.set_title("Survival across quantizer aggressiveness", fontsize=9)
plt.tight_layout()
plt.savefig("paper/fig_survival_bits.pdf", bbox_inches="tight")
plt.savefig("paper/fig_survival_bits.png", dpi=150, bbox_inches="tight")
print("wrote paper/fig_survival_bits.pdf / .png")
