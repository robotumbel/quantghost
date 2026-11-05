"""Detector architecture figure: the sub-quantum residual steganalyzer pipeline,
with the measured ROC-AUC results annotated."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.family": "serif", "font.size": 9, "pdf.fonttype": 42,
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.03})
C = {"navy": "#1f3b73", "teal": "#2e7d6b", "red": "#b23a48", "gray": "#5a5a5a",
     "lite": "#eaf1fb", "litey": "#fdf3e3", "litg": "#eaf3ef"}

fig, ax = plt.subplots(figsize=(7.0, 2.05))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")


def box(x, w, txt, fc, ec, fs=7.4, y=52, h=30):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                 fc=fc, ec=ec, lw=1.1, zorder=2))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs, zorder=3,
            linespacing=1.3)


def arr(x1, x2, y=67, col=C["gray"]):
    ax.add_patch(FancyArrowPatch((x1, y), (x2, y), arrowstyle="-|>", mutation_scale=10,
                 color=col, lw=1.4, zorder=1))


xs = [1, 18.5, 37, 56, 73.5]
ws = [16, 16, 16.5, 14.5, 13.5]
boxes = [
    ("Model under\ntest:\nweights $W$", C["lite"], C["navy"]),
    ("Per-tensor\nresidual\n$r{=}w/s{-}[w/s]$", C["lite"], C["navy"]),
    ("Residual\nhistogram\n($B{=}40$ bins)", C["litg"], C["teal"]),
    ("Mean over\ntensors\n(feature)", C["litg"], C["teal"]),
    ("Logistic\nregression", C["litey"], C["red"]),
]
for (x, w, (t, fc, ec)) in zip(xs, ws, boxes):
    box(x, w, t, fc, ec)
for i in range(len(xs) - 1):
    arr(xs[i] + ws[i], xs[i + 1])
arr(xs[-1] + ws[-1], 99)
ax.text(99.2, 67, "benign /\nstego", ha="right", va="center", fontsize=7.6,
        color=C["red"], weight="bold")

# results strip
ax.add_patch(FancyBboxPatch((1, 4), 98, 30, boxstyle="round,pad=0,rounding_size=1.2",
             fc="#fafafa", ec="0.7", lw=0.9, zorder=0))
ax.text(3, 25, "Detection performance (5-fold CV ROC-AUC):", fontsize=7.6, weight="bold")
ax.text(5, 15.5, "basic QuantGhost", fontsize=7.4)
ax.text(5, 8, "adaptive QuantGhost", fontsize=7.4)
ax.text(40, 15.5, r"$\mathbf{1.000}$", fontsize=8.2, color=C["teal"])
ax.text(40, 8, r"$0.771$", fontsize=8.0, color=C["navy"])
ax.text(56, 11.7, "vs. mantissa-entropy baseline $0.500$ (chance)", fontsize=7.4, color="0.35")
fig.savefig("fig_detector_arch.pdf"); plt.close(fig)
print("wrote fig_detector_arch.pdf")
