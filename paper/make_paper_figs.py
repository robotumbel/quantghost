"""Generate all paper figures (vector PDF) from the real experimental numbers.
Clean, non-overlapping, consistent Elsevier-style design.
Figures: fig_setup (full experimental pipeline), fig_method, fig_trilemma,
         fig_detectors, fig_prune."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
# muted professional palette
C = {"navy": "#1f3b73", "steel": "#3a6ea5", "teal": "#2e7d6b",
     "red": "#b23a48", "amber": "#c98a2b", "gray": "#5a5a5a", "lite": "#e9eef5",
     "lite2": "#eaf3f0", "lite3": "#f6ece0"}


def box(ax, x, y, w, h, text, fc, ec, fs=8.0, tc="black", lw=1.1, round=0.06):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.0,rounding_size={round}",
                       fc=fc, ec=ec, lw=lw, zorder=2)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc, zorder=3, linespacing=1.25)


def arrow(ax, x1, y1, x2, y2, color=C["gray"], lw=1.5, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                 mutation_scale=11, color=color, lw=lw, zorder=1,
                 shrinkA=0, shrinkB=0))


# ============================ Fig 0: experimental pipeline ============================
fig, ax = plt.subplots(figsize=(7.2, 4.3))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

# --- ATTACKER lane (boxes 62-89, label 92) ---
ax.text(1, 92, "ATTACKER", fontsize=8.6, weight="bold", color=C["navy"])
box(ax, 3, 77, 17, 11, "Pre-trained\nFP32 model", C["lite"], C["navy"], 7.5)
box(ax, 3, 63, 17, 11, "Payload\n(encrypt + ECC)", C["lite"], C["navy"], 7.5)
box(ax, 28, 66, 23, 15, "Quantization-aware\nstealthy embed (Alg. 1)\n+ redundancy",
    C["lite"], C["navy"], 7.3)
box(ax, 60, 69, 18, 12, "Stego model\non model hub", C["lite"], C["navy"], 7.5)
arrow(ax, 20, 82, 28, 76); arrow(ax, 20, 68, 28, 71)
arrow(ax, 51, 73.5, 60, 74.5)
arrow(ax, 69, 69, 69, 56, C["teal"])   # hub -> deployment

# --- VICTIM lane (label 57, pills 41-53) ---
ax.text(1, 57.5, "VICTIM DEPLOYMENT", fontsize=8.6, weight="bold", color=C["teal"])
dep = [("INT8\nquant.", 4), ("Pruning\n(unstr./struct.)", 28), ("Fine-tune\n(same / OOD)", 52),
       ("NeuPerm\n(permutation)", 76)]
for t, x in dep:
    box(ax, x, 41, 20, 12, t, C["lite2"], C["teal"], 7.3)
arrow(ax, 50, 41, 50, 32, C["red"])    # deployment -> detection

# --- DEFENDER + RECOVERY lane (label 33, boxes 17-30) ---
ax.text(1, 33.5, "DETECTION & RECOVERY", fontsize=8.6, weight="bold", color=C["red"])
box(ax, 8, 17, 24, 13, "Model X-Ray OSL\n+ byte-level detectors", C["lite3"], C["red"], 7.3)
box(ax, 41, 17, 18, 13, "Extract +\nmajority vote", C["lite3"], C["red"], 7.5)
box(ax, 67, 17, 21, 13, "Payload\nrecovered", C["lite3"], C["red"], 7.5)
ax.plot([84.0, 85.3, 87.6], [27.2, 26.0, 28.6], color=C["teal"], lw=2.3, zorder=4,
        solid_capstyle="round")
arrow(ax, 32, 23.5, 41, 23.5, C["red"]); arrow(ax, 59, 23.5, 67, 23.5, C["red"])

# --- experimental matrix strip (bottom) ---
ax.add_patch(FancyBboxPatch((2, 1), 96, 11, boxstyle="round,pad=0,rounding_size=0.4",
             fc="#fafafa", ec="0.7", lw=0.9, zorder=0))
ax.text(4, 8.8, "Models:", fontsize=7, weight="bold"); ax.text(16, 8.8, "ResNet-50 / -56 / -18, MLP-IDS, GHRP STL-10 zoo (119k models)", fontsize=7)
ax.text(4, 5.6, "Datasets:", fontsize=7, weight="bold"); ax.text(16, 5.6, "ImageNet, CIFAR-10, CICIoMT, CICIoT, STL-10", fontsize=7)
ax.text(4, 2.4, "Metrics:", fontsize=7, weight="bold"); ax.text(16, 2.4, "bit-survival, top-1 / F1, mantissa-byte entropy, detector centroid / AUC", fontsize=7)
fig.savefig("fig_setup.pdf"); plt.close(fig)

# ============================ Fig 1: method schematic ============================
fig, ax = plt.subplots(figsize=(6.6, 2.1))
s = 1.0; codes = np.arange(0, 7)
for q in codes:
    ax.axvspan((q - 0.5) * s, (q + 0.5) * s, color=(C["lite"] if q % 2 else C["lite3"]), zorder=0)
    ax.axvline(q * s, color="0.75", lw=0.8, zorder=1)
ax.axhline(0, color="0.35", lw=1.1, zorder=1)
rng = np.random.default_rng(3)
for q in codes[1:-1]:
    off = rng.uniform(-0.4, 0.4)
    ax.plot([q * s, q * s + off], [0, 0], color=C["steel"], lw=0.9, alpha=0.5, zorder=2)
    ax.plot(q * s + off, 0, "o", ms=7, color=C["navy"], zorder=3)
ax.plot(6 * s, 0, "s", ms=8, color=C["red"], zorder=3)
ax.annotate("pinned weight $|q|{=}q_{\\max}$\n(fixes victim scale $s$)", (6 * s, 0), (5.0, 0.78),
            fontsize=7.2, ha="center", color=C["red"],
            arrowprops=dict(arrowstyle="->", color=C["red"], lw=0.9))
ax.annotate("weight at random sub-quantum\noffset $u\\!\\sim\\!\\mathcal{U}(-0.49,0.49)$\n(natural mantissa)",
            (2 + 0.25, 0), (1.6, 0.80), fontsize=7.2, ha="center", color=C["navy"],
            arrowprops=dict(arrowstyle="->", color=C["navy"], lw=0.9))
for q in codes:
    ax.text(q * s, -0.42, f"$q{{=}}{q}$", ha="center", fontsize=6.8, color="0.45")
ax.text(3.0, -0.78, "bin $[(q{-}0.5)s,(q{+}0.5)s)\\!\\to\\!$ code $q$;   payload bit $=$ LSB$(|q|)$",
        fontsize=7.4, ha="center")
ax.set_xlim(-0.7, 6.8); ax.set_ylim(-0.95, 1.15); ax.set_xticks([]); ax.set_yticks([])
for sp in ax.spines.values():
    sp.set_visible(False)
fig.savefig("fig_method.pdf"); plt.close(fig)

# ============================ Fig 2: pruning ============================
ratios = [30, 50, 70, 90]
un_rec = [1.000, 0.977, 0.535, 0.535]; un_acc = [93.82, 92.05, 78.25, 19.88]
st_rec = [1.000, 0.535, 0.535, 0.535]; st_acc = [10.00, 10.89, 10.00, 10.00]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.8, 2.5), sharey=True)
for ax, rec, acc, ttl in [(a1, un_rec, un_acc, "Unstructured pruning"),
                          (a2, st_rec, st_acc, "Structured pruning")]:
    ax.plot(ratios, rec, "o-", color=C["navy"], lw=1.8, ms=5, label="payload recovery")
    ax.plot(ratios, [x / 100 for x in acc], "s--", color=C["teal"], lw=1.8, ms=5, label="top-1 accuracy")
    ax.axhline(0.5, color="0.6", lw=0.8, ls=":")
    ax.set_title(ttl, fontsize=9); ax.set_xlabel("prune ratio (\\%)"); ax.set_ylim(0, 1.06)
    ax.set_xticks(ratios)
a1.set_ylabel("rate"); a1.legend(fontsize=7.2, loc="lower left", framealpha=0.95)
a2.annotate("model destroyed\n(acc $=$ chance)", (70, 0.10), (62, 0.30), fontsize=7,
            color=C["teal"], ha="center", arrowprops=dict(arrowstyle="->", color=C["teal"], lw=0.8))
fig.tight_layout(); fig.savefig("fig_prune.pdf"); plt.close(fig)

# ============================ Fig 3: detectors ============================
dets = ["OSL\n(diverse)", "MalConv", "hist-KL", "byte\nautocorr", "byte\nentropy", "value\ndist"]
naive = [0.81, 1.00, 1.00, 1.00, 1.00, 0.50]
ours = [0.50, 0.51, 0.46, 0.49, 1.00, 0.84]
x = np.arange(len(dets)); w = 0.38
fig, ax = plt.subplots(figsize=(6.8, 2.5))
ax.bar(x - w / 2, naive, w, label="naive X-LSB", color=C["red"])
ax.bar(x + w / 2, ours, w, label="QuantGhost (ours)", color=C["navy"])
ax.axhline(0.5, color="0.4", lw=0.8, ls=":")
ax.text(len(dets) - 0.45, 0.53, "chance", fontsize=6.8, color="0.4", ha="right")
ax.set_xticks(x); ax.set_xticklabels(dets, fontsize=7.4)
ax.set_ylabel("detection accuracy"); ax.set_ylim(0, 1.18)
ax.legend(fontsize=7.4, loc="upper center", ncol=2, framealpha=0.95, bbox_to_anchor=(0.5, 1.16))
fig.tight_layout(); fig.savefig("fig_detectors.pdf"); plt.close(fig)

# ============================ Fig 4: trilemma ============================
fig, ax = plt.subplots(figsize=(5.6, 4.3))
V = {"H": np.array([0.0, 1.05]), "P": np.array([-1.0, -0.62]), "S": np.array([1.0, -0.62])}
tri = np.array([V["H"], V["P"], V["S"], V["H"]])
ax.fill(tri[:, 0], tri[:, 1], color=C["lite"], alpha=0.5, zorder=0)
ax.plot(tri[:, 0], tri[:, 1], color=C["gray"], lw=1.5, zorder=1)
ax.text(V["H"][0], V["H"][1] + 0.13, "High-capacity\n$\\Theta(P)$ bits", ha="center", va="bottom",
        fontsize=9, weight="bold", color=C["navy"])
ax.text(V["P"][0] - 0.05, V["P"][1] - 0.10, "Permutation-\ninvariant", ha="center", va="top",
        fontsize=9, weight="bold", color=C["teal"])
ax.text(V["S"][0] + 0.05, V["S"][1] - 0.10, "Stealthy +\nquant-robust", ha="center", va="top",
        fontsize=9, weight="bold", color=C["red"])
mid = {"HS": (V["H"] + V["S"]) / 2, "PS": (V["P"] + V["S"]) / 2, "HP": (V["H"] + V["P"]) / 2}
ax.plot(*mid["HS"], "o", ms=10, color=C["navy"], zorder=3)
ax.annotate("per-weight code-LSB\n(QuantGhost)\n$1.0\\!\\to\\!0.51$ vs NeuPerm",
            mid["HS"], (1.30, 0.55), fontsize=7.4, color=C["navy"], ha="left",
            arrowprops=dict(arrowstyle="->", color=C["navy"], lw=0.8))
ax.plot(*mid["PS"], "o", ms=10, color=C["teal"], zorder=3)
ax.annotate("spatial-tap channel\n$\\sum k_h k_w$ bits (196)\nsurvives NeuPerm",
            mid["PS"], (-0.15, -1.32), fontsize=7.4, color=C["teal"], ha="center",
            arrowprops=dict(arrowstyle="->", color=C["teal"], lw=0.8))
ax.plot(*mid["HP"], "o", ms=10, color=C["red"], zorder=3)
ax.annotate("value-histogram coding\n(not stealthy /\nquant-robust)",
            mid["HP"], (-2.35, 0.55), fontsize=7.4, color=C["red"], ha="left",
            arrowprops=dict(arrowstyle="->", color=C["red"], lw=0.8))
ax.text(0, -0.08, "pick at\nmost 2", ha="center", va="center", fontsize=9.5, style="italic", color="0.4")
ax.set_xlim(-2.45, 2.55); ax.set_ylim(-1.55, 1.55); ax.axis("off")
fig.savefig("fig_trilemma.pdf"); plt.close(fig)

print("wrote fig_setup.pdf fig_method.pdf fig_prune.pdf fig_detectors.pdf fig_trilemma.pdf")
