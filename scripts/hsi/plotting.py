"""
Matplotlib styling shared by the experiment figures, so every plot in the
report looks the same: recessive grid, thin lines, fixed categorical order.
"""

import matplotlib

matplotlib.use("Agg")  # figures are saved to the run folder, never shown
import matplotlib.pyplot as plt  # noqa: E402

# Fixed categorical order — series k always gets SERIES[k], never cycled.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK = "#2b2b2a"
MUTED = "#8a8985"
GRID = "#e4e3df"

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "lines.linewidth": 2.0,
    "legend.frameon": False,
    "axes.prop_cycle": matplotlib.cycler(color=SERIES),
})


def save(fig, path):
    fig.savefig(path)
    plt.close(fig)
    print(f"  figure: {path}")
