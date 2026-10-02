import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from scipy import stats

COLORS = {"hist": "#2a78d6", "param": "#eb6834", "t": "#1baf7a", "mc": "#eda100"}
LABELS = {"hist": "Historical", "param": "Gaussian", "t": "Student-t", "mc": "Monte Carlo"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "font.size": 9, "legend.frameon": False,
})


def plot_var_vs_loss(bt, path, alpha=0.99, methods=("hist", "param")):
    """Realised daily loss against the VaR forecast, one panel per method."""
    fig, axes = plt.subplots(len(methods), 1, figsize=(10, 3.2 * len(methods)), sharex=True, sharey=True)
    for ax, m in zip(np.atleast_1d(axes), methods):
        g = bt[(bt.method == m) & np.isclose(bt.alpha, alpha)]
        loss = -g.pnl / 1e3
        ax.plot(g.date, loss, color="#a8a7a1", lw=0.6, label="Realised loss")
        ax.plot(g.date, g["var"] / 1e3, color=COLORS[m], lw=1.5, label=f"{alpha:.0%} VaR forecast")
        exc = g[g.exception]
        ax.scatter(exc.date, -exc.pnl / 1e3, s=14, color=INK, zorder=3,
                   label=f"Exception ({len(exc)})")
        ax.set_title(f"{LABELS[m]}, {alpha:.0%} 1-day VaR", loc="left", fontsize=10)
        ax.set_ylabel("USD thousands")
        ax.legend(loc="upper left", ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_rolling_exceptions(bt, path, alpha=0.99):
    """Exceptions over a trailing 250 days, with the Basel zone boundaries."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axhspan(-0.5, 4.5, color="#1baf7a", alpha=0.07, lw=0)
    ax.axhspan(4.5, 9.5, color="#eda100", alpha=0.10, lw=0)
    ax.axhspan(9.5, 40, color="#e34948", alpha=0.07, lw=0)
    for y, zone in ((2, "green"), (7, "yellow"), (12, "red")):
        ax.text(0.005, y, zone, transform=ax.get_yaxis_transform(), ha="left", va="center", color=MUTED)

    top = 0
    for m in LABELS:
        g = bt[(bt.method == m) & np.isclose(bt.alpha, alpha)].sort_values("date")
        rolling = g.exception.astype(int).rolling(250).sum()
        top = max(top, rolling.max())
        # MC sits on top of the Gaussian line almost everywhere; dash it so both show
        ax.plot(g.date, rolling, color=COLORS[m], lw=1.5, ls="--" if m == "mc" else "-", label=LABELS[m])

    ax.set_ylim(-0.5, top + 2)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel("Exceptions, trailing 250 days")
    ax.set_title(f"Basel traffic light, {alpha:.0%} VaR", loc="left", fontsize=10)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_return_tails(r_p, path):
    """Portfolio return histogram against a fitted normal, log density to show the tails."""
    r = np.asarray(r_p) * 100
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(r, bins=120, density=True, color="#2a78d6", alpha=0.85, label="Empirical")
    x = np.linspace(r.min(), r.max(), 400)
    ax.plot(x, stats.norm.pdf(x, r.mean(), r.std()), color=INK, lw=1.2, label="Normal, same mean/sd")
    ax.set_yscale("log")
    ax.set_ylim(1e-4, None)
    ax.set_xlabel("Daily portfolio log return (%)")
    ax.set_ylabel("Density (log scale)")
    ax.set_title("Portfolio returns vs a fitted normal", loc="left", fontsize=10)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
