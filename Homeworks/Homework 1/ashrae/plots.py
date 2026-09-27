"""Plots used by more than one notebook."""


def plot_curves(ax, curves, title, **plot_kw):
    """One line per entry of `curves` ({label: value per epoch}), on an axis of epochs."""
    n_epochs = max(len(values) for values in curves.values())
    for label, values in curves.items():
        ax.plot(range(1, len(values) + 1), values, marker="o", label=label, **plot_kw)
    ax.set(title=title, xlabel="epoch", xticks=range(1, n_epochs + 1))
    ax.legend(fontsize=8)
