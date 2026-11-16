"""
Figures: an annotated overlay and a size-distribution plot.

Design notes
------------
Particle diameter is a continuous magnitude, so it is encoded with a single
hue stepped light-to-dark rather than a rainbow: with a rainbow the reader
has to consult the legend to order two colours, whereas a one-hue ramp is
ordered by inspection. The ramp starts at a mid step rather than its lightest
because these contours are drawn over a bright micrograph, and the near-white
end of a sequential ramp is designed to recede into the surface -- which is
the right behaviour on a chart and the wrong one on top of an image.

The histogram carries two marks (observed bars, fitted curve), so it gets a
legend; identity is never left to colour alone.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize

# Sequential blue ramp, steps 200-700.
_SEQ_BLUE = ["#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281"]
SIZE_CMAP = LinearSegmentedColormap.from_list("nm_blue", _SEQ_BLUE)

SERIES_1 = "#2a78d6"     # observed
SERIES_2 = "#eb6834"     # fitted
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8982"
SURFACE = "#fcfcfb"


def _style(ax) -> None:
    """Make the frame recede so the data carries the attention."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9, width=0.8)
    ax.grid(True, color=INK_MUTED, alpha=0.18, linewidth=0.7)
    ax.set_axisbelow(True)


def overlay_figure(
    image: np.ndarray,
    labels: np.ndarray,
    diameters: np.ndarray | None = None,
    unit: str = "nm",
    title: str = "Detected particles",
    show_index: bool = False,
    dpi: int = 130,
):
    """Draw particle outlines over the micrograph, coloured by diameter."""
    from skimage.measure import find_contours, regionprops

    h, w = labels.shape
    fig, ax = plt.subplots(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax.imshow(image, cmap="gray", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    regions = regionprops(labels)
    if diameters is not None and len(diameters) == len(regions):
        vals = np.asarray(diameters, dtype=float)
    else:
        vals = np.array([np.sqrt(4 * r.area / np.pi) for r in regions], dtype=float)
        unit = "px"

    finite = vals[np.isfinite(vals)]
    norm = Normalize(
        vmin=float(np.percentile(finite, 2)) if finite.size else 0.0,
        vmax=float(np.percentile(finite, 98)) if finite.size else 1.0,
    )

    for reg, val in zip(regions, vals):
        mask = (labels == reg.label)
        colour = SIZE_CMAP(norm(val)) if np.isfinite(val) else SERIES_1
        for contour in find_contours(mask.astype(float), 0.5):
            ax.plot(contour[:, 1], contour[:, 0], linewidth=1.4, color=colour)
        if show_index:
            ax.text(reg.centroid[1], reg.centroid[0], str(reg.label),
                    color="white", fontsize=6, ha="center", va="center")

    ax.set_title(f"{title}  (n = {len(regions)})",
                 color=INK_PRIMARY, fontsize=11, pad=8)

    if finite.size:
        sm = plt.cm.ScalarMappable(cmap=SIZE_CMAP, norm=norm)
        cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.015)
        cbar.set_label(f"Equivalent diameter ({unit})",
                       color=INK_SECONDARY, fontsize=9)
        cbar.ax.tick_params(colors=INK_SECONDARY, labelsize=8)
        cbar.outline.set_visible(False)

    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    return fig


def distribution_figure(
    diameters: np.ndarray,
    stats=None,
    unit: str = "nm",
    bins: int | str = "auto",
    dpi: int = 130,
):
    """Histogram with the fitted log-normal curve and the percentile triple."""
    d = np.asarray(diameters, dtype=float)
    d = d[np.isfinite(d) & (d > 0)]

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(10.5, 4.0), dpi=dpi,
        gridspec_kw={"width_ratios": [1.35, 1.0]},
    )

    if d.size == 0:
        for a in (ax, ax2):
            a.text(0.5, 0.5, "no particles measured", ha="center", va="center",
                   color=INK_SECONDARY, transform=a.transAxes)
            _style(a)
        fig.patch.set_facecolor(SURFACE)
        fig.tight_layout()
        return fig

    # --- histogram + fit --------------------------------------------------
    counts, edges, patches = ax.hist(
        d, bins=bins, color=SERIES_1, edgecolor=SURFACE, linewidth=1.0,
        density=True, label="Measured",
    )
    if d.size > 2:
        mu, sigma = float(np.mean(np.log(d))), float(np.std(np.log(d), ddof=1))
        if sigma > 0:
            xs = np.linspace(max(d.min() * 0.75, 1e-9), d.max() * 1.15, 400)
            pdf = (1.0 / (xs * sigma * np.sqrt(2 * np.pi))) * np.exp(
                -((np.log(xs) - mu) ** 2) / (2 * sigma**2)
            )
            ax.plot(xs, pdf, color=SERIES_2, linewidth=2.0,
                    label=f"Log-normal fit (GSD {np.exp(sigma):.2f})")

    if stats is not None:
        for value, name in ((stats.d10, "D10"), (stats.d50, "D50"), (stats.d90, "D90")):
            ax.axvline(value, color=INK_MUTED, linestyle="--", linewidth=1.0)
            ax.text(value, ax.get_ylim()[1] * 0.96, f" {name}",
                    color=INK_SECONDARY, fontsize=8, va="top")

    ax.set_xlabel(f"Equivalent circular diameter ({unit})",
                  color=INK_SECONDARY, fontsize=10)
    ax.set_ylabel("Probability density", color=INK_SECONDARY, fontsize=10)
    ax.set_title("Size distribution", color=INK_PRIMARY, fontsize=11, loc="left")
    leg = ax.legend(frameon=False, fontsize=9, loc="upper right")
    for text in leg.get_texts():
        text.set_color(INK_SECONDARY)
    _style(ax)

    # --- cumulative -------------------------------------------------------
    srt = np.sort(d)
    cum = np.arange(1, srt.size + 1) / srt.size * 100.0
    ax2.plot(srt, cum, color=SERIES_1, linewidth=2.0)
    for pct in (10, 50, 90):
        val = float(np.percentile(d, pct))
        ax2.plot([srt.min(), val], [pct, pct], color=INK_MUTED,
                 linestyle="--", linewidth=0.9)
        ax2.plot([val, val], [0, pct], color=INK_MUTED,
                 linestyle="--", linewidth=0.9)
        ax2.annotate(f"D{pct} = {val:.1f}", xy=(val, pct),
                     xytext=(6, -11), textcoords="offset points",
                     color=INK_SECONDARY, fontsize=8)
    ax2.set_ylim(0, 103)
    ax2.set_xlabel(f"Diameter ({unit})", color=INK_SECONDARY, fontsize=10)
    ax2.set_ylabel("Cumulative (%)", color=INK_SECONDARY, fontsize=10)
    ax2.set_title("Cumulative distribution", color=INK_PRIMARY,
                  fontsize=11, loc="left")
    _style(ax2)

    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    return fig


def shape_figure(df, dpi: int = 130):
    """Circularity against diameter -- separates spheres from rods and clumps."""
    fig, ax = plt.subplots(figsize=(5.6, 4.0), dpi=dpi)
    if df is None or df.empty:
        ax.text(0.5, 0.5, "no particles", ha="center", va="center",
                color=INK_SECONDARY, transform=ax.transAxes)
        _style(ax)
        return fig
    x = df["ecd_nm"] if df["ecd_nm"].notna().any() else df["ecd_px"]
    unit = "nm" if df["ecd_nm"].notna().any() else "px"
    ax.scatter(x, df["circularity"], s=26, color=SERIES_1,
               alpha=0.80, edgecolor=SURFACE, linewidth=0.8)
    ax.axhline(0.90, color=INK_MUTED, linestyle="--", linewidth=1.0)
    ax.text(ax.get_xlim()[1], 0.905, "circularity 0.90 ", ha="right",
            fontsize=8, color=INK_SECONDARY)
    ax.set_xlabel(f"Equivalent diameter ({unit})", color=INK_SECONDARY, fontsize=10)
    ax.set_ylabel("Circularity", color=INK_SECONDARY, fontsize=10)
    ax.set_title("Shape against size", color=INK_PRIMARY, fontsize=11, loc="left")
    _style(ax)
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    return fig
