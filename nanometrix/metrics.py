"""
Per-particle measurement and population statistics.

What gets reported, and why these quantities
--------------------------------------------
A size distribution can be summarised in several inequivalent ways, and
quoting the wrong one is a common error in the literature. This module
reports the ones a materials characterisation section is expected to carry:

* Equivalent circular diameter (ECD), d = sqrt(4A/pi). Derived from area
  rather than from a caliper width, because area is the most noise-tolerant
  measurement a segmenter produces -- a boundary pixel misassigned here or
  there barely moves an area, but can visibly move a caliper.
* Log-normal fit by maximum likelihood on ln(d), giving a median and a
  geometric standard deviation. Nucleation-and-growth syntheses produce
  log-normal populations, so a mean-plus-SD quoted as if Gaussian describes
  a shape the data does not have.
* D10 / D50 / D90, the percentile triple used throughout powder technology.
* Sauter mean D[3,2] and De Brouckere mean D[4,3]. These are *not*
  decoration. The arithmetic mean answers "what is a typical particle";
  D[3,2] is surface-area weighted and so governs catalytic and adsorptive
  behaviour, while D[4,3] is volume weighted and is what a laser-diffraction
  instrument reports. For a polydisperse sample the three differ
  substantially, and which one matters depends on the application.
* Specific surface area, estimated from D[3,2] and the bulk density. This is
  the number that makes a nanomaterial interesting -- the reason 20 nm
  particles behave differently from the same mass of bulk powder is that
  surface area per gram scales as 1/d.
* A bootstrap confidence interval on the median, because a size distribution
  measured from one field of view is a sample, and reporting D50 without an
  interval hides how few particles it rests on.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from skimage.measure import regionprops, perimeter_crofton

# Bulk densities (g/cm^3) for the materials most often imaged at this scale.
BULK_DENSITY_G_CM3 = {
    "Au": 19.30, "Ag": 10.49, "Pt": 21.45, "Pd": 12.02, "Cu": 8.96,
    "Fe3O4": 5.17, "Fe2O3": 5.24, "TiO2": 4.23, "ZnO": 5.61,
    "SiO2": 2.65, "CeO2": 7.22, "Al2O3": 3.95, "CdSe": 5.82,
    "polystyrene": 1.05, "C": 2.26,
}


# --------------------------------------------------------------------------
# per-particle table
# --------------------------------------------------------------------------

def measure_particles(
    labels: np.ndarray,
    nm_per_px: float | None = None,
) -> pd.DataFrame:
    """Build a one-row-per-particle table of size and shape descriptors.

    Examples
    --------
    A single 10x10 pixel square particle, calibrated at 2 nm/px. Area-derived
    ECD for a square is slightly larger than its side length (same area as a
    circle of this diameter), and a square's circularity is below 1 because
    its perimeter is less efficient than a circle's for the same area:

    >>> import numpy as np
    >>> from nanometrix.metrics import measure_particles
    >>> labels = np.zeros((20, 20), dtype=int)
    >>> labels[5:15, 5:15] = 1
    >>> df = measure_particles(labels, nm_per_px=2.0)
    >>> int(df.loc[0, "label"])
    1
    >>> round(float(df.loc[0, "ecd_px"]), 2)
    11.28
    >>> round(float(df.loc[0, "ecd_nm"]), 2)
    22.57
    >>> round(float(df.loc[0, "circularity"]), 3)
    0.927

    An all-background label image returns an empty table with the full set
    of columns, rather than one with no columns at all:

    >>> list(measure_particles(np.zeros((10, 10), dtype=int)).columns)[:3]
    ['label', 'centroid_row', 'centroid_col']
    """
    rows = []
    calibrated = bool(nm_per_px and nm_per_px > 0)
    # Written with the `nm_per_px is not None` check inline (rather than just
    # `if calibrated`) so a type checker can narrow `nm_per_px` itself here --
    # it cannot follow the fact that `calibrated` already implies that.
    s = float(nm_per_px) if calibrated and nm_per_px is not None else 1.0

    for reg in regionprops(labels):
        area_px = float(reg.area)
        # Crofton's formula estimates perimeter far better than counting
        # boundary pixels, which overestimates by up to ~10% on a digitised
        # curve and would make every particle look less circular than it is.
        per_px = float(perimeter_crofton(reg.image, directions=4))
        ecd_px = float(np.sqrt(4.0 * area_px / np.pi))

        circularity = (
            float(4.0 * np.pi * area_px / (per_px**2)) if per_px > 0 else np.nan
        )
        major = float(reg.axis_major_length)
        minor = float(reg.axis_minor_length)
        aspect = float(major / minor) if minor > 0 else np.nan

        try:
            feret_max_px = float(reg.feret_diameter_max)
        except Exception:
            feret_max_px = np.nan

        rows.append({
            "label": int(reg.label),
            "centroid_row": float(reg.centroid[0]),
            "centroid_col": float(reg.centroid[1]),
            "area_px": area_px,
            "perimeter_px": per_px,
            "ecd_px": ecd_px,
            "ecd_nm": ecd_px * s if calibrated else np.nan,
            "area_nm2": area_px * s * s if calibrated else np.nan,
            "feret_max_nm": feret_max_px * s if calibrated else np.nan,
            "major_axis_nm": major * s if calibrated else np.nan,
            "minor_axis_nm": minor * s if calibrated else np.nan,
            # Shape descriptors are dimensionless, so they survive an
            # uncalibrated image intact.
            "circularity": min(circularity, 1.0) if np.isfinite(circularity) else np.nan,
            "aspect_ratio": aspect,
            "solidity": float(reg.solidity),
            "eccentricity": float(reg.eccentricity),
            "orientation_deg": float(np.degrees(reg.orientation)),
        })

    columns = [
        "label", "centroid_row", "centroid_col", "area_px", "perimeter_px",
        "ecd_px", "ecd_nm", "area_nm2", "feret_max_nm", "major_axis_nm",
        "minor_axis_nm", "circularity", "aspect_ratio", "solidity",
        "eccentricity", "orientation_deg",
    ]
    # An empty result must still carry its columns, so that callers can index
    # them without special-casing the no-particles-found path.
    df = pd.DataFrame(rows, columns=columns)
    if not df.empty:
        df = df.sort_values("ecd_px", ascending=False).reset_index(drop=True)
    return df


# --------------------------------------------------------------------------
# population statistics
# --------------------------------------------------------------------------

@dataclass
class SizeStats:
    n: int
    unit: str
    mean: float
    std: float
    median: float
    d10: float
    d50: float
    d90: float
    span: float                 # (D90 - D10) / D50, a shape-free width measure
    lognormal_median: float
    lognormal_gsd: float
    sauter_d32: float
    debrouckere_d43: float
    pdi: float                  # (sigma/mean)^2, the DLS convention
    cv_percent: float
    median_ci_low: float
    median_ci_high: float
    ssa_m2_per_g: float | None = None
    density_g_cm3: float | None = None
    material: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _bootstrap_median_ci(
    x: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    if x.size < 3:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    meds = np.median(x[idx], axis=1)
    lo, hi = np.percentile(meds, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def specific_surface_area(d32_nm: float, density_g_cm3: float) -> float:
    """Specific surface area in m^2/g for spheres of Sauter mean d32.

    SSA = 6 / (rho * d). The Sauter mean is the correct diameter to use here
    precisely because it is defined as the diameter of the sphere with the
    same surface-to-volume ratio as the whole population, so a polydisperse
    sample and its D[3,2] sphere share an SSA by construction.

    Examples
    --------
    10 nm gold spheres (bulk density 19.30 g/cm^3):

    >>> from nanometrix.metrics import specific_surface_area
    >>> round(specific_surface_area(d32_nm=10.0, density_g_cm3=19.30), 3)
    31.088

    Halving the diameter doubles the surface area per gram:

    >>> round(specific_surface_area(5.0, 19.30) / specific_surface_area(10.0, 19.30), 3)
    2.0

    An unphysical (non-positive) size or density degrades to NaN rather than
    raising, consistent with the rest of this module:

    >>> import math
    >>> math.isnan(specific_surface_area(d32_nm=0.0, density_g_cm3=1.0))
    True
    """
    if d32_nm <= 0 or density_g_cm3 <= 0:
        return float("nan")
    d_m = d32_nm * 1e-9
    rho_kg_m3 = density_g_cm3 * 1000.0
    return 6.0 / (rho_kg_m3 * d_m) / 1000.0     # m^2/kg -> m^2/g


def summarize_sizes(
    diameters: np.ndarray,
    unit: str = "nm",
    material: str | None = None,
    density_g_cm3: float | None = None,
) -> SizeStats | None:
    """Reduce a set of diameters to the statistics a report needs.

    Examples
    --------
    >>> import numpy as np
    >>> from nanometrix.metrics import summarize_sizes
    >>> d = np.array([10., 11., 9., 12., 10., 11., 9., 13., 10., 12.])
    >>> stats = summarize_sizes(d, unit="nm")
    >>> stats.n
    10
    >>> round(stats.mean, 2)
    10.7
    >>> round(stats.d50, 2)
    10.5
    >>> round(stats.sauter_d32, 2)
    11.0

    An empty or all-invalid input (no positive, finite diameters) returns
    ``None`` instead of a degenerate statistics object:

    >>> summarize_sizes(np.array([])) is None
    True
    >>> summarize_sizes(np.array([-1.0, 0.0, float("nan")])) is None
    True
    """
    d = np.asarray(diameters, dtype=float)
    d = d[np.isfinite(d) & (d > 0)]
    if d.size == 0:
        return None

    mean = float(np.mean(d))
    std = float(np.std(d, ddof=1)) if d.size > 1 else 0.0
    d10, d50, d90 = (float(v) for v in np.percentile(d, [10, 50, 90]))

    # Log-normal MLE is just the moments of the log-transformed data.
    ln = np.log(d)
    mu = float(np.mean(ln))
    sigma = float(np.std(ln, ddof=1)) if d.size > 1 else 0.0

    d32 = float(np.sum(d**3) / np.sum(d**2))
    d43 = float(np.sum(d**4) / np.sum(d**3))

    lo, hi = _bootstrap_median_ci(d)

    rho = density_g_cm3
    if rho is None and material:
        rho = BULK_DENSITY_G_CM3.get(material)
    ssa = (
        specific_surface_area(d32, rho)
        if (rho and unit == "nm") else None
    )

    return SizeStats(
        n=int(d.size),
        unit=unit,
        mean=mean,
        std=std,
        median=float(np.median(d)),
        d10=d10, d50=d50, d90=d90,
        span=float((d90 - d10) / d50) if d50 > 0 else float("nan"),
        lognormal_median=float(np.exp(mu)),
        lognormal_gsd=float(np.exp(sigma)),
        sauter_d32=d32,
        debrouckere_d43=d43,
        pdi=float((std / mean) ** 2) if mean > 0 else float("nan"),
        cv_percent=float(100.0 * std / mean) if mean > 0 else float("nan"),
        median_ci_low=lo, median_ci_high=hi,
        ssa_m2_per_g=ssa,
        density_g_cm3=rho,
        material=material,
    )


def format_report(stats: SizeStats, shape: dict | None = None) -> str:
    """Render the statistics as text fit for pasting into a report."""
    u = stats.unit
    lines = [
        f"Particles measured          : {stats.n}",
        f"Mean +/- SD                 : {stats.mean:.2f} +/- {stats.std:.2f} {u}",
        f"Median (D50)                : {stats.d50:.2f} {u} "
        f"[95% CI {stats.median_ci_low:.2f}-{stats.median_ci_high:.2f}]",
        f"D10 / D90                   : {stats.d10:.2f} / {stats.d90:.2f} {u}",
        f"Span (D90-D10)/D50          : {stats.span:.3f}",
        f"Log-normal median, GSD      : {stats.lognormal_median:.2f} {u}, "
        f"{stats.lognormal_gsd:.3f}",
        f"Sauter mean D[3,2]          : {stats.sauter_d32:.2f} {u}",
        f"De Brouckere mean D[4,3]    : {stats.debrouckere_d43:.2f} {u}",
        f"Coefficient of variation    : {stats.cv_percent:.1f} %",
        f"Polydispersity index (PDI)  : {stats.pdi:.4f}",
    ]
    if stats.ssa_m2_per_g is not None and np.isfinite(stats.ssa_m2_per_g):
        mat = f" ({stats.material}, {stats.density_g_cm3:g} g/cm3)" if stats.material else ""
        lines.append(f"Specific surface area       : {stats.ssa_m2_per_g:.2f} m2/g{mat}")
    if shape:
        lines += [
            f"Mean circularity            : {shape.get('circularity', float('nan')):.3f}",
            f"Mean aspect ratio           : {shape.get('aspect_ratio', float('nan')):.3f}",
            f"Mean solidity               : {shape.get('solidity', float('nan')):.3f}",
        ]
    return "\n".join(lines)
