"""
End-to-end analysis: micrograph in, measurements out.

The one design rule worth stating: the pipeline never refuses to produce a
result. If the scale bar cannot be read, it reports sizes in pixels and says
so, rather than returning nothing. An uncalibrated measurement is still a
perfectly good shape analysis and a perfectly good particle count, and a tool
that throws away both because one OCR call failed is less useful than one
that hands back what it does know.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .calibration import Calibration, calibrate
from .metrics import SizeStats, measure_particles, summarize_sizes, format_report
from .segmentation import (
    SegmentationParams, SegmentationResult, refine_half_max, segment,
)


@dataclass
class AnalysisResult:
    particles: pd.DataFrame
    stats: SizeStats | None
    calibration: Calibration
    segmentation: SegmentationResult
    labels: np.ndarray
    unit: str                      # "nm" when calibrated, else "px"
    size_column: str               # "ecd_nm" or "ecd_px"
    shape: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def n_particles(self) -> int:
        return 0 if self.particles is None else int(len(self.particles))

    @property
    def diameters(self) -> np.ndarray:
        if self.particles is None or self.particles.empty:
            return np.zeros(0)
        return self.particles[self.size_column].to_numpy(dtype=float)

    def report(self) -> str:
        head = [
            f"Calibration : {self.calibration.describe()}",
            f"Modality    : particles are {self.segmentation.polarity} "
            f"against the background",
            f"Rejected    : {self.segmentation.n_border_removed} touching the "
            f"frame edge, {self.segmentation.n_shape_rejected} on size/shape",
        ]
        body = (format_report(self.stats, self.shape)
                if self.stats else "No particles measured.")
        warn = (["", "Warnings:"] + [f"  - {w}" for w in self.warnings]
                if self.warnings else [])
        return "\n".join(head + ["", body] + warn)


def analyze(
    image: np.ndarray,
    nm_per_px: float | None = None,
    params: SegmentationParams | None = None,
    refine_boundaries: bool = False,
    material: str | None = None,
    density_g_cm3: float | None = None,
) -> AnalysisResult:
    """Run calibration, segmentation and measurement over one micrograph.

    `refine_boundaries` re-cuts every particle at its own half-maximum level
    instead of the global Otsu threshold. It defaults to False because it was
    measured against ground truth and lost: the per-particle criterion
    assumes a flat-topped intensity profile, but an SEM particle's brightest
    pixels are its rim rather than its centre, so the "interior" level the
    rule needs is systematically wrong. See validate.py for the numbers.
    """
    warnings: list[str] = []

    cal = calibrate(image, manual_nm_per_px=nm_per_px)
    if not cal.ok:
        warnings.append(
            f"{cal.note}; sizes are reported in pixels. "
            "Pass a nm/px value to get physical units."
        )
    elif cal.source == "ocr" and cal.confidence < 0.45:
        warnings.append(
            f"Scale bar read with low confidence ({cal.confidence:.2f}); "
            "check the calibration before quoting these numbers."
        )

    seg = segment(image, params=params, overlay_mask=cal.overlay_mask)
    labels = seg.labels

    if refine_boundaries and seg.n_kept > 0:
        labels = refine_half_max(image, labels, seg.polarity)

    df = measure_particles(labels, cal.nm_per_px if cal.ok else None)

    unit = "nm" if cal.ok else "px"
    size_col = "ecd_nm" if cal.ok else "ecd_px"

    stats = (
        summarize_sizes(
            df[size_col].to_numpy(dtype=float), unit=unit,
            material=material, density_g_cm3=density_g_cm3,
        )
        if not df.empty else None
    )

    shape: dict = {}
    if not df.empty:
        shape = {
            k: float(df[k].mean())
            for k in ("circularity", "aspect_ratio", "solidity")
        }
        # A population that is mostly non-circular usually means contacts were
        # not fully separated, so say so rather than letting the user quote a
        # diameter for something that is not one particle.
        if shape["circularity"] < 0.80:
            warnings.append(
                f"Mean circularity is {shape['circularity']:.2f}; particles may "
                "be irregular, or touching particles may not have been split. "
                "Inspect the overlay before trusting the size distribution."
            )
    if seg.n_kept == 0:
        warnings.append("No particles survived the size and shape filters.")

    return AnalysisResult(
        particles=df, stats=stats, calibration=cal, segmentation=seg,
        labels=labels, unit=unit, size_column=size_col,
        shape=shape, warnings=warnings,
    )
