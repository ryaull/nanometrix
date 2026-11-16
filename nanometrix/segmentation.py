"""
Particle segmentation: finding individual particles in a micrograph.

The hard part is not finding particles, it is separating them
-----------------------------------------------------------------
Thresholding a micrograph is easy, and on a sparse sample it is the whole
job. Real samples are not sparse: particles dry into contact, and every
touching pair becomes a single connected blob. A blob of two 40 nm spheres
has the area of one 57 nm sphere, so naive thresholding does not merely
miscount, it invents a large-particle tail that was never in the sample and
shifts the reported mean upward. Separating contacts is therefore the
measurement, and that is what the watershed stage does.

Pipeline, and the reason for each stage
---------------------------------------
1. Mask the burnt-in scale bar, which is a perfect high-contrast rectangle
   and would otherwise be measured as one enormous particle.
2. CLAHE, to flatten the illumination gradient and vignetting that make a
   single global threshold cut in the wrong place across the frame.
3. Edge-preserving denoise, because shot noise fragments particle interiors
   into speckle and a blur that removes it also eats the boundary we are
   trying to measure.
4. Polarity detection from the histogram mode, so SEM (bright particles) and
   TEM (dark particles) both work without being told which is which.
5. Otsu threshold, then morphological cleanup.
6. Distance transform plus watershed, which is the contact-splitting step.
7. Border exclusion. A particle clipped by the frame edge shows only part of
   itself; measuring it reports a too-small diameter and, because large
   particles are likelier to touch an edge, biases the whole distribution
   downward. Standard stereological practice is to discard them, and that is
   what we do.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


@dataclass
class SegmentationParams:
    """Knobs for the segmentation pipeline."""

    polarity: str = "auto"            # "auto" | "bright" | "dark"
    clahe_clip: float = 2.0
    clahe_grid: int = 8
    denoise: str = "bilateral"        # "bilateral" | "median" | "none"
    bilateral_d: int = 7
    bilateral_sigma_colour: float = 45.0
    bilateral_sigma_space: float = 7.0
    median_ksize: int = 5

    open_radius: int = 2              # erase speckle
    fill_holes: bool = True

    split_touching: bool = True
    peak_min_distance_frac: float = 0.55  # of the estimated particle radius
    peak_footprint_min: int = 3

    min_area_px: int = 40             # below this, it is noise, not a particle
    max_area_frac: float = 0.25       # above this, it is an artefact or a clump
    min_solidity: float = 0.80        # rejects unsplit clumps and debris
    exclude_border: bool = True


@dataclass
class SegmentationResult:
    labels: np.ndarray                # int32 instance labels, 0 = background
    binary: np.ndarray                # bool mask before splitting
    polarity: str
    n_raw: int                        # regions before rejection filters
    n_kept: int
    n_border_removed: int
    n_shape_rejected: int
    estimated_radius_px: float


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def detect_polarity(gray: np.ndarray, valid: np.ndarray | None = None) -> str:
    """Decide whether particles are brighter or darker than the background.

    Topology, not brightness
    ------------------------
    The tempting rule -- "the background is the commonest intensity, so
    particles are on the other side of the threshold" -- fails on exactly the
    images this tool is built for. Strongly scattering particles saturate, so
    their interiors all land within a bin or two of each other and pile into
    one tall histogram spike, while the background's shot noise spreads its
    pixels across a wide range of bins. The particle phase then *is* the
    mode, and the rule inverts the image: background gets segmented as a
    single enormous particle, every real particle is discarded, and the
    pipeline returns nothing.

    The structural distinction is reliable where the brightness one is not.
    Background is a single connected sheet that reaches around everything;
    particles are many separate compact blobs. So binarise both ways and ask
    which phase is fragmented: for background the largest connected component
    holds nearly all of the phase's area, for particles it holds a small
    fraction. Only when both phases look equally connected -- a densely
    packed sample whose particles all touch -- do we fall back to area.
    """
    mask_valid = (
        valid if valid is not None else np.ones(gray.shape, dtype=bool)
    )
    px = gray[mask_valid]
    if px.size == 0:
        return "bright"

    thresh, _ = cv2.threshold(px.astype(np.uint8), 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    stats_by_polarity: dict[str, tuple[float, float]] = {}
    for name, phase in (("bright", gray >= thresh), ("dark", gray <= thresh)):
        phase = phase & mask_valid
        area = float(phase.sum())
        if area <= 0:
            stats_by_polarity[name] = (1.0, 0.0)
            continue
        n, _, cc_stats, _ = cv2.connectedComponentsWithStats(
            phase.astype(np.uint8), connectivity=8
        )
        if n <= 1:
            stats_by_polarity[name] = (1.0, area / mask_valid.sum())
            continue
        areas = cc_stats[1:, cv2.CC_STAT_AREA].astype(float)
        stats_by_polarity[name] = (
            float(areas.max() / area),      # how connected this phase is
            float(area / mask_valid.sum()),  # how much of the frame it covers
        )

    frag_bright, frac_bright = stats_by_polarity["bright"]
    frag_dark, frac_dark = stats_by_polarity["dark"]

    if abs(frag_bright - frag_dark) > 0.20:
        # The more fragmented phase is the particles.
        return "bright" if frag_bright < frag_dark else "dark"

    # Both equally connected: fall back to the phase covering less area.
    return "bright" if frac_bright < frac_dark else "dark"


def _preprocess(gray: np.ndarray, p: SegmentationParams) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=p.clahe_clip,
                            tileGridSize=(p.clahe_grid, p.clahe_grid))
    out = clahe.apply(gray)
    if p.denoise == "bilateral":
        out = cv2.bilateralFilter(out, p.bilateral_d,
                                  p.bilateral_sigma_colour, p.bilateral_sigma_space)
    elif p.denoise == "median":
        k = p.median_ksize | 1
        out = cv2.medianBlur(out, k)
    return out


def _estimate_radius_px(binary: np.ndarray) -> float:
    """Typical particle radius, read off the distance transform.

    For a disc the distance transform peaks at the radius, so the upper
    percentiles of the distance map inside the foreground estimate the radius
    of the typical particle without anyone having to supply it.
    """
    dist = cv2.distanceTransform(binary.astype(np.uint8), cv2.DIST_L2, 5)
    vals = dist[binary]
    if vals.size == 0:
        return 0.0
    return float(np.percentile(vals, 95))


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def segment(
    image: np.ndarray,
    params: SegmentationParams | None = None,
    overlay_mask: np.ndarray | None = None,
) -> SegmentationResult:
    """Segment particles and return one integer label per particle."""
    p = params or SegmentationParams()
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = gray.astype(np.uint8)
    h, w = gray.shape

    valid = np.ones((h, w), dtype=bool)
    if overlay_mask is not None and overlay_mask.shape == gray.shape:
        valid &= ~overlay_mask

    proc = _preprocess(gray, p)

    polarity = p.polarity
    if polarity == "auto":
        polarity = detect_polarity(proc, valid)

    # Threshold using only valid pixels, so the burnt-in overlay cannot drag
    # the Otsu threshold away from the sample's own statistics.
    sample = proc[valid]
    tval, _ = cv2.threshold(sample, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = (proc >= tval) if polarity == "bright" else (proc <= tval)
    binary &= valid

    if p.open_radius > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * p.open_radius + 1, 2 * p.open_radius + 1)
        )
        binary = cv2.morphologyEx(
            binary.astype(np.uint8), cv2.MORPH_OPEN, k
        ).astype(bool)
    if p.fill_holes:
        binary = ndi.binary_fill_holes(binary)

    radius_px = _estimate_radius_px(binary)

    # --- split touching particles ---------------------------------------
    if p.split_touching and binary.any() and radius_px >= 2:
        dist = cv2.distanceTransform(binary.astype(np.uint8), cv2.DIST_L2, 5)
        # Smoothing the distance map suppresses the spurious local maxima
        # that boundary roughness produces, which would otherwise shatter one
        # particle into several.
        dist = cv2.GaussianBlur(dist, (0, 0), max(radius_px * 0.12, 0.8))

        min_dist = max(int(round(radius_px * p.peak_min_distance_frac)),
                       p.peak_footprint_min)
        coords = peak_local_max(
            dist, min_distance=min_dist, labels=binary,
            exclude_border=False,
        )
        markers = np.zeros(dist.shape, dtype=np.int32)
        for i, (r, c) in enumerate(coords, start=1):
            markers[r, c] = i
        if markers.max() == 0:
            markers, _ = ndi.label(binary)
        labels = watershed(-dist, markers, mask=binary).astype(np.int32)
    else:
        labels, _ = ndi.label(binary)
        labels = labels.astype(np.int32)

    n_raw = int(labels.max())

    # --- rejection filters ----------------------------------------------
    from skimage.measure import regionprops

    max_area = p.max_area_frac * h * w
    keep = np.zeros(labels.max() + 1, dtype=bool)
    n_border = 0
    n_shape = 0

    for reg in regionprops(labels):
        if reg.area < p.min_area_px or reg.area > max_area:
            n_shape += 1
            continue
        if p.exclude_border:
            minr, minc, maxr, maxc = reg.bbox
            if minr <= 0 or minc <= 0 or maxr >= h or maxc >= w:
                n_border += 1
                continue
        if reg.solidity < p.min_solidity:
            n_shape += 1
            continue
        keep[reg.label] = True

    # Renumber so labels run 1..n with no gaps.
    remap = np.zeros(labels.max() + 1, dtype=np.int32)
    remap[np.flatnonzero(keep)] = np.arange(1, int(keep.sum()) + 1)
    clean = remap[labels]

    return SegmentationResult(
        labels=clean,
        binary=binary,
        polarity=polarity,
        n_raw=n_raw,
        n_kept=int(keep.sum()),
        n_border_removed=n_border,
        n_shape_rejected=n_shape,
        estimated_radius_px=float(radius_px),
    )


# --------------------------------------------------------------------------
# sub-pixel boundary refinement
# --------------------------------------------------------------------------

def refine_half_max(
    image: np.ndarray,
    labels: np.ndarray,
    polarity: str,
    pad_px: int = 6,
    min_core_px: int = 6,
) -> np.ndarray:
    """Re-cut each particle's boundary at the half-maximum intensity level.

    Why a single global threshold is not good enough
    ------------------------------------------------
    A particle edge in an electron micrograph is not a step. The finite probe
    size and detector response smear it over several pixels, so the apparent
    boundary depends entirely on where the threshold is placed on that ramp.
    A global Otsu threshold lands wherever the image-wide histogram puts it,
    which is generally *not* the midpoint of any individual particle's edge,
    and the result is a systematic sizing bias -- every particle measured a
    little too small, or a little too large, consistently.

    The metrological convention for a blurred edge is the 50% criterion: the
    boundary sits where intensity crosses the midpoint between the object's
    own interior level and its own local background. Applied per particle, it
    is immune to both illumination drift across the frame and to
    particle-to-particle contrast variation.

    The refinement is confined to each particle's existing watershed
    territory, minus its neighbours' pixels, so widening a boundary can never
    reunite two particles that the splitting stage just separated.
    """
    from skimage.measure import regionprops

    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = gray.astype(np.uint8)
    # Denoise before cutting. The half-max rule is a per-pixel comparison
    # against a level, so on a raw micrograph every pixel whose shot noise
    # straddles that level flips sides independently and the boundary comes
    # out ragged -- which destroys the shape descriptors and, because a
    # ragged outline encloses less area than the smooth one it wanders
    # around, biases the diameter low as well. Edge-preserving smoothing
    # removes the speckle without moving the edge we are trying to locate.
    gray = cv2.bilateralFilter(gray, 7, 45.0, 7.0).astype(np.float32)
    h, w = gray.shape
    bright = (polarity == "bright")

    out = np.zeros_like(labels)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kpad = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * pad_px + 1, 2 * pad_px + 1)
    )

    for reg in regionprops(labels):
        lab = reg.label
        minr, minc, maxr, maxc = reg.bbox
        r0, c0 = max(minr - pad_px - 2, 0), max(minc - pad_px - 2, 0)
        r1, c1 = min(maxr + pad_px + 2, h), min(maxc + pad_px + 2, w)

        win_lbl = labels[r0:r1, c0:c1]
        win_img = gray[r0:r1, c0:c1]
        mine = (win_lbl == lab)
        if mine.sum() < min_core_px:
            out[r0:r1, c0:c1][mine] = lab
            continue
        others = (win_lbl > 0) & ~mine

        # Territory: room to grow into, but never into a neighbour.
        territory = cv2.dilate(mine.astype(np.uint8), kpad).astype(bool)
        territory &= ~others

        # Interior level from an eroded core, so edge pixels do not pollute it.
        core = cv2.erode(mine.astype(np.uint8), k3, iterations=2).astype(bool)
        if core.sum() < min_core_px:
            core = mine
        i_in = float(np.median(win_img[core]))

        # Background level from the ring just outside the current boundary.
        ring = territory & ~cv2.dilate(mine.astype(np.uint8), k3,
                                       iterations=1).astype(bool)
        if ring.sum() < min_core_px:
            out[r0:r1, c0:c1][mine] = lab
            continue
        i_bg = float(np.median(win_img[ring]))

        level = 0.5 * (i_in + i_bg)
        cut = (win_img >= level) if bright else (win_img <= level)
        cut &= territory

        if not cut.any():
            out[r0:r1, c0:c1][mine] = lab
            continue
        cut = ndi.binary_fill_holes(cut)

        # Keep only the blob that actually contains this particle, in case
        # the cut also catches a sliver of something else in the window.
        sub, n = ndi.label(cut)
        if n > 1:
            overlap = np.bincount(sub[mine], minlength=n + 1)
            overlap[0] = 0
            cut = (sub == int(np.argmax(overlap)))
        elif n == 0:
            cut = mine

        dest = out[r0:r1, c0:c1]
        dest[cut & (dest == 0)] = lab

    return out
