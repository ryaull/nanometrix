"""
Unit tests for nanometrix.metrics.

The per-particle descriptors (equivalent circular diameter, circularity,
aspect ratio, solidity) are geometric formulas applied to whatever a
segmenter hands them, so they can be checked against synthetic contours
whose true area/perimeter/axes/convexity we chose ourselves -- no real
micrograph, and no trip through segmentation, is needed to know whether the
formulas in measure_particles() are implemented correctly.

Shapes are drawn directly into small labelled arrays with skimage.draw, at a
large enough scale that pixelation error stays within a few percent of the
continuous-geometry answer (verified empirically; see the tolerances below).
"""

from __future__ import annotations

import numpy as np
import pytest
from skimage.draw import disk, ellipse, rectangle
from skimage.measure import label

from nanometrix.metrics import measure_particles, specific_surface_area, summarize_sizes


# --------------------------------------------------------------------------
# synthetic contour fixtures
# --------------------------------------------------------------------------

def _labels_from_mask(mask: np.ndarray) -> np.ndarray:
    return label(mask.astype(np.uint8))


def _square_labels(side: int = 40, canvas: int = 100) -> np.ndarray:
    mask = np.zeros((canvas, canvas), dtype=np.uint8)
    rr, cc = rectangle((10, 10), (10 + side - 1, 10 + side - 1), shape=mask.shape)
    mask[rr, cc] = 1
    return _labels_from_mask(mask)


def _disk_labels(radius: int = 40, canvas: int = 200) -> np.ndarray:
    mask = np.zeros((canvas, canvas), dtype=np.uint8)
    rr, cc = disk((canvas // 2, canvas // 2), radius, shape=mask.shape)
    mask[rr, cc] = 1
    return _labels_from_mask(mask)


def _ellipse_labels(a: int = 60, b: int = 30, canvas=(200, 300)) -> np.ndarray:
    """a, b are the semi-major/semi-minor axis lengths (full axes 2a, 2b)."""
    mask = np.zeros(canvas, dtype=np.uint8)
    rr, cc = ellipse(canvas[0] // 2, canvas[1] // 2, b, a, shape=mask.shape)
    mask[rr, cc] = 1
    return _labels_from_mask(mask)


def _plus_labels(canvas: int = 60, arm: int = 20) -> np.ndarray:
    """A plus/cross shape: convex hull is much bigger than the shape itself,
    so solidity should come out well below 1 -- unlike every convex fixture
    above."""
    mask = np.zeros((canvas, canvas), dtype=np.uint8)
    lo, hi = arm, canvas - arm
    mask[lo:hi, :] = 1
    mask[:, lo:hi] = 1
    return _labels_from_mask(mask)


# --------------------------------------------------------------------------
# equivalent circular diameter
# --------------------------------------------------------------------------

def test_ecd_matches_area_formula_for_square():
    """ECD is defined purely from area (d = sqrt(4A/pi)), and a pixel-aligned
    square's area is exact, so the reported ecd_px should match the formula
    to high precision -- this is really a check that measure_particles uses
    reg.area rather than, say, a caliper measurement."""
    side = 40
    df = measure_particles(_square_labels(side=side))
    assert len(df) == 1
    area_px = side * side
    expected_ecd = np.sqrt(4.0 * area_px / np.pi)
    assert df.loc[0, "area_px"] == pytest.approx(area_px)
    assert df.loc[0, "ecd_px"] == pytest.approx(expected_ecd, rel=1e-9)


def test_ecd_recovers_disk_diameter_within_pixelation_error():
    radius = 40
    df = measure_particles(_disk_labels(radius=radius))
    assert len(df) == 1
    true_diameter = 2 * radius
    # Pixelating a disk makes its area slightly less than pi*r^2, so the
    # recovered ECD undershoots a little; a few percent is the expected
    # order of the discretisation error at this radius.
    assert df.loc[0, "ecd_px"] == pytest.approx(true_diameter, rel=0.03)


def test_ecd_scales_with_calibration():
    side = 40
    nm_per_px = 0.5
    df = measure_particles(_square_labels(side=side), nm_per_px=nm_per_px)
    row = df.loc[0]
    assert row["ecd_nm"] == pytest.approx(row["ecd_px"] * nm_per_px, rel=1e-9)
    assert row["area_nm2"] == pytest.approx(row["area_px"] * nm_per_px**2, rel=1e-9)


def test_uncalibrated_image_leaves_physical_columns_nan():
    df = measure_particles(_square_labels())
    row = df.loc[0]
    assert np.isnan(row["ecd_nm"])
    assert np.isnan(row["area_nm2"])
    # Shape descriptors are dimensionless and must survive without calibration.
    assert np.isfinite(row["circularity"])
    assert np.isfinite(row["solidity"])


# --------------------------------------------------------------------------
# circularity
# --------------------------------------------------------------------------

def test_circularity_near_one_for_disk():
    """A disk is the maximally circular shape, so circularity should sit at
    (or, via the implementation's min(..., 1.0) clamp, exactly at) 1.0 once
    pixelation noise is accounted for."""
    df = measure_particles(_disk_labels(radius=40))
    circularity = df.loc[0, "circularity"]
    assert 0.95 <= circularity <= 1.0


def test_circularity_clamped_to_one():
    """perimeter_crofton can slightly underestimate a disk's true perimeter,
    which would push the raw 4*pi*A/P^2 ratio above 1 -- a value with no
    physical meaning for a circularity score. measure_particles must clamp
    it, never report circularity > 1."""
    df = measure_particles(_disk_labels(radius=40))
    assert df.loc[0, "circularity"] <= 1.0


def test_circularity_lower_for_square_than_disk():
    """A square is less circular than a disk of comparable size. (The
    textbook value for a square via its *exact* perimeter is 4*pi/16 =
    pi/4 ~= 0.785, but perimeter_crofton is a corner-smoothing estimator by
    design -- see measure_particles's docstring -- so it reports a somewhat
    higher value for a sharp-cornered shape; what must still hold is the
    ranking against a disk, which this test checks directly.)"""
    square_df = measure_particles(_square_labels(side=40))
    disk_df = measure_particles(_disk_labels(radius=23))  # similar area to the square
    square_circ = square_df.loc[0, "circularity"]
    disk_circ = disk_df.loc[0, "circularity"]
    assert square_circ < disk_circ
    assert 0.7 < square_circ < 0.95


# --------------------------------------------------------------------------
# aspect ratio
# --------------------------------------------------------------------------

def test_aspect_ratio_near_one_for_square_and_disk():
    for df in (measure_particles(_square_labels()), measure_particles(_disk_labels())):
        assert df.loc[0, "aspect_ratio"] == pytest.approx(1.0, abs=0.02)


def test_aspect_ratio_matches_ellipse_axis_ratio():
    a, b = 60, 30  # semi-axes -> true axis ratio 2.0
    df = measure_particles(_ellipse_labels(a=a, b=b))
    assert len(df) == 1
    assert df.loc[0, "aspect_ratio"] == pytest.approx(a / b, rel=0.03)


def test_ellipse_axis_lengths_recover_full_axes_in_nm():
    a, b = 60, 30
    nm_per_px = 2.0
    df = measure_particles(_ellipse_labels(a=a, b=b), nm_per_px=nm_per_px)
    row = df.loc[0]
    assert row["major_axis_nm"] == pytest.approx(2 * a * nm_per_px, rel=0.03)
    assert row["minor_axis_nm"] == pytest.approx(2 * b * nm_per_px, rel=0.03)


# --------------------------------------------------------------------------
# solidity
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "labels_fn",
    [lambda: _square_labels(), lambda: _disk_labels(), lambda: _ellipse_labels()],
)
def test_solidity_near_one_for_convex_shapes(labels_fn):
    df = measure_particles(labels_fn())
    assert df.loc[0, "solidity"] == pytest.approx(1.0, abs=0.03)


def test_solidity_well_below_one_for_concave_shape():
    """A plus/cross shape's convex hull covers its missing corners, so
    solidity (area / convex-hull area) must come out noticeably below 1 --
    this is the case a purely area/perimeter-based descriptor would miss."""
    df = measure_particles(_plus_labels())
    assert df.loc[0, "solidity"] < 0.85


# --------------------------------------------------------------------------
# multi-particle / empty-input behaviour of measure_particles itself
# --------------------------------------------------------------------------

def test_empty_label_image_returns_empty_dataframe_with_columns():
    labels = np.zeros((50, 50), dtype=int)
    df = measure_particles(labels)
    assert df.empty
    assert list(df.columns) == [
        "label", "centroid_row", "centroid_col", "area_px", "perimeter_px",
        "ecd_px", "ecd_nm", "area_nm2", "feret_max_nm", "major_axis_nm",
        "minor_axis_nm", "circularity", "aspect_ratio", "solidity",
        "eccentricity", "orientation_deg",
    ]


def test_multiple_particles_are_all_measured_and_sorted_by_size():
    canvas = 300
    mask = np.zeros((canvas, canvas), dtype=np.uint8)
    rr, cc = disk((60, 60), 15, shape=mask.shape)   # small, ecd ~ 30 px
    mask[rr, cc] = 1
    rr, cc = disk((220, 220), 40, shape=mask.shape)  # big, ecd ~ 80 px
    mask[rr, cc] = 1
    labels = _labels_from_mask(mask)

    df = measure_particles(labels)
    assert len(df) == 2
    # Sorted largest-first by ecd_px.
    assert df.loc[0, "ecd_px"] > df.loc[1, "ecd_px"]
    # Both original region labels must be present (order aside).
    assert set(df["label"]) == {1, 2}


# --------------------------------------------------------------------------
# population statistics: summarize_sizes()
# --------------------------------------------------------------------------

def test_summarize_sizes_basic_moments_and_percentiles():
    d = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    stats = summarize_sizes(d, unit="nm")
    assert stats is not None
    assert stats.n == 5
    assert stats.mean == pytest.approx(30.0)
    assert stats.std == pytest.approx(np.std(d, ddof=1))
    assert stats.median == pytest.approx(30.0)
    d10, d50, d90 = np.percentile(d, [10, 50, 90])
    assert stats.d10 == pytest.approx(d10)
    assert stats.d50 == pytest.approx(d50)
    assert stats.d90 == pytest.approx(d90)
    assert stats.span == pytest.approx((d90 - d10) / d50)


def test_summarize_sizes_sauter_and_debrouckere_means():
    """D[3,2] and D[4,3] are moment ratios, independently re-derived here
    from the definitions in the metrics.py docstring rather than by calling
    back into summarize_sizes's own code path."""
    d = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    stats = summarize_sizes(d, unit="nm")
    expected_d32 = float(np.sum(d**3) / np.sum(d**2))
    expected_d43 = float(np.sum(d**4) / np.sum(d**3))
    assert expected_d32 == pytest.approx(40.9090909, rel=1e-6)
    assert expected_d43 == pytest.approx(43.5111111, rel=1e-6)
    assert stats.sauter_d32 == pytest.approx(expected_d32)
    assert stats.debrouckere_d43 == pytest.approx(expected_d43)
    # For a polydisperse set the volume-weighted mean must exceed the
    # surface-weighted mean, which must exceed the plain arithmetic mean.
    assert stats.mean < stats.sauter_d32 < stats.debrouckere_d43


def test_summarize_sizes_pdi_and_cv_percent():
    d = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    stats = summarize_sizes(d, unit="nm")
    std = np.std(d, ddof=1)
    mean = np.mean(d)
    assert stats.pdi == pytest.approx((std / mean) ** 2)
    assert stats.cv_percent == pytest.approx(100.0 * std / mean)


def test_summarize_sizes_monodisperse_population_has_zero_spread():
    d = np.full(10, 25.0)
    stats = summarize_sizes(d, unit="nm")
    assert stats.std == pytest.approx(0.0)
    assert stats.pdi == pytest.approx(0.0)
    assert stats.sauter_d32 == pytest.approx(25.0)
    assert stats.debrouckere_d43 == pytest.approx(25.0)


def test_summarize_sizes_filters_nonpositive_and_nonfinite_values():
    d = np.array([10.0, -5.0, 0.0, np.nan, np.inf, 20.0, 30.0])
    stats = summarize_sizes(d, unit="nm")
    assert stats.n == 3  # only 10, 20, 30 survive


def test_summarize_sizes_empty_input_returns_none():
    assert summarize_sizes(np.array([])) is None
    assert summarize_sizes(np.array([-1.0, 0.0, np.nan])) is None


# --------------------------------------------------------------------------
# specific surface area
# --------------------------------------------------------------------------

def test_specific_surface_area_known_value():
    # SSA = 6 / (rho * d). For a 1 g/cm^3 material with d32 = 10 nm:
    # d_m = 1e-8 m, rho_kg_m3 = 1000, SSA = 6 / (1000 * 1e-8) / 1000 = 600 m^2/g.
    ssa = specific_surface_area(d32_nm=10.0, density_g_cm3=1.0)
    assert ssa == pytest.approx(600.0, rel=1e-9)


def test_specific_surface_area_scales_inversely_with_diameter_and_density():
    base = specific_surface_area(10.0, 1.0)
    assert specific_surface_area(20.0, 1.0) == pytest.approx(base / 2, rel=1e-9)
    assert specific_surface_area(10.0, 2.0) == pytest.approx(base / 2, rel=1e-9)


def test_specific_surface_area_nan_for_nonpositive_inputs():
    assert np.isnan(specific_surface_area(0.0, 1.0))
    assert np.isnan(specific_surface_area(-5.0, 1.0))
    assert np.isnan(specific_surface_area(10.0, 0.0))
