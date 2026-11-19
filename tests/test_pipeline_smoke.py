"""
End-to-end smoke tests for the full analysis pipeline.

Unlike test_metrics.py and test_calibration.py, which exercise one function
each against synthetic, hand-constructed inputs, these tests run the real
path a user takes: load one of the bundled samples/*.png micrographs the
way cli.py does, hand it to nanometrix.analyze(), and check that a complete,
non-empty particle table comes back with the columns downstream code (the
CLI, viz.py, the README's examples) relies on.

These are deliberately loose on *numbers* -- validate.py and the ground
truth in samples/ground_truth.json already cover accuracy. The job here is
narrower: catch a pipeline-breaking regression (an exception, an empty
table, calibration silently failing on an image whose scale bar was always
readable, a missing column) that a pure-function unit test can't see
because it never calls calibrate() -> segment() -> measure_particles()
together on a real image.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cli import load_image, main as cli_main
from nanometrix import analyze

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"
SAMPLE_IMAGES = sorted(SAMPLES_DIR.glob("*.png"))

EXPECTED_COLUMNS = {
    "label", "centroid_row", "centroid_col", "area_px", "perimeter_px",
    "ecd_px", "ecd_nm", "area_nm2", "feret_max_nm", "major_axis_nm",
    "minor_axis_nm", "circularity", "aspect_ratio", "solidity",
    "eccentricity", "orientation_deg",
}


def test_sample_images_are_present():
    # If this fails, every test below is meaningless -- make that obvious
    # rather than letting them all report a confusing "no particles found".
    assert SAMPLE_IMAGES, f"no sample images found under {SAMPLES_DIR}"


@pytest.mark.parametrize("image_path", SAMPLE_IMAGES, ids=lambda p: p.stem)
def test_full_pipeline_returns_nonempty_particle_table(image_path):
    image = load_image(image_path)
    result = analyze(image, material="Au")

    df = result.particles
    assert isinstance(df, pd.DataFrame)
    assert not df.empty, f"{image_path.name}: pipeline found zero particles"
    assert EXPECTED_COLUMNS.issubset(df.columns)

    # Every bundled sample has a genuine, burnt-in scale bar (see
    # samples/ground_truth.json), so a run that falls back to pixel units
    # here is a calibration regression, not an expected outcome.
    assert result.calibration.ok, (
        f"{image_path.name}: scale bar calibration failed "
        f"({result.calibration.note})"
    )
    assert result.unit == "nm"
    assert result.size_column == "ecd_nm"

    # Sizes and shape descriptors must be finite, positive numbers -- NaNs
    # or negatives here would mean a formula blew up on real segmentation
    # output rather than the clean synthetic contours test_metrics.py uses.
    sizes = df["ecd_nm"].to_numpy(dtype=float)
    assert np.all(np.isfinite(sizes))
    assert np.all(sizes > 0)
    for col in ("circularity", "solidity"):
        assert np.all(np.isfinite(df[col].to_numpy(dtype=float)))

    # report() stitches together calibration/segmentation/stats text; it
    # should never raise, and should actually mention how many particles.
    report = result.report()
    assert isinstance(report, str) and report.strip()


def test_cli_runs_end_to_end_on_one_sample(tmp_path):
    """Drive the same code path a user invokes from the command line
    (`python cli.py <image>`), rather than calling analyze() directly."""
    image_path = SAMPLE_IMAGES[0]
    exit_code = cli_main([
        str(image_path),
        "--outdir", str(tmp_path),
        "--no-figures",
    ])
    assert exit_code == 0


def test_cli_batch_mode_writes_combined_csv(tmp_path):
    csv_path = tmp_path / "all.csv"
    exit_code = cli_main([
        str(SAMPLES_DIR),
        "--outdir", str(tmp_path),
        "--no-figures",
        "--csv", str(csv_path),
    ])
    assert exit_code == 0
    assert csv_path.exists()

    combined = pd.read_csv(csv_path)
    assert not combined.empty
    # One "image" value per sample that contributed particles.
    assert combined["image"].nunique() == len(SAMPLE_IMAGES)
