"""
Regression tests for unreadable/corrupt-image handling (see BACKLOG.md).

Two layers are covered, matching the two places a bad image used to produce
a raw, unhelpful traceback instead of a clear error:

- `nanometrix.analyze()` takes an already-decoded numpy array, not a file
  path, so "corrupt" at that layer means a malformed array (wrong type,
  shape, or values) reaching the pipeline -- the kind of thing a flaky
  decoder, a truncated file read partway through, or careless glue code can
  hand it. Before this change each case below failed deep inside OpenCV or
  NumPy with an opaque, implementation-specific error.
- `nanometrix.io.load_grayscale()` (used by both `cli.py` and `app.py`) is
  where an actual bad *file* is caught: a file that doesn't exist, is empty,
  isn't an image at all, or is a truncated/corrupt image that PIL can't
  fully decode.
"""

from __future__ import annotations

import numpy as np
import pytest

from cli import main as cli_main
from nanometrix import analyze
from nanometrix.io import load_grayscale


# ---------------------------------------------------------------------------
# analyze(): malformed in-memory image arrays
# ---------------------------------------------------------------------------

def test_analyze_rejects_non_array_input():
    with pytest.raises(ValueError, match="numpy array"):
        analyze("not an array")


def test_analyze_rejects_none():
    with pytest.raises(ValueError, match="numpy array"):
        analyze(None)


def test_analyze_rejects_1d_array():
    with pytest.raises(ValueError, match="2-D"):
        analyze(np.zeros(100, dtype=np.uint8))


def test_analyze_rejects_empty_array():
    with pytest.raises(ValueError, match="empty"):
        analyze(np.zeros((0, 0), dtype=np.uint8))


def test_analyze_rejects_zero_width_array():
    with pytest.raises(ValueError, match="empty"):
        analyze(np.zeros((50, 0), dtype=np.uint8))


def test_analyze_rejects_wrong_channel_count():
    with pytest.raises(ValueError, match="channel"):
        analyze(np.zeros((20, 20, 5), dtype=np.uint8))


def test_analyze_rejects_non_numeric_dtype():
    with pytest.raises(ValueError, match="numeric"):
        analyze(np.zeros((20, 20), dtype=object))


def test_analyze_rejects_nan_filled_image():
    image = np.full((20, 20), np.nan, dtype=np.float32)
    with pytest.raises(ValueError, match="NaN|infinite"):
        analyze(image)


def test_analyze_rejects_inf_filled_image():
    image = np.full((20, 20), np.inf, dtype=np.float32)
    with pytest.raises(ValueError, match="NaN|infinite"):
        analyze(image)


def test_analyze_accepts_a_well_formed_array():
    # A valid (if boring) array should sail through validation and reach
    # the real pipeline rather than bouncing off the new checks.
    image = np.full((64, 64), 128, dtype=np.uint8)
    result = analyze(image)
    assert result.n_particles == 0
    assert "No particles" in result.report()


# ---------------------------------------------------------------------------
# load_grayscale() / the CLI: actual bad files on disk
# ---------------------------------------------------------------------------

def test_load_grayscale_rejects_missing_file(tmp_path):
    missing = tmp_path / "does_not_exist.png"
    with pytest.raises(ValueError, match="not found"):
        load_grayscale(missing)


def test_load_grayscale_rejects_empty_file(tmp_path):
    empty = tmp_path / "empty.png"
    empty.touch()
    with pytest.raises(ValueError, match="empty"):
        load_grayscale(empty)


def test_load_grayscale_rejects_non_image_file(tmp_path):
    not_an_image = tmp_path / "notes.png"
    not_an_image.write_text("this is plain text, not an image")
    with pytest.raises(ValueError, match="not a recognisable image"):
        load_grayscale(not_an_image)


def test_load_grayscale_rejects_truncated_png(tmp_path):
    truncated = tmp_path / "truncated.png"
    # A valid PNG signature, then a chunk header whose declared length runs
    # past the (short) actual file -- a truncated download or write, which
    # PIL detects while parsing the header in Image.open() itself.
    truncated.write_bytes(b"\x89PNG\r\n\x1a\n" + b"garbagegarbagegarbage")
    with pytest.raises(ValueError, match="corrupt or"):
        load_grayscale(truncated)


def test_cli_skips_corrupt_file_but_still_processes_the_rest(tmp_path):
    """A corrupt image in a batch should be reported and skipped, not take
    down the whole run (the previous behaviour: an unhandled exception)."""
    good = tmp_path / "images"
    good.mkdir()
    corrupt = good / "bad.png"
    corrupt.write_bytes(b"\x89PNG\r\n\x1a\n" + b"garbage" * 8)

    from PIL import Image
    Image.fromarray(np.full((80, 80), 200, dtype=np.uint8)).save(
        good / "fine.png"
    )

    exit_code = cli_main([str(good), "--outdir", str(tmp_path / "out")])
    # One bad file among good ones is a reportable failure (non-zero exit),
    # but it must not have crashed before the good file was analysed.
    assert exit_code == 1
    assert (tmp_path / "out" / "fine_report.txt").exists()
    assert not (tmp_path / "out" / "bad_report.txt").exists()


def test_cli_reports_not_found_path():
    assert cli_main(["/no/such/path/at/all.png"]) == 1
