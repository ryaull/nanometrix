"""
Tests for cli.py's --summary flag (BACKLOG.md: "Add a --batch mode to cli.py
that processes every image in a directory and writes one combined summary
CSV.").

Processing every image in a directory into one combined table already
worked before this change -- `cli.py samples/ --output all.csv` has done
that since `--output-format` was added, one row per *particle*. What was
missing is a true *summary*: one row per *image* (particle count,
calibration, size/shape statistics), which is what `--summary` adds here.
These tests cover: running it over a whole folder, the row count matching
the number of images, the presence of the size/shape columns, and that a
corrupt image in the batch gets its own (empty-stats) row instead of
silently vanishing from the table the way it already does from --output.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from cli import main as cli_main

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"
SAMPLE_IMAGES = sorted(SAMPLES_DIR.glob("*.png"))


def test_summary_has_one_row_per_image_in_a_folder(tmp_path):
    summary_path = tmp_path / "summary.csv"
    exit_code = cli_main([
        str(SAMPLES_DIR), "--outdir", str(tmp_path / "out"),
        "--no-figures", "--summary", str(summary_path),
    ])
    assert exit_code == 0
    assert summary_path.exists()

    summary = pd.read_csv(summary_path)
    assert len(summary) == len(SAMPLE_IMAGES)
    assert set(summary["image"]) == {p.name for p in SAMPLE_IMAGES}


def test_summary_contains_size_and_shape_columns(tmp_path):
    summary_path = tmp_path / "summary.csv"
    one_image = SAMPLE_IMAGES[0]
    assert cli_main([
        str(one_image), "--outdir", str(tmp_path / "out"),
        "--no-figures", "--summary", str(summary_path),
    ]) == 0

    summary = pd.read_csv(summary_path)
    assert len(summary) == 1
    row = summary.iloc[0]
    for col in ("n_particles", "d10", "d50", "d90", "mean", "median",
                "cv_percent", "pdi", "mean_circularity",
                "mean_aspect_ratio", "mean_solidity"):
        assert col in summary.columns, f"missing column: {col}"
    assert row["n_particles"] > 0
    assert row["d50"] > 0


def test_summary_row_count_matches_particle_table_image_count(tmp_path):
    # The per-particle --output table has one row per particle, grouped by
    # "image"; --summary should produce exactly one row per distinct image
    # in that table.
    summary_path = tmp_path / "summary.csv"
    particles_path = tmp_path / "particles.csv"
    assert cli_main([
        str(SAMPLES_DIR), "--outdir", str(tmp_path / "out"), "--no-figures",
        "--summary", str(summary_path), "--output", str(particles_path),
    ]) == 0

    summary = pd.read_csv(summary_path)
    particles = pd.read_csv(particles_path)
    assert len(summary) == particles["image"].nunique()


def test_summary_still_written_when_one_image_is_corrupt(tmp_path):
    folder = tmp_path / "images"
    folder.mkdir()
    Image.fromarray(np.full((80, 80), 200, dtype=np.uint8)).save(
        folder / "fine.png"
    )
    corrupt = folder / "bad.png"
    corrupt.write_bytes(b"\x89PNG\r\n\x1a\n" + b"garbage" * 8)

    summary_path = tmp_path / "summary.csv"
    exit_code = cli_main([
        str(folder), "--outdir", str(tmp_path / "out"), "--no-figures",
        "--summary", str(summary_path),
    ])
    # A corrupt file in the batch is still a reportable failure...
    assert exit_code == 1

    # ...but it must show up in the summary (with its error message and no
    # stats), rather than just vanishing the way it already does from the
    # per-particle --output table.
    summary = pd.read_csv(summary_path)
    assert set(summary["image"]) == {"fine.png", "bad.png"}
    bad_row = summary[summary["image"] == "bad.png"].iloc[0]
    assert bad_row["n_particles"] == 0
    assert isinstance(bad_row["error"], str) and bad_row["error"]

    good_row = summary[summary["image"] == "fine.png"].iloc[0]
    assert good_row["n_particles"] == 0  # a flat 200-value image has none
    assert good_row["error"] == "" or pd.isna(good_row["error"])


def test_no_summary_file_written_without_the_flag(tmp_path):
    summary_path = tmp_path / "summary.csv"
    assert cli_main([
        str(SAMPLE_IMAGES[0]), "--outdir", str(tmp_path / "out"),
        "--no-figures",
    ]) == 0
    assert not summary_path.exists()
