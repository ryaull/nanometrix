"""
Tests for cli.py's --output-format flag.

Backlog: the combined per-particle table written by `--csv`/`--output` was
CSV-only. These tests cover the new JSON export path, the format-from-
extension inference, and the explicit --output-format override -- plus the
--output alias for the (kept, for backward compatibility) --csv flag name.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from cli import main as cli_main

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"
SAMPLE_IMAGE = sorted(SAMPLES_DIR.glob("*.png"))[0]


def _run(tmp_path, *extra_args) -> int:
    return cli_main([
        str(SAMPLE_IMAGE),
        "--outdir", str(tmp_path),
        "--no-figures",
        *extra_args,
    ])


def test_json_extension_is_inferred_and_matches_csv_row_count(tmp_path):
    json_path = tmp_path / "all.json"
    assert _run(tmp_path, "--csv", str(json_path)) == 0
    assert json_path.exists()

    records = json.loads(json_path.read_text())
    assert isinstance(records, list) and records
    assert "ecd_nm" in records[0]
    assert "image" in records[0]

    csv_path = tmp_path / "all.csv"
    assert _run(tmp_path, "--csv", str(csv_path)) == 0
    combined = pd.read_csv(csv_path)
    assert len(records) == len(combined)


def test_output_format_flag_overrides_extension(tmp_path):
    # .csv extension, but an explicit --output-format json should win.
    out_path = tmp_path / "all.csv"
    assert _run(tmp_path, "--csv", str(out_path), "--output-format", "json") == 0

    records = json.loads(out_path.read_text())
    assert isinstance(records, list) and records


def test_output_format_json_flag_with_extensionless_path(tmp_path):
    out_path = tmp_path / "all_particles"
    assert _run(tmp_path, "--output", str(out_path), "--output-format", "json") == 0

    records = json.loads(out_path.read_text())
    assert isinstance(records, list) and records


def test_output_flag_is_an_alias_for_csv(tmp_path):
    out_path = tmp_path / "all.csv"
    assert _run(tmp_path, "--output", str(out_path)) == 0

    combined = pd.read_csv(out_path)
    assert not combined.empty


def test_default_format_without_output_format_flag_is_still_csv(tmp_path):
    # No --output-format and no recognizable extension: must not silently
    # become JSON -- CSV stays the default when nothing says otherwise.
    out_path = tmp_path / "all.dat"
    assert _run(tmp_path, "--csv", str(out_path)) == 0

    # A CSV file, read back as CSV, should parse into the same columns the
    # rest of the suite expects -- reading it as JSON would raise instead.
    combined = pd.read_csv(out_path)
    assert not combined.empty
    assert "ecd_nm" in combined.columns
