# NanoMetrix Backlog

Small, real, incremental improvements. Each entry should be implemented as an
actual working change (code, tests, or docs) before being checked off — no
placeholder or no-op commits.

- [x] Add a `tests/` directory with pytest unit tests for `metrics.py`
      (equivalent circular diameter, circularity, aspect ratio, solidity
      formulas) using small synthetic contours with known values.
- [ ] Add pytest unit tests for `calibration.py`'s `parse_label()` function
      covering common OCR string formats ("100 nm", "100nm", "1 um", "500 Å").
- [ ] Add a `requirements-dev.txt` and a `tests/test_pipeline_smoke.py` that
      runs the full pipeline end-to-end on one of the bundled `samples/*.png`
      images and asserts it returns a non-empty particle table.
- [ ] Add type hints to the public functions in `metrics.py` and `viz.py`
      that are missing them, and run `mypy` (document any issues found).
- [ ] Add a `CONTRIBUTING.md` with setup instructions (venv, requirements,
      Tesseract OCR install per-OS) distilled from the README's troubleshooting
      notes.
- [ ] Add input validation + a clear error message in `pipeline.analyze()`
      for unreadable/corrupt image files, with a regression test.
- [ ] Add a `--output-format` flag to `cli.py` supporting `csv` and `json`
      export of the per-particle table (currently CSV-only).
- [ ] Add a GitHub Actions workflow (`.github/workflows/tests.yml`) that
      installs dependencies and runs the pytest suite on push/PR.
- [ ] Add a second, independent half-max-vs-Otsu comparison at a different
      packing density (20%) to `validate.py` and record the result in the
      README's validation table, to check the Section 5.5 conclusion holds
      outside the originally tested density.
- [ ] Add a `--batch` mode to `cli.py` that processes every image in a
      directory and writes one combined summary CSV.
- [ ] Add docstring examples (doctest-style) to the three or four most-used
      functions in `metrics.py` and `calibration.py`.
- [ ] Review and tighten `segmentation.py`'s watershed parameters against the
      packing-density sweep data already in `outputs/validation_results.csv`;
      document any tuning in the README.
- [ ] Add a minimal `Dockerfile` that bundles Tesseract OCR and the Python
      dependencies, so the CLI runs without manual OCR installation.
- [ ] Add a `--confidence-threshold` CLI flag that warns (or exits non-zero)
      when scale-bar OCR confidence falls below a user-set bar.

When this list empties, add 2-3 new concrete, genuinely useful items here
(not busywork) before implementing the next one.
