# NanoMetrix Backlog

Small, real, incremental improvements. Each entry should be implemented as an
actual working change (code, tests, or docs) before being checked off — no
placeholder or no-op commits.

- [x] Add a `tests/` directory with pytest unit tests for `metrics.py`
      (equivalent circular diameter, circularity, aspect ratio, solidity
      formulas) using small synthetic contours with known values.
- [x] Add pytest unit tests for `calibration.py`'s `parse_label()` function
      covering common OCR string formats ("100 nm", "100nm", "1 um", "500 Å").
- [x] Add a `requirements-dev.txt` and a `tests/test_pipeline_smoke.py` that
      runs the full pipeline end-to-end on one of the bundled `samples/*.png`
      images and asserts it returns a non-empty particle table.
- [x] Add type hints to the public functions in `metrics.py` and `viz.py`
      that are missing them, and run `mypy` (document any issues found).
      Found and fixed a real narrowing bug in `measure_particles()`; see the
      "Static typing" subsection under Engineering notes in the README.
      - mypy also found 5 pre-existing errors in `synth.py` (out of scope for
        this item); see new backlog item below.
- [x] Add a `CONTRIBUTING.md` with setup instructions (venv, requirements,
      Tesseract OCR install per-OS) distilled from the README's troubleshooting
      notes. Also documents a real gotcha found while writing it:
      `test_pipeline_smoke.py` actually exercises real OCR (unlike
      `test_calibration.py`) and fails without Tesseract installed --
      verified by removing the binary and re-running the suite.
- [x] Add input validation + a clear error message in `pipeline.analyze()`
      for unreadable/corrupt image files, with a regression test.
      `analyze()` takes a decoded array, not a path, so it now validates the
      array itself (type/shape/dtype/finite-ness) with a `ValueError` naming
      the problem, instead of an opaque OpenCV/NumPy failure several calls
      deep. Also fixed the matching file-level bug this surfaced: `cli.py`
      and `app.py` each had their own image loader that let a truncated or
      unidentifiable file's PIL exception escape as a raw traceback;
      extracted both into `nanometrix/io.load_grayscale()`, which raises a
      clear `ValueError` instead, and the CLI now skips a bad file in a
      batch rather than aborting the run. See the new "Input validation"
      subsection under Engineering notes in the README.
- [x] Add a `--output-format` flag to `cli.py` supporting `csv` and `json`
      export of the per-particle table (currently CSV-only).
      Added `--output-format {csv,json}`, plus `--output` as a clearer alias
      for the existing `--csv` flag (kept working unchanged for backward
      compatibility). Format is inferred from the path's extension when
      `--output-format` is omitted, defaulting to CSV when neither says
      otherwise. JSON is written as `to_json(orient="records", indent=2)`.
      See the new "Output format" subsection under Engineering notes in the
      README, and `tests/test_cli_output_format.py`.
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
- [ ] Fix the 5 pre-existing `mypy` errors in `synth.py` (an `Optional`
      default on a non-optional dataclass field, two assignment-type
      mismatches, and a tuple passed where `cv2.GaussianBlur` wants a
      kernel-size sequence), found while type-checking `metrics.py`/`viz.py`.

When this list empties, add 2-3 new concrete, genuinely useful items here
(not busywork) before implementing the next one.
