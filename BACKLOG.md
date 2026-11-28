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
- [x] Add a GitHub Actions workflow (`.github/workflows/tests.yml`) that
      installs dependencies and runs the pytest suite on push/PR.
      Runs on `ubuntu-latest` across a Python 3.11/3.12 matrix, installs the
      `tesseract-ocr` apt package (the suite's `test_pipeline_smoke.py`
      exercises real OCR, per `CONTRIBUTING.md` step 3), installs
      `requirements-dev.txt`, and runs `pytest -v` on push and PR to `main`.
      Verified locally first: full suite (79 tests) passes against this
      environment's installed dependencies before adding the workflow.
- [x] Add a second, independent half-max-vs-Otsu comparison at a different
      packing density (20%) to `validate.py` and record the result in the
      README's validation table, to check the Section 5.5 conclusion holds
      outside the originally tested density.
      Generalized `exp_refinement()` to take `n_particles`/`seed_base`
      parameters (default unchanged, still the original ~39% coverage run)
      and added `exp_refinement_density20()`, which picks `n_particles=34`
      for all four modality/size configs -- chosen by directly measuring
      `SynthConfig(...).coverage` for each config (mean 20.0-21.0% across 5
      seeds), not by reusing `exp_density`'s n_particles-to-coverage mapping,
      which uses different particle sizes and so isn't comparable. Full
      `validate.py` run (new coverage ~20%): Otsu 1.17% size error / 0.925
      circularity vs half-max 1.77% / 0.883 -- same ranking as the original
      ~39% run, so Section 5.5's conclusion holds at this lower density too.
      See the new README table under "Boundary criterion".
- [x] Add a `--batch` mode to `cli.py` that processes every image in a
      directory and writes one combined summary CSV.
      Processing every image in a directory already worked (`cli.py
      samples/ --output all.csv` has done that since `--output-format` was
      added), so there was no gap there; what that produces is one row per
      *particle*, not a summary. Added `--summary PATH` instead of a
      same-named `--batch` flag, since the real gap was the summary output,
      not the directory handling: it writes one row per *image* (particle
      count, calibration, D10/D50/D90/mean/median/span/PDI/SSA/..., mean
      circularity/aspect ratio/solidity, warning count) to a CSV, built from
      the same `AnalysisResult` each image already produces. A corrupt image
      in the batch gets its own row (`n_particles=0` plus the error message)
      instead of silently disappearing, which the per-particle `--output`
      table already does for a skipped file. See the new "Batch summary"
      subsection under Engineering notes in the README, and
      `tests/test_cli_batch_summary.py`.
- [x] Add docstring examples (doctest-style) to the three or four most-used
      functions in `metrics.py` and `calibration.py`.
      Added runnable examples to `measure_particles`, `specific_surface_area`
      and `summarize_sizes` (metrics.py) and `parse_label` (calibration.py)
      -- the module's most-referenced public functions, picked by grepping
      call sites across `pipeline.py`/`cli.py`/tests, and ones with
      deterministic output with no image/Tesseract/GPU dependency.
      `calibrate()` was left out: it needs a real scale-bar image and a live
      OCR read, which isn't guaranteed reproducible across Tesseract
      versions. Wired into the test suite via `tests/test_doctests.py`
      (`doctest.testmod()` on just these two modules, not a blanket
      `--doctest-modules`) so a stale example fails CI instead of silently
      rotting. See the new "Doctest examples" subsection under Engineering
      notes in the README.
- [x] Review and tighten `segmentation.py`'s watershed parameters against the
      packing-density sweep data already in `outputs/validation_results.csv`;
      document any tuning in the README.
      Swept `peak_min_distance_frac` (0.55 default through 0.80) against the
      two degraded packing-density points (130/160 particles, ~68%/71%
      coverage) and found the old default was over-splitting: markers too
      close together cut the watershed boundary through a particle's own
      interior rather than its true neighbour contact, undersizing survivors
      and pulling the population D50 down. Raised the default to 0.70 --
      the point past which precision/D50-error gains slow sharply while
      recall is still within a few points of the old default. Verified no
      regression on the baseline, electron-dose, spatial-resolution and
      boundary-criterion (Otsu vs half-max, both coverages) experiments
      first; re-ran the full `validate.py` suite and refreshed
      `outputs/validation_results.csv`/`validation.png` to match. See the
      new "Watershed peak-distance tuning" subsection under Engineering
      notes in the README.
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
