# Contributing to NanoMetrix

Setup, test, and troubleshooting notes for working on the codebase. This is
the condensed, task-oriented version of the "Install and run" and
"Engineering notes" sections of the [README](README.md).

## 1. Set up a virtual environment

Use a venv so NanoMetrix's pinned dependencies don't collide with anything
else on your system.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
```

`.venv/` is already in `.gitignore`, so it's safe to create it inside the
repo.

## 2. Install Python dependencies

```bash
pip install -r requirements.txt        # runtime deps
pip install -r requirements-dev.txt     # + pytest, for running the test suite
```

`requirements-dev.txt` pulls in `requirements.txt` itself (`-r
requirements.txt`), so installing only the dev file is also sufficient during
development.

## 3. Install Tesseract OCR

NanoMetrix reads the scale bar burnt into each micrograph with
[`pytesseract`](https://github.com/madmaze/pytesseract), which is a thin
wrapper around the **Tesseract** binary — `pip install` alone does not
install the engine itself, only the Python bindings.

**Debian / Ubuntu**
```bash
sudo apt install tesseract-ocr
```

**macOS** (Homebrew)
```bash
brew install tesseract
```

**Windows**
1. Install the binary from the [UB-Mannheim Tesseract build](
   https://github.com/UB-Mannheim/tesseract/wiki) (the standard Windows
   installer; it is not on PyPI).
2. Make sure the install directory (e.g.
   `C:\Program Files\Tesseract-OCR`) is on your `PATH`, **or** point
   `pytesseract` at the binary explicitly before calling into NanoMetrix:
   ```python
   import pytesseract
   pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
   ```

Verify the install with:
```bash
tesseract --version
```

**Tesseract is optional, not required, to run NanoMetrix.** If it isn't
installed, `nanometrix/calibration.py`'s OCR step catches the import/launch
failure, returns no votes, and the pipeline falls back to reporting sizes in
pixels with a warning instead of crashing. Install it when you need
nanometre output (which is most real usage). It's also needed to run the
full test suite — see step 4.

## 4. Run the test suite

```bash
pytest
```

Most of the suite does not need Tesseract: `test_calibration.py` checks
`parse_label()` against hand-written OCR-style strings (`"100 nm"`,
`"100nm"`, `"1 um"`, `"500 Å"`, common misreads like a glyph `'O'` for
`'0'`) rather than running OCR itself. **`test_pipeline_smoke.py` is the
exception** — it runs `analyze()` on the bundled `samples/*.png` images
without an explicit nm/px override and asserts that scale-bar calibration
actually succeeds, so it exercises real OCR end-to-end and needs a working
Tesseract install to pass (verified: without `tesseract` on `PATH`, 5 of its
8 cases fail with "scale bar found but label unreadable"). If `pytest`
fails only in `test_pipeline_smoke.py` with that message, the fix is step 3
above, not a code change.

## 5. Optional: type checking

`metrics.py` and `viz.py` are fully type-hinted and are expected to pass
cleanly:

```bash
pip install mypy
mypy --ignore-missing-imports --follow-imports=silent nanometrix/metrics.py nanometrix/viz.py
```

Running `mypy` over the rest of `nanometrix/` currently surfaces five
pre-existing errors confined to `synth.py` (synthetic-micrograph generation
used for testing/validation, not the measurement path) — tracked as a
BACKLOG.md item, not something a new change needs to fix incidentally.

## 6. Quick sanity check

Before sending a change, confirm the demo pipeline still runs end-to-end:

```bash
python cli.py --demo
```

This generates a synthetic micrograph, analyses it, and writes a report —
useful smoke-test coverage for anything touching `segmentation.py`,
`metrics.py`, or `pipeline.py` beyond what the unit tests exercise.

## Common setup problems

- **`pytesseract.pytesseract.TesseractNotFoundError`** — the Tesseract
  *binary* isn't installed or isn't on `PATH`; see step 3. This is the
  single most common first-run error.
- **OCR reads a plausible but wrong value (e.g. "100 nm" as "200 nm") and
  everything downstream is silently off by that factor** — this is exactly
  why `calibration.py` takes a majority vote across several independent
  preprocessings rather than trusting one OCR pass (see the README's
  Engineering notes). If you hit a case the voting still gets wrong, add it
  as a regression case in `tests/test_calibration.py` rather than patching
  around it ad hoc.
- **`ModuleNotFoundError` for `cv2`, `skimage`, etc.** — you're running
  outside the venv, or skipped step 2.
- **`test_pipeline_smoke.py` fails with "scale bar found but label
  unreadable" but everything else passes** — Tesseract isn't installed or
  isn't on `PATH`; see step 3. `test_calibration.py`'s unit tests don't
  touch real OCR and will still pass in that state, which is a useful way
  to tell a missing-Tesseract failure apart from an actual calibration
  regression.
