# NanoMetrix

[![Tests](https://github.com/ryaull/nanometrix/actions/workflows/tests.yml/badge.svg)](https://github.com/ryaull/nanometrix/actions/workflows/tests.yml)

**Automated nanoparticle size and shape metrology from electron micrographs.**

Measures the particle size distribution in an SEM or TEM image automatically,
including reading the instrument's own scale bar so results come back in
nanometres rather than pixels.

---

## The problem

Particle size distribution is the single most important characterisation
metric for any nanomaterial. It controls catalytic activity, optical
response, cellular uptake, sintering behaviour and toxicity — the reason a
20 nm particle behaves differently from the same mass of bulk powder is that
surface area per gram scales as 1/d.

In most labs it is still measured by hand. A researcher opens a micrograph in
ImageJ and clicks across particles one at a time. The procedure is slow
(hundreds of particles per image for decent statistics), subjective, and not
reproducible — two people measuring the same image get different numbers,
and so does the same person on a different day.

NanoMetrix automates the whole path from image to reportable statistics.

## What it does

1. **Finds and reads the scale bar** by OCR, converting pixels to nanometres.
   This is the part that makes it a measurement rather than a picture: a
   diameter in pixels is not a length, since the same particle imaged at two
   magnifications would get two different "sizes".
2. **Segments individual particles**, separating ones that touch — which is
   the hard part, not an afterthought (see below).
3. **Measures each particle**: equivalent circular diameter, Feret diameter,
   circularity, aspect ratio, solidity, eccentricity.
4. **Fits the population**: log-normal median and GSD, D10/D50/D90, Sauter
   mean D[3,2], De Brouckere mean D[4,3], polydispersity index, and a
   bootstrap confidence interval on the median.
5. **Estimates specific surface area** from D[3,2] and the bulk density.
6. **Exports** a per-particle CSV, an annotated overlay, and distribution plots.

### Why separating touching particles is the whole game

Thresholding a micrograph is easy, and on a sparse sample that is the entire
job. Real samples are not sparse: particles dry into contact, and every
touching pair becomes one connected blob. A blob of two 40 nm spheres has the
area of a single 57 nm sphere — so naive thresholding does not merely
miscount, it invents a large-particle tail that was never in the sample and
drags the reported mean upward. A distance-transform watershed splits those
contacts, and the validation below quantifies exactly how well.

Particles clipped by the frame edge are discarded, following standard
stereological practice: only part of such a particle is visible, so measuring
it reports a diameter smaller than the truth, and since larger particles are
likelier to touch an edge the bias falls on the whole distribution.

## Install and run

```bash
pip install -r requirements.txt
sudo apt install tesseract-ocr        # needed for scale-bar OCR
```

```bash
python cli.py --demo                              # generate and analyse
python cli.py samples/sem_au_40nm.png --material Au
python cli.py samples/ --outdir outputs --csv all.csv --material Au
python cli.py samples/ --output all.json --output-format json --material Au
python cli.py samples/ --outdir outputs --summary batch_summary.csv  # one row per image
python cli.py image.tif --nm-per-px 0.37          # skip OCR
streamlit run app.py                              # web interface
python validate.py                                # reproduce the results below
```

As a library:

```python
from nanometrix import analyze
from PIL import Image
import numpy as np

result = analyze(np.array(Image.open("micrograph.png").convert("L")),
                 material="Au")
print(result.report())
result.particles.to_csv("particles.csv", index=False)
```

## Validation

The tool is graded against a **synthetic ground truth**, because real
micrographs have no true answer — the "ground truth" in published datasets is
itself a human with a mouse, and agreeing with it would only prove the tool
reproduces a human's bias. So the problem is inverted: `synth.py` renders
micrographs from diameters *we chose*, applying the degradations a real
instrument imposes (Poisson shot noise, probe blur, illumination gradient,
vignetting, a burnt-in scale bar), and the pipeline is asked to recover the
numbers it started from.

The generator models physics rather than appearance: a log-normal size
population (nucleation-and-growth is multiplicative, so ln d is what adds up),
SEM rim brightening from secondary-electron yield at tilted surfaces, and TEM
mass-thickness contrast as *exponential* attenuation, I = I₀exp(−t/λ).

**Baseline accuracy** — 20 images, SEM and TEM, ~40% areal coverage:

| Metric | Result |
|---|---|
| Scale bar read correctly | 100% of images, 0.18% mean calibration error |
| Detection recall | 99.0% |
| Detection precision | 98.6% |
| Per-particle size error | 1.60% (mean absolute) |
| Population D50 error | 1.82% (mean absolute) |
| True vs measured D50 | r = 0.9996 |

**Packing density** — the dominant error source, and the reason the watershed
stage exists:

| Coverage | Recall | D50 error |
|---|---|---|
| 7% | 100% | +0.7% |
| 28% | 98.2% | −0.0% |
| 44% | 99.2% | −0.3% |
| 60% | 97.6% | −1.1% |
| 68% | 88.8% | −2.2% |
| 71% | 86.5% | −1.1% |

Accuracy holds to about 60% coverage. Beyond that, watershed still cuts into
some touching particles at their contact necks, and recall falls as more
clusters are left merged rather than split — so dilute the sample rather than
trusting a crowded image. The watershed peak-distance parameter was tuned
against this exact sweep to trade a few points of high-density recall for
better size accuracy and precision; see "Watershed peak-distance tuning"
under Engineering notes.

**Spatial resolution** — how few pixels a particle can span:

| Diameter (px) | Size error |
|---|---|
| 13 | 16.9% |
| 20 | 11.0% |
| 30 | 6.7% |
| 45 | 3.5% |
| 70 | 1.1% |
| 100 | 1.0% |

**Image particles at ≥45 px across for better than 5% accuracy**; below about
20 px, blur comparable to the particle itself inflates every diameter.

**Electron dose** — detection stays above 99% down to 15 counts/pixel and
collapses at 8, so low-dose imaging of beam-sensitive samples is viable.

**Boundary criterion** — two methods were implemented and compared rather than
assumed. Re-cutting each particle at its own half-maximum intensity is the
textbook convention for a blurred edge, but measured against ground truth it
*lost* to a global Otsu threshold (1.85% vs 1.37% size error; circularity
0.886 vs 0.944). The reason is specific: the half-max rule assumes a
flat-topped intensity profile, but an SEM particle's brightest pixels are its
rim, not its centre, so the "interior level" the rule needs is systematically
wrong. It remains available via `--refine`, defaulted off.

This comparison was run a second time at a different packing density, to
check the conclusion wasn't an artefact of the ~39% coverage it was first
measured at:

| Coverage | Method | Size error | Bias | D50 error | Circularity |
|---|---|---|---|---|---|
| ~39% | Global Otsu | 1.37% | −0.91% | 0.81% | 0.944 |
| ~39% | Half-maximum | 1.85% | −1.59% | 1.79% | 0.886 |
| ~20% | Global Otsu | 1.17% | +0.00% | 0.78% | 0.925 |
| ~20% | Half-maximum | 1.77% | −1.60% | 1.88% | 0.883 |

Otsu wins on size error and circularity at both densities, by a similar
margin. That's consistent with the stated mechanism: the rim-brightest
profile that breaks the half-max rule doesn't depend on how many particles
are touching, so there was no reason to expect the result to flip at lower
density, and it didn't. See `exp_refinement_density20()` in `validate.py`.

![validation](outputs/validation.png)

## Repository layout

```
nanometrix/
  synth.py          physics-based micrograph generator with exact ground truth
  calibration.py    scale-bar detection, OCR with majority voting, nm/px
  segmentation.py   CLAHE, denoise, polarity, Otsu, watershed, border rejection
  metrics.py        per-particle descriptors, log-normal fit, D32/D43, SSA
  viz.py            annotated overlay, size distribution, shape plots
  pipeline.py       end-to-end orchestration
cli.py              batch command line
app.py              Streamlit web interface
validate.py         the ground-truth experiments above
samples/            synthetic micrographs + ground_truth.json
.github/workflows/  CI: runs the pytest suite on push/PR (tests.yml)
```

## Engineering notes

Three bugs found by validation that would not have surfaced otherwise, each
of which silently produced plausible-looking wrong answers:

- **Polarity detection by histogram mode** inverted on exactly the images the
  tool targets. Strongly scattering particles saturate, so their interiors
  pile into one tall histogram spike while the background's shot noise
  spreads across many bins — making the particles the mode. The background
  was then segmented as a single enormous particle and every real particle
  discarded. Replaced with a topological test: background is one connected
  sheet, particles are many separate blobs, so the more fragmented phase is
  the particles.
- **Single-pass OCR silently doubled the scale.** "100 nm" read as "200 nm"
  is a clean factor-of-two error in every size reported afterwards, with
  nothing downstream to catch it, because 200 nm is just as plausible a bar
  as 100 nm. Fixed by running several independent preprocessings and taking
  the majority; vote agreement now feeds the reported confidence.
- **Half-max refinement thresholded the raw image**, so shot noise straddling
  the level flipped pixels independently and produced ragged boundaries,
  which both wrecked the shape descriptors and biased diameters low.

Known limits: particles must be roughly convex for watershed to split them
correctly; heavy aggregates are reported as single objects and flagged by low
circularity rather than silently measured; the half-max option underperforms
the default; and OCR needs a scale bar with a legible burnt-in label, falling
back to pixel units with a warning when none is found.

### Input validation

`analyze()` takes an already-decoded array, not a file path, so "corrupt
image" isn't a decode error at that layer -- it's a malformed array reaching
the pipeline. Before validation was added, each of the following failed
several calls deep with an opaque, implementation-specific error instead of
a message naming the problem: `None` or a non-array (`AttributeError:
'NoneType' object has no attribute 'ndim'`), a 1-D array (an OpenCV
"Invalid number of channels" assertion from the internal RGB-to-grayscale
conversion), an empty array (an OpenCV `!_src.empty()` assertion from
`morphologyEx`), and a NaN-filled float array (silently cast to garbage
`uint8` values with only a `RuntimeWarning`, no error at all). `analyze()`
now checks the array's type, dimensionality, shape and dtype up front and
raises a `ValueError` describing exactly what's wrong.

The file-loading helpers (`nanometrix.io.load_grayscale`, used by both
`cli.py` and `app.py`) have the matching fix one layer up: a truncated or
unidentifiable file used to propagate PIL's own exception
(`UnidentifiedImageError`, or a bare `OSError("Truncated File Read")`,
triggered lazily by `Image.load()`/`np.asarray()` rather than by
`Image.open()` itself) as a raw traceback. It's now caught and re-raised as
a `ValueError` naming the file, and the CLI skips a bad file in a batch
rather than aborting the whole run.

### Output format

The CLI's combined-table flag was CSV-only; it now also writes JSON.
`--csv`/`--output` take either a `.csv` or `.json` path, and `--output-format
{csv,json}` overrides whatever the extension would otherwise imply (useful
for an extensionless path, or to force JSON onto a `.csv`-named file). `--csv`
is kept as an alias of `--output` rather than removed, so existing scripts
and the README examples above that predate `--output` keep working unchanged.
JSON is written with `DataFrame.to_json(orient="records", indent=2)` -- a
list of per-particle objects, the same shape `pd.read_json` round-trips and
that a web client can `JSON.parse` directly, as opposed to the default
`to_json()` orientation (one object per column) that neither consumer wants.

### Batch summary

Pointing the CLI at a folder already analyses every image in it and can
combine the results into one `--output`/`--csv` table -- but that table is
one row per *particle*, so getting per-image numbers out of a 50-image batch
meant loading it and grouping by `image` yourself. `--summary PATH` writes
the per-image numbers directly: one row per image with its particle count,
calibration (nm/px, OCR source and confidence), the full `SizeStats` table
(mean/median/D10/D50/D90/span/log-normal fit/D32/D43/PDI/CV%/SSA/...), mean
circularity/aspect ratio/solidity, and a warning count -- everything
`result.report()` prints for one image, flattened into a row. A corrupt
image already gets skipped (with a message) rather than aborting the whole
batch; `--summary` now also gives it its own row (`n_particles=0` plus the
error text) instead of letting it disappear from the table the way it
already does from `--output`.

### Static typing

`metrics.py` and `viz.py` are fully type-hinted on their public functions and
pass `mypy --ignore-missing-imports --follow-imports=silent` with zero
issues. Getting there found one real latent bug: `measure_particles()`
computed its pixel-to-nanometre scale factor as
`float(nm_per_px) if calibrated else 1.0`, where `calibrated` is a *separate*
boolean that happens to imply `nm_per_px is not None` -- true at runtime, but
invisible to a type checker, which flagged passing an `Optional[float]` to
`float()`. Harmless in practice (the branch is never taken with `None`), but
the kind of thing that stops being harmless the next time someone edits the
surrounding logic. Fixed by making the `is not None` check part of the
condition itself, so the narrowing is explicit rather than implied.

Running `mypy` over the rest of the package (`nanometrix/`) surfaces five
pre-existing errors confined to `synth.py` (an `Optional` default on a
non-optional dataclass field, two assignment-type mismatches, and a tuple
passed where `cv2.GaussianBlur` wants a kernel-size sequence) -- all in
synthetic-micrograph generation used for testing and validation, not in the
measurement path. They're out of scope for this pass; tracked as a backlog
item.

### Doctest examples

`measure_particles()`, `specific_surface_area()` and `summarize_sizes()` in
`metrics.py`, and `parse_label()` in `calibration.py`, now carry runnable
doctest-style examples in their docstrings -- chosen as the module's
most-called public functions (by reference count across `pipeline.py`,
`cli.py` and the test suite), and ones whose examples can be deterministic
without an image, Tesseract or a GPU. `calibrate()` was deliberately left
out: a meaningful example needs a real scale-bar image and Tesseract OCR,
whose read is not guaranteed bit-for-bit reproducible across OCR engine
versions, which would make the doctest itself flaky rather than a reliable
usage example.

Two numeric traps the examples needed to route around: `round()` on a value
pulled from a pandas `DataFrame` column returns a numpy scalar
(`np.float64`), whose `repr()` under numpy >= 2.0 is `np.float64(11.28)`, not
`11.28` -- so the measure_particles examples cast to `float()` before
rounding, matching what `SizeStats`'s fields already do internally. And a
10x10 pixel square's equivalent circular diameter (ECD, from area) is
*larger* than its side length and its circularity is below 1 -- expected,
not a bug: a square has more perimeter than a circle of the same area, which
is exactly what circularity measures.

These examples are wired into the test suite (not just decorative) by
`tests/test_doctests.py`, which runs `doctest.testmod()` against both
modules and fails if an example's output goes stale. It collects only these
two modules rather than enabling pytest's blanket `--doctest-modules`, so a
`>>>` appearing in another module's explanatory (non-runnable) docstring
text -- several modules in this codebase use `>>>`-free prose instead, but
nothing rules out one doing so later -- can never be mistaken for a test.

### Watershed peak-distance tuning

`segmentation.py`'s watershed split finds one marker per particle by taking
local maxima of the distance transform at least `peak_min_distance_frac *
estimated_radius_px` apart; two markers closer than that collapse into one,
so a lower value splits more aggressively. It defaulted to 0.55, chosen
without reference to the packing-density sweep that `validate.py` now
produces. Sweeping it against that exact sweep (`outputs/validation_results.csv`,
`n_particles_cfg` 130/160, the two points where accuracy was already known to
degrade) shows the old default was over-splitting, not under-splitting: a
closer pair of markers cuts the watershed boundary through a particle's own
interior rather than its true neighbour-to-neighbour contact, which undersizes
both resulting fragments and pulls the population median down.

| peak_min_distance_frac | Recall @130 | Precision @130 | D50 err @130 | Recall @160 | Precision @160 | D50 err @160 |
|---|---|---|---|---|---|---|
| 0.55 (old default) | 91.6% | 88.8% | −3.8% | 89.9% | 88.9% | −2.8% |
| 0.60 | 90.2% | 91.4% | −3.1% | 88.2% | 91.8% | −2.0% |
| 0.65 | 90.2% | 92.9% | −2.9% | 87.1% | 94.3% | −1.5% |
| **0.70 (new default)** | **88.8%** | **95.3%** | **−2.2%** | **86.5%** | **96.0%** | **−1.1%** |
| 0.75 | 87.9% | 96.1% | −1.8% | 86.5% | 97.5% | −0.6% |
| 0.80 | 87.6% | 97.5% | −1.1% | 85.1% | 98.0% | +0.0% |

The trade-off is monotonic in both directions: raising the value keeps
recovering size accuracy and precision all the way to 0.80, at a continuing
recall cost. 0.70 was picked as the point past which precision and D50 error
stop improving quickly (mean abs. per-particle size error at 160 particles:
2.10% at 0.55, down to 2.06% at 0.70, 2.03% at 0.80 -- most of the easy gain
is already had by 0.70) while recall is still within a few points of the old
default, rather than pushed to the extreme that buys the least additional
accuracy for the most additional recall.

Checked for regressions elsewhere before adopting: the baseline, electron-dose
and spatial-resolution sweeps (all well below the ~68% coverage where this
matters) are unchanged or very slightly improved -- e.g. the dose=8
(heaviest-noise) point improves from −9.5% to −2.2% D50 error, since noise
also fragments distance-transform peaks the same way crowding does. The
boundary-criterion A/B (global Otsu vs. half-maximum, both at ~39% and ~20%
coverage, well under the regime this change targets) is unaffected to two
decimal places and Otsu still wins by the same margin, so neither of the
README's two existing boundary-criterion tables needed to change.

## Possible extensions

Swap the classical segmenter for a U-Net or SAM-based one and re-run
`validate.py` unchanged — the harness is model-agnostic, so the comparison is
apples to apples. Other directions: 3D size from tilt series, automatic
aggregate classification, and validation against the public NFFA-EUROPE SEM
dataset (~21,000 annotated nanoscience micrographs).
