"""
Synthetic electron-micrograph generator with exact ground truth.

Why this module exists
----------------------
You cannot validate a measurement tool against real micrographs, because
nobody knows the true diameter of a real nanoparticle to begin with -- the
"ground truth" in published datasets is itself a human with a mouse. So we
invert the problem: draw particles whose diameters we *chose*, render a
micrograph that imitates the degradations of a real instrument, and then ask
the pipeline to recover the numbers we started from.

Physical choices, and the reason for each
-----------------------------------------
* Log-normal size distribution. Nucleation-and-growth syntheses (citrate
  reduction of Au, sol-gel, co-precipitation) produce log-normal, not
  Gaussian, size populations: growth is multiplicative, so the logarithm of
  the diameter is the quantity that adds up. Reporting a log-normal fit is
  standard practice in nanoparticle characterisation.
* SEM contrast: particles bright, with a bright rim. Secondary-electron
  yield rises where the surface tilts away from the detector, so sphere
  edges glow. This rim is exactly what makes naive thresholding merge
  neighbouring particles, which is why the pipeline needs watershed.
* TEM contrast: particles dark on a bright field. Mass-thickness contrast --
  heavy atoms scatter electrons out of the aperture, so thick regions are
  dark, and a sphere is thickest at its centre.
* Poisson noise. Image formation is electron counting, so the noise is shot
  noise and scales as the square root of signal, not additively.
* Vignetting and a slow illumination gradient, which is what breaks a single
  global Otsu threshold and motivates CLAHE plus local thresholding.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------

@dataclass
class SynthConfig:
    """Parameters of one simulated micrograph."""

    width: int = 1024
    height: int = 768

    nm_per_px: float = 0.5           # instrument calibration we will hide in a scale bar
    median_diameter_nm: float = 40.0  # log-normal median (the "D50" we aim to recover)
    gsd: float = 1.25                # geometric standard deviation; 1.0 = perfectly monodisperse
    n_particles: int = 70

    modality: str = "SEM"            # "SEM" (bright particles) or "TEM" (dark particles)
    tem_opacity: float = 3.5         # particle thickness in mean free paths; higher = darker, sharper

    allow_touching: bool = True      # let particles touch, so watershed actually matters
    allow_clipped_edges: bool = True  # let particles run off the frame, as they really do
    min_centre_gap_frac: float = 0.97  # centre distance / summed radii; ~1.0 = just touching
    max_place_attempts: int = 400

    noise_electrons: float = 110.0   # mean electron count per pixel; lower = noisier
    gaussian_read_noise: float = 2.5
    illumination_gradient: float = 0.18  # peak-to-peak fractional brightness drift
    vignette: float = 0.12
    blur_sigma_px: float = 1.1       # finite probe size / point-spread function

    draw_scale_bar: bool = True
    scale_bar_target_frac: float = 0.22  # bar aims to span ~this fraction of image width
    banner_height_px: int = 0        # set >0 to add an instrument info banner

    seed: int | None = 0


@dataclass
class SynthResult:
    """A rendered micrograph plus the truth used to make it."""

    image: np.ndarray                      # uint8, (H, W)
    diameters_nm: np.ndarray               # true diameter of every placed particle
    centres_px: np.ndarray                 # (N, 2) as (row, col)
    clipped: np.ndarray                    # bool, True where the frame cuts the particle
    nm_per_px: float
    scale_bar_nm: float | None
    scale_bar_px: float | None
    config: SynthConfig
    labels: np.ndarray = field(default=None, repr=False)  # instance mask, 0 = background

    @property
    def n_placed(self) -> int:
        return int(len(self.diameters_nm))

    @property
    def diameters_complete_nm(self) -> np.ndarray:
        """Truth restricted to fully visible particles.

        This is the fair comparison set. The analyser deliberately discards
        particles the frame cuts through, because only part of such a
        particle is visible and measuring it would report a diameter smaller
        than the truth. Grading it against clipped particles it never claimed
        to measure would penalise it for being correct.
        """
        return self.diameters_nm[~self.clipped]

    @property
    def coverage(self) -> float:
        """Fraction of the frame occupied by particles.

        This is the single best predictor of how hard an image is to
        segment: below ~0.3 particles are mostly isolated, above ~0.5 they
        form connected clusters that only watershed can separate.
        """
        if self.labels is None:
            return float("nan")
        return float((self.labels > 0).mean())


# --------------------------------------------------------------------------
# size sampling
# --------------------------------------------------------------------------

def sample_lognormal_diameters(
    median_nm: float, gsd: float, n: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw n diameters from a log-normal law specified by median and GSD.

    Parameterising by median and geometric standard deviation is the
    convention in aerosol and colloid science: sigma = ln(GSD) is the
    standard deviation of ln(diameter), and the median is exp(mu). A GSD
    under about 1.2 is normally called monodisperse.
    """
    if gsd <= 1.0:
        return np.full(n, float(median_nm))
    sigma = np.log(gsd)
    mu = np.log(median_nm)
    return rng.lognormal(mean=mu, sigma=sigma, size=n)


# --------------------------------------------------------------------------
# single-particle rendering
# --------------------------------------------------------------------------

def _particle_profile(
    radius_px: float, modality: str, tem_opacity: float = 3.5
) -> np.ndarray:
    """Radial intensity profile of one particle, as a small square patch.

    Values are a signed contrast in [0, 1]; the caller decides whether to add
    or subtract them from the background.
    """
    r = max(radius_px, 0.75)
    size = int(np.ceil(2 * r)) + 5
    yy, xx = np.mgrid[0:size, 0:size]
    cy = cx = (size - 1) / 2.0
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    # Soft-edged disc: a one-pixel-wide ramp stands in for the probe size and
    # keeps the boundary from being aliased into a staircase.
    disc = np.clip(r + 0.5 - dist, 0.0, 1.0)

    if modality.upper() == "TEM":
        # Mass-thickness contrast is *exponential* attenuation, not linear in
        # thickness: electrons are scattered out of the aperture at a rate
        # per unit path length, so transmitted intensity follows
        # I = I0 exp(-t / lambda). Modelling it as linear would make a
        # particle a smooth dome whose half-maximum lies at 0.866 R, baking a
        # 13% sizing underestimate into the ground truth itself. With a
        # realistic opacity the attenuation saturates near the centre and the
        # particle appears as the sharp-edged dark disc that metal
        # nanoparticles actually produce.
        with np.errstate(invalid="ignore"):
            chord = np.sqrt(np.clip(1.0 - (dist / r) ** 2, 0.0, 1.0))
        k = max(tem_opacity, 0.05)
        profile = (1.0 - np.exp(-k * chord)) / (1.0 - np.exp(-k))
        return disc * profile

    # SEM: a fairly flat interior plus a bright rim where the surface tilts.
    rim = np.exp(-0.5 * ((dist - r * 0.88) / max(r * 0.16, 0.7)) ** 2)
    profile = 0.70 + 0.55 * rim
    return disc * profile


def _place_particles(cfg: SynthConfig, diameters_nm: np.ndarray, rng) -> tuple:
    """Choose non-identical centres for particles, honouring the gap rule."""
    radii_px = (diameters_nm / cfg.nm_per_px) / 2.0

    # Largest first: big particles are the hard ones to fit, and placing them
    # while the canvas is empty avoids a biased population where the big end
    # of the distribution silently fails to place.
    order = np.argsort(-radii_px)
    radii_px = radii_px[order]
    diameters_nm = diameters_nm[order]

    top = cfg.banner_height_px
    centres: list[tuple[float, float]] = []
    kept: list[int] = []

    for i, r in enumerate(radii_px):
        pad = r + 2
        if pad * 2 >= min(cfg.width, cfg.height - top):
            continue
        # Real frames cut through particles at the edge. Allowing centres to
        # sit slightly outside reproduces those partial particles, which the
        # analyser must then recognise and discard rather than measure.
        lo_pad = -r * 0.75 if cfg.allow_clipped_edges else pad
        for _ in range(cfg.max_place_attempts):
            cy = rng.uniform(top + lo_pad, cfg.height - lo_pad)
            cx = rng.uniform(lo_pad, cfg.width - lo_pad)
            ok = True
            for (oy, ox), orad in zip(centres, radii_px[kept]):
                gap = np.hypot(cy - oy, cx - ox)
                need = (r + orad) * (
                    cfg.min_centre_gap_frac if cfg.allow_touching else 1.08
                )
                if gap < need:
                    ok = False
                    break
            if ok:
                centres.append((cy, cx))
                kept.append(i)
                break

    kept = np.asarray(kept, dtype=int)
    return (
        np.asarray(centres, dtype=float).reshape(-1, 2),
        diameters_nm[kept],
        radii_px[kept],
    )


# --------------------------------------------------------------------------
# scale bar
# --------------------------------------------------------------------------

_NICE_LENGTHS_NM = [
    1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500,
    1000, 2000, 5000, 10000,
]


def _format_scale_label(nm: float) -> str:
    """Render a length the way an instrument would print it."""
    if nm >= 1000:
        val = nm / 1000.0
        txt = f"{val:g}"
        return f"{txt} um"      # ASCII, so Tesseract has a fair chance
    return f"{nm:g} nm"


def _choose_scale_length(cfg: SynthConfig) -> float:
    target_nm = cfg.width * cfg.scale_bar_target_frac * cfg.nm_per_px
    return float(min(_NICE_LENGTHS_NM, key=lambda v: abs(np.log(v / target_nm))))


def _load_font(px: int):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    ):
        try:
            return ImageFont.truetype(path, px)
        except Exception:
            continue
    return ImageFont.load_default()


def _draw_scale_bar(img_u8: np.ndarray, cfg: SynthConfig) -> tuple:
    """Burn a scale bar and its text label into the bottom-right corner.

    Real instruments overlay the bar on the image, which is why the pipeline
    has to find and mask it rather than being told the calibration.
    """
    length_nm = _choose_scale_length(cfg)
    length_px = length_nm / cfg.nm_per_px
    if not (8 <= length_px <= cfg.width * 0.6):
        return img_u8, None, None

    pil = Image.fromarray(img_u8).convert("L")
    draw = ImageDraw.Draw(pil)

    bar_h = max(4, int(round(cfg.height * 0.009)))
    margin = int(round(cfg.height * 0.035))
    # PIL draws rectangles inclusively, so a box from x0 to x1 covers
    # (x1 - x0 + 1) pixels. Subtracting one keeps the rendered bar exactly
    # length_px wide; without it the ground truth is a pixel narrower than
    # the bar actually drawn, and every calibration test inherits a ~0.5%
    # phantom error that belongs to the generator, not the detector.
    x1 = cfg.width - margin
    x0 = x1 - (length_px - 1)
    y1 = cfg.height - margin
    y0 = y1 - bar_h

    font_px = max(13, int(round(cfg.height * 0.030)))
    font = _load_font(font_px)
    label = _format_scale_label(length_nm)
    tb = draw.textbbox((0, 0), label, font=font)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]

    # Dark plate behind bar and text: both guarantees legibility and gives
    # the detector a high-contrast region to latch onto.
    plate_pad = max(6, font_px // 3)
    plate = [
        min(x0, x1 - tw) - plate_pad,
        y0 - th - plate_pad * 2,
        x1 + plate_pad,
        y1 + plate_pad,
    ]
    draw.rectangle(plate, fill=0)
    draw.rectangle([x0, y0, x1, y1], fill=255)
    draw.text(((x0 + x1) / 2 - tw / 2, y0 - th - plate_pad * 1.3), label,
              fill=255, font=font)

    return np.asarray(pil, dtype=np.uint8), length_nm, float(length_px)


# --------------------------------------------------------------------------
# main entry point
# --------------------------------------------------------------------------

def generate(cfg: SynthConfig | None = None, **overrides) -> SynthResult:
    """Render one synthetic micrograph and return it with its ground truth."""
    cfg = cfg or SynthConfig()
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise TypeError(f"unknown SynthConfig field: {k}")
        setattr(cfg, k, v)

    rng = np.random.default_rng(cfg.seed)

    wanted = sample_lognormal_diameters(
        cfg.median_diameter_nm, cfg.gsd, cfg.n_particles, rng
    )
    centres, diameters_nm, radii_px = _place_particles(cfg, wanted, rng)

    # --- background ------------------------------------------------------
    base = 0.22 if cfg.modality.upper() == "SEM" else 0.80
    field_ = np.full((cfg.height, cfg.width), base, dtype=np.float64)

    yy, xx = np.mgrid[0 : cfg.height, 0 : cfg.width]
    ny = yy / max(cfg.height - 1, 1)
    nx = xx / max(cfg.width - 1, 1)
    ang = rng.uniform(0, 2 * np.pi)
    ramp = np.cos(ang) * (nx - 0.5) + np.sin(ang) * (ny - 0.5)
    field_ *= 1.0 + cfg.illumination_gradient * ramp

    rad = np.sqrt((nx - 0.5) ** 2 + (ny - 0.5) ** 2) / 0.7071
    field_ *= 1.0 - cfg.vignette * rad**2

    # --- particles -------------------------------------------------------
    labels = np.zeros((cfg.height, cfg.width), dtype=np.int32)
    sign = 1.0 if cfg.modality.upper() == "SEM" else -1.0
    amp = 0.62 if cfg.modality.upper() == "SEM" else 0.58

    # Particles are composited by taking the maximum contrast, not by adding.
    # Adding would be wrong physics: two opaque particles that touch cannot
    # produce a region brighter (SEM) or darker (TEM) than either alone -- the
    # one nearer the detector simply occludes the other. Summing instead
    # paints an impossible lens-shaped hot spot at every contact, and those
    # artefacts would hand the segmenter a cue that real images never contain.
    particle_layer = np.zeros((cfg.height, cfg.width), dtype=np.float64)

    for idx, ((cy, cx), r) in enumerate(zip(centres, radii_px), start=1):
        patch = _particle_profile(r, cfg.modality, cfg.tem_opacity)
        ph, pw = patch.shape
        r0 = int(round(cy - (ph - 1) / 2.0))
        c0 = int(round(cx - (pw - 1) / 2.0))
        r1, c1 = r0 + ph, c0 + pw
        sr0, sc0 = max(r0, 0), max(c0, 0)
        sr1, sc1 = min(r1, cfg.height), min(c1, cfg.width)
        if sr0 >= sr1 or sc0 >= sc1:
            continue
        sub = patch[sr0 - r0 : sr1 - r0, sc0 - c0 : sc1 - c0]
        view = particle_layer[sr0:sr1, sc0:sc1]
        np.maximum(view, sub, out=view)
        # Instance label: later particles win the contested pixels, which is
        # what occlusion by the nearer particle looks like from above.
        region = labels[sr0:sr1, sc0:sc1]
        region[sub > 0.5] = idx

    field_ += sign * amp * particle_layer

    # --- instrument degradations ----------------------------------------
    field_ = np.clip(field_, 0.0, 1.5)

    if cfg.blur_sigma_px > 0:
        import cv2
        k = int(2 * np.ceil(3 * cfg.blur_sigma_px) + 1)
        field_ = cv2.GaussianBlur(field_, (k, k), cfg.blur_sigma_px)

    if cfg.noise_electrons and cfg.noise_electrons > 0:
        # Shot noise: sample counts, then normalise back to a 0-1 scale.
        counts = rng.poisson(np.clip(field_, 0, None) * cfg.noise_electrons)
        field_ = counts / cfg.noise_electrons
    if cfg.gaussian_read_noise:
        field_ = field_ + rng.normal(0, cfg.gaussian_read_noise / 255.0,
                                     field_.shape)

    img = np.clip(field_, 0.0, 1.0)
    img_u8 = (img * 255.0 + 0.5).astype(np.uint8)

    bar_nm = bar_px = None
    if cfg.draw_scale_bar:
        img_u8, bar_nm, bar_px = _draw_scale_bar(img_u8, cfg)

    # A particle is clipped when its disc crosses any frame boundary.
    if len(centres):
        clipped = (
            (centres[:, 0] - radii_px < 0)
            | (centres[:, 0] + radii_px > cfg.height - 1)
            | (centres[:, 1] - radii_px < 0)
            | (centres[:, 1] + radii_px > cfg.width - 1)
        )
    else:
        clipped = np.zeros(0, dtype=bool)

    return SynthResult(
        image=img_u8,
        diameters_nm=diameters_nm,
        centres_px=centres,
        clipped=clipped,
        nm_per_px=cfg.nm_per_px,
        scale_bar_nm=bar_nm,
        scale_bar_px=bar_px,
        config=cfg,
        labels=labels,
    )
