"""
Scale-bar detection and OCR: turning pixels into nanometres.

The point of this module
------------------------
A micrograph on its own carries no physical length. The magnification lives
in the scale bar that the instrument burns into the image, so any size
reported without reading that bar is in pixels, and pixels are not a unit of
length -- the same particle imaged at two magnifications would get two
different "sizes". Automating the bar is therefore not a convenience
feature; it is what lets the tool report nanometres at all, and it removes
the step where a human types the magnification in and occasionally gets it
wrong.

Strategy
--------
1. Find candidate bars geometrically: a scale bar is a long, thin, strongly
   contrasting horizontal blob, almost always in the lower part of the frame.
   This is a shape query, so it needs no OCR and rarely fails.
2. Read the label with Tesseract, restricted to a character whitelist and
   single-line mode.
3. Parse the number and unit, convert to nanometres, and divide by the bar's
   pixel length.
4. Return a confidence and an occlusion mask, so the caller can both judge
   the result and exclude the burnt-in overlay from particle analysis.

Every stage degrades rather than throws: if OCR is unavailable or
unreadable, the caller is told so and can supply nm/px manually.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import cv2
import numpy as np

# Conversion to nanometres. Tesseract routinely confuses the micro sign, so
# several spellings of micrometre are accepted.
_UNIT_TO_NM = {
    "nm": 1.0,
    "um": 1_000.0, "µm": 1_000.0, "μm": 1_000.0, "mm": 1_000_000.0,
    "a": 0.1, "å": 0.1,          # angstrom
    "pm": 0.001,
}

# Digits, separators and the letters that appear in length units only.
_OCR_WHITELIST = "0123456789.,munApµμÅ"

_LABEL_RE = re.compile(
    r"(?P<value>\d+(?:[.,]\d+)?)\s*(?P<unit>nm|um|µm|μm|mm|pm|å|a)\b",
    re.IGNORECASE,
)


@dataclass
class Calibration:
    """Result of trying to read the physical scale off an image."""

    nm_per_px: float | None
    bar_length_px: float | None = None
    bar_length_nm: float | None = None
    label_text: str | None = None
    confidence: float = 0.0
    source: str = "none"          # "ocr" | "manual" | "none"
    overlay_mask: np.ndarray | None = None  # True where burnt-in annotation sits
    note: str = ""

    @property
    def ok(self) -> bool:
        return self.nm_per_px is not None and self.nm_per_px > 0

    def describe(self) -> str:
        if not self.ok:
            return f"uncalibrated ({self.note or 'no scale bar found'})"
        if self.source == "manual":
            return f"{self.nm_per_px:.4g} nm/px (supplied manually)"
        return (
            f"{self.nm_per_px:.4g} nm/px "
            f"(read '{self.label_text}' over {self.bar_length_px:.0f} px, "
            f"confidence {self.confidence:.2f})"
        )


# --------------------------------------------------------------------------
# bar geometry
# --------------------------------------------------------------------------

def _bar_candidates(gray: np.ndarray, search_frac: float = 0.35) -> list[dict]:
    """Find long thin horizontal blobs in the lower part of the image."""
    h, w = gray.shape
    y0 = int(h * (1.0 - search_frac))
    strip = gray[y0:, :]

    found: list[dict] = []
    # A bar may be white-on-dark or black-on-light, so test both polarities.
    for polarity, img in (("bright", strip), ("dark", 255 - strip)):
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Close small gaps along the bar without fattening it vertically.
        binary = cv2.morphologyEx(
            binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
        )
        n, _, stats, cents = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            if bw < w * 0.03 or bw > w * 0.85:
                continue
            if bh < 1 or bh > max(3, h * 0.035):
                continue
            aspect = bw / float(bh)
            if aspect < 6:
                continue
            fill = area / float(bw * bh)
            if fill < 0.72:                 # a bar is a solid rectangle
                continue
            found.append({
                "x": int(x), "y": int(y + y0), "w": int(bw), "h": int(bh),
                "aspect": float(aspect), "fill": float(fill),
                "polarity": polarity,
                "cx": float(cents[i][0]), "cy": float(cents[i][1] + y0),
            })

    # Prefer long, solid, low-lying candidates.
    found.sort(
        key=lambda c: (c["aspect"] * c["fill"] * (c["cy"] / h)),
        reverse=True,
    )
    return found[:6]


# --------------------------------------------------------------------------
# label OCR
# --------------------------------------------------------------------------

def _ocr_votes(patch: np.ndarray) -> list[tuple[float, str]]:
    """Read a length label many ways and return every reading obtained.

    OCR on a burnt-in instrument label is not reliable enough to trust once.
    The glyphs are small, the lettering is often bolded into near-contact
    with itself, and a single misread digit does not produce an obviously
    wrong answer -- it produces a plausible one. '100 nm' read as '200 nm' is
    a clean factor-of-two error in every size the tool subsequently reports,
    with nothing downstream to flag it, because 200 nm is just as reasonable
    a scale bar as 100 nm.

    Single attempts cannot catch that, so this returns all of them and lets
    the caller take a majority. Independent preprocessings rarely make the
    *same* misreading, so the correct value is normally the modal one, and
    the spread doubles as an honest confidence signal.
    """
    votes: list[tuple[float, str]] = []
    if patch.size == 0:
        return votes
    try:
        import pytesseract
    except Exception:
        return votes

    # Each attempt is a separate Tesseract process, so exhausting every
    # variant costs dozens of subprocess launches per image and dominates the
    # runtime of a batch. Once several independent preprocessings have
    # already agreed, further attempts cannot change the majority, so stop.
    def decided() -> bool:
        if len(votes) < 4:
            return False
        from collections import Counter
        top = Counter(v[0] for v in votes).most_common(1)[0][1]
        return top / len(votes) >= 0.8

    for scale_to in (40.0, 64.0):
        scale = max(1.0, scale_to / max(patch.shape[0], 1))
        big = cv2.resize(patch, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_CUBIC)
        for img in (big, 255 - big):       # light-on-dark and dark-on-light
            work = cv2.GaussianBlur(img, (3, 3), 0)
            _, work = cv2.threshold(work, 0, 255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            work = cv2.copyMakeBorder(work, 14, 14, 14, 14,
                                      cv2.BORDER_CONSTANT, value=255)
            for psm in (7, 6, 13):         # line, block, raw line
                cfg = (f"--psm {psm} --oem 3 "
                       f"-c tessedit_char_whitelist={_OCR_WHITELIST}")
                try:
                    txt = pytesseract.image_to_string(work, config=cfg)
                except Exception:
                    continue
                parsed = parse_label(txt.strip())
                if parsed:
                    votes.append(parsed)
                    if decided():
                        return votes
    return votes


def _ocr_text(patch: np.ndarray) -> str:
    """Backwards-compatible single reading: the modal vote, or ''."""
    votes = _ocr_votes(patch)
    if not votes:
        return ""
    from collections import Counter
    return Counter(v[1] for v in votes).most_common(1)[0][0]


def parse_label(text: str) -> tuple[float, str] | None:
    """Pull a length and unit out of OCR output, tolerating its usual errors."""
    if not text:
        return None
    cleaned = (text.replace("\n", " ")
                   .replace("|", "1")
                   .replace("O", "0").replace("o", "0")
                   .strip())

    for candidate in (cleaned, cleaned.replace(" ", "")):
        m = _LABEL_RE.search(candidate)
        if not m:
            continue
        raw = m.group("value").replace(",", ".")
        try:
            value = float(raw)
        except ValueError:
            continue
        unit = m.group("unit").lower()
        factor = _UNIT_TO_NM.get(unit)
        if factor is None:
            continue
        if value <= 0:
            continue
        return value * factor, f"{m.group('value')} {m.group('unit')}"
    return None


# --------------------------------------------------------------------------
# annotation plate
# --------------------------------------------------------------------------

def _plate_bbox(gray: np.ndarray, bar: dict) -> tuple[int, int, int, int] | None:
    """Locate the solid plate an instrument draws behind its annotation.

    This matters more than it sounds. OCR here is a two-class problem --
    text against plate -- but only if the crop contains nothing else. Let a
    strip of sample into the crop and a third brightness population joins in;
    Otsu then splits sample-from-plate instead of text-from-plate, and the
    lettering is absorbed into whichever class it resembles. That is how a
    label a human reads effortlessly becomes unreadable to the machine.

    The plate is found by growing outward from the bar rather than by
    connected components. Components fail here for a specific reason: a
    strongly scattering particle is saturated to the same near-black value as
    the plate itself, so any tolerance wide enough to hold the plate together
    also welds it to every dark particle that happens to touch it, and the
    box then swallows a band of sample. Growing outward instead asks a
    sharper question of each row -- does this row consist *only* of plate and
    lettering? -- which sample texture fails immediately however dark it is.
    """
    h, w = gray.shape
    img = gray.astype(np.float32)
    bx0, bx1 = bar["x"], bar["x"] + bar["w"]
    by0, by1 = bar["y"], bar["y"] + bar["h"]

    ribbons = []
    above = img[max(by0 - 3, 0):max(by0 - 1, 1), bx0:bx1]
    below = img[min(by1 + 1, h - 1):min(by1 + 3, h), bx0:bx1]
    for rib in (above, below):
        if rib.size:
            ribbons.append(rib.reshape(-1))
    if not ribbons:
        return None
    ring = np.concatenate(ribbons)
    plate_val = float(np.median(ring))
    if float(np.std(ring)) > 30.0:            # the bar does not sit on a plate
        return None

    bar_px = img[by0:by1, bx0:bx1]
    text_val = float(np.median(bar_px)) if bar_px.size else 255.0 - plate_val
    if abs(text_val - plate_val) < 40.0:      # no usable plate/text contrast
        return None

    tol_plate, tol_text = 30.0, 60.0

    def belongs(block: np.ndarray) -> float:
        """Fraction of a block explained by plate colour or text colour."""
        if block.size == 0:
            return 0.0
        ok = (np.abs(block - plate_val) < tol_plate) | \
             (np.abs(block - text_val) < tol_text)
        return float(ok.mean())

    purity = 0.90
    max_v = int(h * 0.30)
    max_hspan = int(w * 0.60)

    y_top = by0
    for y in range(by0 - 1, max(by0 - max_v, -1), -1):
        if belongs(img[y, bx0:bx1]) < purity:
            break
        y_top = y
    y_bot = by1
    for y in range(by1, min(by1 + max_v, h)):
        if belongs(img[y, bx0:bx1]) < purity:
            break
        y_bot = y + 1

    x_left = bx0
    for x in range(bx0 - 1, max(bx0 - max_hspan, -1), -1):
        if belongs(img[y_top:y_bot, x]) < purity:
            break
        x_left = x
    x_right = bx1
    for x in range(bx1, min(bx1 + max_hspan, w)):
        if belongs(img[y_top:y_bot, x]) < purity:
            break
        x_right = x + 1

    ph, pw = y_bot - y_top, x_right - x_left
    if ph < bar["h"] + 4 or pw < bar["w"] * 0.9:
        return None                            # too thin to hold a label
    return int(x_left), int(y_top), int(pw), int(ph)


# --------------------------------------------------------------------------
# overlay mask
# --------------------------------------------------------------------------

def _overlay_mask(
    gray: np.ndarray,
    bar: dict | None,
    plate: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """Mark burnt-in annotation so it is never measured as a particle.

    Instruments stamp the bar, its label and often a dark info plate onto the
    image. Those pixels are graphics, not sample, and a bar is a perfect
    high-contrast rectangle that any thresholder will happily report as an
    enormous elongated 'particle'.
    """
    h, w = gray.shape
    mask = np.zeros((h, w), dtype=bool)
    if bar is None:
        return mask

    # A detected plate is the exact extent of the burnt-in graphics, so when
    # one exists there is no need to guess at padding.
    if plate is not None:
        px, py, pw, ph = plate
        pad = 3
        mask[max(py - pad, 0):min(py + ph + pad, h),
             max(px - pad, 0):min(px + pw + pad, w)] = True
        return mask

    # Start from the bar box, generously padded to swallow the text label.
    pad_x = int(max(bar["w"] * 0.35, w * 0.02))
    pad_up = int(max(bar["h"] * 14, h * 0.065))
    pad_dn = int(max(bar["h"] * 4, h * 0.02))
    x0 = max(bar["x"] - pad_x, 0)
    x1 = min(bar["x"] + bar["w"] + pad_x, w)
    y0 = max(bar["y"] - pad_up, 0)
    y1 = min(bar["y"] + bar["h"] + pad_dn, h)
    mask[y0:y1, x0:x1] = True

    # Many instruments draw a solid plate behind the annotation. Where such a
    # near-uniform block touches the bar, mask the whole plate.
    region = gray[y0:y1, x0:x1]
    if region.size:
        plate_val = float(np.median(region[region < np.percentile(region, 35)])) \
            if (region < np.percentile(region, 35)).any() else None
        if plate_val is not None and plate_val < 60:
            flat = (np.abs(gray.astype(np.int16) - plate_val) < 18)
            n, lbl, stats, _ = cv2.connectedComponentsWithStats(
                flat.astype(np.uint8), connectivity=8
            )
            touching = np.unique(lbl[y0:y1, x0:x1])
            for i in touching:
                if i == 0:
                    continue
                bx, by, bw_, bh_, area = stats[i]
                # Only absorb compact rectangular plates, never the background.
                if area > 0.25 * h * w:
                    continue
                if bw_ * bh_ > 0 and area / float(bw_ * bh_) > 0.75:
                    mask |= (lbl == i)
    return mask


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------

def calibrate(
    image: np.ndarray,
    manual_nm_per_px: float | None = None,
    search_frac: float = 0.35,
) -> Calibration:
    """Determine nm-per-pixel for a micrograph.

    If `manual_nm_per_px` is given it is trusted outright, but the scale bar
    is still located so that its pixels can be excluded from analysis.
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = gray.astype(np.uint8)
    h, w = gray.shape

    candidates = _bar_candidates(gray, search_frac=search_frac)

    best: dict | None = None
    best_parse: tuple[float, str] | None = None
    best_text = ""
    vote_share = 0.0

    plates: dict[int, tuple[int, int, int, int]] = {}

    for ci, cand in enumerate(candidates):
        plate = _plate_bbox(gray, cand)
        if plate is not None:
            plates[ci] = plate

        # Preferred crops come from inside the annotation plate, split just
        # above and just below the bar. Only if no plate exists do we fall
        # back to padding blindly into the sample.
        crops: list[tuple[int, int, int, int]] = []
        patches: list[np.ndarray] = []
        if plate is not None:
            px, py, pw, ph = plate
            # First attempt: the entire plate with the bar painted out in
            # plate colour. Splitting the plate just above and just below the
            # bar risks slicing through the lettering when the plate is
            # short, and a clipped digit is worse than no digit -- '5' with
            # its top row cut reads as a comma. Blanking the bar instead
            # leaves every glyph whole.
            whole = gray[py:py + ph, px:px + pw].copy()
            border = np.concatenate([whole[0, :], whole[-1, :]])
            fill = int(np.median(border))
            br0 = max(cand["y"] - py, 0)
            br1 = min(cand["y"] + cand["h"] - py, whole.shape[0])
            if br1 > br0:
                whole[br0:br1, :] = fill
            patches.append(whole)
            if cand["y"] - py > 6:
                crops.append((px, py, px + pw, cand["y"] - 1))
            if (py + ph) - (cand["y"] + cand["h"]) > 6:
                crops.append((px, cand["y"] + cand["h"] + 1, px + pw, py + ph))
        else:
            pad_x = int(max(cand["w"] * 0.30, 8))
            x0 = max(cand["x"] - pad_x, 0)
            x1 = min(cand["x"] + cand["w"] + pad_x, w)
            above_y0 = max(cand["y"] - int(max(cand["h"] * 12, h * 0.055)), 0)
            below_y1 = min(
                cand["y"] + cand["h"] + int(max(cand["h"] * 10, h * 0.05)), h
            )
            if cand["y"] - 1 - above_y0 > 6:
                crops.append((x0, above_y0, x1, cand["y"] - 1))
            if below_y1 - (cand["y"] + cand["h"] + 1) > 6:
                crops.append((x0, cand["y"] + cand["h"] + 1, x1, below_y1))

        for (cx0, cy0, cx1, cy1) in crops:
            cx0, cy0 = max(cx0, 0), max(cy0, 0)
            cx1, cy1 = min(cx1, w), min(cy1, h)
            if cx1 - cx0 < 8 or cy1 - cy0 < 6:
                continue
            patches.append(gray[cy0:cy1, cx0:cx1])

        all_votes: list[tuple[float, str]] = []
        for patch in patches:
            all_votes.extend(_ocr_votes(patch))
            if len(all_votes) >= 4:
                from collections import Counter
                top = Counter(v[0] for v in all_votes).most_common(1)[0][1]
                if top / len(all_votes) >= 0.8:
                    break

        if all_votes:
            from collections import Counter
            tally = Counter(v[0] for v in all_votes)
            winner_nm, winner_count = tally.most_common(1)[0]
            pretty = next(lbl for nm, lbl in all_votes if nm == winner_nm)
            best = cand
            best_parse = (winner_nm, pretty)
            best_text = pretty
            vote_share = winner_count / len(all_votes)
            break

    bar = best or (candidates[0] if candidates else None)
    bar_index = candidates.index(bar) if bar in candidates else None
    mask = _overlay_mask(gray, bar, plates.get(bar_index) if bar_index is not None else None)

    if manual_nm_per_px:
        return Calibration(
            nm_per_px=float(manual_nm_per_px), source="manual",
            bar_length_px=float(bar["w"]) if bar else None,
            overlay_mask=mask, confidence=1.0,
            note="calibration supplied by user",
        )

    if best is None or best_parse is None:
        return Calibration(
            nm_per_px=None, source="none", overlay_mask=mask,
            bar_length_px=float(bar["w"]) if bar else None,
            note=("scale bar found but label unreadable"
                  if bar is not None else "no scale bar found"),
        )

    length_nm, pretty = best_parse
    nm_per_px = length_nm / float(best["w"])

    # Confidence combines how bar-like the blob was, how strongly the OCR
    # attempts agreed, and whether the result is physically sane for electron
    # microscopy. Agreement carries real weight here: a label read the same
    # way by every preprocessing is trustworthy, whereas a split vote is the
    # signature of the silent factor-of-two misread this scheme exists to
    # catch, and the user needs to see that in the number.
    shape_score = min(best["aspect"] / 20.0, 1.0) * best["fill"]
    plausible = 0.01 <= nm_per_px <= 200.0
    agreement = vote_share if vote_share > 0 else 0.5
    confidence = float(np.clip(
        shape_score * agreement * (1.0 if plausible else 0.25), 0, 1
    ))

    return Calibration(
        nm_per_px=float(nm_per_px),
        bar_length_px=float(best["w"]),
        bar_length_nm=float(length_nm),
        label_text=pretty,
        confidence=confidence,
        source="ocr",
        overlay_mask=mask,
        note=("" if plausible else "calibration outside the usual EM range")
             or ("" if agreement >= 0.6 else
                 f"OCR readings disagreed ({agreement:.0%} agreement); "
                 "verify the scale bar"),
    )
