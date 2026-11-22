"""
Decoding a micrograph file into the uint8 grayscale array `analyze()` wants.

`cli.py` and `app.py` each had their own copy of this (`load_image` /
`load_uint8`), and neither handled a bad file: `Image.open()` defers the
real decode, so a truncated or unidentifiable file raised PIL's own
exception (`UnidentifiedImageError`, or a bare `OSError("Truncated File
Read")`) straight out of whichever call forced the decode, as a multi-frame
traceback with no indication of which file or what was wrong with it.

`load_grayscale` is the one place that now does this decode, so both
callers get the same behaviour and a corrupt or unreadable file fails with
one clear, specific `ValueError` instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError


def load_grayscale(source: Any) -> np.ndarray:
    """Decode an image file into the array `nanometrix.analyze()` expects.

    `source` is a path (str or Path) or a file-like object, as accepted by
    `PIL.Image.open` (e.g. the object Streamlit's `file_uploader` hands
    back). Non-8-bit images are rescaled to the full 0-255 range.

    Raises `ValueError` -- naming the file and the problem -- if `source`
    does not exist, is empty, or cannot be decoded as an image, rather than
    letting PIL's or the OS's own exception propagate as a raw traceback.
    """
    name = getattr(source, "name", source)

    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise ValueError(f"image file not found: {path}")
        if path.is_file() and path.stat().st_size == 0:
            raise ValueError(f"image file is empty: {path}")

    try:
        img: Image.Image = Image.open(source)
        img.load()  # force the real decode now, not lazily inside analyze()
    except UnidentifiedImageError as exc:
        raise ValueError(
            f"'{name}' is not a recognisable image file "
            "(unsupported format, or not an image at all)"
        ) from exc
    except OSError as exc:
        raise ValueError(
            f"'{name}' could not be decoded -- the file looks corrupt or "
            f"truncated ({exc})"
        ) from exc

    if img.mode not in ("L", "I;16", "I"):
        img = img.convert("L")
    arr = np.asarray(img)
    if arr.dtype != np.uint8:               # 16-bit micrographs are common
        arr = arr.astype(np.float32)
        lo, hi = float(arr.min()), float(arr.max())
        arr = ((arr - lo) / (hi - lo) * 255.0) if hi > lo else np.zeros_like(arr)
        arr = arr.astype(np.uint8)
    return arr
