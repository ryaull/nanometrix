#!/usr/bin/env python3
"""
NanoMetrix command line: analyse one micrograph or a whole folder.

Examples
--------
    python cli.py samples/sem_40nm.png --material Au
    python cli.py samples/ --outdir outputs --material Au --csv all.csv
    python cli.py samples/ --outdir outputs --output all.json --output-format json
    python cli.py micrograph.tif --nm-per-px 0.37     # skip OCR
    python cli.py --demo                              # generate and analyse
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nanometrix import BULK_DENSITY_G_CM3, analyze            # noqa: E402
from nanometrix.io import load_grayscale                      # noqa: E402
from nanometrix.segmentation import SegmentationParams        # noqa: E402
from nanometrix.viz import (                                  # noqa: E402
    distribution_figure, overlay_figure, shape_figure,
)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def load_image(path: Path) -> np.ndarray:
    return load_grayscale(path)


def analyse_one(path: Path, args, outdir: Path) -> pd.DataFrame:
    image = load_image(path)
    params = SegmentationParams(
        polarity=args.polarity,
        min_area_px=args.min_area,
        split_touching=not args.no_split,
        exclude_border=not args.keep_border,
    )
    result = analyze(
        image,
        nm_per_px=args.nm_per_px,
        params=params,
        refine_boundaries=args.refine,
        material=args.material,
        density_g_cm3=args.density,
    )

    print(f"\n=== {path.name} " + "=" * max(4, 58 - len(path.name)))
    print(result.report())

    stem = path.stem
    if not args.no_figures:
        outdir.mkdir(parents=True, exist_ok=True)
        f = overlay_figure(image, result.labels, result.diameters,
                           result.unit, title=f"Detected particles - {stem}")
        f.savefig(outdir / f"{stem}_overlay.png", dpi=150, bbox_inches="tight")
        g = distribution_figure(result.diameters, result.stats, result.unit)
        g.savefig(outdir / f"{stem}_distribution.png", dpi=150, bbox_inches="tight")
        h = shape_figure(result.particles)
        h.savefig(outdir / f"{stem}_shape.png", dpi=150, bbox_inches="tight")
        import matplotlib.pyplot as plt
        plt.close("all")

    df = result.particles.copy()
    if not df.empty:
        df.insert(0, "image", path.name)
        if result.calibration.ok:
            df["nm_per_px"] = result.calibration.nm_per_px
    if not args.no_figures:
        (outdir / f"{stem}_report.txt").write_text(result.report())
    return df


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Automated nanoparticle size and shape analysis "
                    "from electron micrographs.")
    ap.add_argument("path", nargs="?", help="image file or folder of images")
    ap.add_argument("--demo", action="store_true",
                    help="generate a synthetic micrograph and analyse it")
    ap.add_argument("--outdir", default="outputs", help="where figures go")
    ap.add_argument("--csv", "--output", dest="output", default=None,
                    help="write the combined per-particle table to this path "
                         "(format set by --output-format, else guessed from "
                         "the extension, defaulting to CSV)")
    ap.add_argument("--output-format", default=None, choices=["csv", "json"],
                    help="file format for --csv/--output, overriding "
                         "whatever its extension would imply")
    ap.add_argument("--nm-per-px", type=float, default=None,
                    help="supply calibration instead of reading the scale bar")
    ap.add_argument("--material", default=None,
                    choices=sorted(BULK_DENSITY_G_CM3),
                    help="material, used for the specific-surface-area estimate")
    ap.add_argument("--density", type=float, default=None,
                    help="bulk density in g/cm3, overrides --material")
    ap.add_argument("--polarity", default="auto",
                    choices=["auto", "bright", "dark"])
    ap.add_argument("--min-area", type=int, default=40,
                    help="smallest accepted particle, in pixels")
    ap.add_argument("--no-split", action="store_true",
                    help="skip watershed separation of touching particles")
    ap.add_argument("--keep-border", action="store_true",
                    help="keep particles clipped by the frame (biases sizes low)")
    ap.add_argument("--refine", action="store_true",
                    help="re-cut boundaries at each particle's half-maximum")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)

    outdir = Path(args.outdir)
    had_error = False

    if args.demo:
        from nanometrix import SynthConfig, generate
        outdir.mkdir(parents=True, exist_ok=True)
        demo = outdir / "demo_micrograph.png"
        r = generate(SynthConfig(seed=7, modality="SEM",
                                 median_diameter_nm=40, nm_per_px=0.5,
                                 n_particles=70))
        Image.fromarray(r.image).save(demo)
        print(f"Synthetic micrograph written to {demo}")
        print(f"Ground truth: {len(r.diameters_complete_nm)} complete particles, "
              f"true D50 = {np.median(r.diameters_complete_nm):.2f} nm")
        frames = [analyse_one(demo, args, outdir)]
    else:
        if not args.path:
            ap.error("give an image or folder, or use --demo")
        target = Path(args.path)
        if target.is_dir():
            files = sorted(p for p in target.iterdir()
                           if p.suffix.lower() in IMAGE_SUFFIXES
                           and not p.name.startswith("_"))
            if not files:
                print(f"no images found in {target}", file=sys.stderr)
                return 1
        elif target.is_file():
            files = [target]
        else:
            print(f"not found: {target}", file=sys.stderr)
            return 1

        # An unreadable or corrupt image shouldn't take the whole batch
        # down: report it clearly, skip it, and keep analysing the rest.
        frames = []
        for p in files:
            try:
                frames.append(analyse_one(p, args, outdir))
            except ValueError as exc:
                print(f"skipping {p.name}: {exc}", file=sys.stderr)
                had_error = True

    frames = [f for f in frames if not f.empty]
    if args.output and frames:
        allp = pd.concat(frames, ignore_index=True)
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fmt = args.output_format or (
            "json" if out_path.suffix.lower() == ".json" else "csv"
        )
        if fmt == "json":
            allp.to_json(out_path, orient="records", indent=2)
        else:
            allp.to_csv(out_path, index=False)
        print(f"\n{len(allp)} particles written to {out_path} ({fmt})")
    return 1 if had_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
