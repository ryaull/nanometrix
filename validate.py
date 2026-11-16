#!/usr/bin/env python3
"""
Ground-truth validation of the NanoMetrix pipeline.

Why a synthetic ground truth
----------------------------
A sizing tool cannot be validated on real micrographs, because the "true"
diameter of a real nanoparticle is itself the output of a measurement -- in
practice a human drawing lines in ImageJ, which is the very procedure this
tool exists to replace. Agreeing with that reference would prove only that
the tool reproduces a human's bias. So the comparison is inverted: generate
micrographs from diameters we chose, degrade them the way an instrument
does, and ask how well the chosen numbers are recovered.

What is measured
----------------
1. Baseline accuracy at realistic imaging conditions.
2. Packing density, which tests whether the watershed stage holds up as
   particles come into contact.
3. Spatial resolution, i.e. how few pixels a particle can span before its
   diameter stops being trustworthy.
4. Noise, i.e. how low the electron dose can go before detection fails.
5. Global Otsu against per-particle half-maximum boundary cutting.

Run:  python validate.py            (full)
      python validate.py --quick    (fewer seeds)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from nanometrix import SynthConfig, analyze, generate
from nanometrix.viz import INK_MUTED, INK_PRIMARY, INK_SECONDARY, SERIES_1, SERIES_2, SURFACE


# --------------------------------------------------------------------------
# matching measured particles to ground truth
# --------------------------------------------------------------------------

def match_particles(result, truth, tol_frac: float = 0.6):
    """Pair measured particles with the true ones by centroid proximity.

    Greedy nearest-neighbour matching, closest pair first, each truth and
    each detection used at most once. A pair is only allowed if the centroids
    agree to within a fraction of the true radius, which stops a detection
    being credited to a particle it merely happens to lie nearest to.
    """
    if result.particles is None or result.particles.empty:
        return np.zeros(0, int), np.zeros(0, int), 0

    meas = result.particles[["centroid_row", "centroid_col"]].to_numpy(float)
    keep = ~truth.clipped                       # only fully visible particles
    true_c = truth.centres_px[keep]
    true_d = truth.diameters_nm[keep]
    if len(true_c) == 0 or len(meas) == 0:
        return np.zeros(0, int), np.zeros(0, int), len(meas)

    true_r_px = (true_d / truth.nm_per_px) / 2.0
    d = np.linalg.norm(true_c[:, None, :] - meas[None, :, :], axis=2)
    tol = true_r_px[:, None] * tol_frac

    pairs = []
    used_t, used_m = set(), set()
    order = np.dstack(np.unravel_index(np.argsort(d, axis=None), d.shape))[0]
    for ti, mi in order:
        if ti in used_t or mi in used_m:
            continue
        if d[ti, mi] > tol[ti, 0]:
            continue
        pairs.append((ti, mi))
        used_t.add(ti); used_m.add(mi)
    if not pairs:
        return np.zeros(0, int), np.zeros(0, int), len(meas)
    ti, mi = map(np.array, zip(*pairs))
    return ti, mi, len(meas)


def evaluate(cfg: SynthConfig, refine: bool = False, params=None) -> dict | None:
    """Generate one image, analyse it, and score the result against truth."""
    truth = generate(cfg)
    res = analyze(truth.image, refine_boundaries=refine, params=params)

    complete = truth.diameters_complete_nm
    if complete.size == 0:
        return None

    row = {
        "modality": cfg.modality,
        "median_nm": cfg.median_diameter_nm,
        "nm_per_px": cfg.nm_per_px,
        "diameter_px": cfg.median_diameter_nm / cfg.nm_per_px,
        "n_particles_cfg": cfg.n_particles,
        "noise_electrons": cfg.noise_electrons,
        "coverage": truth.coverage,
        "n_true": int(complete.size),
        "n_measured": res.n_particles,
        "calibrated": bool(res.calibration.ok),
        "calib_error_pct": (
            abs(res.calibration.nm_per_px - truth.nm_per_px) / truth.nm_per_px * 100
            if res.calibration.ok else np.nan
        ),
    }

    if res.stats is None or not res.calibration.ok:
        row.update({"recall": 0.0, "precision": 0.0, "d50_error_pct": np.nan,
                    "size_mape": np.nan, "size_bias_pct": np.nan,
                    "circularity": np.nan})
        return row

    ti, mi, n_meas = match_particles(res, truth)
    keep = ~truth.clipped
    true_d = truth.diameters_nm[keep]

    row["recall"] = len(ti) / max(len(true_d), 1)
    row["precision"] = len(mi) / max(n_meas, 1)

    if len(ti):
        md = res.particles.iloc[mi]["ecd_nm"].to_numpy(float)
        td = true_d[ti]
        rel = (md - td) / td * 100.0
        row["size_mape"] = float(np.mean(np.abs(rel)))
        row["size_bias_pct"] = float(np.mean(rel))
    else:
        row["size_mape"] = row["size_bias_pct"] = np.nan

    true_d50 = float(np.median(complete))
    row["d50_error_pct"] = (res.stats.d50 - true_d50) / true_d50 * 100.0
    row["true_d50"] = true_d50
    row["meas_d50"] = res.stats.d50
    row["circularity"] = res.shape.get("circularity", np.nan)
    return row


# --------------------------------------------------------------------------
# experiments
# --------------------------------------------------------------------------

def _run(configs, refine=False) -> pd.DataFrame:
    rows = [evaluate(c, refine=refine) for c in configs]
    return pd.DataFrame([r for r in rows if r is not None])


def exp_baseline(seeds: int) -> pd.DataFrame:
    """Realistic conditions: moderate density, ordinary dose."""
    cfgs = []
    for s in range(seeds):
        for mod, med, npx in (("SEM", 40, 0.5), ("SEM", 90, 1.1),
                              ("TEM", 18, 0.22), ("TEM", 45, 0.55)):
            cfgs.append(SynthConfig(seed=100 + s, modality=mod,
                                    median_diameter_nm=med, nm_per_px=npx,
                                    n_particles=70, gsd=1.25))
    return _run(cfgs)


def exp_density(seeds: int) -> pd.DataFrame:
    cfgs = [
        SynthConfig(seed=200 + s, modality="SEM", median_diameter_nm=40,
                    nm_per_px=0.5, n_particles=n)
        for n in (12, 25, 45, 70, 100, 130, 160) for s in range(seeds)
    ]
    return _run(cfgs)


def exp_resolution(seeds: int) -> pd.DataFrame:
    """Vary how many pixels a particle spans, holding everything else fixed.

    The minimum-area filter is scaled to the particles being imaged rather
    than left at its default. Otherwise this experiment would not measure the
    resolution limit of the segmentation at all -- at 6 px across, a particle
    covers 28 px^2 and the default 40 px^2 floor discards every real particle
    while keeping the noise blobs that happen to exceed it. That is a
    misconfiguration, not a resolution limit, and conflating the two would
    report the wrong number.
    """
    from nanometrix.segmentation import SegmentationParams

    rows = []
    for px in (6, 9, 13, 20, 30, 45, 70, 100):
        expected_area = np.pi * (px / 2.0) ** 2
        params = SegmentationParams(min_area_px=max(int(0.25 * expected_area), 6))
        for s in range(seeds):
            med = 40.0
            cfg = SynthConfig(seed=300 + s, modality="SEM",
                              median_diameter_nm=med, nm_per_px=med / px,
                              n_particles=70)
            row = evaluate(cfg, params=params)
            if row is not None:
                rows.append(row)
    return pd.DataFrame(rows)


def exp_noise(seeds: int) -> pd.DataFrame:
    cfgs = [
        SynthConfig(seed=400 + s, modality="SEM", median_diameter_nm=40,
                    nm_per_px=0.5, n_particles=70, noise_electrons=ne)
        for ne in (8, 15, 30, 60, 120, 250, 500) for s in range(seeds)
    ]
    return _run(cfgs)


def exp_refinement(seeds: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfgs = []
    for s in range(seeds):
        for mod, med, npx in (("SEM", 40, 0.5), ("TEM", 18, 0.22),
                              ("SEM", 90, 1.1), ("TEM", 45, 0.55)):
            cfgs.append(SynthConfig(seed=500 + s, modality=mod,
                                    median_diameter_nm=med, nm_per_px=npx,
                                    n_particles=70))
    return _run(cfgs, refine=False), _run(cfgs, refine=True)


# --------------------------------------------------------------------------
# figure
# --------------------------------------------------------------------------

def make_figure(density, resolution, noise, baseline, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def style(ax):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(INK_MUTED); ax.spines[side].set_linewidth(0.8)
        ax.tick_params(colors=INK_SECONDARY, labelsize=9, width=0.8)
        ax.grid(True, color=INK_MUTED, alpha=0.18, linewidth=0.7)
        ax.set_axisbelow(True)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=130)

    ax = axes[0, 0]
    g = density.groupby("n_particles_cfg").agg(
        cov=("coverage", "mean"), bias=("d50_error_pct", "mean"),
        sd=("d50_error_pct", "std")).reset_index()
    ax.errorbar(g["cov"] * 100, g["bias"], yerr=g["sd"], color=SERIES_1,
                linewidth=2.0, marker="o", markersize=6, capsize=3,
                markeredgecolor=SURFACE)
    ax.axhline(0, color=INK_MUTED, linewidth=1.0, linestyle="--")
    ax.set_xlabel("Areal coverage (%)", color=INK_SECONDARY, fontsize=10)
    ax.set_ylabel("D50 error (%)", color=INK_SECONDARY, fontsize=10)
    ax.set_title("Packing density drives the sizing bias",
                 color=INK_PRIMARY, fontsize=11, loc="left")
    style(ax)

    ax = axes[0, 1]
    g = resolution.groupby("diameter_px").agg(
        mape=("size_mape", "mean"), rec=("recall", "mean")).reset_index()
    ax.plot(g["diameter_px"], g["mape"], color=SERIES_1, linewidth=2.0,
            marker="o", markersize=6, markeredgecolor=SURFACE,
            label="Per-particle size error")
    ax.axhline(5, color=INK_MUTED, linewidth=1.0, linestyle="--")
    ax.text(g["diameter_px"].max(), 5.4, "5% ", ha="right", fontsize=8,
            color=INK_SECONDARY)
    ax.set_xscale("log")
    ax.set_xlabel("Particle diameter (pixels)", color=INK_SECONDARY, fontsize=10)
    ax.set_ylabel("Mean abs. size error (%)", color=INK_SECONDARY, fontsize=10)
    ax.set_title("Resolution limit", color=INK_PRIMARY, fontsize=11, loc="left")
    style(ax)

    ax = axes[1, 0]
    g = noise.groupby("noise_electrons").agg(
        rec=("recall", "mean"), mape=("size_mape", "mean")).reset_index()
    ax.plot(g["noise_electrons"], g["rec"] * 100, color=SERIES_1,
            linewidth=2.0, marker="o", markersize=6, markeredgecolor=SURFACE,
            label="Detection recall")
    ax.plot(g["noise_electrons"], g["mape"], color=SERIES_2, linewidth=2.0,
            marker="s", markersize=6, markeredgecolor=SURFACE,
            label="Size error (%)")
    ax.set_xscale("log")
    ax.set_xlabel("Electron dose (counts/pixel)", color=INK_SECONDARY, fontsize=10)
    ax.set_ylabel("Percent", color=INK_SECONDARY, fontsize=10)
    ax.set_title("Tolerance to low dose", color=INK_PRIMARY, fontsize=11, loc="left")
    leg = ax.legend(frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(INK_SECONDARY)
    style(ax)

    ax = axes[1, 1]
    b = baseline.dropna(subset=["true_d50", "meas_d50"])
    ax.scatter(b["true_d50"], b["meas_d50"], s=34, color=SERIES_1,
               alpha=0.8, edgecolor=SURFACE, linewidth=0.8)
    if len(b):
        lim = [min(b["true_d50"].min(), b["meas_d50"].min()) * 0.9,
               max(b["true_d50"].max(), b["meas_d50"].max()) * 1.05]
        ax.plot(lim, lim, color=INK_MUTED, linestyle="--", linewidth=1.2)
        ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("True D50 (nm)", color=INK_SECONDARY, fontsize=10)
    ax.set_ylabel("Measured D50 (nm)", color=INK_SECONDARY, fontsize=10)
    ax.set_title("Measured against true, all baseline runs",
                 color=INK_PRIMARY, fontsize=11, loc="left")
    style(ax)

    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--outdir", default="outputs")
    args = ap.parse_args()
    seeds = 2 if args.quick else 5
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)

    print("Running validation. Every number below is measured against a "
          "ground truth\nthe generator chose, not against a human annotation.\n")

    print("[1/5] baseline accuracy ...")
    baseline = exp_baseline(seeds)
    print("[2/5] packing density ...")
    density = exp_density(seeds)
    print("[3/5] spatial resolution ...")
    resolution = exp_resolution(seeds)
    print("[4/5] electron dose ...")
    noise = exp_noise(seeds)
    print("[5/5] boundary criterion A/B ...")
    ref_off, ref_on = exp_refinement(seeds)

    pd.concat([
        baseline.assign(experiment="baseline"),
        density.assign(experiment="density"),
        resolution.assign(experiment="resolution"),
        noise.assign(experiment="noise"),
        ref_off.assign(experiment="refine_off"),
        ref_on.assign(experiment="refine_on"),
    ], ignore_index=True).to_csv(out / "validation_results.csv", index=False)

    line = "=" * 68
    print(f"\n{line}\nBASELINE (70 particles, ~40% coverage, ordinary dose)\n{line}")
    b = baseline
    print(f"  images analysed            : {len(b)}")
    print(f"  scale bar read             : {b['calibrated'].mean()*100:.0f}% "
          f"of images, mean error {b['calib_error_pct'].mean():.2f}%")
    print(f"  detection recall           : {b['recall'].mean()*100:.1f}%")
    print(f"  detection precision        : {b['precision'].mean()*100:.1f}%")
    print(f"  per-particle size error    : {b['size_mape'].mean():.2f}% (mean abs)")
    print(f"  per-particle bias          : {b['size_bias_pct'].mean():+.2f}%")
    print(f"  population D50 error       : {b['d50_error_pct'].abs().mean():.2f}% "
          f"(mean abs), bias {b['d50_error_pct'].mean():+.2f}%")
    if len(b.dropna(subset=['true_d50'])) > 2:
        r = np.corrcoef(b['true_d50'].dropna(), b['meas_d50'].dropna())[0, 1]
        print(f"  true vs measured D50       : r = {r:.4f}")

    print(f"\n{line}\nPACKING DENSITY\n{line}")
    g = density.groupby("n_particles_cfg").agg(
        coverage=("coverage", "mean"), recall=("recall", "mean"),
        d50_err=("d50_error_pct", "mean"), mape=("size_mape", "mean")).reset_index()
    print(f"  {'particles':>9s} {'coverage':>9s} {'recall':>8s} "
          f"{'D50 err':>9s} {'size err':>9s}")
    for _, r in g.iterrows():
        print(f"  {int(r['n_particles_cfg']):9d} {r['coverage']*100:8.1f}% "
              f"{r['recall']*100:7.1f}% {r['d50_err']:+8.2f}% {r['mape']:8.2f}%")

    print(f"\n{line}\nSPATIAL RESOLUTION\n{line}")
    g = resolution.groupby("diameter_px").agg(
        recall=("recall", "mean"), mape=("size_mape", "mean"),
        bias=("size_bias_pct", "mean")).reset_index()
    print(f"  {'diam (px)':>9s} {'recall':>8s} {'size err':>9s} {'bias':>9s}")
    for _, r in g.iterrows():
        print(f"  {r['diameter_px']:9.0f} {r['recall']*100:7.1f}% "
              f"{r['mape']:8.2f}% {r['bias']:+8.2f}%")

    print(f"\n{line}\nELECTRON DOSE\n{line}")
    g = noise.groupby("noise_electrons").agg(
        recall=("recall", "mean"), mape=("size_mape", "mean"),
        calib=("calibrated", "mean")).reset_index()
    print(f"  {'dose':>8s} {'recall':>8s} {'size err':>9s} {'scale bar read':>15s}")
    for _, r in g.iterrows():
        print(f"  {int(r['noise_electrons']):8d} {r['recall']*100:7.1f}% "
              f"{r['mape']:8.2f}% {r['calib']*100:14.0f}%")

    print(f"\n{line}\nBOUNDARY CRITERION: global Otsu vs per-particle half-max\n{line}")
    for name, d in (("global Otsu   ", ref_off), ("half-maximum  ", ref_on)):
        print(f"  {name} size err {d['size_mape'].mean():5.2f}%  "
              f"bias {d['size_bias_pct'].mean():+5.2f}%  "
              f"D50 err {d['d50_error_pct'].abs().mean():5.2f}%  "
              f"circularity {d['circularity'].mean():.3f}")

    make_figure(density, resolution, noise, baseline, out / "validation.png")
    print(f"\nWrote {out/'validation_results.csv'} and {out/'validation.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
