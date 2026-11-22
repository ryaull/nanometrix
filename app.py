"""
NanoMetrix web app.

Run with:  streamlit run app.py

Upload a micrograph, get a size distribution. The scale bar is read
automatically, with a manual override for images whose bar is unreadable or
absent.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from nanometrix import BULK_DENSITY_G_CM3, SynthConfig, analyze, generate
from nanometrix.io import load_grayscale
from nanometrix.segmentation import SegmentationParams
from nanometrix.viz import distribution_figure, overlay_figure, shape_figure

st.set_page_config(page_title="NanoMetrix", page_icon="*", layout="wide")


def load_uint8(data) -> np.ndarray:
    return load_grayscale(data)


st.title("NanoMetrix")
st.caption(
    "Automated particle size and shape analysis for SEM and TEM micrographs. "
    "The scale bar is located and read automatically, so results come back in "
    "nanometres rather than pixels."
)

with st.sidebar:
    st.header("Input")
    uploaded = st.file_uploader(
        "Micrograph", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"]
    )
    use_demo = st.checkbox("Use a synthetic demo image", value=uploaded is None)

    st.header("Calibration")
    auto_cal = st.checkbox("Read the scale bar automatically", value=True)
    manual_nm_px = None
    if not auto_cal:
        manual_nm_px = st.number_input(
            "nm per pixel", min_value=0.0001, max_value=1000.0,
            value=0.50, step=0.01, format="%.4f",
        )

    st.header("Segmentation")
    polarity = st.selectbox(
        "Particle contrast", ["auto", "bright", "dark"], index=0,
        help="Bright for most SEM, dark for most TEM. Auto decides from the image.",
    )
    split = st.checkbox("Separate touching particles (watershed)", value=True)
    exclude_border = st.checkbox(
        "Discard particles clipped by the frame", value=True,
        help="A clipped particle only shows part of itself, so measuring it "
             "biases the size distribution downward.",
    )
    min_area = st.slider("Minimum particle area (px)", 5, 500, 40, step=5)
    refine = st.checkbox(
        "Re-cut boundaries at half-maximum", value=False,
        help="Alternative boundary criterion. Measured against ground truth "
             "it performed worse than the default on these images.",
    )

    st.header("Material")
    material = st.selectbox(
        "For the surface-area estimate",
        ["(none)"] + sorted(BULK_DENSITY_G_CM3), index=0,
    )

if use_demo and uploaded is None:
    c1, c2, c3 = st.columns(3)
    with c1:
        demo_mod = st.selectbox("Modality", ["SEM", "TEM"])
    with c2:
        demo_d = st.slider("True median diameter (nm)", 5, 200, 40, step=5)
    with c3:
        demo_n = st.slider("Particles", 20, 160, 70, step=10)
    truth = generate(SynthConfig(
        seed=1, modality=demo_mod, median_diameter_nm=float(demo_d),
        nm_per_px=demo_d / 80.0, n_particles=demo_n,
    ))
    image = truth.image
    st.info(
        f"Synthetic image. Ground truth: {len(truth.diameters_complete_nm)} "
        f"fully visible particles, true D50 = "
        f"{np.median(truth.diameters_complete_nm):.2f} nm, "
        f"true scale {truth.nm_per_px:.4g} nm/px."
    )
elif uploaded is not None:
    try:
        image = load_uint8(uploaded)
    except ValueError as exc:
        st.error(f"Could not read the uploaded file: {exc}")
        st.stop()
else:
    st.stop()

params = SegmentationParams(
    polarity=polarity, split_touching=split,
    exclude_border=exclude_border, min_area_px=int(min_area),
)

with st.spinner("Analysing..."):
    result = analyze(
        image, nm_per_px=manual_nm_px, params=params,
        refine_boundaries=refine,
        material=None if material == "(none)" else material,
    )

for w in result.warnings:
    st.warning(w)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Particles", result.n_particles)
if result.stats:
    k2.metric(f"D50 ({result.unit})", f"{result.stats.d50:.2f}")
    k3.metric("Log-normal GSD", f"{result.stats.lognormal_gsd:.3f}")
    if result.stats.ssa_m2_per_g and np.isfinite(result.stats.ssa_m2_per_g):
        k4.metric("Surface area (m2/g)", f"{result.stats.ssa_m2_per_g:.2f}")
    else:
        k4.metric("Mean circularity", f"{result.shape.get('circularity', 0):.3f}")

st.subheader("Detected particles")
st.pyplot(overlay_figure(image, result.labels, result.diameters, result.unit),
          use_container_width=True)

st.subheader("Size distribution")
st.pyplot(distribution_figure(result.diameters, result.stats, result.unit),
          use_container_width=True)

left, right = st.columns([1, 1])
with left:
    st.subheader("Shape")
    st.pyplot(shape_figure(result.particles), use_container_width=True)
with right:
    st.subheader("Summary")
    st.code(result.report(), language="text")

st.subheader("Per-particle measurements")
st.dataframe(result.particles, use_container_width=True, height=320)
if not result.particles.empty:
    st.download_button(
        "Download CSV", result.particles.to_csv(index=False).encode(),
        file_name="nanometrix_particles.csv", mime="text/csv",
    )
