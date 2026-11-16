"""NanoMetrix -- automated nanoparticle metrology from electron micrographs."""

from .calibration import Calibration, calibrate
from .metrics import (
    BULK_DENSITY_G_CM3, SizeStats, format_report, measure_particles,
    summarize_sizes,
)
from .pipeline import AnalysisResult, analyze
from .segmentation import SegmentationParams, segment, refine_half_max
from .synth import SynthConfig, SynthResult, generate

__version__ = "1.0.0"
__all__ = [
    "analyze", "AnalysisResult", "calibrate", "Calibration",
    "segment", "SegmentationParams", "refine_half_max",
    "measure_particles", "summarize_sizes", "SizeStats", "format_report",
    "BULK_DENSITY_G_CM3", "generate", "SynthConfig", "SynthResult",
]
