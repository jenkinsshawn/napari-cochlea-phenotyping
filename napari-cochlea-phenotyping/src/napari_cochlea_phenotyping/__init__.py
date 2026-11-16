"""Measure hair cells, stereocilia bundles and nuclei in confocal images of the cochlea."""
__version__ = "0.1.0"

from ._device import Device, detect_device, cellpose_model_kwargs, run_with_memory_fallback
from ._resampling import distinct_resamples, cluster_bootstrap, compare_groups_by_mouse, field_summary, small_sample_factor
from ._io import ImageInfo, ChannelInfo, read_info, read_stack
from ._standardize import DatasetSettings, guess_stains, brightness, qc_flags, scale_to_background

__all__ = ["Device", "detect_device", "cellpose_model_kwargs", "run_with_memory_fallback", "distinct_resamples",
           "cluster_bootstrap", "compare_groups_by_mouse", "field_summary", "small_sample_factor", "ImageInfo", "ChannelInfo",
           "read_info", "read_stack", "DatasetSettings", "guess_stains", "brightness", "qc_flags", "scale_to_background"]
