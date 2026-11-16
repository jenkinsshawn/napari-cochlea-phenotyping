import os

import numpy as np
import pytest

from napari_cochlea_phenotyping._surface import find_surface


def layered(dz, dx, depth_um=lambda x, y: 6.0 + 0.02 * x + 0.01 * y, size_um=110.0, thick_um=12.0, seed=0):
    """Synthetic phalloidin stack: dim tissue with a bright, speckled actin layer at depth_um (um)."""
    rng = np.random.default_rng(seed); n = int(size_um / dx); nz = int(thick_um / dz)
    yy, xx = np.mgrid[0:n, 0:n] * dx; d = depth_um(xx, yy); z = (np.arange(nz) * dz)[:, None, None]
    layer = np.exp(-0.5 * ((z - d[None]) / 0.5) ** 2) * (1 + 3 * (rng.random((1, n, n)) > 0.97))
    return (1000 + 300 * rng.random((nz, n, n)) + 4000 * layer).astype(np.uint16), d


@pytest.mark.parametrize("dz,dx", [(0.3, 0.1083), (0.6, 0.1083), (0.3, 0.2166), (1.0, 0.2)])
def test_surface_depth_at_different_scales(dz, dx):
    st, d = layered(dz, dx); s = find_surface(st, (dz, dx, dx))
    err = np.abs(s.depth_um(dz) - d)
    assert np.median(err) <= max(0.2, dz / 2) and s.good_squares > 0


def test_odd_depth_smoothing_keeps_surface_centered():
    st, d = layered(0.6, 0.1083, depth_um=lambda x, y: 6.0 + 0 * x)
    assert abs(np.median(find_surface(st, (0.6, 0.1083, 0.1083)).depth_um(0.6) - d)) <= 0.15


def test_no_clear_layer_falls_back():
    st = (1000 + 10 * np.random.default_rng(1).random((20, 128, 128))).astype(np.uint16)
    s = find_surface(st, (0.3, 0.1083, 0.1083)); assert s.planes.shape == (128, 128) and s.good_squares < 6


WORK = os.environ.get("COCHLEA_TEST_WORK", "")    # work folder whose fields\<field>.npz hold the pipeline's surfaces
RAW = os.environ.get("COCHLEA_TEST_RAW", "")


@pytest.mark.skipif(not (os.path.isdir(WORK) and os.path.isdir(RAW)), reason="local P1 images not available")
def test_local_surfaces_identical_to_pipeline():
    pytest.importorskip("nd2")
    from napari_cochlea_phenotyping._io import read_info, read_stack
    for f in ["AR524 Ocm, Calb2 Phall P1 60X BASE", "AR47 Calb2 PMCA2 Phall P1 60X BASE"]:
        z = np.load(os.path.join(WORK, "fields", f + ".npz")); st = [str(s) for s in z["stains"]]
        p = os.path.join(RAW, f + ".nd2"); info = read_info(p)
        s = find_surface(read_stack(p)[:, st.index("Phalloidin")], info.voxel_um)
        assert np.array_equal(s.planes.astype(z["surface"].dtype), z["surface"])   # stored as float32
