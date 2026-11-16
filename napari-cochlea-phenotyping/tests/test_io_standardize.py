import json
import os

import numpy as np
import pytest
import tifffile

from napari_cochlea_phenotyping._io import ChannelInfo, ImageInfo, read_info, read_stack
from napari_cochlea_phenotyping._standardize import (DatasetSettings, brightness, guess_stains, qc_flags,
                                                     scale_real_voxels, scale_to_background, stains_in_name)


def stack(z=5, c=3, y=32, x=40):
    a = np.arange(z * c * y * x, dtype=np.uint16).reshape(z, c, y, x)
    return a


def test_imagej_tif(tmp_path):
    p = str(tmp_path / "AR1 Calb2 Phall 60X BASE.tif"); a = stack()
    tifffile.imwrite(p, a, imagej=True, resolution=(1 / 0.1083, 1 / 0.1083), metadata={"spacing": 0.3, "unit": "um", "axes": "ZCYX"})
    info = read_info(p)
    assert info.shape_zcyx == (5, 3, 32, 40) and info.voxel_um[0] == pytest.approx(0.3)
    assert info.voxel_um[2] == pytest.approx(0.1083, rel=1e-4) and not info.warnings
    assert np.array_equal(read_stack(p), a)


def test_ome_tif(tmp_path):
    p = str(tmp_path / "f.ome.tif"); a = stack()
    tifffile.imwrite(p, a, ome=True, metadata={"axes": "ZCYX", "PhysicalSizeX": 0.2, "PhysicalSizeY": 0.2,
                                                 "PhysicalSizeZ": 0.5, "Channel": {"Name": ["DAPI", "TRITC", "Cy5"]}})
    info = read_info(p)
    assert info.voxel_um == pytest.approx((0.5, 0.2, 0.2)) and [c.name for c in info.channels] == ["DAPI", "TRITC", "Cy5"]
    assert np.array_equal(read_stack(p), a)


def test_tif_without_scale_warns(tmp_path):
    p = str(tmp_path / "plain.tif"); tifffile.imwrite(p, stack()[:, 0])
    info = read_info(p)
    assert any("voxel size" in w for w in info.warnings)
    assert read_stack(p).shape == (5, 1, 32, 40)


def test_time_series_refused(tmp_path):
    p = str(tmp_path / "t.tif"); tifffile.imwrite(p, np.zeros((2, 3, 8, 8), np.uint16), imagej=True, metadata={"axes": "TZYX"})
    with pytest.raises(ValueError):
        read_info(p)


def test_unsupported_format(tmp_path):
    with pytest.raises(ValueError):
        read_info(str(tmp_path / "x.czi"))


def info_with(name, filters, em=None):
    ch = [ChannelInfo(i, f, emission_nm=(em[i] if em else None)) for i, f in enumerate(filters)]
    return ImageInfo("x", name, "nd2", (10, len(ch), 8, 8), "uint16", 16, (0.3, 0.1083, 0.1083), ch)


def test_webber_stain_rule():
    g = [s.stain for s in guess_stains(info_with("AR524 Ocm, Calb2 Phall P1 60X BASE", ["EGFP", "TRITC", "Cy5", "DAPI"]))]
    assert g == ["Ocm", "Calb2", "Phalloidin", "DAPI"]
    g = [s.stain for s in guess_stains(info_with("JW5436 Calb2 PMCA2 Phall P1 60X APEX", ["FITC", "TRITC", "CY5", "DAPI"]))]
    assert g == ["PMCA2", "Calb2", "Phalloidin", "DAPI"]
    g = guess_stains(info_with("AR1065 Calb2 Prestin CtBP2", ["EGFP", "TRITC", "Cy5", "DAPI"]))
    assert [s.stain for s in g] == ["to check", "to check", "to check", "DAPI"] and "Prestin" in g[0].reason


def test_unknown_filter_uses_emission():
    g = guess_stains(info_with("sample", ["Alexa 647"], em=[668.0]))
    assert g[0].stain == "to check" and "far red" in g[0].reason


def test_names():
    assert stains_in_name("P1 Oncm calb2 phall") == ["Ocm", "Calb2", "Phalloidin"]


def test_brightness_ignores_zero_fill_and_counts_saturation():
    rng = np.random.default_rng(0); v = rng.integers(100, 200, (4, 50, 50)).astype(np.uint16)
    v[:, :, :10] = 0; v[0, 0, 20] = 4095
    b = brightness(v, saturation_value=4095)
    assert b["background"] >= 100 and b["zero_share"] == pytest.approx(0.2) and b["saturated_share"] > 0


def test_scale_to_background():
    x = np.array([0.0, 100.0, 1100.0]); y = scale_to_background(x, 100.0, 1100.0, floor=0.042)
    assert y[0] == 0 and y[1] == pytest.approx(0.042) and y[2] == pytest.approx(1.0)


def test_qc_flags():
    i = info_with("AR1 Calb2 Phall", ["TRITC", "Cy5"])
    st = {"Phalloidin": dict(saturated_share=0.01, zero_share=0.0, contrast=1.5)}
    f = qc_flags(i, ["Calb2", "Phalloidin"], st, expected_voxel_um=(0.5, 0.1083, 0.1083))
    assert any("saturated" in x for x in f) and any("contrast" in x for x in f) and any("plane spacing" in x for x in f)
    assert any("no phalloidin" in x for x in qc_flags(i, ["Calb2", "DAPI"], {}))


def test_settings_roundtrip(tmp_path):
    s = DatasetSettings((0.3, 0.1083, 0.1083), {"Cy5": "Phalloidin", "DAPI": "DAPI"}, bundle_side=-1, confirmed_by="tester")
    p = str(tmp_path / "s.json"); s.save(p); r = DatasetSettings.load(p)
    assert r.voxel_um == (0.3, 0.1083, 0.1083) and r.stains_by_channel_name["Cy5"] == "Phalloidin" and r.plugin_version
    d = json.load(open(p)); d["settings_version"] = 99; json.dump(d, open(p, "w"))
    with pytest.raises(ValueError):
        DatasetSettings.load(p)


def test_settings_validation():
    with pytest.raises(ValueError):
        DatasetSettings((0.3, None, 0.1), {"Cy5": "Phalloidin"}).validate()
    with pytest.raises(ValueError):
        DatasetSettings((0.3, 0.1, 0.1), {"Cy5": "Calb2"}).validate()
    with pytest.raises(ValueError):
        DatasetSettings((0.3, 0.1, 0.1), {"Cy5": "Phalloidin"}, bundle_side=0).validate()


RAW = os.environ.get("COCHLEA_TEST_RAW", "")   # folder of local .nd2 images; the test is skipped without it


@pytest.mark.skipif(not os.path.isdir(RAW), reason="local P1 images not available")
def test_local_nd2_metadata_matches_field_table():
    import glob
    import pandas as pd
    pytest.importorskip("nd2")
    table = os.environ.get("COCHLEA_TEST_FIELD_TABLE", "")
    if not os.path.exists(table):
        pytest.skip("field table not given (COCHLEA_TEST_FIELD_TABLE)")
    ft = pd.read_csv(table)
    for p in sorted(glob.glob(os.path.join(RAW, "*.nd2"))):
        info = read_info(p); r = ft[ft.field == info.field].iloc[0]
        assert round(info.voxel_um[2], 4) == r.pixel_um and round(info.voxel_um[0], 3) == r.z_step_um
        assert ";".join(s.stain for s in guess_stains(info)) == r.stains


def test_scale_real_voxels_ignores_fill_share():
    rng = np.random.default_rng(2); v = rng.gamma(2.0, 300.0, (20, 30, 30)) + 100
    a = scale_real_voxels(v, floor=0.042); w = v.copy(); w[:, :, :18] = 0          # 60% empty fill
    b = scale_real_voxels(w, floor=0.042)
    assert (b[w == 0] == 0).all() and np.percentile(a, 1) == pytest.approx(0.042, abs=1e-3)
    # the same real voxels keep nearly the same scale whatever the share of fill
    assert np.median(np.abs(b[:, :, 18:] - scale_real_voxels(v[:, :, 18:], floor=0.042))) < 1e-9
