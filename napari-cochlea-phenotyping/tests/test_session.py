import numpy as np
import pytest
import tifffile

from napari_cochlea_phenotyping._session import StandardizeSession, side_view


def write(p, z=6, c=3, spacing=0.3, px=0.1083, names=("DAPI", "TRITC", "Cy5")):
    a = np.random.default_rng(0).integers(100, 3000, (z, c, 24, 24)).astype(np.uint16)
    tifffile.imwrite(p, a, ome=True, metadata={"axes": "ZCYX", "PhysicalSizeX": px, "PhysicalSizeY": px,
                                                 "PhysicalSizeZ": spacing, "Channel": {"Name": list(names)}})
    return a


def test_session_flow(tmp_path):
    for n in ("AR1 Calb2 Phall BASE", "AR2 Calb2 Phall APEX"):
        write(str(tmp_path / f"{n}.ome.tif"))
    write(str(tmp_path / "AR3 Calb2 Phall MIDDLE.ome.tif"), spacing=0.5)
    (tmp_path / "notes.txt").write_text("not an image")
    s = StandardizeSession(); added = s.add_paths([str(tmp_path)])
    assert len(added) == 3 and s.voxel_um == pytest.approx((0.3, 0.1083, 0.1083))
    assert s.fields_off_scale() == ["AR3 Calb2 Phall MIDDLE"]
    assert s.stains == {("Calb2, Phalloidin", "DAPI"): "DAPI", ("Calb2, Phalloidin", "TRITC"): "Calb2",
                        ("Calb2, Phalloidin", "Cy5"): "Phalloidin"}
    assert s.mouse_of == {"AR1 Calb2 Phall BASE": "AR1", "AR2 Calb2 Phall APEX": "AR2", "AR3 Calb2 Phall MIDDLE": "AR3"}
    with pytest.raises(ValueError):
        s.settings()                                            # nothing confirmed yet
    s.confirm_stains(); s.set_bundle_side(-1)
    q = s.check_field("AR1 Calb2 Phall BASE")
    assert "Phalloidin" in q["stats"] and q["stats"]["Phalloidin"]["contrast"] > 0
    rows = {r["field"]: r for r in s.qc_rows()}
    assert rows["AR1 Calb2 Phall BASE"]["checked"] and not rows["AR2 Calb2 Phall APEX"]["checked"]
    paths = s.save(str(tmp_path / "results"), confirmed_by="tester")
    assert all(p.endswith((".json", ".csv")) for p in paths)


def test_editing_a_stain_needs_reconfirming(tmp_path):
    write(str(tmp_path / "AR1 Calb2 Phall.ome.tif")); s = StandardizeSession(); s.add_paths([str(tmp_path)])
    s.confirm_stains(); s.set_stain("TRITC", "PMCA2")
    assert not s.stains_confirmed
    s.set_stain("Cy5", "to check")
    with pytest.raises(ValueError):
        s.confirm_stains()


def test_unreadable_files_reported(tmp_path):
    (tmp_path / "broken.tif").write_bytes(b"not a tif")
    s = StandardizeSession(); assert s.add_paths([str(tmp_path)]) == [] and len(s.errors) == 1


def test_side_view():
    v = np.zeros((5, 4, 3)); v[1, 2, 0] = 7
    sv = side_view(v); assert sv.shape == (5, 3) and sv[1, 0] == 7


def test_one_filter_different_stains_in_two_panels(tmp_path):
    """Webber lab images: EGFP carries Ocm in some files and PMCA2 in others; each set keeps its own stain."""
    names = ("DAPI", "EGFP", "TRITC", "Cy5")
    for n in ("AR1064 Ocm Calb2 Phall P1 60X APEX", "AR524 Ocm, Calb2 Phall P1 60X BASE"):
        write(str(tmp_path / f"{n}.ome.tif"), c=4, names=names)
    for n in ("AR454 Calb2 PMCA2 Phall P1 60X MIDDLE", "AR47 Calb2 PMCA2 Phall P1 60X BASE", "JW5436 (2)Calb2 PMCA2 Phall P1 60X APEX"):
        write(str(tmp_path / f"{n}.ome.tif"), c=4, names=names)
    s = StandardizeSession(); s.add_paths([str(tmp_path)])
    assert sorted(s.panels()) == ["Calb2, Ocm, Phalloidin", "Calb2, PMCA2, Phalloidin"]
    assert s.stains_of("AR1064 Ocm Calb2 Phall P1 60X APEX") == ["DAPI", "Ocm", "Calb2", "Phalloidin"]
    assert s.stains_of("AR47 Calb2 PMCA2 Phall P1 60X BASE") == ["DAPI", "PMCA2", "Calb2", "Phalloidin"]
    s.set_stain("EGFP", "other", panel="Calb2, Ocm, Phalloidin")           # one panel only
    assert s.stains_of("AR454 Calb2 PMCA2 Phall P1 60X MIDDLE")[1] == "PMCA2"
    s.set_stain("EGFP", "Ocm", panel="Calb2, Ocm, Phalloidin")
    assert s.mouse_of["JW5436 (2)Calb2 PMCA2 Phall P1 60X APEX"] == "JW5436 (2)"
    for m, g in (("AR1064", "control"), ("AR524", "control"), ("AR454", "Smad3"), ("AR47", "beta-catenin"), ("JW5436 (2)", "Smad3")):
        s.set_group(m, g)
    s.control_group = "control"; s.confirm_stains(); s.set_bundle_side(-1)
    st = s.settings(confirmed_by="tester")
    assert st.stains_for("AR1064 Ocm Calb2 Phall P1 60X APEX", list(names))[1] == "Ocm"
    assert st.stains_for("AR47 Calb2 PMCA2 Phall P1 60X BASE", list(names))[1] == "PMCA2"
    assert "EGFP" not in st.stains_by_channel_name and st.stains_by_channel_name["Cy5"] == "Phalloidin"
    assert st.group_of("JW5436 (2)Calb2 PMCA2 Phall P1 60X APEX") == "Smad3"
    p = s.save(str(tmp_path / "results"))[0]
    from napari_cochlea_phenotyping._standardize import DatasetSettings
    back = DatasetSettings.load(p)
    assert back.stains_by_panel == st.stains_by_panel and back.group_of_mouse == st.group_of_mouse and back.control_group == "control"


def test_control_group_must_have_mice(tmp_path):
    write(str(tmp_path / "AR1 Calb2 Phall.ome.tif")); s = StandardizeSession(); s.add_paths([str(tmp_path)])
    s.confirm_stains(); s.set_bundle_side(-1); s.control_group = "control"
    with pytest.raises(ValueError):
        s.settings()
    s.set_group("AR1", "control"); assert s.settings().control_group == "control"


def test_mouse_guess():
    from napari_cochlea_phenotyping._session import guess_mouse
    assert guess_mouse("AR1383-2 Calb2 PMCA2 Phall P1 60X APEX (4)") == "AR1383-2"
    assert guess_mouse("AR170 (CRE bcatMUT) Calb2, oncm, phall P1 60X BASE") == "AR170"
    assert guess_mouse("JW5436 Calb2 PMCA2 Phall P1 60X Apex") == "JW5436"
    assert guess_mouse("sample 3 cochlea") == "sample"


def test_version_1_settings_still_load(tmp_path):
    import json
    from napari_cochlea_phenotyping._standardize import DatasetSettings
    p = tmp_path / "dataset_settings.json"
    p.write_text(json.dumps(dict(voxel_um=[0.3, 0.1083, 0.1083], stains_by_channel_name={"Cy5": "Phalloidin", "DAPI": "DAPI"},
                                 bundle_side=-1, confirmed_by="x", confirmed_on="", plugin_version="0.1.0", device={},
                                 notes="", settings_version=1)))
    s = DatasetSettings.load(str(p)); assert s.stains_for("any field", ["DAPI", "Cy5"]) == ["DAPI", "Phalloidin"]
