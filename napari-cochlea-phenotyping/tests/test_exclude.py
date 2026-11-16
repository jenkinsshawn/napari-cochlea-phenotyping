import os

import numpy as np
import pytest
import tifffile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_APP = []


def _app():
    from qtpy.QtWidgets import QApplication
    if not _APP:
        _APP.append(QApplication.instance() or QApplication([]))
    return _APP[0]


def _images(tmp_path):
    for n in ("AR1 Calb2 Phall BASE", "AR2 Calb2 Phall APEX"):
        a = np.random.default_rng(0).integers(100, 3000, (6, 3, 24, 24)).astype(np.uint16)
        tifffile.imwrite(str(tmp_path / f"{n}.ome.tif"), a, ome=True, metadata={"axes": "ZCYX", "PhysicalSizeX": 0.1083,
                         "PhysicalSizeY": 0.1083, "PhysicalSizeZ": 0.3, "Channel": {"Name": ["DAPI", "TRITC", "Cy5"]}})
    return tmp_path


def test_leave_out_and_include_again(tmp_path):
    _app()
    from qtpy.QtCore import Qt
    from napari_cochlea_phenotyping._standardize_tab import StandardizeTab
    from napari_cochlea_phenotyping._standardize import DatasetSettings
    t = StandardizeTab(None); t.add_paths([str(_images(tmp_path))])
    assert t.exclude_selected() == [] and "Select an image" in t.status.text()
    t.images.item(0).setSelected(True); f = t.images.item(0).data(Qt.UserRole)
    assert t.exclude_selected(reason="folded") == [f] and t.session.excluded == {f: "folded"}
    assert "left out of comparisons: folded" in t.images.item(0).text()
    p = t.confirm(str(tmp_path / "results")); assert p
    assert DatasetSettings.load(p).excluded_fields == {f: "folded"}
    t.images.item(0).setSelected(True); t.include_selected(); assert t.session.excluded == {}
    assert "left out" not in t.images.item(0).text()


def test_unknown_left_out_image_rejected():
    from napari_cochlea_phenotyping._standardize import DatasetSettings
    s = DatasetSettings((0.3, 0.1083, 0.1083), {"Cy5": "Phalloidin"}, mouse_of_field={"AR1 BASE": "AR1"}, excluded_fields={"AR9 APEX": "torn"})
    with pytest.raises(ValueError, match="left-out images not in the dataset"):
        s.validate()


def test_session_exclude_needs_a_known_image(tmp_path):
    from napari_cochlea_phenotyping._session import StandardizeSession
    s = StandardizeSession(); s.add_paths([str(_images(tmp_path))])
    with pytest.raises(KeyError):
        s.exclude("not an image", "torn")
    f = sorted(s.infos)[0]; s.exclude(f, "  "); assert s.excluded[f] == "left out"
