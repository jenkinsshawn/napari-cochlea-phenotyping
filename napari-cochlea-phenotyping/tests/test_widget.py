import os

import numpy as np
import tifffile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class FakeViewer:
    def __init__(self): self.layers = []
    def add_image(self, data, **kw): self.layers.append((data, kw))


_APP = []


def _app():
    """Create the Qt application once and keep a reference (Qt aborts if it is garbage collected)."""
    from qtpy.QtWidgets import QApplication
    if not _APP:
        _APP.append(QApplication.instance() or QApplication([]))
    return _APP[0]


def test_panel_opens_and_finds_a_device():
    _app()
    from napari_cochlea_phenotyping._widget import CochleaPhenotyping
    w = CochleaPhenotyping(None, detect_in_background=False)
    assert w.device is not None and w.tabs.count() == 4 and w.device_label.text()
    w.close()


def test_standardize_tab(tmp_path):
    _app()
    from napari_cochlea_phenotyping._standardize_tab import StandardizeTab
    for n in ("AR1 Calb2 Phall BASE", "AR2 Calb2 Phall APEX"):
        a = np.random.default_rng(0).integers(100, 3000, (6, 3, 24, 24)).astype(np.uint16)
        tifffile.imwrite(str(tmp_path / f"{n}.ome.tif"), a, ome=True, metadata={"axes": "ZCYX", "PhysicalSizeX": 0.1083,
                         "PhysicalSizeY": 0.1083, "PhysicalSizeZ": 0.3, "Channel": {"Name": ["DAPI", "TRITC", "Cy5"]}})
    v = FakeViewer(); t = StandardizeTab(v); t.add_paths([str(tmp_path)])
    assert t.images.count() == 2 and t.stain_table.rowCount() == 3 and t.scale_box.isHidden()
    sv = t.show_side_view(); assert sv.shape == (6, 24) and len(v.layers) == 1 and v.layers[0][1]["scale"] == (0.3, 0.1083)
    p = t.confirm(str(tmp_path / "results"))
    assert p and os.path.exists(p) and "Confirmed" in t.status.text() and t.session.bundle_side == -1
    t.side_higher.setChecked(True); assert t.bundle_side() == 1
    assert t.mouse_table.rowCount() == 2 and t.mouse_table.item(0, 0).text() == "AR1"
    t.mouse_table.item(0, 2).setText("control"); t.mouse_table.item(1, 2).setText("mutant")
    assert t.session.group_of_mouse == {"AR1": "control", "AR2": "mutant"} and t.control_cb.count() == 3
    t.control_cb.setCurrentText("control"); assert t.session.control_group == "control" and not t.session.stains_confirmed
    p = t.confirm(str(tmp_path / "results")); assert p and "Confirmed" in t.status.text()
    from napari_cochlea_phenotyping._standardize import DatasetSettings
    assert DatasetSettings.load(p).group_of("AR2 Calb2 Phall APEX") == "mutant"
    cy5 = [r for r in range(t.stain_table.rowCount()) if t.stain_table.item(r, 1).text() == "Cy5"][0]
    t.stain_table.cellWidget(cy5, 2).setCurrentText("to check")        # Cy5 -> no phalloidin
    assert not t.session.stains_confirmed and t.confirm(str(tmp_path / "results")) is None


def test_cells_tab_finds_surface(tmp_path):
    _app()
    from napari_cochlea_phenotyping._widget import CochleaPhenotyping
    from test_surface import layered
    st, d = layered(0.3, 0.1083, size_um=40.0)
    arr = np.stack([st, st // 2], 1)                                   # (z, c, y, x): Cy5 phalloidin, TRITC
    tifffile.imwrite(str(tmp_path / "AR1 Calb2 Phall BASE.ome.tif"), arr, ome=True, metadata={"axes": "ZCYX", "PhysicalSizeX": 0.1083,
                     "PhysicalSizeY": 0.1083, "PhysicalSizeZ": 0.3, "Channel": {"Name": ["Cy5", "TRITC"]}})
    v = FakeViewer(); w = CochleaPhenotyping(v, detect_in_background=False); w.standardize.add_paths([str(tmp_path)])
    r = w.cells.run_surface()
    assert r is not None and abs(np.median(r["surface"].depth_um(0.3) - d)) < 0.3 and len(v.layers) == 2
    assert "clear actin layer" in w.cells.status.text()


def test_run_tab_measures_with_the_pipeline(tmp_path, monkeypatch):
    """The Run tab writes the run's settings, starts the pipeline as a separate process and follows its steps."""
    import sys, time
    from qtpy.QtWidgets import QApplication
    from napari_cochlea_phenotyping._standardize_tab import StandardizeTab
    from napari_cochlea_phenotyping._run_tab import RunTab
    from test_run import FAKE
    app = _app()
    scripts = tmp_path / "scripts"; scripts.mkdir(); (scripts / "run_pipeline.py").write_text(FAKE)
    models = tmp_path / "models"; models.mkdir()
    monkeypatch.setenv("COCHLEA_PIPELINE_SCRIPTS", str(scripts)); monkeypatch.setenv("COCHLEA_PIPELINE_MODELS", str(models))
    imgs = tmp_path / "images"; imgs.mkdir()
    for n in ("AR1 Calb2 Phall BASE", "AR2 Calb2 Phall APEX"):
        a = np.random.default_rng(0).integers(100, 3000, (6, 3, 24, 24)).astype(np.uint16)
        tifffile.imwrite(str(imgs / f"{n}.ome.tif"), a, ome=True, metadata={"axes": "ZCYX", "PhysicalSizeX": 0.1083,
                         "PhysicalSizeY": 0.1083, "PhysicalSizeZ": 0.3, "Channel": {"Name": ["DAPI", "TRITC", "Cy5"]}})
    st = StandardizeTab(FakeViewer()); st.add_paths([str(imgs)])
    rt = RunTab(session_getter=lambda: st.session, results_getter=lambda: st.results_folder)
    assert not rt.start(python=sys.executable) and "Confirm" in rt.status.text()      # nothing confirmed yet
    assert st.confirm()
    got = []; rt.finished.connect(got.append)
    assert rt.start(python=sys.executable)
    t0 = time.time()
    while rt.timer.isActive() and time.time() - t0 < 60:
        app.processEvents(); time.sleep(0.05)
    assert rt.monitor.finished and got == [st.results_folder] and "Finished" in rt.status.text()
    assert [rt.steps.item(i).text() for i in range(rt.steps.count())] == ["fields: done earlier", "cells_v0: done (1.5 min)", "checks: done (0.2 min)"]
    assert "fewer than 3 cells" in rt.warnings.text() and os.path.exists(os.path.join(st.results_folder, "pipeline_settings_run.py"))


def test_run_tab_stop_ends_the_run(tmp_path, monkeypatch):
    import sys, time
    from napari_cochlea_phenotyping._standardize_tab import StandardizeTab
    from napari_cochlea_phenotyping._run_tab import RunTab
    app = _app()
    scripts = tmp_path / "scripts"; scripts.mkdir(); models = tmp_path / "models"; models.mkdir()
    (scripts / "run_pipeline.py").write_text(
        "import sys, subprocess, time\n"
        "if '--list' in sys.argv: print(' 1 slow              -     slow.py'); sys.exit(0)\n"
        "print('2026-11-13 10:00 CST  start slow: python slow.py', flush=True)\n"
        "subprocess.run([sys.executable, '-c', 'import time; time.sleep(120)'])\n")
    monkeypatch.setenv("COCHLEA_PIPELINE_SCRIPTS", str(scripts)); monkeypatch.setenv("COCHLEA_PIPELINE_MODELS", str(models))
    imgs = tmp_path / "images"; imgs.mkdir()
    a = np.random.default_rng(0).integers(100, 3000, (6, 3, 24, 24)).astype(np.uint16)
    tifffile.imwrite(str(imgs / "AR1 Calb2 Phall BASE.ome.tif"), a, ome=True, metadata={"axes": "ZCYX", "PhysicalSizeX": 0.1083,
                     "PhysicalSizeY": 0.1083, "PhysicalSizeZ": 0.3, "Channel": {"Name": ["DAPI", "TRITC", "Cy5"]}})
    st = StandardizeTab(FakeViewer()); st.add_paths([str(imgs)]); assert st.confirm()
    rt = RunTab(session_getter=lambda: st.session, results_getter=lambda: st.results_folder)
    assert rt.start(python=sys.executable)
    t0 = time.time()
    while rt.monitor.current != "slow" and time.time() - t0 < 30:
        app.processEvents(); time.sleep(0.05)
    t1 = time.time(); rt.stop()
    while rt.timer.isActive() and time.time() - t1 < 30:
        app.processEvents(); time.sleep(0.05)
    assert time.time() - t1 < 25 and not rt.running() and "Stopped by the user" in rt.status.text()
