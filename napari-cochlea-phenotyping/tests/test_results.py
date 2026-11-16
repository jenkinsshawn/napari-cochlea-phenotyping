import json
import os

import numpy as np
import pandas as pd

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from napari_cochlea_phenotyping._results import comparisons, open_results, smallest_p, status_level


class FakeViewer:
    def __init__(self): self.points = []
    def add_points(self, data, **kw): self.points.append((data, kw))


_APP = []


def _app():
    from qtpy.QtWidgets import QApplication
    if not _APP:
        _APP.append(QApplication.instance() or QApplication([]))
    return _APP[0]


def _results_folder(tmp, n_control=2, n_group=3):
    mice = [(f"C{i}", "control") for i in range(n_control)] + [(f"M{i}", "mutant") for i in range(n_group)]
    ft = pd.DataFrame([dict(field=f"{m} BASE", mouse=m, group=g, turn="BASE", damaged=False, ihc_per_100um=11.0 + i, pillar_width_sd_um=0.1)
                       for i, (m, g) in enumerate(mice)])
    ft.to_csv(tmp / "feature_table_fields.csv", index=False)
    G = pd.DataFrame([
        dict(measure="ihc_per_100um", vs_control="mutant", control_median=11.5, group_median=13.0, mouse_level_difference=2.5, p_permutation=0.1,
             p_model=0.001, validation="validated: 93% of clicked inner hair cells found", imaging_settings="",
             model_note="spread between mice estimated at zero: model interval and P too narrow; use the mouse-level P", p_permutation_adjusted=0.2),
        dict(measure="pillar_width_sd_um", vs_control="mutant", control_median=0.1, group_median=0.1, mouse_level_difference=0.0, p_permutation=0.9,
             p_model=0.9, validation="not validated (needs single-head accuracy)", imaging_settings="", model_note="", p_permutation_adjusted=0.9)])
    G.to_csv(tmp / "group_comparisons.csv", index=False)
    pd.DataFrame([dict(measure="inner hair cells found", agreement_rule="within 4 um", cells=348, fields=26, mice=8, agreement_pct=93.4,
                       interval_low_pct=88.0, interval_high_pct=97.0, documented_pct=93.4, interval_method="resampling mice",
                       reference="clicks", assessed_on="this dataset")]).to_csv(tmp / "validation_cards.csv", index=False)
    pd.DataFrame([dict(field="C0 BASE", cell=1, x=10.0, y=20.0, height_um=5.0, category="standing")]).to_csv(tmp / "ihc_stereocilia_cells_v2.csv", index=False)
    pd.DataFrame([dict(field="C0 BASE", x=30.0, y=40.0, row="OHC1"), dict(field="C0 BASE", x=31.0, y=60.0, row="OHC2"),
                  dict(field="C0 BASE", x=32.0, y=70.0, row="inside Calb2 band")]).to_csv(tmp / "hair_cell_rows_cells.csv", index=False)
    pd.DataFrame([dict(field="C0 BASE", pixel_um=0.1)]).to_csv(tmp / "field_table.csv", index=False)
    json.dump({"started": "2026-11-14 09:00 CST", "plugin_version": "0.test", "graphics_card": "none",
               "settings": {"CONTROL_GROUP": "control"}}, open(tmp / "run_info_2026-11-14_0900.json", "w"))
    return tmp


def test_status_levels():
    assert status_level(None, "validated: 91% found") == "validated"
    assert status_level(None, "usable, more variable: base raised") == "validated, more variable"
    assert status_level(None, "validated, more variable: field median") == "validated, more variable"
    assert status_level(None, "middle and base: within target") == "partly checked"
    assert status_level(None, "as nucleus area; volume not checked against outlines") == "not validated"
    assert status_level(None, "not at target: within 1 um in 64%") == "not validated"
    assert status_level(None, "quality check") == "quality check"
    assert status_level("partly checked", "validated") == "partly checked"     # an explicit status column wins


def test_smallest_p():
    assert abs(smallest_p(8, 2, 3) - 1 / 560) < 1e-12
    assert abs(smallest_p(4, 2, 2) - 1 / 6) < 1e-12
    assert smallest_p(3, 2, 2) != smallest_p(3, 2, 2)      # not possible: nan


def test_comparison_warnings(tmp_path):
    r = open_results(str(_results_folder(tmp_path, n_control=2, n_group=2)))
    cs = r.comparisons(); by = {c.measure: c for c in cs}
    assert cs[0].measure == "ihc_per_100um"                              # validated measures listed first
    w = by["ihc_per_100um"].warnings
    assert any("spread between mice" in x for x in w) and any("smallest possible P is 0.17" in x for x in w)
    assert by["pillar_width_sd_um"].level == "not validated" and by["pillar_width_sd_um"].warnings[0].startswith("not validated")
    assert by["ihc_per_100um"].mice_control == 2 and by["ihc_per_100um"].mice_group == 2
    r5 = open_results(str(_results_folder(tmp_path, n_control=2, n_group=3)))        # 5 mice: 1 of 10 arrangements
    assert any("smallest possible P is 0.10" in x for x in {c.measure: c for c in r5.comparisons()}["ihc_per_100um"].warnings)
    r6 = open_results(str(_results_folder(tmp_path, n_control=3, n_group=3)))        # 6 mice: 1 of 20, exactly 0.05
    assert not any("smallest possible" in x for x in {c.measure: c for c in r6.comparisons()}["ihc_per_100um"].warnings)


def test_results_tab(tmp_path):
    _app()
    from napari_cochlea_phenotyping._results_tab import ResultsTab
    v = FakeViewer(); t = ResultsTab(v, selected_getter=lambda: "C0 BASE")
    assert not t.load(str(tmp_path / "missing")) and "no results folder" in t.status.text()
    assert t.load(str(_results_folder(tmp_path)))
    assert t.cards.rowCount() == 1 and "93.4%" in t.cards.item(0, 1).text()
    assert t.table.rowCount() == 1 and "1 comparisons of quality checks" in t.warn.text()   # unvalidated measures hidden by default
    t.all_box.setChecked(True); assert t.table.rowCount() == 2
    assert t.table.item(0, 0).text().startswith("! ") and "(2 v 3 mice)" in t.table.item(0, 6).text()
    t.group_box.setCurrentIndex(1); assert t.table.rowCount() == 2 and t.group_box.currentText() == "mutant against control"
    data = t.show_layers()
    names = [d[0] for d in data]
    assert names == ["inner hair cells", "outer hair cells OHC1", "outer hair cells OHC2"] and len(v.points) == 3
    assert np.allclose(data[0][1], [[2.0, 1.0]])                      # (y, x) in um at 0.1 um pixels
    assert t.open_table("not a table") is None
