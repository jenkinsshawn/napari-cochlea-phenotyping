"""Redo-statistics command for a finished run (no Qt needed)."""
import os

import pytest

from napari_cochlea_phenotyping._run import SETTINGS_COPY, groups_in_table, redo_statistics_command


def test_groups_in_table(tmp_path):
    g = tmp_path / "groups.csv"
    g.write_text("mouse,group\nA1,WT\nA2,WT\nB1,mutant\nC1,\n", encoding="utf-8")
    assert groups_in_table(str(g)) == ["WT", "mutant"]


def test_redo_command(tmp_path):
    sc = tmp_path / "scripts"; sc.mkdir(); (sc / "phenotype_statistics.py").write_text("", encoding="utf-8")
    res = tmp_path / "results"; res.mkdir(); (res / SETTINGS_COPY).write_text("", encoding="utf-8")
    g = tmp_path / "groups.csv"; g.write_text("mouse,group\nA1,WT\n", encoding="utf-8")
    argv, env = redo_statistics_command(str(res), str(g), "WT", scripts_folder=str(sc), python="py")
    assert argv == ["py", str(sc / "phenotype_statistics.py"), "--with-noise-adjusted", "--report", "--groups", str(g), "--control", "WT"]
    assert env["CONFOCAL_SETTINGS"] == str(res / SETTINGS_COPY) and env["PYTHONPATH"] == ""


def test_redo_command_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        redo_statistics_command(str(tmp_path), str(tmp_path / "none.csv"), scripts_folder=str(tmp_path))


def test_redo_command_drop_share(tmp_path):
    sc = tmp_path / "scripts"; sc.mkdir(); (sc / "phenotype_statistics.py").write_text("", encoding="utf-8")
    argv, _ = redo_statistics_command(str(tmp_path), "", scripts_folder=str(sc), python="py", drop_share=0.2)
    assert argv == ["py", str(sc / "phenotype_statistics.py"), "--with-noise-adjusted", "--report", "--drop-least-reliable", "0.2"]
    with pytest.raises(ValueError):
        redo_statistics_command(str(tmp_path), "", scripts_folder=str(sc), python="py", drop_share=0.8)
