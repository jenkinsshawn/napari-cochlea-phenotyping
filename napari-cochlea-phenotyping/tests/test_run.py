import os
import subprocess
import sys

import pytest

from napari_cochlea_phenotyping._run import (RunMonitor, RunPlan, default_models_folder, image_folder_of,
                                             parse_step_list)

FAKE = """import sys, time
args = sys.argv[1:]
settings = args[args.index("--settings") + 1]
assert open(settings).read().startswith("# Written by the Harp panel")
if "--list" in args:
    print(" 1 fields            done  make_field_pictures.py --redo")
    print(" 2 cells_v0          -     outline_cells.py --tag v0")
    print(" 3 checks            -     run_checks.py"); sys.exit(0)
print("2026-11-13 10:00 CST  run record: x.json")
print("2026-11-13 10:00 CST  skip fields (done)")
print("2026-11-13 10:00 CST  start cells_v0: python outline_cells.py")
print("2026-11-13 10:01 CST  note: cells_v0: study field skipped: A1: fewer than 3 cells")
print("2026-11-13 10:02 CST  done cells_v0 in 1.5 min")
print("2026-11-13 10:02 CST  start checks: python run_checks.py")
if "--fail" in open(settings).read():
    print("2026-11-13 10:03 CST  FAILED in checks: A1: failed: boom")
    print("2026-11-13 10:03 CST  STOPPED at checks: exit code 1, 1 failed study fields, after 0.1 min"); sys.exit(1)
print("2026-11-13 10:03 CST  done checks in 0.2 min")
print("2026-11-13 10:03 CST  pipeline finished")
"""


def make(tmp_path, fail=False):
    scripts = tmp_path / "scripts"; scripts.mkdir(); (scripts / "run_pipeline.py").write_text(FAKE)
    (scripts / "pipeline_settings.py").write_text('DATA_ROOT = r"%s"\n' % str(tmp_path / "data"))
    (tmp_path / "data" / "models").mkdir(parents=True); imgs = tmp_path / "images"; imgs.mkdir()
    sj = tmp_path / "res" / "dataset_settings.json"; sj.parent.mkdir(); sj.write_text("{}")
    plan = RunPlan(str(sj), str(imgs), str(tmp_path / "res"), str(scripts), default_models_folder(str(scripts)),
                   excluded_fields={"A2": "torn"} if not fail else {"--fail": "x"})
    return plan


def test_settings_copy_and_problems(tmp_path):
    plan = make(tmp_path); assert plan.problems() == []
    p = plan.write_settings(); txt = open(p).read()
    vals = {}; exec(compile(txt, p, "exec"), vals)                  # the copy is valid Python with the right folders
    assert (vals["DATASET_SETTINGS"], vals["RAW"], vals["WORK"], vals["RESULTS"], vals["ANNOTATIONS"]) == (
        plan.settings_json, plan.image_folder, plan.work_folder, plan.results_folder, plan.annotations_folder)
    assert vals["MODELS"] == plan.models_folder and vals["DAMAGED"] == {"A2": "torn"} and "\\\\" not in vals["RESULTS"]
    assert os.path.isdir(plan.work_folder) and os.path.isdir(plan.annotations_folder)
    bad = RunPlan(str(tmp_path / "missing.json"), str(tmp_path / "nowhere"), str(tmp_path / "r2"), "", "")
    assert len(bad.problems()) == 4
    with pytest.raises(ValueError):
        bad.write_settings()


def test_run_and_monitor(tmp_path):
    plan = make(tmp_path); plan.write_settings()
    out = subprocess.run(plan.command(list_steps=True), capture_output=True, text=True).stdout
    steps = parse_step_list(out); assert steps == [("fields", True), ("cells_v0", False), ("checks", False)]
    mon = RunMonitor([s for s, _ in steps])
    p = subprocess.Popen(plan.command(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout: mon.feed(line)
    assert p.wait() == 0 and mon.finished
    assert mon.status == {"fields": "done earlier", "cells_v0": "done", "checks": "done"} and mon.minutes["cells_v0"] == 1.5
    assert mon.notes == ["cells_v0: study field skipped: A1: fewer than 3 cells"] and mon.summary().startswith("finished")


def test_stopped_run_is_reported(tmp_path):
    plan = make(tmp_path, fail=True); plan.write_settings(); mon = RunMonitor(["fields", "cells_v0", "checks"])
    p = subprocess.Popen(plan.command(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout: mon.feed(line)
    assert p.wait() == 1 and not mon.finished and mon.status["checks"] == "stopped"
    assert mon.failures == ["checks: A1: failed: boom"] and "STOPPED at checks" in mon.summary()


def test_one_image_folder(tmp_path):
    a = tmp_path / "a"; b = tmp_path / "b"
    assert image_folder_of([str(a / "x.nd2"), str(a / "y.nd2")]) == str(a)
    with pytest.raises(ValueError):
        image_folder_of([str(a / "x.nd2"), str(b / "y.nd2")])
