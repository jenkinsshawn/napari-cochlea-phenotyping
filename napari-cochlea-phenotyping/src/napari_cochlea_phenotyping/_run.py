r"""Run step: the validated measurement pipeline on a confirmed dataset, in a separate process.

The pipeline is the folder of step scripts (run_pipeline.py and the scripts it calls). It is found from the environment
variable COCHLEA_PIPELINE_SCRIPTS, else beside the plugin source (Projects\confocal-phenotyping\stereocilia\scripts).
A run writes a settings copy into the results folder that names the dataset_settings.json the user confirmed, the image
folder, a work folder, the results folder, an annotations folder for this dataset's hand references (empty at first)
and the trained models, then runs run_pipeline.py with it. Trained models are applied, never retrained, by a run.

Kept apart from the panel so it can be tested without a window.
"""
from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

SCRIPTS_ENV = "COCHLEA_PIPELINE_SCRIPTS"
MODELS_ENV = "COCHLEA_PIPELINE_MODELS"
SETTINGS_COPY = "pipeline_settings_run.py"


def find_scripts_folder() -> Optional[str]:
    """Folder holding run_pipeline.py: COCHLEA_PIPELINE_SCRIPTS, else the project layout beside the plugin source."""
    cands = [os.environ.get(SCRIPTS_ENV, "")]
    here = os.path.dirname(os.path.abspath(__file__))                       # ...\napari-cochlea-phenotyping\src\napari_cochlea_phenotyping
    cands.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(here))), "stereocilia", "scripts"))
    for c in cands:
        if c and os.path.exists(os.path.join(c, "run_pipeline.py")):
            return c
    return None


def default_models_folder(scripts_folder: Optional[str]) -> Optional[str]:
    """COCHLEA_PIPELINE_MODELS, else the MODELS folder of the pipeline's own settings file."""
    if os.environ.get(MODELS_ENV):
        return os.environ[MODELS_ENV]
    if scripts_folder:
        p = os.path.join(scripts_folder, "pipeline_settings.py")
        if os.path.exists(p):
            txt = open(p, encoding="utf-8").read()
            m = re.search(r'^DATA_ROOT\s*=\s*r?["\'](.+?)["\']', txt, re.M)
            if m:
                return os.path.join(m.group(1), "models")
    return None


@dataclass
class RunPlan:
    settings_json: str                       # dataset_settings.json saved by the Standardize step
    image_folder: str
    results_folder: str
    scripts_folder: str
    models_folder: str
    work_folder: str = ""                    # default: <results folder>\work
    annotations_folder: str = ""             # default: <results folder>\annotations (this dataset's hand references)
    excluded_fields: Dict[str, str] = field(default_factory=dict)   # field -> reason (left out of group comparisons)

    def __post_init__(self):
        self.work_folder = self.work_folder or os.path.join(self.results_folder, "work")
        self.annotations_folder = self.annotations_folder or os.path.join(self.results_folder, "annotations")

    def problems(self) -> List[str]:
        out = []
        if not os.path.exists(self.settings_json): out.append("confirm the dataset in the Standardize tab first")
        if not os.path.isdir(self.image_folder): out.append(f"image folder not found: {self.image_folder}")
        if not (self.scripts_folder and os.path.exists(os.path.join(self.scripts_folder, "run_pipeline.py"))):
            out.append(f"pipeline not found; set {SCRIPTS_ENV} to the folder with run_pipeline.py")
        if not (self.models_folder and os.path.isdir(self.models_folder)):
            out.append(f"trained models not found ({self.models_folder or 'no folder'}); set {MODELS_ENV}")
        return out

    def write_settings(self) -> str:
        """Write the settings copy for this run into the results folder and return its path."""
        bad = self.problems()
        if bad:
            raise ValueError("; ".join(bad))
        for d in (self.results_folder, self.work_folder, self.annotations_folder):
            os.makedirs(d, exist_ok=True)
        lines = ["# Written by the Harp panel for one run; values not given here keep the pipeline's defaults.",
                 f"DATASET_SETTINGS = {self.settings_json!r}", f"RAW = {self.image_folder!r}",
                 f"WORK = {self.work_folder!r}", f"RESULTS = {self.results_folder!r}",
                 f"ANNOTATIONS = {self.annotations_folder!r}", f"MODELS = {self.models_folder!r}",
                 f"DAMAGED = {dict(self.excluded_fields)!r}"]
        p = os.path.join(self.results_folder, SETTINGS_COPY)
        with open(p, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        return p

    def command(self, python: Optional[str] = None, only: Optional[str] = None, list_steps: bool = False) -> List[str]:
        cmd = [python or sys.executable, "-u", os.path.join(self.scripts_folder, "run_pipeline.py"),
               "--settings", os.path.join(self.results_folder, SETTINGS_COPY)]
        if list_steps: cmd.append("--list")
        elif only: cmd += ["--only", only]
        return cmd


def image_folder_of(paths: List[str]) -> str:
    """The one folder holding all images of a dataset (the pipeline reads one image folder)."""
    folders = sorted({os.path.dirname(os.path.abspath(p)) for p in paths})
    if len(folders) != 1:
        raise ValueError(f"the images are in {len(folders)} folders; put them in one folder for a run")
    return folders[0]


_PATTERNS = [("start", re.compile(r"\bstart (\w+):")), ("done", re.compile(r"\bdone (\w+) in ([\d.]+) min")),
             ("skip", re.compile(r"\bskip (\w+) \(done\)")), ("failed", re.compile(r"\bFAILED in (\w+): (.*)")),
             ("stopped", re.compile(r"\bSTOPPED (?:at (\w+)|before starting): ?(.*)")), ("note", re.compile(r"\bnote: (\w+): (.*)")),
             ("finished", re.compile(r"\bpipeline finished\b"))]


class RunMonitor:
    """State of a run from the pipeline's printed progress lines."""
    def __init__(self, steps: Optional[List[str]] = None):
        self.steps = list(steps or []); self.status: Dict[str, str] = {s: "waiting" for s in self.steps}
        self.minutes: Dict[str, float] = {}; self.notes: List[str] = []; self.failures: List[str] = []
        self.current: Optional[str] = None; self.finished = False; self.stopped: Optional[str] = None; self.tail: List[str] = []

    def _set(self, step, st):
        if step not in self.status: self.steps.append(step)
        self.status[step] = st

    def feed(self, line: str):
        line = line.rstrip("\r\n"); self.tail = (self.tail + [line])[-200:]
        for kind, rx in _PATTERNS:
            m = rx.search(line)
            if not m: continue
            if kind == "start": self.current = m.group(1); self._set(m.group(1), "running")
            elif kind == "done": self._set(m.group(1), "done"); self.minutes[m.group(1)] = float(m.group(2)); self.current = None
            elif kind == "skip": self._set(m.group(1), "done earlier")
            elif kind == "failed": self.failures.append(f"{m.group(1)}: {m.group(2)}")
            elif kind == "note": self.notes.append(f"{m.group(1)}: {m.group(2)}")
            elif kind == "stopped":
                self.stopped = line.strip()
                if m.group(1): self._set(m.group(1), "stopped")
            elif kind == "finished": self.finished = True
            break

    def summary(self) -> str:
        if self.finished: return f"finished: {sum(v == 'done' for v in self.status.values())} steps run"
        if self.stopped: return self.stopped
        if self.current: return f"running {self.current} ({self.steps.index(self.current) + 1} of {len(self.steps)})" if self.current in self.steps else f"running {self.current}"
        return "not started"


def parse_step_list(text: str) -> List[tuple]:
    """(step, done) from the output of run_pipeline.py --list."""
    out = []
    for line in text.splitlines():
        m = re.match(r"\s*\d+\s+(\w+)\s+(done|-)\s", line)
        if m: out.append((m.group(1), m.group(2) == "done"))
    return out


def redo_statistics_command(results_folder: str, groups_csv: str = "", control: str = "",
                            scripts_folder: Optional[str] = None, python: Optional[str] = None, drop_share: float = 0.0):
    """Command and environment that redo the statistics of a finished run without measuring again, with another
    genotype table (columns mouse, group; empty keeps the run's genotypes) and leaving out the given share (0-0.5) of
    the least reliable top-view bundles in each cell type. Both the unadjusted and the noise-adjusted tests are run and
    the results report is rewritten. Returns (argv, env) or raises FileNotFoundError."""
    import sys
    sc = scripts_folder or find_scripts_folder()
    if not sc or not os.path.exists(os.path.join(sc, "phenotype_statistics.py")):
        raise FileNotFoundError(f"pipeline scripts not found; set {SCRIPTS_ENV}")
    if groups_csv and not os.path.exists(groups_csv):
        raise FileNotFoundError(groups_csv)
    if not 0 <= drop_share <= 0.5:
        raise ValueError("drop_share must be between 0 and 0.5")
    argv = [python or sys.executable, os.path.join(sc, "phenotype_statistics.py"), "--with-noise-adjusted", "--report"]
    if groups_csv:
        argv += ["--groups", groups_csv]
    if drop_share > 0:
        argv += ["--drop-least-reliable", f"{drop_share:g}"]
    if control:
        argv += ["--control", control]
    env = dict(os.environ); env["PYTHONPATH"] = ""
    env.setdefault("JOBLIB_MULTIPROCESSING", "0")
    run_copy = os.path.join(results_folder, SETTINGS_COPY)
    if os.path.exists(run_copy):
        env["CONFOCAL_SETTINGS"] = run_copy
    return argv, env


def groups_in_table(groups_csv: str) -> List[str]:
    """Group names in a genotype table (second column), in order of first appearance."""
    import csv
    out: List[str] = []
    with open(groups_csv, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.reader(fh))
    for r in rows[1:]:
        if len(r) >= 2 and r[1].strip() and r[1].strip() not in out:
            out.append(r[1].strip())
    return out
