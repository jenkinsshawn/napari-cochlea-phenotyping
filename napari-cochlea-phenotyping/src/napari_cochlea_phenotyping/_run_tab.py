"""Run tab: measure every image of the confirmed dataset with the validated pipeline.

One button starts the run; the steps, their status and any warnings are shown as it goes. The run is a separate
process, so napari stays responsive and a failing step cannot take the viewer down. Finished steps are kept, so a
stopped run continues where it stopped.
"""
from __future__ import annotations

import os
from typing import Callable, Optional

import queue
import subprocess
import sys
import threading

from qtpy.QtCore import Qt, QTimer, Signal
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QPlainTextEdit, QPushButton,
                            QVBoxLayout, QWidget)

from ._run import RunMonitor, RunPlan, default_models_folder, find_scripts_folder, image_folder_of, parse_step_list

OK, WARN, RUN, GRAY = "#2e7d32", "#b26a00", "#1565c0", "#757575"
COLORS = {"done": OK, "done earlier": OK, "running": RUN, "stopped": WARN, "waiting": GRAY}


class RunTab(QWidget):
    finished = Signal(str)                      # results folder, when a run finishes

    def __init__(self, session_getter: Callable = lambda: None, results_getter: Callable = lambda: "",
                 excluded_getter: Callable = lambda: {}, device_getter: Callable = lambda: None):
        super().__init__()
        self.session_getter, self.results_getter, self.excluded_getter = session_getter, results_getter, excluded_getter
        self.device_getter = device_getter
        self.proc: Optional[subprocess.Popen] = None; self.timer = None; self._lines = queue.Queue(); self.monitor = RunMonitor(); self.plan: Optional[RunPlan] = None
        lay = QVBoxLayout(self)
        row = QHBoxLayout(); self.start_btn = QPushButton("Measure all images"); self.stop_btn = QPushButton("Stop")
        self.start_btn.clicked.connect(lambda: self.start()); self.stop_btn.clicked.connect(lambda: self.stop()); self.stop_btn.setEnabled(False)
        row.addWidget(self.start_btn); row.addWidget(self.stop_btn); lay.addLayout(row)
        self.status = QLabel("Confirm the dataset in the Standardize tab, then measure."); self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.RichText); lay.addWidget(self.status)
        self.steps = QListWidget(); lay.addWidget(self.steps, 1)
        self.warnings = QLabel(""); self.warnings.setWordWrap(True); self.warnings.setTextFormat(Qt.RichText); lay.addWidget(self.warnings)
        self.log_btn = QPushButton("Show log"); self.log_btn.setCheckable(True); lay.addWidget(self.log_btn)
        self.log = QPlainTextEdit(); self.log.setReadOnly(True); self.log.setMaximumBlockCount(5000); self.log.setVisible(False)
        self.log_btn.toggled.connect(self.log.setVisible); lay.addWidget(self.log, 1)

    # plan -------------------------------------------------------------------------------------------------------
    def make_plan(self) -> RunPlan:
        s = self.session_getter(); res = self.results_getter()
        if s is None or not s.infos:
            raise ValueError("add images in the Standardize tab first")
        if not (s.stains_confirmed and s.bundle_side_confirmed):
            raise ValueError("confirm the stains, groups and bundle direction in the Standardize tab first")
        sc = find_scripts_folder()
        return RunPlan(os.path.join(res, "dataset_settings.json"), image_folder_of([i.path for i in s.infos.values()]),
                       res, sc or "", default_models_folder(sc) or "", excluded_fields=dict(self.excluded_getter() or {}))

    def _show_steps(self):
        self.steps.clear()
        for st in self.monitor.steps:
            v = self.monitor.status.get(st, "waiting"); extra = f" ({self.monitor.minutes[st]:.1f} min)" if st in self.monitor.minutes else ""
            it = QListWidgetItem(f"{st}: {v}{extra}"); it.setForeground(QColor(COLORS.get(v, GRAY))); self.steps.addItem(it)
        w = [f"<span style='color:{WARN}'>{x}</span>" for x in self.monitor.failures] + \
            [f"<span style='color:{GRAY}'>{x}</span>" for x in self.monitor.notes[-8:]]
        self.warnings.setText("<br>".join(w))

    # running ----------------------------------------------------------------------------------------------------
    def start(self, python: Optional[str] = None) -> bool:
        try:
            self.plan = self.make_plan(); self.plan.write_settings()
        except ValueError as e:
            self.status.setText(f"<span style='color:{WARN}'>{str(e)[:1].upper() + str(e)[1:]}</span>"); return False
        try:
            out = subprocess.run(self.plan.command(python, list_steps=True), cwd=self.plan.scripts_folder, capture_output=True,
                                 text=True, timeout=300, **_no_window())
            steps = parse_step_list(out.stdout)
        except Exception as e:
            self.status.setText(f"<span style='color:{WARN}'>The pipeline could not list its steps: {e}</span>"); return False
        self.monitor = RunMonitor([s for s, _ in steps])
        for s, done in steps:
            if done: self.monitor.status[s] = "done earlier"
        self._show_steps(); self.log.clear(); self._lines = queue.Queue()
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        d = self.device_getter()
        if d is not None and not d.gpu:      # processor chosen (or no usable card): the steps must not find a card on their own
            env["CUDA_VISIBLE_DEVICES"] = ""
        self.proc = subprocess.Popen(self.plan.command(python), cwd=self.plan.scripts_folder, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1, env=env,
                                     **_no_window(new_group=True))
        threading.Thread(target=self._pump, args=(self.proc, self._lines), daemon=True).start()
        self.timer = QTimer(self); self.timer.timeout.connect(self._poll); self.timer.start(200)
        self.start_btn.setEnabled(False); self.stop_btn.setEnabled(True)
        self.status.setText(f"<span style='color:{RUN}'>Measuring {len(self.session_getter().infos)} images into {self.plan.results_folder}</span>")
        return True

    @staticmethod
    def _pump(proc, q):
        for line in proc.stdout:
            q.put(line)
        q.put(None)

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def _poll(self):
        ended = False
        while True:
            try:
                line = self._lines.get_nowait()
            except queue.Empty:
                break
            if line is None:
                ended = True; break
            self.monitor.feed(line); self.log.appendPlainText(line.rstrip("\r\n"))
        self._show_steps()
        if self.monitor.current and not ended: self.status.setText(f"<span style='color:{RUN}'>{self.monitor.summary()}</span>")
        if ended:
            self.timer.stop(); self._done(self.proc.wait())

    def _done(self, code=0):
        self._show_steps(); self.start_btn.setEnabled(True); self.stop_btn.setEnabled(False)
        if self.monitor.finished and code == 0:
            self.status.setText(f"<span style='color:{OK}'>Finished; results in {self.plan.results_folder}</span>")
            self.finished.emit(self.plan.results_folder)
        else:
            why = self.monitor.stopped or f"the run ended with exit code {code}"
            self.status.setText(f"<span style='color:{WARN}'>{why}. Press Measure again to continue from the stopped step.</span>")

    def stop(self):
        """Stop the run and every step it started (the pipeline runs each step as its own process)."""
        if not self.running(): return
        self.monitor.stopped = "Stopped by the user"
        try:
            import psutil                                   # installed with napari
            parent = psutil.Process(self.proc.pid); procs = parent.children(recursive=True) + [parent]
            for p in procs:
                try:
                    p.kill()
                except psutil.NoSuchProcess:
                    pass
            psutil.wait_procs(procs, timeout=20)
        except ImportError:                                 # without psutil: the operating system's own tree stop
            if os.name == "nt":
                subprocess.run(["taskkill", "/T", "/F", "/PID", str(self.proc.pid)], capture_output=True, **_no_window())
            else:
                import signal
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
        try:
            self.proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def _no_window(new_group: bool = False) -> dict:
    """No console window on Windows; a separate process group so a stop reaches every step."""
    if os.name == "nt":
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | (getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if new_group else 0)
        return dict(creationflags=flags)
    return dict(start_new_session=True) if new_group else {}
