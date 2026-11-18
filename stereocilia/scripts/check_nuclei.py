"""Blind check of hair cell nucleus outlines (items from make_nucleus_check.py).

Each item shows one nucleus (DAPI) from above on the left and from the side on the right, with the automatic outline in
cyan. Press:
  1   the outline fits one whole nucleus
  2   the outline is cut short, spills over, or joins two nuclei
  0   cannot tell
  Backspace   back one item
Answers save after every key to ANNOTATIONS\nucleus_check\answers.csv. Answered items are skipped on restart.

Run (imaging env, scripts folder):  python check_nuclei.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline_settings as S
from datetime import datetime, timezone, timedelta
import numpy as np, pandas as pd
import napari

DIR = os.path.join(S.ANNOTATIONS, "nucleus_check"); ANS = os.path.join(DIR, "answers.csv")
CHOICES = {"1": "fits", "2": "cut or merged", "0": "cannot tell"}


def central_time():
    now = datetime.now(timezone.utc); dst = 3 <= now.month <= 10
    return (now + timedelta(hours=-5 if dst else -6)).strftime("%Y-%m-%d %H:%M ") + ("CDT" if dst else "CST")


class Check:
    def __init__(self):
        self.imgs = np.load(os.path.join(DIR, "items.npz"))["images"]; self.n = len(self.imgs)
        self.ans = pd.read_csv(ANS) if os.path.exists(ANS) else pd.DataFrame(columns=["item", "answer", "saved"])
        done = set(self.ans.item); self.k = next((i for i in range(self.n) if i + 1 not in done), self.n - 1)
        self.v = napari.Viewer(title="Nucleus outline check")
        self.layer = self.v.add_image(self.imgs[self.k], rgb=True, name="item")
        keys = {k: (lambda a_: (lambda *args: self.answer(a_)))(a) for k, a in CHOICES.items()}
        keys["Backspace"] = lambda *args: self.back()
        for k, fn in keys.items():                       # the selected layer's own shortcuts (1, 2, 3 ...) win over the viewer's,
            self.v.bind_key(k, fn, overwrite=True)       # so the keys are bound on the layer as well
            self.layer.bind_key(k, fn, overwrite=True)
        self.v.reset_view()
        self.v.text_overlay.visible = True; self.v.text_overlay.font_size = 13; self.show()

    def show(self, msg=""):
        self.layer.data = self.imgs[self.k]
        done = len(set(self.ans.item))
        self.v.text_overlay.text = (f"item {self.k + 1} of {self.n}  ({done} answered)   1 fits   2 cut or merged   0 cannot tell   Backspace back  {msg}")

    def answer(self, a):
        self.ans = self.ans[self.ans.item != self.k + 1]
        self.ans = pd.concat([self.ans, pd.DataFrame([dict(item=self.k + 1, answer=a, saved=central_time())])], ignore_index=True)
        self.ans.sort_values("item").to_csv(ANS, index=False)
        if len(set(self.ans.item)) >= self.n: self.show("all items answered; you can close the window"); return
        self.k = next((i for i in list(range(self.k + 1, self.n)) + list(range(0, self.k)) if i + 1 not in set(self.ans.item)), self.k); self.show()

    def back(self):
        self.k = max(self.k - 1, 0); self.show()


if __name__ == "__main__":
    Check(); napari.run()
