"""Bundle detector on every field, in one run (about 30 minutes on this workstation).

Steps (the image stacks are skipped where they already exist):
  1. image stacks at fixed heights from the cell surface for every field (paint_bundles.py --prepare --all)
  2. train the detector on Shawn's strokes, 200 trees, all processor cores
     (train_bundle_detector.py --train; adding the VASCilia outlines gave slightly worse
     outlines in the 6 Nov check, so they are not used here)
  3. bundle probability for every field (train_bundle_detector.py --predict)
  4. bundle outlines and measures per hair cell (measure_bundles.py --detector, cutoff 0.85)
Progress, with Central times, goes to E:\\Webber_confocal\\work\\run_bundles_all_log.txt.

Run from the scripts folder:
    python run_bundles_all.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline_settings as S
import os, subprocess, sys
from datetime import datetime, timezone, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(S.WORK, 'run_bundles_all_log.txt')


def central_time():
    now = datetime.now(timezone.utc); dst = 3 <= now.month <= 10
    return (now + timedelta(hours=-5 if dst else -6)).strftime("%Y-%m-%d %H:%M ") + ("CDT" if dst else "CST")


def say(msg):
    line = f"{central_time()}  {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def step(name, args):
    say(f"start: {name}")
    r = subprocess.run([sys.executable] + args, cwd=HERE)
    if r.returncode != 0:
        say(f"STOPPED: {name} failed (exit {r.returncode}); the lines above show why"); sys.exit(1)
    say(f"done: {name}")


def main():
    missing = []
    for mod in ("nd2", "sklearn", "joblib", "skimage"):
        try:
            __import__(mod)
        except ImportError:
            missing.append({"sklearn": "scikit-learn", "skimage": "scikit-image"}.get(mod, mod))
    if missing:
        print("Missing packages. Run this once, then run_bundles_all.py again:")
        print("    python -m pip install " + " ".join(missing)); sys.exit(1)
    os.environ.pop("BUNDLE_JOBS", None); os.environ.pop("BUNDLE_TREES", None)   # all cores, 200 trees
    step("image stacks for every field", ["paint_bundles.py", "--prepare", "--all"])
    detector = os.path.join(S.MODELS, "bundle_detector_strokes.joblib")
    if "--train" in sys.argv or not os.path.exists(detector):      # training is explicit; a run applies the saved detector
        step("train detector (200 trees, Shawn's strokes)", ["train_bundle_detector.py", "--train"])
    else:
        say(f"using the saved detector {detector} (run_bundles_all.py --train retrains it on the painted strokes)")
    step("bundle probability for every field", ["train_bundle_detector.py", "--predict"])
    step("bundle outlines and measures", ["measure_bundles.py", "--detector"])
    say("finished; summary in results\\bundle_fields_detector.csv")


if __name__ == "__main__":
    main()
