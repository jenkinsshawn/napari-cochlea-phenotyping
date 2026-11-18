"""Run the whole confocal measurement pipeline in order, from the .nd2 files to the feature table and the checks.

Each step is one script from this folder. When a step finishes, a marker is written to
E:\\Webber_confocal\\work\\pipeline_done\\<step>.txt, and finished steps are skipped on the next run.
Progress, with Central times, goes to the screen and to E:\\Webber_confocal\\work\\pipeline_log.txt.

Steps, in the order their inputs require (approximate times on the Creighton workstation, 28-32 fields):
   1 fields            make_field_pictures.py --redo             stacks, cell surface, channel table           30 min
   2 cells_v0          outline_cells.py --tag v0 --redo --gpu    cell outlines from phalloidin (Cellpose-SAM)  15 min
   3 ihc_cells         ihc_cellpose.py --gpu                     inner hair cells from the Calb2 projection
   4 cells_v1          outline_cells.py --tag v1 --base v0 --redo --gpu  v0 outlines plus inner hair cells from Calb2
   5 cell_types        assign_cell_types.py                      cell type and row of each outline              2 min
   6 ohc_bundles       run_bundles_all.py                        stacks at fixed heights, bundle detector, bundle measures  30 min
   7 ihc_stacks_v1     ihc_stacks.py                             inner hair cell cut-outs (first frame; field list for rows)  15-20 min
   8 rows              hair_cell_rows_classify.py                outer hair cell rows (trained classifiers)
   9 ihc_stacks        ihc_stacks.py --v2                        inner hair cell cut-outs (second frame)       15-20 min
  10 ihc_stereocilia   ihc_stereocilia.py --v2                   inner hair cell bundle base, tip, height      about 6 s per cell
  11 ihc_bundle_shape  ihc_bundle_shape.py                       inner hair cell bundle 3D shape (front view)  1 min
  12 ohc_bundle_shape  ohc_bundles_v2.py                         outer hair cell bundle span and bend
  13 spatial           spatial_stats.py                          spacing, alignment and neighbor measures
  14 row_order         row_order.py                              straightness and order of the outer hair cell rows
  15 ohc1_stacks       ohc1_stacks.py                            first-row outer hair cell cut-outs
  16 ohc1_height       ohc1_bundle_height.py                     first-row bundle height                        1 min
  17 nuclei            nuclei_cellpose.py --gpu                  nucleus outlines (graphics card)               about 17 s per field
  18 ihc_nuclei        ihc_nuclei_calb2.py                       inner hair cell nuclei and Calb2               40 min
  19 chromatin         nucleus_chromatin.py                      nucleus size, shape, DAPI and chromatin        about 50 s per field
  20 hair_cell_nuclei  hair_cell_nuclei.py                       nucleus of each inner and outer hair cell
  21 identity         hair_cell_identity.py                     inner-like and outer-like hair cells (Calb2, Ocm, PMCA2)
  22 pillars           pillar_heads.py                           pillar cell heads
  23 settings          imaging_settings.py                       imaging settings and comparability warnings    1 min
  24 field_quality     field_quality.py                          image quality and surface depth and slope       1 min
  25 features          build_feature_table.py                    field table and group comparisons              1 min
  26 bundle_phenotypes bundle_phenotypes.py                      bundle outline, texture, height and volume     5 min
  27 cell_phenotypes   cell_phenotypes.py                        cell surface, row packing, polarity, nuclei
  28 phenotype_statistics phenotype_statistics.py --with-noise-adjusted  group tests, unadjusted and noise-adjusted  12 min
  29 checks            run_checks.py                             agreement with the hand references             5 min
  30 report            report_run.py                             results report (Word) and four CSV tables
Steps 3, 7, 9, 15 and 17 (and the stacks inside step 6) skip fields whose outputs already exist. To recompute
everything, run into a fresh work and results folder with a settings copy (--settings); the done-markers are kept
in that work folder, so a fresh folder also starts the step list from the top.

Run (imaging env, from the scripts folder):
    python run_pipeline.py                     every step not yet done
    python run_pipeline.py --list              show the steps and which are done
    python run_pipeline.py --from ihc_stacks   rerun from this step to the end
    python run_pipeline.py --only checks       rerun one step
    python run_pipeline.py --settings E:\\NewStudy\\pipeline_settings_newstudy.py    use a copied settings file (new dataset)
For a new dataset, copy pipeline_settings.py, edit the folders, image scale and mouse table, and pass it with --settings.
"""
import os, sys
if "--settings" in sys.argv:   # must be set before the settings file is read
    os.environ["CONFOCAL_SETTINGS"] = os.path.abspath(sys.argv[sys.argv.index("--settings") + 1])
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline_settings as S
import os, re, subprocess, sys, time
from datetime import datetime, timezone, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = S.WORK
DONE = os.path.join(WORK, "pipeline_done"); LOG = os.path.join(WORK, "pipeline_log.txt")

STEPS = [("fields", ["make_field_pictures.py", "--redo"]), ("cells_v0", ["outline_cells.py", "--tag", "v0", "--redo", "--gpu"]),
         ("ihc_cells", ["ihc_cellpose.py", "--gpu"]), ("cells_v1", ["outline_cells.py", "--tag", "v1", "--base", "v0", "--redo", "--gpu"]),
         ("cell_types", ["assign_cell_types.py"]), ("ohc_bundles", ["run_bundles_all.py"]), ("ihc_stacks_v1", ["ihc_stacks.py"]),
         ("rows", ["hair_cell_rows_classify.py"]), ("ihc_stacks", ["ihc_stacks.py", "--v2"]), ("ihc_stereocilia", ["ihc_stereocilia.py", "--v2"]),
         ("ihc_bundle_shape", ["ihc_bundle_shape.py"]),
         ("ohc_bundle_shape", ["ohc_bundles_v2.py"]), ("spatial", ["spatial_stats.py"]), ("row_order", ["row_order.py"]), ("ohc1_stacks", ["ohc1_stacks.py"]), ("ohc1_height", ["ohc1_bundle_height.py"]),
         ("nuclei", ["nuclei_cellpose.py", "--gpu"]), ("ihc_nuclei", ["ihc_nuclei_calb2.py"]), ("chromatin", ["nucleus_chromatin.py"]),
         ("hair_cell_nuclei", ["hair_cell_nuclei.py"]), ("identity", ["hair_cell_identity.py"]), ("pillars", ["pillar_heads.py"]),
         ("settings", ["imaging_settings.py"]), ("field_quality", ["field_quality.py"]),
         ("features", ["build_feature_table.py"]),
         ("bundle_phenotypes", ["bundle_phenotypes.py"]), ("cell_phenotypes", ["cell_phenotypes.py"]),
         ("phenotype_statistics", ["phenotype_statistics.py", "--with-noise-adjusted"]),
         ("checks", ["run_checks.py"]), ("report", ["report_run.py"])]


def central_now():
    t = datetime.now(timezone.utc); y = t.year
    dst_start = datetime(y, 3, 8, 8, tzinfo=timezone.utc); dst_start += timedelta(days=(6 - dst_start.weekday()) % 7)    # 2nd Sunday March, 2 am CST
    dst_end = datetime(y, 11, 1, 7, tzinfo=timezone.utc); dst_end += timedelta(days=(6 - dst_end.weekday()) % 7)          # 1st Sunday Nov, 2 am CDT
    off, name = (-5, "CDT") if dst_start <= t < dst_end else (-6, "CST")
    return (t + timedelta(hours=off)).strftime("%Y-%m-%d %H:%M ") + name



def _field_names():
    folder = os.path.join(S.WORK, "fields")
    names = [f[:-4] for f in os.listdir(folder) if f.endswith(".npz")] if os.path.isdir(folder) else []
    names += S.raw_fields()
    return sorted(set(names), key=len, reverse=True)

def any_field_named(line):
    return any(n in line for n in _field_names())

def study_field_named(line):
    """True when the line names a field whose mouse is in the mouse table (pipeline_settings.MICE)."""
    return any(n in line and S.group_of(n) is not None for n in _field_names())

def say(msg):
    line = f"{central_now()}  {msg}"; print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh: fh.write(line + "\n")


REQUIRED = ["numpy", "pandas", "scipy", "skimage", "sklearn", "joblib", "matplotlib", "nd2", "torch", "cellpose", "statsmodels"]

def missing_packages():
    """Packages the steps import that this Python cannot find (checked before any step starts)."""
    import importlib.util
    return [m for m in REQUIRED if importlib.util.find_spec(m) is None]

def write_run_info():
    """Record how this run was made: settings values, a checksum of every script and plugin file, the plugin version,
    the graphics card, and the start time (Central). Written to RESULTS\\run_info_<date time>.json."""
    import hashlib, json, glob as _g
    files = sorted(_g.glob(os.path.join(HERE, "*.py")))
    try:
        import napari_cochlea_phenotyping as ncp; pkg_dir = os.path.dirname(ncp.__file__); version = ncp.__version__
        files += sorted(_g.glob(os.path.join(pkg_dir, "*.py")))
    except Exception as e:
        version = f"not importable ({type(e).__name__})"
    md5 = {os.path.relpath(f, os.path.dirname(HERE)): hashlib.md5(open(f, "rb").read()).hexdigest() for f in files}
    try:
        import torch; card = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none visible to PyTorch"
    except Exception as e:
        card = f"PyTorch not importable ({type(e).__name__})"
    sett = {k: (v if isinstance(v, (int, float, str, bool)) else str(v)) for k, v in vars(S).items()
            if k.isupper() and not k.startswith("_")}
    info = dict(started=central_now(), python=sys.version.split()[0], plugin_version=version, graphics_card=card,
                settings_file=os.environ.get("CONFOCAL_SETTINGS", "default"), settings=sett, file_md5=md5)
    os.makedirs(S.RESULTS, exist_ok=True); stamp = central_now()[:16].replace(":", "").replace(" ", "_")
    p = os.path.join(S.RESULTS, f"run_info_{stamp}.json")
    with open(p, "w", encoding="utf-8") as fh: json.dump(info, fh, indent=1)
    return p

def main(argv):
    os.makedirs(DONE, exist_ok=True); names = [s for s, _ in STEPS]
    if "--list" in argv:
        for i, (s, cmd) in enumerate(STEPS, 1): print(f"{i:2d} {s:17s} {'done' if os.path.exists(os.path.join(DONE, s + '.txt')) else '-':5s} {' '.join(cmd)}")
        return 0
    todo = names
    if "--only" in argv: todo = [argv[argv.index("--only") + 1]]
    elif "--from" in argv: todo = names[names.index(argv[argv.index("--from") + 1]):]
    force = "--only" in argv or "--from" in argv
    miss = missing_packages()
    if miss and "--list" not in argv:
        names = {"skimage": "scikit-image", "sklearn": "scikit-learn"}
        say("STOPPED before starting: missing packages " + ", ".join(miss) + ". In the imaging env: pip install " + " ".join(names.get(m, m) for m in miss))
        return 1
    say(f"run record: {write_run_info()}")
    for s, cmd in STEPS:
        if s not in todo: continue
        marker = os.path.join(DONE, s + ".txt")
        if os.path.exists(marker) and not force: say(f"skip {s} (done)"); continue
        say(f"start {s}: python {' '.join(cmd)}"); t0 = time.time()
        # run the step, show its output live, and collect lines reporting a failed field
        proc = subprocess.Popen([sys.executable] + cmd, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        failed, ignored, tail = [], [], []
        for line in proc.stdout:
            print(line, end="", flush=True)
            tail = (tail + [line.rstrip()])[-20:]
            if study_field_named(line) and re.search(r":\s*(no |fewer than|not enough|skipped)", line, re.I) and not re.search(r"\bfailed\b", line, re.I):
                say(f"note: {s}: study field skipped: {line.strip()[:200]}")   # visible in the log, not only on screen
            if re.search(r"\bfailed\b", line, re.I):
                (failed if study_field_named(line) or not any_field_named(line) else ignored).append(line.strip())
        rc = proc.wait()
        for x in ignored: say(f"note: {s}: failure on a field outside the mouse table (not used): {x[:160]}")
        if rc != 0 or failed:
            for x in failed: say(f"FAILED in {s}: {x[:200]}")
            if rc != 0:
                say(f"last output of {s}:")
                for x in tail: say("    " + x[:220])
            say(f"STOPPED at {s}: exit code {rc}, {len(failed)} failed study fields, after {(time.time() - t0) / 60:.1f} min"); return rc or 1
        with open(marker, "w", encoding="utf-8") as fh: fh.write(central_now() + "\n")
        say(f"done {s} in {(time.time() - t0) / 60:.1f} min")
    say("pipeline finished"); return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
