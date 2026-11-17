# Cochlea Phenotyping

Automatic measurement of hair cells and stereocilia bundles in confocal images of mouse cochlear whole mounts, as a
napari plugin.

## Install (Windows)

Download this repository (Code > Download ZIP), unzip it, type `powershell` in the File Explorer address bar of the
unzipped folder and run:

```
powershell -ExecutionPolicy Bypass -File .\install_cochlea_phenotyping.ps1
```

Then open napari from the **Cochlea Phenotyping** shortcut on the Desktop. See
`napari-cochlea-phenotyping/Cochlea_Phenotyping_user_guide.docx` for a step-by-step guide.

## Contents

- `napari-cochlea-phenotyping/`: the napari plugin and its tests.
- `stereocilia/scripts/`: the measurement steps the plugin runs (`run_pipeline.py` runs them in order).
- `models/`: the trained models.
