# Cochlea Phenotyping

Automatic measurement of hair cells and stereocilia bundles in confocal images of mouse cochlear whole mounts,
as a napari plugin (Plugins > Cochlea Phenotyping) and the pipeline scripts it runs.

## Layout

- `napari-cochlea-phenotyping/`: the napari plugin (Standardize, Cells, Measure and Results tabs) and its tests.
- `stereocilia/scripts/`: the pipeline steps the Measure tab runs (`run_pipeline.py` runs them in order;
  `pipeline_settings.py` holds the folders, image scale and mouse table; `run_checks.py` compares the outputs with the
  hand references).

The trained models (bundle detector, hair cell and row classifiers, inner hair cell bundle classifier, pillar detector)
and the validation reference are in `models/`; copy them into the folder named by `MODELS` in `pipeline_settings.py`.

## Install

```
conda activate imaging
cd napari-cochlea-phenotyping
pip install -e ".[pipeline]"
napari
```

The plugin finds the pipeline scripts in `stereocilia/scripts` beside its own folder, or in the folder named by the
environment variable `COCHLEA_PIPELINE_SCRIPTS`.

## Run the pipeline without napari

```
conda activate imaging
cd stereocilia/scripts
python run_pipeline.py --settings <your settings file>
python run_checks.py
```
