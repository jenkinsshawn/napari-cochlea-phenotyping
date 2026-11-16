# napari-cochlea-phenotyping

A napari plugin for measuring confocal images of the cochlea step by step: hair cells and their rows, stereocilia
bundles, nuclei and pillar heads. It is being built from the validated measurement scripts used for P1 mouse cochleae
(phalloidin, Calb2, PMCA2, Ocm and DAPI). Each step writes an editable napari layer that the next step reads, so you
can correct an outline before moving on, or bring in a layer from another plugin.

The panel only asks for what a person has to decide, once per dataset: the stain in each channel, which way the
bundles point (from a side view), and the scale when a file does not give it. Brightness standardization, quality
checks, the choice of graphics card and the statistics run on their own; anything unusual appears as a warning.

Version 0.1.0 has four tabs. The Cells tab so far finds the cell surface of one image; hair cell outlines and rows
are measured by the Measure tab and shown in the Results tab.

## Using the panel

1. **Standardize.** Add the images or a folder. Check the stain of each channel (guessed from the filter and the file
   name), the group of each mouse (read from the start of the file name), the control group, and the bundle
   direction on the side view; the scale is read from the files. Select an image with folded or torn tissue and press
   **Leave out of comparisons**: it is still measured but not used in the group comparisons. Press **Confirm**.
2. **Cells.** Shows the cell surface of the selected image, with a warning when the actin layer is clear in too few
   places.
3. **Measure.** One button measures every image with the validated pipeline (the scripts folder of
   confocal-phenotyping\stereocilia, found automatically). Steps run in order in a separate process, and a stopped
   run continues from the step where it stopped. About 3-4 hours for 30 images on a graphics card.
4. **Results.** Opens when a run finishes (or open any results folder). It shows how well each measure agreed with
   the hand references, the group comparisons with one value per mouse (status of each measure, mouse-level P,
   adjusted P, and notes such as too few mice for a P below 0.05), the result tables, and the measured cells of the
   selected image as layers. Measures that are not validated are hidden unless you tick the box.

## Install

```
conda activate imaging
pip install -e "C:\path\to\napari-cochlea-phenotyping[pipeline]"
napari
```

Then open **Plugins > Cochlea Phenotyping**.

## Graphics card

The plugin uses an NVIDIA graphics card when PyTorch can see one, then Apple graphics, and otherwise the processor;
the top of the panel shows which (hover for the reason). To force the processor, set the environment variable
`COCHLEA_PHENOTYPING_DEVICE=cpu` before starting napari. If you have an NVIDIA card and the panel says the processor,
PyTorch was probably installed without graphics card support; reinstall it with CUDA from https://pytorch.org.

Outlines from the graphics card and the processor differ slightly, so a batch runs on one device and the device is
written into the results. If the card runs out of memory on a large field, the field is retried on the card with
smaller batches.

## Intervals on the results

Cells from the same mouse are more alike than cells from different mice, and neighboring cells along a row are more
alike than distant ones. Intervals that treat every cell as independent are therefore too narrow. Intervals and comparisons therefore resample mice, not cells:

* **Mouse and group summaries:** resampling whole mice, keeping each mouse's cells together, with a small-sample
  correction that widens the interval by sqrt(k/(k-1)) x t(k-1)/z for k mice (`cluster_bootstrap`).
* **Comparing two groups:** one value per mouse; an exact P value from every way of relabeling the mice, the
  difference with each mouse left out in turn, and a Welch t interval from the mouse values
  (`compare_groups_by_mouse`).
* **One field:** cell count, median and middle half, without an interval (`field_summary`). Neighboring cells along
  a row are alike (next-door correlation about 0.2-0.5 in P1 data), and with 10-25 cells per field none of the
  resampling schemes tested held its stated coverage.

How often 95% intervals contained the true mean in simulation (15 cells per mouse; spread between mice 0.4 um and
between cells 0.7 um). Columns 2, 3 and 5: interval_coverage_simulation_2026-11-12.csv, 2,000 simulated studies per
row; column 4: interval_coverage_correction_2026-11-12.csv, 2,000 per row, separate draws.

| Mice | Cells resampled | Mice resampled | Mice resampled, corrected | t interval on mouse means |
|---|---|---|---|---|
| 2 | 59% | 48% | not run | 94% |
| 3 | 63% | 76% | 94% | 95% |
| 5 | 62% | 83% | 95% | 95% |
| 8 | 62% | 87% | 94% | 94% |
| 12 | 62% | 90% | 94% | 94% |

With unequal cells per mouse (3-18), heavy-tailed differences between mice or skewed cell values, the corrected
interval covered 92-97%; resampling fields and cells within mice as well gave 98-100% (too wide), and the Welch
interval for group differences 96-98% (interval_robustness_simulation_2026-11-12.csv, 800 simulated studies per
scenario, scenarios with 3, 5 and 8 mice, using the plugin's own functions).

## Tests

```
pip install -e ".[testing]"
python -m pytest
```

## License

BSD 3-Clause.
