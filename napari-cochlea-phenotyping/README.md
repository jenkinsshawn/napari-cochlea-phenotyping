# Harp

A napari plugin that measures confocal images of the cochlea: inner and outer hair cells and their rows, the order and
straightness of the outer hair cell rows, stereocilia bundles (size, shape, height, texture, and inner hair cell bundles
in 3D), spacing and direction of the cells, nucleus size and chromatin, and hair cell identity from Calb2, Ocm or
PMCA2. It compares groups with one value per mouse, with and without adjustment for imaging depth and tissue slope, and
writes every measure to CSV tables and a Word report.

## Install (Windows)

Download the repository (Code > Download ZIP) and unzip it. In the unzipped folder, type `powershell` in the File
Explorer address bar and run:

```
powershell -ExecutionPolicy Bypass -File .\install_cochlea_phenotyping.ps1
```

No administrator rights are needed. The installer sets up everything, including graphics card support when an NVIDIA
card is present, checks that the trained models load, runs the tests, and puts a **Harp** shortcut on
the Desktop. A step-by-step guide is in `Harp_user_guide.docx`.

## Use

1. **Standardize.** Add a folder of .nd2 images. Check the stain in each channel, the group of each mouse (or load a
   table with columns `mouse, group`), the control group and the bundle direction. Mark folded or torn fields with
   **Leave out of comparisons**. Press **Confirm**.
2. **Cells.** Optional quick look at the cell surface of one image.
3. **Measure.** **Measure all images** runs every step in the background; a stopped run continues where it stopped.
   About 3-4 hours for 30 images on a graphics card.
4. **Results.** Agreement with hand references, group comparisons, the result tables, and the measured cells drawn on
   the selected image. **Redo statistics with a genotype table...** recomputes the statistics with other groups
   without measuring again; **Leave out least reliable bundles (%)** repeats them without the least certain bundles.
   **Open the results report** opens the Word report.

Results are written to `cochlea_phenotyping_results` next to the images.

## Statistics

Each measure is summarized per field, then per mouse; groups are compared on mouse values (difference with a Welch 95%
interval, exact permutation P over all relabelings of the mice, three-group F), with false discovery rate adjustment
within each family of measures. Where a measure follows the local depth or slope of the tissue surface within fields
(P below 0.01), the tests are repeated with that effect removed.

## Graphics card

An NVIDIA card is used when available, otherwise the processor. Set `COCHLEA_PHENOTYPING_DEVICE=cpu` to force the
processor.

## Tests

```
python -m pytest tests
```

## License

BSD 3-Clause.
