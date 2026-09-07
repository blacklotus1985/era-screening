# Presentation build

`ERA_overview_presentation.pptx` is the final, intentionally rasterized
presentation. The four charts are PNG images so their rendered error bars stay
faithful across presentation software. Text, shapes, notes, and the model
family tree remain editable.

## Provenance

- Slide 5 charts 1 and 2 use the three-seed means for full-vocabulary `B` and
  `Delta SI` for Pythia-70M and OPT-350M. Their error bars are the sample
  standard deviations `0.004386929079948761`, `0.019552720258314248`,
  `0.042354428361323686`, and `0.022765224286368756`.
- Slide 5 chart 3 uses the between-group and within-group shares of the mean
  target-word change: `17.2/82.8` for Pythia-70M and `52.9/47.1` for OPT-350M.
- Slide 6 uses the 11-model `Delta SI` panel. Its values and per-model sample
  standard deviations are recorded in `prepare_charts.py`.
- The numerical sources are `docs/RESULTS.md` and
  `results/reference_metrics/v2_balanced_r2/public_summary.json`.
- The editable layout and slide text are in `deck_source/build.js`.

## Build path

Run these commands from the repository root.

1. Generate the editable chart package from the JavaScript source. The
   generator writes its output in the current directory, so use a temporary
   working directory:

   ```powershell
   New-Item -ItemType Directory -Force .deck-work
  Copy-Item docs\deck_source\era_logo.png .deck-work\era_logo.png
   Push-Location .deck-work
   node ..\docs\deck_source\build.js
   Pop-Location
   ```

2. Add the recorded error bars and the highlighted Pythia-70M point to the
   editable chart package:

   ```powershell
   python docs\deck_source\prepare_charts.py `
     .deck-work\ERA_overview_presentation.pptx `
     .deck-work\ERA_overview_presentation_prepared.pptx
   ```

3. Manually open the prepared file in LibreOffice Impress. Convert only the
   four charts to images, preserving their rendered error bars: three charts
   on slide 5 and one chart on slide 6. Leave all text, shapes, notes, and the
   model-family tree editable. Save the resulting presentation as
   `docs/ERA_overview_presentation.pptx`.

4. Run the final-package validator:

   ```powershell
   python docs\deck_source\post.py
   ```

The validator fails if chart XML is present in the final package, if the four
images are not attached to slides 5 and 6, or if the final slide 2, 5, and 7
content is missing. It also checks that the matching content remains in
`build.js`. It does not claim that the LibreOffice rasterization step is
automatic.

The committed PPTX is the reviewed final artifact. Re-run step 3 whenever the
charts or their error bars change; do not replace the final images by editable
charts without updating this process and the validator.
