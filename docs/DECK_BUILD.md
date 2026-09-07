# Presentation build

`ERA_overview_presentation.pdf` is the published presentation and the only
deck artifact this repository carries. It is exported from an intentionally
rasterized PPTX: the four charts are PNG images so their rendered error bars
stay faithful across presentation software, while text, shapes, notes, and the
model family tree remain editable in that intermediate file.

The intermediate PPTX is a build product. It is reproduced from
`deck_source/build.js` by the steps below and is not committed, so the
published PDF is the reviewed artifact.

The deck records the study overview prepared for rc2; rc3 updates software
compatibility and model-loading safety, not those historical measurements.

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

Run these commands from the repository root with Node.js and npm installed.
The commands below use PowerShell. On other shells, run the same tools from
the corresponding directories.

Install the locked presentation dependencies once:

```powershell
npm ci --ignore-scripts --prefix docs/deck_source
```

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
   `.deck-work\ERA_overview_presentation_final.pptx`.

4. Check that rasterized file before exporting it, by passing it to the
   validator:

   ```powershell
   python docs\deck_source\post.py .deck-work\ERA_overview_presentation_final.pptx
   ```

   For the PPTX the validator reads every ZIP member and verifies all XML and
   internal file references. It fails on damaged archives, missing assets, a
   slide count other than nine, or chart XML in the package. It also fails if
   the four images are not attached to slides 5 and 6, or if the final slide
   2, 5, and 7 content is missing. It does not claim that the LibreOffice
   rasterization step is automatic.

5. Export the same presentation to `docs/ERA_overview_presentation.pdf`. Open
   the PDF and check all nine slides, including the charts on slides 5 and 6.

6. Run the validator again with no argument, to check the published PDF:

   ```powershell
   python docs\deck_source\post.py
   ```

   For the PDF it resolves `startxref` and every cross-reference offset back
   to a real object, and fails on a truncated file or a slide count other than
   nine. In both modes it also checks that the matching slide content remains
   in `build.js`.

Re-run steps 3 to 5 whenever the charts or their error bars change; do not
replace the final images by editable charts without updating this process and
the validator. Keep the intermediate PPTX out of the commit: the published PDF
is the reviewed artifact.

## Check the file that Git will publish

`.gitattributes` marks PPTX and PDF files as binary. This exception must stay
after the general text rule for `docs/`; text normalization corrupts these
documents even when some slides remain readable.

After staging the reviewed presentation, validate the staged bytes too:

```powershell
python -c "import pathlib, subprocess, sys; sys.path.insert(0, 'docs/deck_source'); import post; data = subprocess.check_output(['git', 'show', ':docs/ERA_overview_presentation.pdf']); pathlib.Path('.deck-work/staged.pdf').write_bytes(data); post.validate_pdf('.deck-work/staged.pdf'); print('STAGED PDF INTEGRITY PASS')"
```

Run `post.py` again in a fresh checkout of the pushed commit. A local file that
opens correctly is not sufficient evidence that the committed bytes are intact.

## Known issue in the local build tools

The locked `pptxgenjs@4.0.1` dependency uses `image-size@1.2.1`. npm reports two
HIGH entries for that dependency chain. The underlying issues are infinite
loops when parsing crafted ICNS, JXL or HEIF images:
[GHSA-w3rx-r6r6-pgpr](https://github.com/advisories/GHSA-w3rx-r6r6-pgpr) and
[GHSA-5p2g-fcmc-qvqq](https://github.com/advisories/GHSA-5p2g-fcmc-qvqq).

As checked on 2026-09-07, no patched `image-size` release is listed. This build
uses the reviewed local PNG logo; use trusted local assets only. The Node
tooling is not included in the Python package and does not run when someone
opens the presentation. Keep this exception visible until an upstream fix can
be tested. Do not use `npm audit fix --force`: the proposed downgrade changes
the presentation API and is not a verified fix for this build.
