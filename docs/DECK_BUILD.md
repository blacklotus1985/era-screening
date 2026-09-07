# Presentation build

`ERA_overview_presentation.pdf` is the published presentation and the only
deck artifact this repository carries. It is exported from an intentionally
rasterized PPTX: the four charts are PNG images so their rendered error bars
stay faithful across presentation software, while text, shapes, notes, and the
model family tree remain editable in that intermediate file.

The intermediate PPTX is a build product. It is reproduced from
`deck_source/build.js` by the steps below and is not committed, so the
published PDF is the reviewed artifact.

Regenerating the deck is **not automatic**. Step 3 converts the four charts to
images by hand in LibreOffice Impress, and step 5 exports and reviews the PDF
by hand. Nothing in this repository performs those two steps, and the checks
in step 7 do not claim otherwise: they confirm that an approved file reached
the repository unchanged, not that it was built correctly.

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
   checker:

   ```powershell
   python docs\deck_source\post.py .deck-work\ERA_overview_presentation_final.pptx
   ```

   For the PPTX it reads every ZIP member and verifies all XML and internal
   file references. It fails on damaged archives, missing assets, a slide
   count other than nine, or chart XML in the package. It also fails if the
   four images are not attached to slides 5 and 6, or if the final slide 2, 5,
   and 7 content is missing.

5. Export the same presentation to `docs/ERA_overview_presentation.pdf`. Open
   the PDF and check all nine slides, including the charts on slides 5 and 6.
   This is the reviewing step: the checks below confirm that the approved file
   reaches the repository unchanged, not that its contents are correct.

6. Record the approved bytes in `deck_source/published_deck.sha256`:

   ```powershell
   $hash = (Get-FileHash docs\ERA_overview_presentation.pdf -Algorithm SHA256).Hash.ToLower()
   "$hash  ERA_overview_presentation.pdf" | Out-File -Encoding utf8 docs\deck_source\published_deck.sha256
   ```

7. Run the published-deck checks:

   ```powershell
   python docs\deck_source\post.py
   ```

Re-run steps 3 to 6 whenever the charts or their error bars change; do not
replace the final images by editable charts without updating this process and
the checks. Keep the intermediate PPTX out of the commit: the published PDF is
the reviewed artifact.

## What the published-deck checks cover

Two checks run against `docs/ERA_overview_presentation.pdf`, because neither
covers what the other does.

`check_published_bytes` compares the file with the SHA-256 recorded in
`deck_source/published_deck.sha256`. This is what would notice Git text
normalization. `.gitattributes` marks PDF files as binary to prevent it, and
that exception must stay after the general text rule for `docs/`; the digest
is what proves the protection held.

`check_deck_content` opens the deck with `pypdf` and confirms that it opens,
carries nine pages, and still shows the expected text on slides 2, 5, and 7.
It also checks that the matching slide text remains in `build.js`.

This second check reads the document; it does not validate the PDF format, and
it is not a substitute for the digest. `pypdf` reconstructs a damaged
cross-reference table while reading, so a normalized deck still opens, still
reports nine pages, and still shows readable text on slide 2. A test in
`tests/test_presentation_package.py` records that behaviour so the two checks
are not later collapsed into one.

`pypdf` is declared in the `dev` extra. ERA never opens a PDF at runtime, so
it is not needed to use the package.

## Check the file that Git will publish

A working copy that opens correctly is not evidence about what Git will
publish. After staging the reviewed presentation, compare the staged bytes
against the recorded digest:

```powershell
python docs\deck_source\post.py --staged
```

CI runs `post.py` on a fresh checkout of the pushed commit, which repeats the
digest comparison against the bytes Git actually delivers.

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
