# Sample labels for reviewers

A small, representative set covering each verdict path. Upload one at
**https://treasury.liam.cool** (or your local instance), enter the application
values below, and hit **Verify**. Each label is shown with a labelled,
colour-coded box per region, so the result is auditable at a glance.

> Tip: you can also leave the application fields blank — the tool still extracts
> every field and checks the government warning (which has no application input).

| File | Scenario | Application values to enter | Expected |
|---|---|---|---|
| `L006_gin_compliant.png` | Fully compliant | Brand `Red Barn Distillery` · Class `London Dry Gin` · ABV `47` · Net `750 mL` | **pass** |
| `L000_gin_abv_mismatch.png` | ABV mismatch (label says 43%, app says 44) | Brand `Wandering Provisions` · Class `London Dry Gin` · ABV `44` · Net `1 L` | **fail** (ABV) |
| `L002_rum_warning_missing_clause.png` | Government warning missing clause (2) | Brand `Black Fox Barrel Co.` · Class `Aged Caribbean Rum` · ABV `41` · Net `355 mL` | **fail** (warning) |
| `L024_tequila_warning_title_case.png` | "Government Warning" not in all-caps (Jenny's reject) | Brand `Copper House` · Class `Tequila Reposado` · ABV `38` · Net `375 mL` | **fail** (warning caps) |
| `L013_scotch_brand_variant.png` | Brand casing variant — `Iron Crest Cellars` vs `IRON CREST CELLARS` (Dave's case) | Brand `Iron Crest Cellars` · Class `Single Malt Scotch Whisky` · ABV `44` · Net `375 mL` | **needs_review** |

These are from the pixel-exact "hybrid" test set (background art generated with an
image model, then field + warning text composited with Pillow so the ground truth
is precise — see `test_images/generate_labels_hybrid.py`). The full ~145-label set
is reproducible via that script; only this handful is committed (the rest are large
and gitignored).
