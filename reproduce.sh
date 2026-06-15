#!/usr/bin/env bash
# Reproduce the AlphaEarth-vs-geography solar pilot, scripts 01 -> 15.
#
# Prerequisites:
#   python -m venv .venv && source .venv/bin/activate
#   pip install -r requirements.txt
#   earthengine authenticate            # for the Earth Engine steps
#   export EE_PROJECT=your-ee-project-id
#
# Steps tagged [NET] need internet (DWD download); [EE] need Earth Engine.
# To ONLY re-verify the result against committed intermediates, skip to step 15.
set -euo pipefail

PY="${PY:-python}"
: "${EE_PROJECT:?set EE_PROJECT to your Earth Engine GCP project id}"

run() { echo; echo "=== $* ==="; "$PY" "$@"; }

run scripts/01_download_dwd.py                                   # [NET]
run scripts/02_add_clear_sky.py
run scripts/03_extract_alphaearth.py --project "$EE_PROJECT"     # [EE]
run scripts/04_sanity_plots.py
run scripts/05_layer2_regression.py
run scripts/06_layer3_residual_hourly.py
run scripts/07_layer2_spatial_block.py
run scripts/08_layer2_spatial_block_perfold.py
run scripts/09_pool_relaxed_stations.py                          # [NET]
run scripts/10_extract_added_embeddings.py --project "$EE_PROJECT"  # [EE]
run scripts/11_pool_retest.py
run scripts/12_geo_easy_diagnostic.py
run scripts/13_power_analysis.py
run scripts/14_cluster_diagnostic.py                             # [EE] (SRTM)
run scripts/15_verify_all.py                                     # adversarial re-derivation

echo; echo "All steps complete. See VERIFICATION.md and spatial_holdout/FINDINGS.md."
