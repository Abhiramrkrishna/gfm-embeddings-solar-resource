# Preregistration

Do geospatial-foundation-model embeddings encode solar resource beyond geographic
coordinates?

This document records the confirmatory hypothesis, primary target, estimator, test,
and decision rule that were fixed in advance and that governed the executed pilot.
Everything else carried out in this repository (heterogeneity analysis, power
analysis, diagnostics) is exploratory and is labelled as such in the findings log;
it is not reported as confirmatory.

## Design principle

One confirmatory hypothesis, one primary target, one primary test, and one decision
rule, fixed before touching the held-out data. A confirmatory result is reported
against that rule whatever it says.

## Primary hypothesis (H1)

Across ground stations evaluated under spatial-block leave-stations-out
cross-validation, AlphaEarth embeddings predict the detrended mean clear-sky index
(the residual after removing an OLS latitude/longitude/elevation trend) with lower
error than a shuffled-embedding null.

- **Primary model:** AlphaEarth (frozen 64-dimensional annual embedding).
- **Primary target:** detrended mean clear-sky index (kt_cs), with the geographic
  detrend refit on training stations only, inside each fold (no leakage).
- **Primary estimator:** Ridge regression with generalized-cross-validation alpha
  selection. (Justified empirically: gradient-boosted trees overfit and PLS is
  over-parameterized at this n and d.)
- **Cross-validation:** spatial-block leave-stations-out. Stations are clustered into
  contiguous geographic blocks; each block is held out in turn so test stations have
  no nearby training stations.
- **Primary test:** one-sided Wilcoxon signed-rank on per-station |error|, embedding
  versus the shuffled-embedding null.

## Decision rule (fixed in advance)

H1 is supported if and only if both:

1. the per-station embedding-beats-shuffle rate exceeds 60%, AND
2. the Wilcoxon p < 0.01 (stricter than 0.05 to account for pooling).

- **GO:** both conditions met.
- **KILL:** effect below the floor or not significant.

**Secondary confirmatory check.** The embedding must also beat the geographic
baseline (not only the shuffled null) on the same test, at p < 0.05. Beating the
shuffle shows the embedding contains information; beating geography shows it adds
information beyond coordinates. Both are required for the claim that matters.

## Pre-committed contingency

If the effect is too small to detect at an achievable sample size — established by a
power analysis on the pilot effect size — the work is reported as a limitation study
(a calibrated negative result) rather than expanded into a larger data collection.
This contingency is fixed in advance so that a negative outcome is not reframed
after the fact.

## Status

Pre-registration of the executed pilot. The larger multi-network, multi-model
expansion that this design anticipated was not carried out; this repository is the
pilot and its verification.
