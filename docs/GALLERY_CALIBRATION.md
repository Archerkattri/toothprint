# Gallery-level identity calibration

ToothPrint's older identity certificate calibrated pairwise scores. That is not
the same random variable as the score selected after searching a gallery: with
40 candidates, a single query gets 40 chances to produce an unusually small
distance. `toothprint.identity.gallery` calibrates the minimum distance over the
complete declared gallery for each non-enrolled calibration query.

## Decision contract

```python
from toothprint.identity.gallery import (
    GalleryCalibration,
    decide_gallery_identity,
    evaluate_gallery_calibration,
)

calibration = GalleryCalibration.fit(
    non_enrolled_distance_matrix,  # shape (queries, complete_gallery)
    alpha=0.01,
    site_id="clinic-a",
    created_utc="2026-09-14T00:00:00Z",
    selector_id="complete_gallery_argmin_distance",
)

decision = decide_gallery_identity(
    query_distance_row,
    calibration,
    gallery_labels=gallery_patient_ids,
)
```

Distances are lower-is-better. The threshold is a lower-tail order statistic
and acceptance uses strict `<` at the boundary, so ties do not inflate the
calibration false-accept rate. The decision accepts only one distinct identity;
multiple identities under the threshold produce an explicit identity set and
`abstain`. A top-two ambiguity margin can add a stricter operational gate.

The calibration object binds the site, selector, gallery size, data fingerprint,
method version and creation time. A gallery-size or scoring-pipeline mismatch is
a hard error. `evaluate_gallery_calibration` reports held-out gallery-level
false-accept rate separately from empirical genuine correct-accept, abstention
and wrong-accept rates. Genuine performance is not covered by the false-accept
bound.

This is a research identity-safety primitive, not a clinical or forensic
validation. Gallery growth, scanner changes, registration changes and population
shift require recalibration or a separately justified update. Longitudinal
same-patient validation remains gated by the restricted data described in
`evaluation/DATA_GATE.md`.

## Reproduce the CPU pilot

```powershell
python evaluation/scripts/gallery_calibration_pilot.py
```

The pilot compares the correct complete-gallery calibration with a deliberately
invalid pairwise threshold reused after a 40-way search. It uses synthetic
distances and makes no claim about real patients or external benchmarks.
