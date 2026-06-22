"""Phone capture guidance — required and optional views per case.

The five standard protocol views map onto the surface region types defined in
``schemas.py``.  ``missing_views`` and ``coverage_per_region`` are the two
main consumer entry points.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from toothprint.bench.dmc.coverage import coverage_from_point_cloud, synthetic_coverage


@dataclass(frozen=True)
class ViewSpec:
    """Specification for a single capture view in the protocol.

    Attributes:
        view_name:       Short identifier used as a key (e.g. ``"anterior_close"``).
        description:     Human-readable guidance for the photographer.
        required:        Whether the view must be present for a complete capture.
        target_regions:  Region-type strings (matching ``RegionType`` from
            ``schemas.py``) that this view primarily covers.
    """

    view_name: str
    description: str
    required: bool
    target_regions: list[str] = field(default_factory=list)


STANDARD_PROTOCOL: list[ViewSpec] = [
    ViewSpec(
        view_name="anterior_close",
        description=(
            "Close-up frontal view of the anterior teeth. "
            "Lip retractors or a gentle lip pull recommended."
        ),
        required=True,
        target_regions=["anterior_crown", "visible_gingival_margin"],
    ),
    ViewSpec(
        view_name="left_buccal",
        description=(
            "Left-side buccal view showing premolars and molars. "
            "Retract the left cheek gently."
        ),
        required=True,
        target_regions=["buccal_crown", "visible_gingival_margin"],
    ),
    ViewSpec(
        view_name="right_buccal",
        description=(
            "Right-side buccal view showing premolars and molars. "
            "Retract the right cheek gently."
        ),
        required=True,
        target_regions=["buccal_crown", "visible_gingival_margin"],
    ),
    ViewSpec(
        view_name="upper_occlusal",
        description=(
            "Overhead occlusal view of the upper arch. "
            "Tilt the phone or use a dental mirror."
        ),
        required=False,
        target_regions=["occlusal_or_incisal"],
    ),
    ViewSpec(
        view_name="lower_occlusal",
        description=(
            "Chin-up occlusal view of the lower arch. "
            "Tilt the phone or use a dental mirror."
        ),
        required=False,
        target_regions=["occlusal_or_incisal"],
    ),
]


def missing_views(
    captured_view_names: list[str],
    protocol: list[ViewSpec] = STANDARD_PROTOCOL,
) -> list[ViewSpec]:
    """Return required views that are absent from *captured_view_names*.

    Args:
        captured_view_names: Names of views already captured (order irrelevant).
        protocol:            Protocol to check against; defaults to
            ``STANDARD_PROTOCOL``.

    Returns:
        List of ``ViewSpec`` objects for each required view not yet captured.
        Empty list if all required views are present.
    """
    captured = set(captured_view_names)
    return [
        spec for spec in protocol if spec.required and spec.view_name not in captured
    ]


def coverage_per_region(
    view_names: list[str],
    quality_tags_per_view: dict[str, list[str]],
    protocol: list[ViewSpec] = STANDARD_PROTOCOL,
    image_paths_per_view: dict[str, Path] | None = None,
    reconstruction_points: np.ndarray | None = None,
    region_bboxes: dict[str, tuple] | None = None,
) -> dict[str, float]:
    """Estimate coverage fraction for each region type in the protocol.

    For every region type that appears across all ``ViewSpec`` entries, the
    function aggregates ``synthetic_coverage`` scores from each view that
    targets that region, then clamps the sum to [0, 1].

    When *image_paths_per_view* is provided, each view image is analysed with
    :mod:`dentalmapcert.image_quality` and the detected tags are merged into
    *quality_tags_per_view* (caller-supplied explicit tags always take
    precedence over auto-detected ones).

    When both *reconstruction_points* (an ``(N, 3)`` numpy array) and
    *region_bboxes* (a mapping of region_id → 6-tuple bbox) are provided,
    :func:`~dentalmapcert.coverage.coverage_from_point_cloud` is used for
    each region that has a bbox entry, replacing the synthetic heuristic with
    a real reconstruction-based score.

    Args:
        view_names:             Names of views that were captured.
        quality_tags_per_view:  Mapping of view_name -> list of quality tags
            (e.g. ``{"anterior_close": ["glare"]}``) for views that have issues.
            Missing keys are treated as having no quality issues.
        protocol:               Protocol to use; defaults to ``STANDARD_PROTOCOL``.
        image_paths_per_view:   Optional mapping of view_name -> ``Path`` to the
            captured image.  When supplied, :func:`~dentalmapcert.image_quality\
.analyze_view_quality` is called for views that lack explicit quality tags,
            and the detected tags are added automatically.
        reconstruction_points:  Optional ``(N, 3)`` ndarray of reconstructed
            3-D points (in metres or mm — must match *region_bboxes* units).
            When provided alongside *region_bboxes*, enables reconstruction-based
            coverage scoring for each region with a bbox entry.
        region_bboxes:          Optional mapping of region_id ->
            ``(x_min, y_min, z_min, x_max, y_max, z_max)`` bbox.  Required when
            *reconstruction_points* is supplied to enable real coverage scoring.

    Returns:
        Mapping of region-type string -> estimated coverage fraction in [0, 1].
    """
    # Auto-detect quality tags from images when paths are provided.
    if image_paths_per_view:
        from toothprint.bench.dmc.image_quality import (
            analyze_view_quality,
        )  # lazy import

        merged_tags: dict[str, list[str]] = dict(quality_tags_per_view)
        for view_name, img_path in image_paths_per_view.items():
            if view_name not in merged_tags:
                detected = analyze_view_quality(img_path)
                if detected:
                    merged_tags[view_name] = detected
        quality_tags_per_view = merged_tags

    captured = set(view_names)

    # Collect all region types mentioned anywhere in the protocol.
    all_regions: set[str] = set()
    for spec in protocol:
        all_regions.update(spec.target_regions)

    region_coverage: dict[str, float] = {r: 0.0 for r in all_regions}

    # Aggregate per-region: count views that cover each region and collect their tags.
    region_view_counts: dict[str, int] = {r: 0 for r in all_regions}
    region_quality_tags: dict[str, list[str]] = {r: [] for r in all_regions}
    for spec in protocol:
        if spec.view_name not in captured:
            continue
        tags = quality_tags_per_view.get(spec.view_name, [])
        for region in spec.target_regions:
            region_view_counts[region] += 1
            region_quality_tags[region].extend(tags)

    use_reconstruction = (
        reconstruction_points is not None
        and region_bboxes is not None
        and reconstruction_points.shape[0] > 0
    )

    for region in all_regions:
        n_views = region_view_counts[region]

        # Try reconstruction-based coverage when point cloud and bbox are available.
        if use_reconstruction and region in region_bboxes:  # type: ignore[operator]
            bbox = region_bboxes[region]  # type: ignore[index]
            x_min, y_min, z_min, x_max, y_max, z_max = (float(v) for v in bbox)
            pts = reconstruction_points  # type: ignore[union-attr]
            mask = (
                (pts[:, 0] >= x_min)
                & (pts[:, 0] <= x_max)
                & (pts[:, 1] >= y_min)
                & (pts[:, 1] <= y_max)
                & (pts[:, 2] >= z_min)
                & (pts[:, 2] <= z_max)
            )
            pts_in_region = [(float(p[0]), float(p[1]), float(p[2])) for p in pts[mask]]
            score = coverage_from_point_cloud(region, pts_in_region, bbox)
            region_coverage[region] = score.coverage_fraction
            continue

        # Fallback: synthetic heuristic.
        if n_views == 0:
            continue
        score = synthetic_coverage(
            surface_region_id=region,
            n_views=n_views,
            quality_tags=region_quality_tags[region],
        )
        region_coverage[region] = score.coverage_fraction

    return region_coverage
