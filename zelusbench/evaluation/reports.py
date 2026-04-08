"""Generate diagnostic profiles and aggregate reports."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from .scorer import QueryScore


@dataclass
class DiagnosticProfile:
    """Full diagnostic breakdown for a benchmark run."""

    # Raw scores
    all_scores: list[dict] = field(default_factory=list)

    # 1. Attention Decay Curve: accuracy vs chain depth
    accuracy_by_depth: dict[int, float] = field(default_factory=dict)

    # 2. Density Robustness: accuracy by number of points
    accuracy_by_num_points: dict[int, float] = field(default_factory=dict)

    # 3. Transform Adaptation: accuracy by transform count
    accuracy_by_transform_count: dict[int, float] = field(default_factory=dict)

    # 4. Positional Bias: accuracy by query position in sequence
    accuracy_by_query_position: dict[int, float] = field(default_factory=dict)

    # 5. Topology Sensitivity: accuracy by leaf bias
    accuracy_by_leaf_bias: dict[float, float] = field(default_factory=dict)

    # 6. Dimensionality: accuracy by spatial dimension
    accuracy_by_dim: dict[int, float] = field(default_factory=dict)

    # Overall score
    overall_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "accuracy_by_depth": self.accuracy_by_depth,
            "accuracy_by_num_points": self.accuracy_by_num_points,
            "accuracy_by_transform_count": self.accuracy_by_transform_count,
            "accuracy_by_query_position": self.accuracy_by_query_position,
            "accuracy_by_leaf_bias": self.accuracy_by_leaf_bias,
            "accuracy_by_dim": self.accuracy_by_dim,
            "overall_score": self.overall_score,
            "num_queries": len(self.all_scores),
        }


def _mean_score(scores: list[dict]) -> float:
    if not scores:
        return 0.0
    return sum(s["score"] for s in scores) / len(scores)


def _weighted_harmonic_mean(values: list[float], min_val: float = 1e-10) -> float:
    """Weighted harmonic mean (equal weights) for overall score."""
    values = [max(v, min_val) for v in values if v is not None]
    if not values:
        return 0.0
    return len(values) / sum(1.0 / v for v in values)


def build_diagnostic_profile(
    scores: list[QueryScore],
    scenario_metadata: list[dict],
) -> DiagnosticProfile:
    """Build a full diagnostic profile from scored queries.

    Args:
        scores: List of QueryScore objects from all scenarios.
        scenario_metadata: List of dicts with keys like 'dag_structure',
                          'distractor_ratio', 'num_transforms', 'dim'.
                          One per scenario, indexed by scenario order.
    """
    profile = DiagnosticProfile()

    # Build a flat list of score dicts with metadata
    score_dicts = []
    for score in scores:
        sd = score.to_dict()
        score_dicts.append(sd)
    profile.all_scores = score_dicts

    # 1. Accuracy by chain depth
    by_depth: dict[int, list[dict]] = defaultdict(list)
    for sd in score_dicts:
        by_depth[sd["chain_depth"]].append(sd)
    profile.accuracy_by_depth = {d: _mean_score(ss) for d, ss in sorted(by_depth.items())}

    # 2. Accuracy by num_points (from scenario metadata)
    by_num_points: dict[int, list[dict]] = defaultdict(list)
    for sd, meta in _zip_scores_metadata(score_dicts, scenario_metadata):
        np_val = meta.get("num_points", 0)
        by_num_points[np_val].append(sd)
    profile.accuracy_by_num_points = {
        d: _mean_score(ss) for d, ss in sorted(by_num_points.items())
    }

    # 3. Accuracy by transform count
    by_transforms: dict[int, list[dict]] = defaultdict(list)
    for sd, meta in _zip_scores_metadata(score_dicts, scenario_metadata):
        count = meta.get("num_transforms", 0)
        by_transforms[count].append(sd)
    profile.accuracy_by_transform_count = {
        d: _mean_score(ss) for d, ss in sorted(by_transforms.items())
    }

    # 4. Accuracy by query position
    by_position: dict[int, list[dict]] = defaultdict(list)
    for sd in score_dicts:
        by_position[sd["query_index"]].append(sd)
    profile.accuracy_by_query_position = {
        d: _mean_score(ss) for d, ss in sorted(by_position.items())
    }

    # 5. Accuracy by leaf bias
    by_lb: dict[float, list[dict]] = defaultdict(list)
    for sd, meta in _zip_scores_metadata(score_dicts, scenario_metadata):
        lb = meta.get("leaf_bias", 0.5)
        by_lb[round(lb, 2)].append(sd)
    profile.accuracy_by_leaf_bias = {d: _mean_score(ss) for d, ss in sorted(by_lb.items())}

    # 6. Accuracy by dimension
    by_dim: dict[int, list[dict]] = defaultdict(list)
    for sd, meta in _zip_scores_metadata(score_dicts, scenario_metadata):
        dim = meta.get("dim", 3)
        by_dim[dim].append(sd)
    profile.accuracy_by_dim = {d: _mean_score(ss) for d, ss in sorted(by_dim.items())}

    # Overall score: weighted harmonic mean of dimension averages
    dimension_scores = [
        np.mean(list(profile.accuracy_by_depth.values())) if profile.accuracy_by_depth else 0,
        np.mean(list(profile.accuracy_by_num_points.values())) if profile.accuracy_by_num_points else 0,
        np.mean(list(profile.accuracy_by_transform_count.values())) if profile.accuracy_by_transform_count else 0,
        np.mean(list(profile.accuracy_by_query_position.values())) if profile.accuracy_by_query_position else 0,
        np.mean(list(profile.accuracy_by_leaf_bias.values())) if profile.accuracy_by_leaf_bias else 0,
        np.mean(list(profile.accuracy_by_dim.values())) if profile.accuracy_by_dim else 0,
    ]
    profile.overall_score = float(_weighted_harmonic_mean(dimension_scores))

    return profile


def _zip_scores_metadata(
    score_dicts: list[dict],
    scenario_metadata: list[dict],
) -> list[tuple[dict, dict]]:
    """Pair each score dict with its scenario metadata.

    If metadata is shorter, cycles through it. This handles the common case
    where each scenario produces multiple queries.
    """
    if not scenario_metadata:
        return [(sd, {}) for sd in score_dicts]

    result = []
    meta_idx = 0
    queries_per_scenario = max(1, len(score_dicts) // len(scenario_metadata))
    for i, sd in enumerate(score_dicts):
        meta_idx = min(i // queries_per_scenario, len(scenario_metadata) - 1)
        result.append((sd, scenario_metadata[meta_idx]))
    return result
