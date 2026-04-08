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

    # 2. Distractor Robustness: accuracy at each distractor level
    accuracy_by_distractor_ratio: dict[int, float] = field(default_factory=dict)

    # 3. Transform Adaptation: accuracy by transform count
    accuracy_by_transform_count: dict[int, float] = field(default_factory=dict)

    # 4. Positional Bias: accuracy by query position in sequence
    accuracy_by_query_position: dict[int, float] = field(default_factory=dict)

    # 5. Topology Sensitivity: accuracy by DAG structure
    accuracy_by_dag_structure: dict[str, float] = field(default_factory=dict)

    # 6. Dimensionality: accuracy by spatial dimension
    accuracy_by_dim: dict[int, float] = field(default_factory=dict)

    # Overall score
    overall_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "accuracy_by_depth": self.accuracy_by_depth,
            "accuracy_by_distractor_ratio": self.accuracy_by_distractor_ratio,
            "accuracy_by_transform_count": self.accuracy_by_transform_count,
            "accuracy_by_query_position": self.accuracy_by_query_position,
            "accuracy_by_dag_structure": self.accuracy_by_dag_structure,
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

    # 2. Accuracy by distractor ratio (from scenario metadata)
    by_distractor: dict[int, list[dict]] = defaultdict(list)
    for sd, meta in _zip_scores_metadata(score_dicts, scenario_metadata):
        ratio = meta.get("distractor_ratio", 0)
        by_distractor[ratio].append(sd)
    profile.accuracy_by_distractor_ratio = {
        d: _mean_score(ss) for d, ss in sorted(by_distractor.items())
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

    # 5. Accuracy by DAG structure
    by_dag: dict[str, list[dict]] = defaultdict(list)
    for sd, meta in _zip_scores_metadata(score_dicts, scenario_metadata):
        structure = meta.get("dag_structure", "UNKNOWN")
        by_dag[structure].append(sd)
    profile.accuracy_by_dag_structure = {d: _mean_score(ss) for d, ss in sorted(by_dag.items())}

    # 6. Accuracy by dimension
    by_dim: dict[int, list[dict]] = defaultdict(list)
    for sd, meta in _zip_scores_metadata(score_dicts, scenario_metadata):
        dim = meta.get("dim", 3)
        by_dim[dim].append(sd)
    profile.accuracy_by_dim = {d: _mean_score(ss) for d, ss in sorted(by_dim.items())}

    # Overall score: weighted harmonic mean of dimension averages
    dimension_scores = [
        np.mean(list(profile.accuracy_by_depth.values())) if profile.accuracy_by_depth else 0,
        np.mean(list(profile.accuracy_by_distractor_ratio.values())) if profile.accuracy_by_distractor_ratio else 0,
        np.mean(list(profile.accuracy_by_transform_count.values())) if profile.accuracy_by_transform_count else 0,
        np.mean(list(profile.accuracy_by_query_position.values())) if profile.accuracy_by_query_position else 0,
        np.mean(list(profile.accuracy_by_dag_structure.values())) if profile.accuracy_by_dag_structure else 0,
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
