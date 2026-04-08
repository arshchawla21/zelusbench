"""Scenario configuration — all difficulty knobs."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto


class DAGStructure(Enum):
    LINEAR = auto()
    BRANCHING = auto()
    MERGING = auto()
    DIAMOND = auto()


class QueryType(Enum):
    POSITION = auto()      # "What is the position of X?"
    DISTANCE = auto()      # "What is the distance from X to Y?"
    BOOLEAN = auto()        # "Is X closer to Y or Z?"


class DistractorLevel(Enum):
    CLEAN = 0       # 0:1
    LOW = 1         # 1:1
    HIGH = 3        # 3:1
    EXTREME = 10    # 10:1


class TransformDensity(Enum):
    STATIC = 0
    LIGHT = 1       # 1-2
    HEAVY = 3       # 3-5
    EXTREME = 6     # 6+


@dataclass
class ScenarioConfig:
    """All parameters controlling scenario generation."""

    # Spatial
    dim: int = 3

    # Chain depth (sustained attention)
    min_chain_depth: int = 3
    max_chain_depth: int = 7

    # DAG topology
    dag_structure: DAGStructure = DAGStructure.LINEAR

    # Distractors (selective attention)
    distractor_level: DistractorLevel = DistractorLevel.LOW

    # Transforms (attention updating)
    transform_density: TransformDensity = TransformDensity.LIGHT
    transform_types: list[str] = field(default_factory=lambda: ["rotation", "translation"])

    # Query configuration
    query_types: list[QueryType] = field(default_factory=lambda: [QueryType.POSITION, QueryType.DISTANCE])
    num_queries: int = 3
    num_splits: int = 3  # number of statement blocks between queries

    # Point definition types to use
    point_def_types: list[str] = field(default_factory=lambda: [
        "cartesian_offset", "magnitude_direction", "magnitude_polar",
        "midpoint", "weighted_centroid",
    ])

    # Coordinate ranges
    coord_min: float = -10.0
    coord_max: float = 10.0
    magnitude_min: float = 1.0
    magnitude_max: float = 8.0

    # Seed for reproducibility
    seed: int | None = None

    @property
    def distractor_ratio(self) -> int:
        return self.distractor_level.value

    @property
    def num_transforms(self) -> int:
        match self.transform_density:
            case TransformDensity.STATIC:
                return 0
            case TransformDensity.LIGHT:
                return 2
            case TransformDensity.HEAVY:
                return 4
            case TransformDensity.EXTREME:
                return 7


# --- Presets ---

def easy_config(**overrides) -> ScenarioConfig:
    defaults = dict(
        dim=2, min_chain_depth=2, max_chain_depth=3,
        dag_structure=DAGStructure.LINEAR,
        distractor_level=DistractorLevel.CLEAN,
        transform_density=TransformDensity.STATIC,
        num_queries=2, num_splits=2,
    )
    defaults.update(overrides)
    return ScenarioConfig(**defaults)


def medium_config(**overrides) -> ScenarioConfig:
    defaults = dict(
        dim=3, min_chain_depth=4, max_chain_depth=7,
        dag_structure=DAGStructure.BRANCHING,
        distractor_level=DistractorLevel.LOW,
        transform_density=TransformDensity.LIGHT,
        num_queries=3, num_splits=3,
    )
    defaults.update(overrides)
    return ScenarioConfig(**defaults)


def hard_config(**overrides) -> ScenarioConfig:
    defaults = dict(
        dim=3, min_chain_depth=7, max_chain_depth=12,
        dag_structure=DAGStructure.DIAMOND,
        distractor_level=DistractorLevel.HIGH,
        transform_density=TransformDensity.HEAVY,
        num_queries=5, num_splits=4,
        transform_types=["rotation", "translation", "reflection", "scaling"],
    )
    defaults.update(overrides)
    return ScenarioConfig(**defaults)
