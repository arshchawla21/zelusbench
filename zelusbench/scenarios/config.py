"""Scenario configuration — all difficulty knobs."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


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

    # Query depth targeting — force queries to specific depths
    query_target_depth: int | None = None  # exact depth match
    query_min_depth: int | None = None     # at-least-this-deep

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

    @classmethod
    def randomize_except(
        cls,
        rng: random.Random,
        pinned: dict[str, Any],
    ) -> "ScenarioConfig":
        """Build a config with everything randomized except pinned fields.

        Handles interdependencies (e.g. no magnitude_spherical in 2D).
        """
        dim = pinned.get("dim", rng.choice([2, 3]))

        depth_brackets = [(2, 3), (4, 6), (7, 10), (12, 16)]
        lo, hi = rng.choice(depth_brackets)
        min_depth = pinned.get("min_chain_depth", lo)
        max_depth = pinned.get("max_chain_depth", hi)
        if max_depth < min_depth:
            max_depth = min_depth

        dag = pinned.get("dag_structure",
                         rng.choice(list(DAGStructure)))
        dist = pinned.get("distractor_level",
                          rng.choice(list(DistractorLevel)))
        td = pinned.get("transform_density",
                        rng.choice(list(TransformDensity)))

        # Transform types scale with density
        if td == TransformDensity.STATIC:
            tt = []
        elif td in (TransformDensity.LIGHT,):
            tt = ["rotation", "translation"]
        else:
            tt = ["rotation", "translation", "reflection", "scaling"]
        tt = pinned.get("transform_types", tt)

        # Query types — always include POSITION
        all_qt = list(QueryType)
        qt = pinned.get("query_types",
                        [QueryType.POSITION] + rng.sample(
                            [q for q in all_qt if q != QueryType.POSITION],
                            k=rng.randint(0, len(all_qt) - 1)))

        # Point def types — at least 2, filter spherical for 2D
        all_pdt = ["cartesian_offset", "magnitude_direction",
                   "magnitude_polar", "midpoint", "weighted_centroid"]
        if dim == 3:
            all_pdt.append("magnitude_spherical")
        else:
            all_pdt = [t for t in all_pdt if t != "magnitude_spherical"]
        k_pdt = rng.randint(2, len(all_pdt))
        pdt = pinned.get("point_def_types", rng.sample(all_pdt, k_pdt))

        # Coordinate ranges
        coord_choices = [(-5, 5), (-10, 10), (-20, 20)]
        cmin, cmax = rng.choice(coord_choices)
        cmin = pinned.get("coord_min", cmin)
        cmax = pinned.get("coord_max", cmax)

        mag_choices = [(0.5, 3.0), (1.0, 8.0), (2.0, 15.0)]
        mmin, mmax = rng.choice(mag_choices)
        mmin = pinned.get("magnitude_min", mmin)
        mmax = pinned.get("magnitude_max", mmax)

        nq = pinned.get("num_queries", rng.choice([2, 3, 4]))
        ns = pinned.get("num_splits", min(max_depth, 5))

        return cls(
            dim=dim,
            min_chain_depth=min_depth,
            max_chain_depth=max_depth,
            dag_structure=dag,
            distractor_level=dist,
            transform_density=td,
            transform_types=tt,
            query_types=qt,
            num_queries=nq,
            num_splits=ns,
            point_def_types=pdt,
            coord_min=cmin,
            coord_max=cmax,
            magnitude_min=mmin,
            magnitude_max=mmax,
            query_target_depth=pinned.get("query_target_depth"),
            query_min_depth=pinned.get("query_min_depth"),
            seed=pinned.get("seed"),
        )


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
