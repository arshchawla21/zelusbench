"""Scenario configuration — all difficulty knobs."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class QueryType(Enum):
    POSITION = auto()      # "What is the position of X?"
    DISTANCE = auto()      # "What is the distance from X to Y?"
    BOOLEAN = auto()        # "Is X closer to Y or Z?"


@dataclass
class ScenarioConfig:
    """All parameters controlling scenario generation."""

    # Spatial
    dim: int = 3

    # Chain depth (sustained attention)
    min_chain_depth: int = 3
    max_chain_depth: int = 7

    # Topology — probability of extending a leaf vs branching from any point
    leaf_bias: float = 0.5  # 0.0=bushy/random, 1.0=always extend leaves (linear)

    # Density — total points to generate (more = more noise)
    num_points: int = 8

    # Transforms — probability each step triggers a transform
    transform_prob: float = 0.1  # 0.0=static, higher=more transforms
    transform_types: list[str] = field(default_factory=lambda: ["rotation", "translation"])

    # Query configuration
    query_types: list[QueryType] = field(default_factory=lambda: [QueryType.POSITION, QueryType.DISTANCE])
    num_queries: int = 3

    # Point definition types to use
    point_def_types: list[str] = field(default_factory=lambda: [
        "cartesian_offset", "magnitude_direction",
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

        leaf_bias = pinned.get("leaf_bias",
                               rng.choice([0.0, 0.25, 0.5, 0.75, 1.0]))

        # num_points scales with actual depth (not the random bracket)
        if max_depth <= 3:
            pt_lo, pt_hi = 3, 6
        elif max_depth <= 6:
            pt_lo, pt_hi = 5, 10
        elif max_depth <= 10:
            pt_lo, pt_hi = 10, 20
        else:
            pt_lo, pt_hi = 15, 30
        num_points = pinned.get("num_points", rng.randint(pt_lo, pt_hi))

        transform_prob = pinned.get("transform_prob",
                                    rng.choice([0.0, 0.05, 0.1, 0.15, 0.2]))

        # Transform types scale with probability
        if transform_prob == 0.0:
            tt = []
        elif transform_prob <= 0.1:
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

        # Point def types — at least 2, filter by dimension
        # magnitude_polar (single angle) is 2D only
        # magnitude_spherical (theta + phi) is 3D only
        all_pdt = ["cartesian_offset", "magnitude_direction",
                   "midpoint", "weighted_centroid"]
        if dim == 2:
            all_pdt.append("magnitude_polar")
        if dim >= 3:
            all_pdt.append("magnitude_spherical")
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

        return cls(
            dim=dim,
            min_chain_depth=min_depth,
            max_chain_depth=max_depth,
            leaf_bias=leaf_bias,
            num_points=num_points,
            transform_prob=transform_prob,
            transform_types=tt,
            query_types=qt,
            num_queries=nq,
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
        leaf_bias=1.0, num_points=4, transform_prob=0.0,
        num_queries=2,
        point_def_types=["cartesian_offset", "magnitude_direction",
                         "magnitude_polar", "midpoint", "weighted_centroid"],
    )
    defaults.update(overrides)
    return ScenarioConfig(**defaults)


def medium_config(**overrides) -> ScenarioConfig:
    defaults = dict(
        dim=3, min_chain_depth=4, max_chain_depth=7,
        leaf_bias=0.7, num_points=10, transform_prob=0.1,
        num_queries=3,
    )
    defaults.update(overrides)
    return ScenarioConfig(**defaults)


def hard_config(**overrides) -> ScenarioConfig:
    defaults = dict(
        dim=3, min_chain_depth=7, max_chain_depth=12,
        leaf_bias=0.3, num_points=20, transform_prob=0.2,
        num_queries=5,
        transform_types=["rotation", "translation", "reflection", "scaling"],
    )
    defaults.update(overrides)
    return ScenarioConfig(**defaults)
