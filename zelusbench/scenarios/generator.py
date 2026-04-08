"""Scenario generator — generative step-by-step construction."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..geometry.space import Space
from ..geometry.point import (
    PointDefinition, PointDefType,
    cartesian_offset, magnitude_direction, magnitude_polar,
    magnitude_spherical, midpoint, weighted_centroid, projection,
)
from ..geometry.transforms import (
    Transform, TransformType,
    rotation, translation, reflection, scaling,
)
from ..geometry.vectors import distance

from .config import ScenarioConfig, QueryType
from .templates import (
    render_system_prompt, render_point_definition, render_transform,
    render_query_position, render_query_distance, render_query_boolean,
)


@dataclass
class Query:
    query_id: str
    query_type: QueryType
    target_points: list[str]  # points involved in the query
    ground_truth: Any = None  # resolved after scenario is built
    chain_depth: int = 0
    query_index: int = 0

    def to_dict(self) -> dict:
        truth = self.ground_truth
        if isinstance(truth, np.ndarray):
            truth = truth.tolist()
        return {
            "query_id": self.query_id,
            "query_type": self.query_type.name,
            "target_points": self.target_points,
            "ground_truth": truth,
            "chain_depth": self.chain_depth,
            "query_index": self.query_index,
        }


@dataclass
class Scenario:
    scenario_id: str
    config: ScenarioConfig
    prompt: str
    queries: list[Query]
    space_snapshot: dict  # serialized Space after all operations

    def to_dict(self) -> dict:
        return {
            "scenario_id": self.scenario_id,
            "dim": self.config.dim,
            "prompt": self.prompt,
            "queries": [q.to_dict() for q in self.queries],
            "space": self.space_snapshot,
            "metadata": {
                "leaf_bias": self.config.leaf_bias,
                "num_points": self.config.num_points,
                "transform_prob": self.config.transform_prob,
                "dim": self.config.dim,
            },
        }


class ScenarioGenerator:
    """Generates geometric scenarios via a step-by-step generative process."""

    def __init__(self, config: ScenarioConfig):
        self.config = config
        self.rng = random.Random(config.seed)
        self._name_counter = 0

    def _next_name(self) -> str:
        """Generate next point name: A, B, ..., Z (skip O), then A1, B1, ..."""
        while True:
            self._name_counter += 1
            idx = self._name_counter - 1
            if idx < 26:
                name = chr(ord('A') + idx)
            else:
                name = chr(ord('A') + idx % 26) + str(idx // 26)
            if not name.startswith('O'):
                return name

    # ------------------------------------------------------------------
    # Core generative loop
    # ------------------------------------------------------------------

    def generate(self, scenario_id: str) -> Scenario:
        """Generate a scenario by stepping through point/transform/query actions."""
        cfg = self.config
        space = Space(dim=cfg.dim)
        lines: list[str] = []

        # Header
        lines.append(render_system_prompt(cfg.dim))
        lines.append("")
        lines.append("---")
        lines.append("")

        # Budget
        num_points = max(cfg.num_points, cfg.min_chain_depth)
        points_placed = 0
        max_depth = 0

        final_queries: list[Query] = []
        used_targets: set[str] = set()
        query_idx = 0
        queries_remaining = cfg.num_queries

        # Spread queries evenly: emit after every query_interval points
        query_interval = max(1, num_points // (queries_remaining + 1))

        # Phase 1: Build until min_chain_depth is reached
        # Use configured leaf_bias when there's slack, force linear only when
        # remaining budget is tight (must extend deepest chain to hit target)
        while points_placed < num_points and max_depth < cfg.min_chain_depth:
            depth_deficit = cfg.min_chain_depth - max_depth
            points_left = num_points - points_placed
            must_extend = depth_deficit >= points_left  # no slack, must go linear
            if must_extend:
                # Extend from the deepest point specifically
                name, defn = self._gen_point_from_deepest(space)
            else:
                name, defn = self._gen_point(space, leaf_bias=cfg.leaf_bias)
            lines.append(render_point_definition(name, defn))
            points_placed += 1
            max_depth = max(max_depth, space.chain_depth(name))

        # Phase 2: Generative loop for remaining points
        while points_placed < num_points:
            # Roll for transform
            pts = space.non_origin_points()
            if (cfg.transform_prob > 0 and len(pts) >= 2
                    and cfg.transform_types
                    and self.rng.random() < cfg.transform_prob):
                t = self._plan_single_transform(pts)
                t.apply(space)
                lines.append(render_transform(t))
                continue  # transform doesn't consume a point slot

            # Add a point
            # If max_chain_depth reached, bias toward branching (low leaf_bias)
            effective_lb = cfg.leaf_bias
            if max_depth >= cfg.max_chain_depth:
                effective_lb = min(effective_lb, 0.2)

            name, defn = self._gen_point(space, leaf_bias=effective_lb)
            lines.append(render_point_definition(name, defn))
            points_placed += 1
            max_depth = max(max_depth, space.chain_depth(name))

            # Interleave query if interval hit and valid target exists
            if (queries_remaining > 0
                    and points_placed % query_interval == 0
                    and len(space.non_origin_points()) >= 2
                    and self._has_valid_query_target(space)):
                lines.append("")
                q = self._plan_single_query(
                    space, space.non_origin_points(), used_targets, query_idx,
                )
                q = self._recompute_ground_truth(q, space)
                used_targets.add(q.target_points[0])
                lines.append(self._render_query(q, cfg.dim))
                lines.append("")
                final_queries.append(q)
                query_idx += 1
                queries_remaining -= 1

        # Phase 3: Remaining queries (leftover from interval misses)
        while queries_remaining > 0:
            pts = space.non_origin_points()
            if not pts:
                break
            lines.append("")
            q = self._plan_single_query(space, pts, used_targets, query_idx)
            q = self._recompute_ground_truth(q, space)
            used_targets.add(q.target_points[0])
            lines.append(self._render_query(q, cfg.dim))
            final_queries.append(q)
            query_idx += 1
            queries_remaining -= 1

        return Scenario(
            scenario_id=scenario_id,
            config=cfg,
            prompt="\n".join(lines).strip(),
            queries=final_queries,
            space_snapshot=space.to_dict(),
        )

    # ------------------------------------------------------------------
    # Point generation
    # ------------------------------------------------------------------

    def _gen_point_from_deepest(self, space: Space) -> tuple[str, PointDefinition]:
        """Generate a point extending from the deepest point (guaranteed depth growth)."""
        name = self._next_name()
        pts = space.non_origin_points()
        if pts:
            anchor = max(pts, key=lambda p: space.chain_depth(p))
        else:
            anchor = "O"
        defn = self._random_point_def(anchor, self.config.dim, space,
                                       force_single_anchor=True)
        space.define_point(name, defn)
        return name, defn

    def _gen_point(
        self, space: Space, leaf_bias: float,
        force_single_anchor: bool = False,
    ) -> tuple[str, PointDefinition]:
        """Generate a single point using leaf_bias to pick the anchor."""
        name = self._next_name()
        anchor = self._pick_anchor(space, leaf_bias)
        defn = self._random_point_def(anchor, self.config.dim, space,
                                       force_single_anchor=force_single_anchor)
        space.define_point(name, defn)
        return name, defn

    def _pick_anchor(self, space: Space, leaf_bias: float) -> str:
        """Pick an anchor point based on leaf_bias.

        leaf_bias=1.0: always pick a leaf (extends chains linearly)
        leaf_bias=0.0: pick any existing point (creates bushy graphs)
        """
        all_points = space.non_origin_points()
        if not all_points:
            return "O"

        leaves = space.leaf_nodes()
        if not leaves:
            leaves = all_points

        if self.rng.random() < leaf_bias:
            return self.rng.choice(leaves)
        else:
            return self.rng.choice(all_points)

    def _random_point_def(
        self, anchor: str, dim: int, space: Space,
        force_single_anchor: bool = False,
    ) -> PointDefinition:
        """Generate a random point definition relative to anchor.

        For multi-anchor types (midpoint, weighted_centroid), picks
        additional anchors from existing points in space.
        """
        available = list(self.config.point_def_types)
        if dim == 2:
            available = [t for t in available if t != "magnitude_spherical"]
        if dim >= 3:
            available = [t for t in available if t != "magnitude_polar"]
        if force_single_anchor:
            available = [t for t in available
                         if t not in ("midpoint", "weighted_centroid")]
            if not available:
                available = ["cartesian_offset"]

        choice = self.rng.choice(available)
        all_pts = space.non_origin_points()

        match choice:
            case "cartesian_offset":
                offset = [round(self.rng.uniform(self.config.coord_min / 3,
                                                  self.config.coord_max / 3), 1)
                          for _ in range(dim)]
                return cartesian_offset(anchor, offset)

            case "magnitude_direction":
                direction = [round(self.rng.gauss(0, 1), 2) for _ in range(dim)]
                if all(abs(d) < 0.01 for d in direction):
                    direction[0] = 1.0
                mag = round(self.rng.uniform(self.config.magnitude_min,
                                             self.config.magnitude_max), 1)
                return magnitude_direction(anchor, mag, direction)

            case "magnitude_polar":
                mag = round(self.rng.uniform(self.config.magnitude_min,
                                             self.config.magnitude_max), 1)
                angle = round(self.rng.uniform(0, 360), 0)
                return magnitude_polar(anchor, mag, angle)

            case "magnitude_spherical":
                mag = round(self.rng.uniform(self.config.magnitude_min,
                                             self.config.magnitude_max), 1)
                theta = round(self.rng.uniform(0, 180), 0)
                phi = round(self.rng.uniform(0, 360), 0)
                return magnitude_spherical(anchor, mag, theta, phi)

            case "midpoint":
                # Pick 2-3 anchors from existing points
                if len(all_pts) >= 2:
                    k = self.rng.randint(2, min(3, len(all_pts)))
                    anchors = self.rng.sample(all_pts, k)
                    return midpoint(*anchors)
                # Fallback: offset from anchor
                offset = [round(self.rng.uniform(-5, 5), 1) for _ in range(dim)]
                return cartesian_offset(anchor, offset)

            case "weighted_centroid":
                if len(all_pts) >= 2:
                    k = self.rng.randint(2, min(4, len(all_pts)))
                    anchors = self.rng.sample(all_pts, k)
                    weights = [round(self.rng.uniform(0.1, 1.0), 2) for _ in range(k)]
                    return weighted_centroid(anchors, weights)
                offset = [round(self.rng.uniform(-5, 5), 1) for _ in range(dim)]
                return cartesian_offset(anchor, offset)

            case _:
                offset = [round(self.rng.uniform(-5, 5), 1) for _ in range(dim)]
                return cartesian_offset(anchor, offset)

    # ------------------------------------------------------------------
    # Query planning
    # ------------------------------------------------------------------

    def _has_valid_query_target(self, space: Space) -> bool:
        """Check if a valid query target exists given depth constraints."""
        cfg = self.config
        pts = space.non_origin_points()
        if not pts:
            return False
        if cfg.query_target_depth is not None:
            return any(space.chain_depth(p) == cfg.query_target_depth for p in pts)
        if cfg.query_min_depth is not None:
            return any(space.chain_depth(p) >= cfg.query_min_depth for p in pts)
        return True

    def _pick_deep_point(self, space: Space, points: list[str]) -> str:
        """Pick a point biased toward the deeper end of the chain."""
        by_depth = sorted(points, key=lambda p: space.chain_depth(p), reverse=True)
        deep_half = by_depth[:max(1, len(by_depth) // 2)]
        return self.rng.choice(deep_half)

    def _pick_point_at_depth(
        self, space: Space, points: list[str], target_depth: int,
    ) -> str | None:
        """Pick a random point at exactly the specified chain depth."""
        at_depth = [p for p in points if space.chain_depth(p) == target_depth]
        return self.rng.choice(at_depth) if at_depth else None

    def _select_query_target(
        self, space: Space, available_points: list[str],
        used_targets: set[str],
    ) -> str:
        """Select a query target respecting depth constraints from config."""
        cfg = self.config

        if cfg.query_target_depth is not None:
            pt = self._pick_point_at_depth(space, available_points, cfg.query_target_depth)
            if pt is not None:
                return pt
            return self._pick_deep_point(space, available_points)

        if cfg.query_min_depth is not None:
            deep_enough = [p for p in available_points
                           if space.chain_depth(p) >= cfg.query_min_depth]
            if deep_enough:
                return self.rng.choice(deep_enough)
            return self._pick_deep_point(space, available_points)

        # Default: biased toward deep, prefer unused targets
        candidates = [p for p in available_points if p not in used_targets]
        if not candidates:
            candidates = list(available_points)
        return self._pick_deep_point(space, candidates)

    def _plan_single_query(
        self, space: Space, available_points: list[str],
        used_targets: set[str], query_idx: int,
    ) -> Query:
        """Plan one query targeting only points defined so far."""
        qtype = self.rng.choice(self.config.query_types)
        query_id = f"q_{query_idx:03d}"

        target = self._select_query_target(space, available_points, used_targets)

        match qtype:
            case QueryType.POSITION:
                return Query(
                    query_id=query_id, query_type=qtype,
                    target_points=[target],
                    ground_truth=space.get_position(target),
                    chain_depth=space.chain_depth(target),
                    query_index=query_idx,
                )

            case QueryType.DISTANCE:
                others = [p for p in available_points if p != target]
                b = self.rng.choice(others) if others else "O"
                return Query(
                    query_id=query_id, query_type=qtype,
                    target_points=[target, b],
                    ground_truth=float(distance(
                        space.get_position(target), space.get_position(b))),
                    chain_depth=max(space.chain_depth(target),
                                    space.chain_depth(b)),
                    query_index=query_idx,
                )

            case QueryType.BOOLEAN:
                others = [p for p in available_points if p != target]
                if len(others) >= 2:
                    a, b = self.rng.sample(others, 2)
                elif len(others) == 1:
                    a, b = others[0], "O"
                else:
                    a, b = "O", "O"
                d_a = distance(space.get_position(target), space.get_position(a))
                d_b = distance(space.get_position(target), space.get_position(b))
                answer = a if d_a < d_b else b
                return Query(
                    query_id=query_id, query_type=qtype,
                    target_points=[target, a, b],
                    ground_truth=answer,
                    chain_depth=max(space.chain_depth(target),
                                    space.chain_depth(a), space.chain_depth(b)),
                    query_index=query_idx,
                )

    # ------------------------------------------------------------------
    # Transform planning
    # ------------------------------------------------------------------

    def _plan_single_transform(self, available_points: list[str]) -> Transform:
        """Plan one transform targeting only points that have been defined."""
        cfg = self.config
        ttype = self.rng.choice(cfg.transform_types)
        k = self.rng.randint(1, max(1, len(available_points) // 2))
        pts = self.rng.sample(available_points, min(k, len(available_points)))

        match ttype:
            case "rotation":
                center = [0.0] * cfg.dim
                angle = self.rng.choice([45, 90, 180, -90, 30, 60, 120])
                axis = None
                if cfg.dim >= 3:
                    axis_choices = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
                    axis = self.rng.choice(axis_choices)
                return rotation(pts, center, angle, axis)

            case "translation":
                disp = [round(self.rng.uniform(-3, 3), 1) for _ in range(cfg.dim)]
                return translation(pts, disp)

            case "reflection":
                normal = [0.0] * cfg.dim
                normal[self.rng.randint(0, cfg.dim - 1)] = 1.0
                plane_pt = [0.0] * cfg.dim
                return reflection(pts, normal, plane_pt)

            case "scaling":
                center = [0.0] * cfg.dim
                factor = self.rng.choice([0.5, 2.0, 1.5, 0.25, 3.0])
                return scaling(pts, center, factor)

            case _:
                disp = [round(self.rng.uniform(-3, 3), 1) for _ in range(cfg.dim)]
                return translation(pts, disp)

    # ------------------------------------------------------------------
    # Ground truth & rendering
    # ------------------------------------------------------------------

    def _recompute_ground_truth(self, query: Query, space: Space) -> Query:
        """Recompute ground truth after transforms may have changed positions."""
        match query.query_type:
            case QueryType.POSITION:
                query.ground_truth = space.get_position(query.target_points[0])
            case QueryType.DISTANCE:
                a, b = query.target_points
                query.ground_truth = distance(
                    space.get_position(a), space.get_position(b)
                )
            case QueryType.BOOLEAN:
                target, a, b = query.target_points
                d_a = distance(space.get_position(target), space.get_position(a))
                d_b = distance(space.get_position(target), space.get_position(b))
                query.ground_truth = a if d_a < d_b else b
        query.chain_depth = max(space.chain_depth(p) for p in query.target_points)
        return query

    def _render_query(self, query: Query, dim: int) -> str:
        match query.query_type:
            case QueryType.POSITION:
                return render_query_position(query.target_points[0], query.query_id, dim)
            case QueryType.DISTANCE:
                return render_query_distance(query.target_points[0], query.target_points[1], query.query_id)
            case QueryType.BOOLEAN:
                return render_query_boolean(
                    query.target_points[0], query.target_points[1],
                    query.target_points[2], query.query_id,
                )


def generate_scenario_batch(
    configs: list[ScenarioConfig],
    id_prefix: str = "s",
) -> list[Scenario]:
    """Generate a batch of scenarios from a list of configs."""
    scenarios = []
    for i, cfg in enumerate(configs):
        gen = ScenarioGenerator(cfg)
        scenario = gen.generate(f"{id_prefix}_{i:04d}")
        scenarios.append(scenario)
    return scenarios
