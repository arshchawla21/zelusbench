"""Scenario generator — builds randomized scenarios from config."""

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

from .config import ScenarioConfig, DAGStructure, QueryType
from .templates import (
    render_system_prompt, render_point_definition, render_transform,
    render_query_position, render_query_distance, render_query_boolean,
)
from .distractors import (
    generate_disconnected_distractors, generate_branch_distractors,
    generate_restatement_distractors,
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
class ScenarioBlock:
    """A block of statements, transforms, or queries."""
    block_type: str  # "statements", "distractors", "transform", "query", "restatement"
    content: list[str]  # natural language lines
    point_definitions: list[tuple[str, dict]] = field(default_factory=list)
    transform: dict | None = None
    query: dict | None = None


@dataclass
class Scenario:
    scenario_id: str
    config: ScenarioConfig
    prompt: str
    queries: list[Query]
    space_snapshot: dict  # serialized Space after all operations
    blocks: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "scenario_id": self.scenario_id,
            "dim": self.config.dim,
            "prompt": self.prompt,
            "queries": [q.to_dict() for q in self.queries],
            "space": self.space_snapshot,
            "metadata": {
                "dag_structure": self.config.dag_structure.name,
                "distractor_ratio": self.config.distractor_ratio,
                "num_transforms": self.config.num_transforms,
                "dim": self.config.dim,
            },
        }


class ScenarioGenerator:
    """Generates randomized geometric scenarios from a config."""

    def __init__(self, config: ScenarioConfig):
        self.config = config
        self.rng = random.Random(config.seed)
        self._name_counter = 0

    def _next_name(self) -> str:
        self._name_counter += 1
        # Use A, B, C, ... then A1, B1, ...
        idx = self._name_counter - 1
        if idx < 26:
            return chr(ord('A') + idx)
        return chr(ord('A') + idx % 26) + str(idx // 26)

    def generate(self, scenario_id: str) -> Scenario:
        """Generate a complete scenario."""
        cfg = self.config
        space = Space(dim=cfg.dim)

        # Phase 1: Build the main point chain based on DAG structure
        main_points = self._build_main_graph(space)

        # Phase 2: Generate distractors
        distractor_points = self._generate_distractors(space, main_points)

        # Phase 3: Determine query placement and transforms
        queries = self._plan_queries(space, main_points)

        # Phase 4: Plan transforms
        transforms = self._plan_transforms(space, main_points)

        # Phase 5: Assemble the prompt with interleaved blocks
        prompt, final_queries = self._assemble_prompt(
            space, main_points, distractor_points, queries, transforms
        )

        return Scenario(
            scenario_id=scenario_id,
            config=cfg,
            prompt=prompt,
            queries=final_queries,
            space_snapshot=space.to_dict(),
        )

    def _build_main_graph(self, space: Space) -> list[str]:
        """Build the main dependency graph. Returns list of point names."""
        cfg = self.config
        depth = self.rng.randint(cfg.min_chain_depth, cfg.max_chain_depth)

        match cfg.dag_structure:
            case DAGStructure.LINEAR:
                return self._build_linear(space, depth)
            case DAGStructure.BRANCHING:
                return self._build_branching(space, depth)
            case DAGStructure.MERGING:
                return self._build_merging(space, depth)
            case DAGStructure.DIAMOND:
                return self._build_diamond(space, depth)

    def _random_point_def(self, anchor: str, dim: int) -> PointDefinition:
        """Generate a random point definition relative to anchor."""
        available = list(self.config.point_def_types)
        # Filter out types that don't apply
        if dim == 2:
            available = [t for t in available if t != "magnitude_spherical"]
        choice = self.rng.choice(available)

        match choice:
            case "cartesian_offset":
                offset = [round(self.rng.uniform(self.config.coord_min / 3, self.config.coord_max / 3), 1)
                          for _ in range(dim)]
                return cartesian_offset(anchor, offset)

            case "magnitude_direction":
                direction = [round(self.rng.gauss(0, 1), 2) for _ in range(dim)]
                # Avoid zero vector
                if all(abs(d) < 0.01 for d in direction):
                    direction[0] = 1.0
                mag = round(self.rng.uniform(self.config.magnitude_min, self.config.magnitude_max), 1)
                return magnitude_direction(anchor, mag, direction)

            case "magnitude_polar":
                mag = round(self.rng.uniform(self.config.magnitude_min, self.config.magnitude_max), 1)
                angle = round(self.rng.uniform(0, 360), 0)
                return magnitude_polar(anchor, mag, angle)

            case "magnitude_spherical":
                mag = round(self.rng.uniform(self.config.magnitude_min, self.config.magnitude_max), 1)
                theta = round(self.rng.uniform(0, 180), 0)
                phi = round(self.rng.uniform(0, 360), 0)
                return magnitude_spherical(anchor, mag, theta, phi)

            case _:
                # Default to cartesian offset
                offset = [round(self.rng.uniform(-5, 5), 1) for _ in range(dim)]
                return cartesian_offset(anchor, offset)

    def _build_linear(self, space: Space, depth: int) -> list[str]:
        names = []
        prev = "O"
        for _ in range(depth):
            name = self._next_name()
            defn = self._random_point_def(prev, self.config.dim)
            space.define_point(name, defn)
            names.append(name)
            prev = name
        return names

    def _build_branching(self, space: Space, depth: int) -> list[str]:
        """A -> B -> C, A -> D -> E. Two branches from a common root."""
        names = []
        # Main trunk (half the depth)
        trunk_len = max(2, depth // 2)
        prev = "O"
        for _ in range(trunk_len):
            name = self._next_name()
            defn = self._random_point_def(prev, self.config.dim)
            space.define_point(name, defn)
            names.append(name)
            prev = name

        branch_point = names[self.rng.randint(0, len(names) - 1)]

        # Branch 1
        prev = branch_point
        remaining = depth - trunk_len
        b1_len = max(1, remaining // 2)
        for _ in range(b1_len):
            name = self._next_name()
            defn = self._random_point_def(prev, self.config.dim)
            space.define_point(name, defn)
            names.append(name)
            prev = name

        # Branch 2
        prev = branch_point
        b2_len = remaining - b1_len
        for _ in range(max(1, b2_len)):
            name = self._next_name()
            defn = self._random_point_def(prev, self.config.dim)
            space.define_point(name, defn)
            names.append(name)
            prev = name

        return names

    def _build_merging(self, space: Space, depth: int) -> list[str]:
        """Two chains merge via midpoint."""
        names = []
        chain_len = max(2, depth // 2)

        # Chain 1
        prev = "O"
        chain1_end = None
        for _ in range(chain_len):
            name = self._next_name()
            defn = self._random_point_def(prev, self.config.dim)
            space.define_point(name, defn)
            names.append(name)
            prev = name
            chain1_end = name

        # Chain 2
        prev = "O"
        chain2_end = None
        for _ in range(chain_len):
            name = self._next_name()
            defn = self._random_point_def(prev, self.config.dim)
            space.define_point(name, defn)
            names.append(name)
            prev = name
            chain2_end = name

        # Merge point
        merge_name = self._next_name()
        space.define_point(merge_name, midpoint(chain1_end, chain2_end))
        names.append(merge_name)

        # Continue from merge
        prev = merge_name
        for _ in range(max(1, depth - 2 * chain_len - 1)):
            name = self._next_name()
            defn = self._random_point_def(prev, self.config.dim)
            space.define_point(name, defn)
            names.append(name)
            prev = name

        return names

    def _build_diamond(self, space: Space, depth: int) -> list[str]:
        """A -> B, A -> C, D = f(B, C), then continue from D."""
        names = []

        # Root
        root = self._next_name()
        defn = self._random_point_def("O", self.config.dim)
        space.define_point(root, defn)
        names.append(root)

        # Left branch
        prev = root
        left_len = max(1, (depth - 2) // 2)
        left_end = None
        for _ in range(left_len):
            name = self._next_name()
            defn = self._random_point_def(prev, self.config.dim)
            space.define_point(name, defn)
            names.append(name)
            prev = name
            left_end = name

        # Right branch
        prev = root
        right_len = max(1, (depth - 2) // 2)
        right_end = None
        for _ in range(right_len):
            name = self._next_name()
            defn = self._random_point_def(prev, self.config.dim)
            space.define_point(name, defn)
            names.append(name)
            prev = name
            right_end = name

        # Diamond merge
        diamond = self._next_name()
        space.define_point(diamond, midpoint(left_end, right_end))
        names.append(diamond)

        # Tail
        prev = diamond
        tail_len = max(0, depth - left_len - right_len - 2)
        for _ in range(tail_len):
            name = self._next_name()
            defn = self._random_point_def(prev, self.config.dim)
            space.define_point(name, defn)
            names.append(name)
            prev = name

        return names

    def _generate_distractors(self, space: Space, main_points: list[str]) -> list[str]:
        """Generate distractor points based on config."""
        cfg = self.config
        num_relevant = len(main_points)
        num_distractors = num_relevant * cfg.distractor_ratio

        if num_distractors == 0:
            return []

        existing = set(space.point_names)
        distractor_names = []

        # Split between disconnected and branch distractors
        num_disconnected = num_distractors // 2
        num_branch = num_distractors - num_disconnected

        # Disconnected subgraph
        disconnected = generate_disconnected_distractors(
            max(1, num_disconnected), cfg.dim, existing, self.rng,
            (cfg.coord_min, cfg.coord_max),
        )
        for name, defn in disconnected:
            space.define_point(name, defn)
            distractor_names.append(name)
            existing.add(name)

        # Branch distractors
        branch = generate_branch_distractors(
            max(1, num_branch), cfg.dim, main_points, existing, self.rng,
            (cfg.coord_min, cfg.coord_max),
        )
        for name, defn in branch:
            space.define_point(name, defn)
            distractor_names.append(name)
            existing.add(name)

        return distractor_names

    def _plan_queries(self, space: Space, main_points: list[str]) -> list[Query]:
        """Plan queries distributed across the scenario."""
        cfg = self.config
        queries = []

        for i in range(cfg.num_queries):
            qtype = self.rng.choice(cfg.query_types)
            query_id = f"q_{i:03d}"

            match qtype:
                case QueryType.POSITION:
                    target = self.rng.choice(main_points)
                    pos = space.get_position(target)
                    queries.append(Query(
                        query_id=query_id,
                        query_type=qtype,
                        target_points=[target],
                        ground_truth=pos,
                        chain_depth=space.chain_depth(target),
                        query_index=i,
                    ))

                case QueryType.DISTANCE:
                    if len(main_points) >= 2:
                        a, b = self.rng.sample(main_points, 2)
                    else:
                        a, b = main_points[0], "O"
                    dist = distance(space.get_position(a), space.get_position(b))
                    queries.append(Query(
                        query_id=query_id,
                        query_type=qtype,
                        target_points=[a, b],
                        ground_truth=float(dist),
                        chain_depth=max(space.chain_depth(a), space.chain_depth(b)),
                        query_index=i,
                    ))

                case QueryType.BOOLEAN:
                    if len(main_points) >= 3:
                        target, a, b = self.rng.sample(main_points, 3)
                    else:
                        target = main_points[0]
                        a, b = "O", main_points[-1] if len(main_points) > 1 else "O"
                    d_a = distance(space.get_position(target), space.get_position(a))
                    d_b = distance(space.get_position(target), space.get_position(b))
                    answer = a if d_a < d_b else b
                    queries.append(Query(
                        query_id=query_id,
                        query_type=qtype,
                        target_points=[target, a, b],
                        ground_truth=answer,
                        chain_depth=max(space.chain_depth(target),
                                        space.chain_depth(a), space.chain_depth(b)),
                        query_index=i,
                    ))

        return queries

    def _plan_transforms(self, space: Space, main_points: list[str]) -> list[Transform]:
        """Plan transform events based on config."""
        cfg = self.config
        num = cfg.num_transforms
        if num == 0:
            return []

        transforms = []
        for _ in range(num):
            ttype = self.rng.choice(cfg.transform_types)
            # Pick a subset of points to transform
            k = self.rng.randint(1, max(1, len(main_points) // 2))
            pts = self.rng.sample(main_points, min(k, len(main_points)))

            match ttype:
                case "rotation":
                    center = [0.0] * cfg.dim
                    angle = self.rng.choice([45, 90, 180, -90, 30, 60, 120])
                    axis = None
                    if cfg.dim >= 3:
                        # Random axis from standard basis
                        axis_choices = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
                        axis = self.rng.choice(axis_choices)
                    transforms.append(rotation(pts, center, angle, axis))

                case "translation":
                    disp = [round(self.rng.uniform(-3, 3), 1) for _ in range(cfg.dim)]
                    transforms.append(translation(pts, disp))

                case "reflection":
                    normal = [0.0] * cfg.dim
                    normal[self.rng.randint(0, cfg.dim - 1)] = 1.0
                    plane_pt = [0.0] * cfg.dim
                    transforms.append(reflection(pts, normal, plane_pt))

                case "scaling":
                    center = [0.0] * cfg.dim
                    factor = self.rng.choice([0.5, 2.0, 1.5, 0.25, 3.0])
                    transforms.append(scaling(pts, center, factor))

        return transforms

    def _assemble_prompt(
        self,
        space: Space,
        main_points: list[str],
        distractor_points: list[str],
        queries: list[Query],
        transforms: list[Transform],
    ) -> tuple[str, list[Query]]:
        """Assemble the full prompt with interleaved blocks."""
        cfg = self.config
        lines: list[str] = []

        # System prompt
        lines.append(render_system_prompt(cfg.dim))
        lines.append("")

        # Distribute main points across splits
        all_points = list(main_points)
        splits = self._split_list(all_points, cfg.num_splits)

        # Distribute distractors across splits
        dist_splits = self._split_list(distractor_points, cfg.num_splits)

        # Distribute transforms between queries
        transform_placements = self._split_list(transforms, max(1, cfg.num_queries - 1))

        # Track which queries need ground truth recomputed after transforms
        applied_transforms = 0
        final_queries: list[Query] = []

        for split_idx in range(cfg.num_splits):
            # Statement block
            if split_idx < len(splits):
                for name in splits[split_idx]:
                    defn = space.get_definition(name)
                    lines.append(render_point_definition(name, defn))

            # Distractor block
            if split_idx < len(dist_splits) and dist_splits[split_idx]:
                lines.append("")
                for name in dist_splits[split_idx]:
                    defn = space.get_definition(name)
                    lines.append(render_point_definition(name, defn))

                # Add restatement distractors
                restatements = generate_restatement_distractors(
                    main_points, min(2, cfg.distractor_ratio), self.rng
                )
                for r in restatements:
                    lines.append(r)

            lines.append("")

            # Query checkpoint (if we have queries left)
            if split_idx < len(queries):
                q = queries[split_idx]
                lines.append(self._render_query(q, cfg.dim))
                lines.append("")

                # Recompute ground truth based on current space state
                q = self._recompute_ground_truth(q, space)
                final_queries.append(q)

            # Apply transforms after query (except after last)
            if split_idx < len(transform_placements) and transform_placements[split_idx]:
                for t in transform_placements[split_idx]:
                    t.apply(space)
                    lines.append(render_transform(t))
                    applied_transforms += 1
                lines.append("")

        # Any remaining queries after all splits
        for q in queries[len(final_queries):]:
            lines.append(self._render_query(q, cfg.dim))
            lines.append("")
            q = self._recompute_ground_truth(q, space)
            final_queries.append(q)

        return "\n".join(lines), final_queries

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

    @staticmethod
    def _split_list(lst: list, n: int) -> list[list]:
        """Split a list into n roughly equal parts."""
        if n <= 0:
            return [lst]
        k, m = divmod(len(lst), n)
        return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


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
