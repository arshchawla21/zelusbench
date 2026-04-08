"""Tests for scenario generation, parsing, and scoring."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

import random

from zelusbench.scenarios.config import (
    ScenarioConfig, QueryType,
    easy_config, medium_config, hard_config,
)
from zelusbench.scenarios.generator import ScenarioGenerator, Scenario, generate_scenario_batch
from zelusbench.evaluation.parser import parse_coordinates, parse_distance, parse_boolean
from zelusbench.evaluation.scorer import (
    score_query, ScoreTier, relative_error_vec, relative_error_scalar,
)


class TestScenarioGeneration:
    def test_basic_generation(self):
        cfg = easy_config(seed=42)
        gen = ScenarioGenerator(cfg)
        scenario = gen.generate("test_001")
        assert scenario.scenario_id == "test_001"
        assert len(scenario.queries) > 0
        assert len(scenario.prompt) > 0

    def test_deterministic(self):
        cfg = easy_config(seed=42)
        gen1 = ScenarioGenerator(cfg)
        s1 = gen1.generate("test")
        gen2 = ScenarioGenerator(cfg)
        s2 = gen2.generate("test")
        assert s1.prompt == s2.prompt

    def test_leaf_bias_spectrum(self):
        """Different leaf_bias values should produce valid scenarios."""
        for lb in [0.0, 0.25, 0.5, 0.75, 1.0]:
            cfg = ScenarioConfig(
                dim=3, min_chain_depth=4, max_chain_depth=6,
                leaf_bias=lb, num_points=8,
                transform_prob=0.0,
                num_queries=2, seed=42,
            )
            gen = ScenarioGenerator(cfg)
            scenario = gen.generate(f"test_lb_{lb}")
            assert len(scenario.queries) >= 1
            for q in scenario.queries:
                assert q.ground_truth is not None

    def test_many_points(self):
        """Many points should produce a longer prompt."""
        cfg = ScenarioConfig(
            dim=3, min_chain_depth=3, max_chain_depth=5,
            num_points=20, leaf_bias=0.5,
            transform_prob=0.0,
            num_queries=2, seed=42,
        )
        gen = ScenarioGenerator(cfg)
        scenario = gen.generate("test_many")
        assert len(scenario.prompt) > 0

    def test_with_transforms(self):
        cfg = ScenarioConfig(
            dim=3, min_chain_depth=3, max_chain_depth=5,
            num_points=8, leaf_bias=0.5,
            transform_prob=0.15,
            transform_types=["rotation", "translation"],
            num_queries=3, seed=42,
        )
        gen = ScenarioGenerator(cfg)
        scenario = gen.generate("test_transform")
        assert len(scenario.queries) >= 1

    def test_2d_generation(self):
        cfg = easy_config(dim=2, seed=42)
        gen = ScenarioGenerator(cfg)
        scenario = gen.generate("test_2d")
        for q in scenario.queries:
            if isinstance(q.ground_truth, np.ndarray):
                assert len(q.ground_truth) == 2

    def test_batch_generation(self):
        configs = [easy_config(seed=i) for i in range(5)]
        scenarios = generate_scenario_batch(configs)
        assert len(scenarios) == 5
        ids = [s.scenario_id for s in scenarios]
        assert len(set(ids)) == 5

    def test_query_types(self):
        cfg = ScenarioConfig(
            dim=3, min_chain_depth=4, max_chain_depth=6,
            query_types=[QueryType.POSITION, QueryType.DISTANCE, QueryType.BOOLEAN],
            num_points=8, num_queries=6, seed=42,
        )
        gen = ScenarioGenerator(cfg)
        scenario = gen.generate("test_qtypes")
        types = {q.query_type for q in scenario.queries}
        assert len(types) >= 1

    def test_min_depth_reached(self):
        """Generator should always reach min_chain_depth."""
        for seed in range(10):
            cfg = ScenarioConfig(
                dim=3, min_chain_depth=6, max_chain_depth=8,
                leaf_bias=0.3, num_points=10,
                transform_prob=0.0, num_queries=2, seed=seed,
            )
            gen = ScenarioGenerator(cfg)
            scenario = gen.generate(f"depth_test_{seed}")
            from zelusbench.geometry.space import Space
            space = Space.from_dict(scenario.space_snapshot)
            max_depth = max(space.chain_depth(p) for p in space.non_origin_points())
            assert max_depth >= 6, f"seed={seed}: max_depth={max_depth}, expected >= 6"

    def test_high_leaf_bias_is_linear(self):
        """leaf_bias=1.0 should produce mostly linear chains (high max depth)."""
        cfg = ScenarioConfig(
            dim=2, min_chain_depth=5, max_chain_depth=10,
            leaf_bias=1.0, num_points=8,
            transform_prob=0.0, num_queries=2, seed=42,
        )
        gen = ScenarioGenerator(cfg)
        scenario = gen.generate("linear_test")
        from zelusbench.geometry.space import Space
        space = Space.from_dict(scenario.space_snapshot)
        pts = space.non_origin_points()
        max_depth = max(space.chain_depth(p) for p in pts)
        # With lb=1.0 and 8 points, depth should be close to 8
        assert max_depth >= 5


class TestParsing:
    def test_parse_coordinates_parens(self):
        result = parse_coordinates("The position is (1.5, -2.3, 7.1)", dim=3)
        assert result is not None
        assert_allclose(result, [1.5, -2.3, 7.1])

    def test_parse_coordinates_brackets(self):
        result = parse_coordinates("Answer: [3.0, 4.0, 0.0]", dim=3)
        assert result is not None
        assert_allclose(result, [3.0, 4.0, 0.0])

    def test_parse_coordinates_2d(self):
        result = parse_coordinates("Position is (5.0, -3.0)", dim=2)
        assert result is not None
        assert_allclose(result, [5.0, -3.0])

    def test_parse_coordinates_labeled(self):
        result = parse_coordinates("x=1.0, y=2.0, z=3.0", dim=3)
        assert result is not None
        assert_allclose(result, [1.0, 2.0, 3.0])

    def test_parse_distance(self):
        result = parse_distance("The distance is 5.385")
        assert result is not None
        assert abs(result - 5.385) < 0.001

    def test_parse_distance_units(self):
        result = parse_distance("approximately 7.2 units")
        assert result is not None
        assert abs(result - 7.2) < 0.001

    def test_parse_boolean(self):
        assert parse_boolean("Point A", "A", "B") == "A"
        assert parse_boolean("B", "A", "B") == "B"
        assert parse_boolean("The answer is Point B", "A", "B") == "B"

    def test_parse_unparseable(self):
        result = parse_coordinates("I don't know", dim=3)
        assert result is None


class TestScoring:
    def test_exact_match(self):
        query = {"query_id": "q_001", "query_type": "POSITION",
                 "ground_truth": [1.0, 2.0, 3.0], "chain_depth": 1, "query_index": 0}
        parsed = {"parsed_value": np.array([1.0, 2.0, 3.0]), "parse_success": True}
        score = score_query(query, parsed)
        assert score.tier == ScoreTier.EXACT
        assert score.score == 1.0

    def test_close_match(self):
        query = {"query_id": "q_001", "query_type": "POSITION",
                 "ground_truth": [10.0, 0.0, 0.0], "chain_depth": 1, "query_index": 0}
        parsed = {"parsed_value": np.array([10.3, 0.0, 0.0]), "parse_success": True}
        score = score_query(query, parsed)
        assert score.tier == ScoreTier.CLOSE
        assert score.score == 0.7

    def test_wrong(self):
        query = {"query_id": "q_001", "query_type": "POSITION",
                 "ground_truth": [1.0, 0.0, 0.0], "chain_depth": 1, "query_index": 0}
        parsed = {"parsed_value": np.array([5.0, 5.0, 5.0]), "parse_success": True}
        score = score_query(query, parsed)
        assert score.tier == ScoreTier.WRONG
        assert score.score == 0.0

    def test_refused(self):
        query = {"query_id": "q_001", "query_type": "POSITION",
                 "ground_truth": [1.0, 0.0, 0.0], "chain_depth": 1, "query_index": 0}
        parsed = {"parsed_value": None, "parse_success": False}
        score = score_query(query, parsed)
        assert score.tier == ScoreTier.REFUSED
        assert score.score == 0.0

    def test_distance_scoring(self):
        query = {"query_id": "q_001", "query_type": "DISTANCE",
                 "ground_truth": 10.0, "chain_depth": 1, "query_index": 0}
        parsed = {"parsed_value": 10.0, "parse_success": True}
        score = score_query(query, parsed)
        assert score.tier == ScoreTier.EXACT
        assert score.score == 1.0

    def test_boolean_scoring(self):
        query = {"query_id": "q_001", "query_type": "BOOLEAN",
                 "ground_truth": "A", "chain_depth": 1, "query_index": 0}
        parsed = {"parsed_value": "A", "parse_success": True}
        score = score_query(query, parsed)
        assert score.score == 1.0

        parsed_wrong = {"parsed_value": "B", "parse_success": True}
        score_wrong = score_query(query, parsed_wrong)
        assert score_wrong.score == 0.0

    def test_relative_error_near_origin(self):
        err = relative_error_vec(np.array([0.1, 0, 0]), np.array([0, 0, 0]))
        assert err < 0.15


class TestQueryTargetDepth:
    def test_exact_depth_targeting(self):
        """All queries should target exactly the configured depth."""
        cfg = ScenarioConfig(
            dim=3, min_chain_depth=8, max_chain_depth=8,
            leaf_bias=1.0, num_points=10,
            transform_prob=0.0,
            query_types=[QueryType.POSITION],
            num_queries=3,
            query_target_depth=8, seed=42,
        )
        gen = ScenarioGenerator(cfg)
        scenario = gen.generate("depth_target_test")
        for q in scenario.queries:
            assert q.chain_depth == 8, (
                f"Query {q.query_id} has depth {q.chain_depth}, expected 8"
            )

    def test_min_depth_targeting(self):
        """Queries should target points at or above min depth."""
        cfg = ScenarioConfig(
            dim=3, min_chain_depth=10, max_chain_depth=10,
            leaf_bias=1.0, num_points=12,
            transform_prob=0.0,
            query_types=[QueryType.POSITION],
            num_queries=3,
            query_min_depth=7, seed=42,
        )
        gen = ScenarioGenerator(cfg)
        scenario = gen.generate("min_depth_test")
        for q in scenario.queries:
            assert q.chain_depth >= 7, (
                f"Query {q.query_id} has depth {q.chain_depth}, expected >= 7"
            )

    def test_depth_targeting_with_distance_query(self):
        """Distance queries: primary target at target depth."""
        cfg = ScenarioConfig(
            dim=3, min_chain_depth=6, max_chain_depth=6,
            leaf_bias=1.0, num_points=8,
            transform_prob=0.0,
            query_types=[QueryType.DISTANCE],
            num_queries=3,
            query_target_depth=6, seed=99,
        )
        gen = ScenarioGenerator(cfg)
        scenario = gen.generate("dist_depth_test")
        for q in scenario.queries:
            from zelusbench.geometry.space import Space
            space = Space.from_dict(scenario.space_snapshot)
            primary_depth = space.chain_depth(q.target_points[0])
            assert primary_depth == 6, (
                f"Query {q.query_id} primary target at depth {primary_depth}, expected 6"
            )


class TestRandomizeExcept:
    def test_pinned_fields_preserved(self):
        """Pinned fields should not change across seeds."""
        for seed in range(5):
            rng = random.Random(seed)
            cfg = ScenarioConfig.randomize_except(rng, pinned={
                "dim": 3,
                "min_chain_depth": 8,
                "max_chain_depth": 8,
                "seed": seed,
            })
            assert cfg.dim == 3
            assert cfg.min_chain_depth == 8
            assert cfg.max_chain_depth == 8

    def test_unpinned_fields_vary(self):
        """Unpinned fields should vary across seeds."""
        configs = []
        for seed in range(20):
            rng = random.Random(seed)
            cfg = ScenarioConfig.randomize_except(rng, pinned={
                "min_chain_depth": 4, "max_chain_depth": 4, "seed": seed,
            })
            configs.append(cfg)
        dims = {c.dim for c in configs}
        leaf_biases = {c.leaf_bias for c in configs}
        assert len(dims) >= 2, "Expected both 2D and 3D across 20 seeds"
        assert len(leaf_biases) >= 2, "Expected multiple leaf_bias values"

    def test_no_spherical_in_2d(self):
        """2D configs should never include magnitude_spherical."""
        for seed in range(10):
            rng = random.Random(seed)
            cfg = ScenarioConfig.randomize_except(rng, pinned={"dim": 2, "seed": seed})
            assert "magnitude_spherical" not in cfg.point_def_types


class TestEndToEnd:
    def test_generate_and_mock_score(self):
        """Full pipeline: generate scenario, mock-answer with ground truth, score."""
        import re
        from zelusbench.evaluation.parser import parse_model_response

        cfg = easy_config(seed=42)
        gen = ScenarioGenerator(cfg)
        scenario = gen.generate("e2e_test")

        # Simulate a perfect model by returning ground truth
        responses = []
        for q in scenario.queries:
            truth = q.ground_truth
            if isinstance(truth, np.ndarray):
                coords = ", ".join(f"{x:.4f}" for x in truth)
                responses.append(f"[Answer {q.query_id}] ({coords})")
            elif isinstance(truth, float):
                responses.append(f"[Answer {q.query_id}] {truth:.4f}")
            else:
                responses.append(f"[Answer {q.query_id}] {truth}")
        mock_response = "\n".join(responses)

        # Split by [Answer q_ID] and score
        parts = re.split(r'\[Answer\s+q_\d+\]', mock_response)
        if len(parts) > 1:
            parts = parts[1:]

        query_dicts = [q.to_dict() for q in scenario.queries]
        assert len(parts) == len(query_dicts)

        for qd, rp in zip(query_dicts, parts):
            parsed = parse_model_response(rp, qd)
            score = score_query(qd, parsed)
            assert score.score == 1.0, f"Mock should score perfectly: {score}"
