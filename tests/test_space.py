"""Tests for Space — world state with dependency DAG."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from zelusbench.geometry.space import Space
from zelusbench.geometry.point import (
    cartesian_offset, midpoint, magnitude_direction, origin,
)
from zelusbench.geometry.vectors import vec


class TestSpaceBasics:
    def test_origin_exists(self):
        space = Space(dim=3)
        assert space.has_point("O")
        assert_allclose(space.get_position("O"), [0, 0, 0])

    def test_define_point(self):
        space = Space(dim=3)
        space.define_point("A", cartesian_offset("O", [1, 2, 3]))
        assert_allclose(space.get_position("A"), [1, 2, 3])

    def test_chain(self):
        space = Space(dim=3)
        space.define_point("A", cartesian_offset("O", [1, 0, 0]))
        space.define_point("B", cartesian_offset("A", [0, 1, 0]))
        space.define_point("C", cartesian_offset("B", [0, 0, 1]))
        assert_allclose(space.get_position("C"), [1, 1, 1])

    def test_undefined_point_raises(self):
        space = Space(dim=3)
        with pytest.raises(KeyError):
            space.get_position("X")


class TestDependencyDAG:
    def test_chain_depth(self):
        space = Space(dim=3)
        space.define_point("A", cartesian_offset("O", [1, 0, 0]))
        space.define_point("B", cartesian_offset("A", [0, 1, 0]))
        space.define_point("C", cartesian_offset("B", [0, 0, 1]))
        assert space.chain_depth("O") == 0
        assert space.chain_depth("A") == 1
        assert space.chain_depth("B") == 2
        assert space.chain_depth("C") == 3

    def test_get_dependents(self):
        space = Space(dim=3)
        space.define_point("A", cartesian_offset("O", [1, 0, 0]))
        space.define_point("B", cartesian_offset("A", [0, 1, 0]))
        space.define_point("C", cartesian_offset("B", [0, 0, 1]))
        deps = space.get_dependents("A")
        assert "B" in deps
        assert "C" in deps

    def test_get_dependencies(self):
        space = Space(dim=3)
        space.define_point("A", cartesian_offset("O", [1, 0, 0]))
        space.define_point("B", cartesian_offset("A", [0, 1, 0]))
        space.define_point("C", cartesian_offset("B", [0, 0, 1]))
        deps = space.get_dependencies("C")
        assert "B" in deps
        assert "A" in deps
        assert "O" in deps


class TestPropagation:
    def test_redefine_propagates(self):
        space = Space(dim=3)
        space.define_point("A", cartesian_offset("O", [1, 0, 0]))
        space.define_point("B", cartesian_offset("A", [1, 0, 0]))
        assert_allclose(space.get_position("B"), [2, 0, 0])

        # Redefine A
        space.define_point("A", cartesian_offset("O", [5, 0, 0]))
        assert_allclose(space.get_position("A"), [5, 0, 0])
        assert_allclose(space.get_position("B"), [6, 0, 0])

    def test_direct_position_propagates(self):
        space = Space(dim=3)
        space.define_point("A", cartesian_offset("O", [1, 0, 0]))
        space.define_point("B", cartesian_offset("A", [1, 0, 0]))
        space.define_point("C", cartesian_offset("B", [1, 0, 0]))

        space.set_position_direct("A", vec(10, 0, 0))
        # B's definition is still "A + (1,0,0)" so B should now resolve from A's new pos
        # But set_position_direct changes A's resolved position, then propagates
        # B is re-resolved from its definition using A's new position
        assert_allclose(space.get_position("B"), [11, 0, 0])
        assert_allclose(space.get_position("C"), [12, 0, 0])

    def test_midpoint_propagation(self):
        space = Space(dim=3)
        space.define_point("A", cartesian_offset("O", [0, 0, 0]))
        space.define_point("B", cartesian_offset("O", [4, 0, 0]))
        space.define_point("M", midpoint("A", "B"))
        assert_allclose(space.get_position("M"), [2, 0, 0])

        # Move B
        space.set_position_direct("B", vec(10, 0, 0))
        assert_allclose(space.get_position("M"), [5, 0, 0])


class TestSerialization:
    def test_round_trip(self):
        space = Space(dim=3)
        space.define_point("A", cartesian_offset("O", [1, 2, 3]))
        space.define_point("B", cartesian_offset("A", [4, 5, 6]))
        space.define_point("M", midpoint("A", "B"))

        d = space.to_dict()
        restored = Space.from_dict(d)

        assert_allclose(restored.get_position("A"), space.get_position("A"))
        assert_allclose(restored.get_position("B"), space.get_position("B"))
        assert_allclose(restored.get_position("M"), space.get_position("M"))

    def test_copy(self):
        space = Space(dim=3)
        space.define_point("A", cartesian_offset("O", [1, 0, 0]))
        copy = space.copy()
        copy.define_point("B", cartesian_offset("A", [0, 1, 0]))
        assert not space.has_point("B")
        assert copy.has_point("B")
