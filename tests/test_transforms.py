"""Tests for geometric transforms."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from zelusbench.geometry.space import Space
from zelusbench.geometry.point import cartesian_offset, midpoint
from zelusbench.geometry.transforms import (
    rotation, translation, reflection, scaling, frame_shift,
    apply_rotation, apply_translation, apply_reflection, apply_scaling,
    apply_frame_shift,
)
from zelusbench.geometry.vectors import vec


class TestRotation:
    def test_90_degrees_2d(self):
        space = Space(dim=2)
        space.define_point("A", cartesian_offset("O", [1, 0]))
        t = rotation(["A"], [0, 0], 90)
        t.apply(space)
        assert_allclose(space.get_position("A"), [0, 1], atol=1e-10)

    def test_180_degrees_2d(self):
        space = Space(dim=2)
        space.define_point("A", cartesian_offset("O", [1, 0]))
        t = rotation(["A"], [0, 0], 180)
        t.apply(space)
        assert_allclose(space.get_position("A"), [-1, 0], atol=1e-10)

    def test_90_degrees_3d_z_axis(self):
        space = Space(dim=3)
        space.define_point("A", cartesian_offset("O", [1, 0, 0]))
        t = rotation(["A"], [0, 0, 0], 90, [0, 0, 1])
        t.apply(space)
        assert_allclose(space.get_position("A"), [0, 1, 0], atol=1e-10)

    def test_rotation_around_center(self):
        space = Space(dim=2)
        space.define_point("A", cartesian_offset("O", [2, 0]))
        # Rotate around (1, 0) by 180 degrees
        t = rotation(["A"], [1, 0], 180)
        t.apply(space)
        assert_allclose(space.get_position("A"), [0, 0], atol=1e-10)

    def test_rotation_propagates(self):
        space = Space(dim=3)
        space.define_point("A", cartesian_offset("O", [1, 0, 0]))
        space.define_point("B", cartesian_offset("A", [1, 0, 0]))
        t = rotation(["A"], [0, 0, 0], 90, [0, 0, 1])
        t.apply(space)
        # A moves to (0,1,0), B should re-resolve to A + (1,0,0) = (1,1,0)
        assert_allclose(space.get_position("A"), [0, 1, 0], atol=1e-10)
        assert_allclose(space.get_position("B"), [1, 1, 0], atol=1e-10)


class TestTranslation:
    def test_simple(self):
        space = Space(dim=3)
        space.define_point("A", cartesian_offset("O", [1, 2, 3]))
        t = translation(["A"], [1, 1, 1])
        t.apply(space)
        assert_allclose(space.get_position("A"), [2, 3, 4])

    def test_multiple_points(self):
        space = Space(dim=3)
        space.define_point("A", cartesian_offset("O", [1, 0, 0]))
        space.define_point("B", cartesian_offset("O", [0, 1, 0]))
        t = translation(["A", "B"], [5, 5, 5])
        t.apply(space)
        assert_allclose(space.get_position("A"), [6, 5, 5])
        assert_allclose(space.get_position("B"), [5, 6, 5])


class TestReflection:
    def test_reflect_across_yz_plane(self):
        space = Space(dim=3)
        space.define_point("A", cartesian_offset("O", [3, 2, 1]))
        t = reflection(["A"], [1, 0, 0], [0, 0, 0])
        t.apply(space)
        assert_allclose(space.get_position("A"), [-3, 2, 1])

    def test_reflect_across_xz_plane(self):
        space = Space(dim=3)
        space.define_point("A", cartesian_offset("O", [3, 2, 1]))
        t = reflection(["A"], [0, 1, 0], [0, 0, 0])
        t.apply(space)
        assert_allclose(space.get_position("A"), [3, -2, 1])

    def test_reflect_2d(self):
        space = Space(dim=2)
        space.define_point("A", cartesian_offset("O", [3, 4]))
        t = reflection(["A"], [1, 0], [0, 0])
        t.apply(space)
        assert_allclose(space.get_position("A"), [-3, 4])


class TestScaling:
    def test_scale_from_origin(self):
        space = Space(dim=3)
        space.define_point("A", cartesian_offset("O", [2, 3, 4]))
        t = scaling(["A"], [0, 0, 0], 2.0)
        t.apply(space)
        assert_allclose(space.get_position("A"), [4, 6, 8])

    def test_scale_from_center(self):
        space = Space(dim=3)
        space.define_point("A", cartesian_offset("O", [4, 0, 0]))
        t = scaling(["A"], [2, 0, 0], 0.5)
        t.apply(space)
        assert_allclose(space.get_position("A"), [3, 0, 0])


class TestFrameShift:
    def test_shift_to_point(self):
        space = Space(dim=3)
        space.define_point("A", cartesian_offset("O", [5, 3, 1]))
        space.define_point("B", cartesian_offset("O", [10, 6, 2]))
        t = frame_shift("A")
        t.apply(space)
        assert_allclose(space.get_position("A"), [0, 0, 0])
        assert_allclose(space.get_position("B"), [5, 3, 1])


class TestTransformSerialization:
    def test_round_trip(self):
        t = rotation(["A", "B"], [0, 0, 0], 90, [0, 0, 1])
        d = t.to_dict()
        from zelusbench.geometry.transforms import Transform
        restored = Transform.from_dict(d)
        assert restored.transform_type == t.transform_type
        assert restored.params["angle_deg"] == 90
