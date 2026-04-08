"""Tests for point definitions and resolution."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from zelusbench.geometry.point import (
    PointDefinition, PointDefType,
    origin, cartesian_offset, magnitude_direction, magnitude_polar,
    magnitude_spherical, midpoint, weighted_centroid, projection,
)
from zelusbench.geometry.vectors import vec


class TestOrigin:
    def test_origin_2d(self):
        defn = origin(dim=2)
        pos = defn.resolve({})
        assert_allclose(pos, [0, 0])

    def test_origin_3d(self):
        defn = origin(dim=3)
        pos = defn.resolve({})
        assert_allclose(pos, [0, 0, 0])


class TestCartesianOffset:
    def test_simple_offset(self):
        positions = {"A": vec(1, 2, 3)}
        defn = cartesian_offset("A", [3, -1, 2])
        assert_allclose(defn.resolve(positions), [4, 1, 5])

    def test_offset_from_origin(self):
        positions = {"O": vec(0, 0, 0)}
        defn = cartesian_offset("O", [5, 5, 5])
        assert_allclose(defn.resolve(positions), [5, 5, 5])

    def test_2d_offset(self):
        positions = {"A": vec(1, 2)}
        defn = cartesian_offset("A", [3, 4])
        assert_allclose(defn.resolve(positions), [4, 6])


class TestMagnitudeDirection:
    def test_unit_direction(self):
        positions = {"O": vec(0, 0, 0)}
        defn = magnitude_direction("O", 5.0, [1, 0, 0])
        assert_allclose(defn.resolve(positions), [5, 0, 0])

    def test_diagonal_direction(self):
        positions = {"O": vec(0, 0, 0)}
        defn = magnitude_direction("O", np.sqrt(2), [1, 1, 0])
        result = defn.resolve(positions)
        assert_allclose(result, [1, 1, 0], atol=1e-10)

    def test_from_nonorigin(self):
        positions = {"A": vec(1, 0, 0)}
        defn = magnitude_direction("A", 3.0, [0, 1, 0])
        assert_allclose(defn.resolve(positions), [1, 3, 0])


class TestMagnitudePolar:
    def test_0_degrees(self):
        positions = {"O": vec(0, 0, 0)}
        defn = magnitude_polar("O", 5.0, 0)
        result = defn.resolve(positions)
        assert_allclose(result[:2], [5, 0], atol=1e-10)

    def test_90_degrees(self):
        positions = {"O": vec(0, 0, 0)}
        defn = magnitude_polar("O", 5.0, 90)
        result = defn.resolve(positions)
        assert_allclose(result[:2], [0, 5], atol=1e-10)

    def test_45_degrees(self):
        positions = {"O": vec(0, 0)}
        defn = magnitude_polar("O", np.sqrt(2), 45)
        result = defn.resolve(positions)
        assert_allclose(result, [1, 1], atol=1e-10)


class TestMagnitudeSpherical:
    def test_along_z(self):
        positions = {"O": vec(0, 0, 0)}
        defn = magnitude_spherical("O", 5.0, 0, 0)
        result = defn.resolve(positions)
        assert_allclose(result, [0, 0, 5], atol=1e-10)

    def test_along_x(self):
        positions = {"O": vec(0, 0, 0)}
        defn = magnitude_spherical("O", 5.0, 90, 0)
        result = defn.resolve(positions)
        assert_allclose(result, [5, 0, 0], atol=1e-10)

    def test_along_y(self):
        positions = {"O": vec(0, 0, 0)}
        defn = magnitude_spherical("O", 5.0, 90, 90)
        result = defn.resolve(positions)
        assert_allclose(result, [0, 5, 0], atol=1e-10)


class TestMidpoint:
    def test_two_points(self):
        positions = {"A": vec(0, 0, 0), "B": vec(4, 4, 4)}
        defn = midpoint("A", "B")
        assert_allclose(defn.resolve(positions), [2, 2, 2])

    def test_three_points(self):
        positions = {"A": vec(0, 0, 0), "B": vec(3, 0, 0), "C": vec(0, 3, 0)}
        defn = midpoint("A", "B", "C")
        assert_allclose(defn.resolve(positions), [1, 1, 0])


class TestWeightedCentroid:
    def test_equal_weights(self):
        positions = {"A": vec(0, 0, 0), "B": vec(4, 4, 4)}
        defn = weighted_centroid(["A", "B"], [1, 1])
        assert_allclose(defn.resolve(positions), [2, 2, 2])

    def test_unequal_weights(self):
        positions = {"A": vec(0, 0, 0), "B": vec(4, 0, 0)}
        defn = weighted_centroid(["A", "B"], [3, 1])
        assert_allclose(defn.resolve(positions), [1, 0, 0])


class TestProjection:
    def test_project_onto_x_axis(self):
        positions = {"C": vec(3, 4, 0), "A": vec(0, 0, 0), "B": vec(1, 0, 0)}
        defn = projection("C", "A", "B")
        assert_allclose(defn.resolve(positions), [3, 0, 0])

    def test_project_onto_diagonal(self):
        positions = {"C": vec(1, 0, 0), "A": vec(0, 0, 0), "B": vec(1, 1, 0)}
        defn = projection("C", "A", "B")
        assert_allclose(defn.resolve(positions), [0.5, 0.5, 0], atol=1e-10)


class TestSerialization:
    def test_round_trip(self):
        defn = cartesian_offset("A", [1.0, 2.0, 3.0])
        d = defn.to_dict()
        restored = PointDefinition.from_dict(d)
        assert restored.def_type == defn.def_type
        assert restored.anchors == defn.anchors
        assert restored.params == defn.params
