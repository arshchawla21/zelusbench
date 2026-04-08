"""Vector math utilities for geometric operations."""

from __future__ import annotations

import math
import numpy as np
from numpy.typing import NDArray

Vec = NDArray[np.float64]


def vec(*components: float) -> Vec:
    return np.array(components, dtype=np.float64)


def magnitude(v: Vec) -> float:
    return float(np.linalg.norm(v))


def normalize(v: Vec) -> Vec:
    m = magnitude(v)
    if m < 1e-12:
        return np.zeros_like(v)
    return v / m


def dot(a: Vec, b: Vec) -> float:
    return float(np.dot(a, b))


def cross(a: Vec, b: Vec) -> Vec:
    """Cross product (3D only)."""
    return np.cross(a, b)


def distance(a: Vec, b: Vec) -> float:
    return float(np.linalg.norm(a - b))


def deg2rad(degrees: float) -> float:
    return math.radians(degrees)


def rad2deg(radians: float) -> float:
    return math.degrees(radians)


def polar_to_cartesian(mag: float, angle_deg: float) -> Vec:
    """2D: magnitude + angle (degrees) -> (x, y)."""
    theta = deg2rad(angle_deg)
    return vec(mag * math.cos(theta), mag * math.sin(theta))


def spherical_to_cartesian(mag: float, theta_deg: float, phi_deg: float) -> Vec:
    """3D: magnitude + spherical angles -> (x, y, z).
    theta = polar angle from +z, phi = azimuthal angle from +x in xy-plane.
    """
    theta = deg2rad(theta_deg)
    phi = deg2rad(phi_deg)
    return vec(
        mag * math.sin(theta) * math.cos(phi),
        mag * math.sin(theta) * math.sin(phi),
        mag * math.cos(theta),
    )


def rotation_matrix_2d(angle_deg: float) -> NDArray[np.float64]:
    theta = deg2rad(angle_deg)
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def rotation_matrix_3d(axis: Vec, angle_deg: float) -> NDArray[np.float64]:
    """Rodrigues' rotation formula as a 3x3 matrix."""
    theta = deg2rad(angle_deg)
    axis = normalize(axis)
    c, s = math.cos(theta), math.sin(theta)
    ux, uy, uz = axis
    return np.array([
        [c + ux * ux * (1 - c),      ux * uy * (1 - c) - uz * s, ux * uz * (1 - c) + uy * s],
        [uy * ux * (1 - c) + uz * s,  c + uy * uy * (1 - c),      uy * uz * (1 - c) - ux * s],
        [uz * ux * (1 - c) - uy * s,  uz * uy * (1 - c) + ux * s, c + uz * uz * (1 - c)],
    ], dtype=np.float64)


def rotate_point(point: Vec, center: Vec, angle_deg: float, axis: Vec | None = None) -> Vec:
    """Rotate a point around a center. 2D if len==2, else 3D with axis."""
    relative = point - center
    if len(point) == 2:
        R = rotation_matrix_2d(angle_deg)
    else:
        if axis is None:
            axis = vec(0, 0, 1)
        R = rotation_matrix_3d(axis, angle_deg)
        if len(relative) > 3:
            rotated = R @ relative[:3]
            result = relative.copy()
            result[:3] = rotated
            return center + result
    return center + R @ relative


def reflect_point(point: Vec, plane_normal: Vec, plane_point: Vec) -> Vec:
    """Reflect a point across a plane (3D) or line (2D)."""
    n = normalize(plane_normal)
    v = point - plane_point
    return point - 2 * dot(v, n) * n


def project_onto_line(point: Vec, line_a: Vec, line_b: Vec) -> Vec:
    """Project point onto line defined by two points."""
    ab = line_b - line_a
    ap = point - line_a
    t = dot(ap, ab) / max(dot(ab, ab), 1e-12)
    return line_a + t * ab