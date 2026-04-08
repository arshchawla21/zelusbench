"""Core geometric engine — source of truth for all computations."""

from .vectors import Vec, vec, magnitude, normalize, distance, dot, cross
from .point import (
    PointDefinition, PointDefType,
    origin, cartesian_offset, magnitude_direction, magnitude_polar,
    magnitude_spherical, midpoint, weighted_centroid, projection,
)
from .space import Space
from .transforms import (
    Transform, TransformType,
    rotation, translation, reflection, scaling, invalidation, frame_shift,
)

__all__ = [
    "Vec", "vec", "magnitude", "normalize", "distance", "dot", "cross",
    "PointDefinition", "PointDefType",
    "origin", "cartesian_offset", "magnitude_direction", "magnitude_polar",
    "magnitude_spherical", "midpoint", "weighted_centroid", "projection",
    "Space",
    "Transform", "TransformType",
    "rotation", "translation", "reflection", "scaling", "invalidation", "frame_shift",
]
