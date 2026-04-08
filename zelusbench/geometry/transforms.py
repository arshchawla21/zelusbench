"""Geometric transforms that mutate Space state."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

import numpy as np

from .space import Space
from .vectors import Vec, vec, rotate_point, reflect_point, normalize


class TransformType(Enum):
    ROTATION = auto()
    TRANSLATION = auto()
    REFLECTION = auto()
    SCALING = auto()
    INVALIDATION = auto()
    FRAME_SHIFT = auto()


@dataclass
class Transform:
    """A transform event that mutates the space."""
    transform_type: TransformType
    params: dict[str, Any]

    def apply(self, space: Space):
        match self.transform_type:
            case TransformType.ROTATION:
                apply_rotation(space, **self.params)
            case TransformType.TRANSLATION:
                apply_translation(space, **self.params)
            case TransformType.REFLECTION:
                apply_reflection(space, **self.params)
            case TransformType.SCALING:
                apply_scaling(space, **self.params)
            case TransformType.INVALIDATION:
                apply_invalidation(space, **self.params)
            case TransformType.FRAME_SHIFT:
                apply_frame_shift(space, **self.params)

    def to_dict(self) -> dict:
        result = {"transform_type": self.transform_type.name, "params": {}}
        for k, v in self.params.items():
            if isinstance(v, np.ndarray):
                result["params"][k] = v.tolist()
            elif hasattr(v, 'to_dict'):
                result["params"][k] = v.to_dict()
            else:
                result["params"][k] = v
        return result

    @classmethod
    def from_dict(cls, d: dict) -> Transform:
        return cls(
            transform_type=TransformType[d["transform_type"]],
            params=d["params"],
        )


def apply_rotation(space: Space, points: list[str], center: list[float],
                   angle_deg: float, axis: list[float] | None = None):
    center_v = np.array(center, dtype=np.float64)
    axis_v = np.array(axis, dtype=np.float64) if axis is not None else None
    for name in points:
        pos = space.get_position(name)
        new_pos = rotate_point(pos, center_v, angle_deg, axis_v)
        space.set_position_direct(name, new_pos)


def apply_translation(space: Space, points: list[str], displacement: list[float]):
    disp = np.array(displacement, dtype=np.float64)
    for name in points:
        pos = space.get_position(name)
        space.set_position_direct(name, pos + disp)


def apply_reflection(space: Space, points: list[str],
                     plane_normal: list[float], plane_point: list[float]):
    normal = np.array(plane_normal, dtype=np.float64)
    pt = np.array(plane_point, dtype=np.float64)
    for name in points:
        pos = space.get_position(name)
        new_pos = reflect_point(pos, normal, pt)
        space.set_position_direct(name, new_pos)


def apply_scaling(space: Space, points: list[str],
                  center: list[float], factor: float):
    center_v = np.array(center, dtype=np.float64)
    for name in points:
        pos = space.get_position(name)
        new_pos = center_v + factor * (pos - center_v)
        space.set_position_direct(name, new_pos)


def apply_invalidation(space: Space, point: str, new_definition: dict):
    from .point import PointDefinition
    if isinstance(new_definition, dict):
        defn = PointDefinition.from_dict(new_definition)
    else:
        defn = new_definition
    space.define_point(point, defn)


def apply_frame_shift(space: Space, new_origin: str):
    """Shift all positions so new_origin becomes the origin."""
    origin_pos = space.get_position(new_origin)
    # Compute all new positions first, then apply without intermediate propagation
    new_positions = {}
    for name in space.point_names:
        pos = space.get_position(name)
        new_positions[name] = pos - origin_pos
    # Apply all at once (set _positions directly to avoid cascading re-resolves)
    for name, pos in new_positions.items():
        space._positions[name] = pos


# --- Convenience constructors ---

def rotation(points: list[str], center: list[float], angle_deg: float,
             axis: list[float] | None = None) -> Transform:
    params = {"points": points, "center": center, "angle_deg": angle_deg}
    if axis is not None:
        params["axis"] = axis
    return Transform(TransformType.ROTATION, params)


def translation(points: list[str], displacement: list[float]) -> Transform:
    return Transform(TransformType.TRANSLATION, {"points": points, "displacement": displacement})


def reflection(points: list[str], plane_normal: list[float],
               plane_point: list[float]) -> Transform:
    return Transform(TransformType.REFLECTION,
                     {"points": points, "plane_normal": plane_normal, "plane_point": plane_point})


def scaling(points: list[str], center: list[float], factor: float) -> Transform:
    return Transform(TransformType.SCALING,
                     {"points": points, "center": center, "factor": factor})


def invalidation(point: str, new_definition: dict) -> Transform:
    return Transform(TransformType.INVALIDATION,
                     {"point": point, "new_definition": new_definition})


def frame_shift(new_origin: str) -> Transform:
    return Transform(TransformType.FRAME_SHIFT, {"new_origin": new_origin})
