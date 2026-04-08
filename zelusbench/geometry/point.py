"""Point definitions — every point is defined relative to existing points."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np

from .vectors import Vec, vec, normalize, polar_to_cartesian, spherical_to_cartesian, dot, project_onto_line


class PointDefType(Enum):
    ORIGIN = auto()
    CARTESIAN_OFFSET = auto()
    MAGNITUDE_DIRECTION = auto()
    MAGNITUDE_POLAR = auto()
    MAGNITUDE_SPHERICAL = auto()
    MIDPOINT = auto()
    WEIGHTED_CENTROID = auto()
    PROJECTION = auto()


@dataclass
class PointDefinition:
    """Base for all point definition types."""
    def_type: PointDefType
    anchors: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)

    def resolve(self, positions: dict[str, Vec]) -> Vec:
        match self.def_type:
            case PointDefType.ORIGIN:
                dim = self.params.get("dim", 3)
                return np.zeros(dim, dtype=np.float64)

            case PointDefType.CARTESIAN_OFFSET:
                anchor = positions[self.anchors[0]]
                offset = np.array(self.params["offset"], dtype=np.float64)
                return anchor + offset

            case PointDefType.MAGNITUDE_DIRECTION:
                anchor = positions[self.anchors[0]]
                mag = self.params["magnitude"]
                direction = np.array(self.params["direction"], dtype=np.float64)
                return anchor + mag * normalize(direction)

            case PointDefType.MAGNITUDE_POLAR:
                anchor = positions[self.anchors[0]]
                mag = self.params["magnitude"]
                angle = self.params["angle_deg"]
                offset_2d = polar_to_cartesian(mag, angle)
                # Embed in higher dims if needed
                result = np.zeros_like(anchor)
                result[:2] = offset_2d
                return anchor + result

            case PointDefType.MAGNITUDE_SPHERICAL:
                anchor = positions[self.anchors[0]]
                mag = self.params["magnitude"]
                theta = self.params["theta_deg"]
                phi = self.params["phi_deg"]
                offset_3d = spherical_to_cartesian(mag, theta, phi)
                result = np.zeros_like(anchor)
                result[:3] = offset_3d
                return anchor + result

            case PointDefType.MIDPOINT:
                pts = [positions[a] for a in self.anchors]
                return np.mean(pts, axis=0)

            case PointDefType.WEIGHTED_CENTROID:
                pts = [positions[a] for a in self.anchors]
                weights = np.array(self.params["weights"], dtype=np.float64)
                weighted = sum(w * p for w, p in zip(weights, pts))
                return weighted / weights.sum()

            case PointDefType.PROJECTION:
                # Project anchors[0] onto line defined by anchors[1], anchors[2]
                point = positions[self.anchors[0]]
                line_a = positions[self.anchors[1]]
                line_b = positions[self.anchors[2]]
                return project_onto_line(point, line_a, line_b)

            case _:
                raise ValueError(f"Unknown definition type: {self.def_type}")

    def dependency_names(self) -> list[str]:
        return list(self.anchors)

    def to_dict(self) -> dict:
        return {
            "def_type": self.def_type.name,
            "anchors": self.anchors,
            "params": {k: v.tolist() if isinstance(v, np.ndarray) else v
                       for k, v in self.params.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> PointDefinition:
        return cls(
            def_type=PointDefType[d["def_type"]],
            anchors=d["anchors"],
            params=d["params"],
        )


# --- Convenience constructors ---

def origin(dim: int = 3) -> PointDefinition:
    return PointDefinition(PointDefType.ORIGIN, params={"dim": dim})


def cartesian_offset(anchor: str, offset: list[float]) -> PointDefinition:
    return PointDefinition(PointDefType.CARTESIAN_OFFSET, anchors=[anchor], params={"offset": offset})


def magnitude_direction(anchor: str, magnitude: float, direction: list[float]) -> PointDefinition:
    return PointDefinition(PointDefType.MAGNITUDE_DIRECTION, anchors=[anchor],
                           params={"magnitude": magnitude, "direction": direction})


def magnitude_polar(anchor: str, magnitude: float, angle_deg: float) -> PointDefinition:
    return PointDefinition(PointDefType.MAGNITUDE_POLAR, anchors=[anchor],
                           params={"magnitude": magnitude, "angle_deg": angle_deg})


def magnitude_spherical(anchor: str, magnitude: float, theta_deg: float, phi_deg: float) -> PointDefinition:
    return PointDefinition(PointDefType.MAGNITUDE_SPHERICAL, anchors=[anchor],
                           params={"magnitude": magnitude, "theta_deg": theta_deg, "phi_deg": phi_deg})


def midpoint(*anchors: str) -> PointDefinition:
    return PointDefinition(PointDefType.MIDPOINT, anchors=list(anchors))


def weighted_centroid(anchors: list[str], weights: list[float]) -> PointDefinition:
    return PointDefinition(PointDefType.WEIGHTED_CENTROID, anchors=anchors, params={"weights": weights})


def projection(point: str, line_a: str, line_b: str) -> PointDefinition:
    return PointDefinition(PointDefType.PROJECTION, anchors=[point, line_a, line_b])
