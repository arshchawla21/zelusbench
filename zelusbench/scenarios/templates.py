"""Natural language templates for rendering geometric scenarios."""

from __future__ import annotations

import numpy as np

from ..geometry.point import PointDefinition, PointDefType
from ..geometry.transforms import Transform, TransformType
from ..geometry.vectors import rad2deg


def _fmt_vec(v: np.ndarray) -> str:
    """Format a vector as a readable tuple."""
    parts = [f"{x:.1f}" for x in v]
    return f"({', '.join(parts)})"


def _fmt_coord_labels(dim: int) -> str:
    labels = ["x", "y", "z", "w"][:dim]
    return ", ".join(labels)


def render_system_prompt(dim: int) -> str:
    coord_labels = _fmt_coord_labels(dim)
    return (
        f"Process the following {dim}D spatial reasoning scenario. "
        f"Statements are chronological — propagate all transformations before answering.\n"
        f"Format: [Answer q_ID] value — e.g. [Answer q_001] ({', '.join(['0.0'] * dim)}) or [Answer q_002] 5.385 or [Answer q_003] B"
    )


def render_point_definition(name: str, defn: PointDefinition) -> str:
    """Render a point definition as a natural language statement."""
    match defn.def_type:
        case PointDefType.ORIGIN:
            return f"Point O is at the origin."

        case PointDefType.CARTESIAN_OFFSET:
            anchor = defn.anchors[0]
            offset = defn.params["offset"]
            return f"Point {name} is at offset {_fmt_vec(np.array(offset))} from Point {anchor}."

        case PointDefType.MAGNITUDE_DIRECTION:
            anchor = defn.anchors[0]
            mag = defn.params["magnitude"]
            direction = defn.params["direction"]
            return (f"Point {name} is {mag:.1f} units from Point {anchor} "
                    f"in the direction {_fmt_vec(np.array(direction))}.")

        case PointDefType.MAGNITUDE_POLAR:
            anchor = defn.anchors[0]
            mag = defn.params["magnitude"]
            angle = defn.params["angle_deg"]
            return (f"Point {name} is {mag:.1f} units from Point {anchor} "
                    f"at angle {angle:.0f} degrees.")

        case PointDefType.MAGNITUDE_SPHERICAL:
            anchor = defn.anchors[0]
            mag = defn.params["magnitude"]
            theta = defn.params["theta_deg"]
            phi = defn.params["phi_deg"]
            return (f"Point {name} is {mag:.1f} units from Point {anchor} "
                    f"at polar angle {theta:.0f} degrees and azimuthal angle {phi:.0f} degrees.")

        case PointDefType.MIDPOINT:
            anchors = " and ".join(f"Point {a}" for a in defn.anchors)
            return f"Point {name} is the midpoint of {anchors}."

        case PointDefType.WEIGHTED_CENTROID:
            weights = defn.params["weights"]
            parts = [f"Point {a} (weight {w:.1f})" for a, w in zip(defn.anchors, weights)]
            return f"Point {name} is the weighted centroid of {', '.join(parts)}."

        case PointDefType.PROJECTION:
            return (f"Point {name} is the projection of Point {defn.anchors[0]} "
                    f"onto the line through Point {defn.anchors[1]} and Point {defn.anchors[2]}.")

        case _:
            return f"Point {name} is defined."


def render_transform(transform: Transform) -> str:
    """Render a transform event as natural language."""
    p = transform.params

    match transform.transform_type:
        case TransformType.ROTATION:
            points_str = ", ".join(f"Point {n}" for n in p["points"])
            center = _fmt_vec(np.array(p["center"]))
            angle = p["angle_deg"]
            axis_str = ""
            if "axis" in p and p["axis"] is not None:
                axis_str = f" around the axis {_fmt_vec(np.array(p['axis']))}"
            return (f"Rotate {points_str} by {angle:.0f} degrees "
                    f"around center {center}{axis_str}.")

        case TransformType.TRANSLATION:
            points_str = ", ".join(f"Point {n}" for n in p["points"])
            disp = _fmt_vec(np.array(p["displacement"]))
            return f"Translate {points_str} by displacement {disp}."

        case TransformType.REFLECTION:
            points_str = ", ".join(f"Point {n}" for n in p["points"])
            normal = _fmt_vec(np.array(p["plane_normal"]))
            pt = _fmt_vec(np.array(p["plane_point"]))
            return (f"Reflect {points_str} across the plane with normal {normal} "
                    f"passing through {pt}.")

        case TransformType.SCALING:
            points_str = ", ".join(f"Point {n}" for n in p["points"])
            center = _fmt_vec(np.array(p["center"]))
            factor = p["factor"]
            return f"Scale {points_str} by factor {factor:.1f} relative to center {center}."

        case TransformType.INVALIDATION:
            point = p["point"]
            return f"Point {point} is now redefined. Forget its old position and update accordingly."

        case TransformType.FRAME_SHIFT:
            new_origin = p["new_origin"]
            return f"From now on, describe everything relative to Point {new_origin} as the new origin."

        case _:
            return "A transformation has occurred."


def render_query_position(point_name: str, query_id: str, dim: int) -> str:
    coord_labels = _fmt_coord_labels(dim)
    return f"[Query {query_id}] Position of {point_name}? ({coord_labels})"


def render_query_distance(point_a: str, point_b: str, query_id: str) -> str:
    return f"[Query {query_id}] Distance from {point_a} to {point_b}?"


def render_query_boolean(point_target: str, point_a: str, point_b: str, query_id: str) -> str:
    return f"[Query {query_id}] Is {point_target} closer to {point_a} or {point_b}?"
