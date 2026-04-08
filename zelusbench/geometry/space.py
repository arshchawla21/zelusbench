"""World state: resolves all points, tracks dependency DAG."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from .point import PointDefinition, PointDefType, origin
from .vectors import Vec


@dataclass
class Space:
    """Manages a dependency DAG of points and resolves positions."""

    dim: int = 3
    _definitions: dict[str, PointDefinition] = field(default_factory=dict)
    _positions: dict[str, Vec] = field(default_factory=dict)
    _children: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    _definition_order: list[str] = field(default_factory=list)

    def __post_init__(self):
        if "O" not in self._definitions:
            self.define_point("O", origin(self.dim))

    @property
    def points(self) -> dict[str, Vec]:
        """All resolved point positions."""
        return dict(self._positions)

    @property
    def point_names(self) -> list[str]:
        """All defined point names in definition order."""
        return list(self._definition_order)

    def define_point(self, name: str, definition: PointDefinition) -> Vec:
        """Define a new point or redefine an existing one. Returns resolved position."""
        # Remove old dependency edges if redefining
        if name in self._definitions:
            old_deps = self._definitions[name].dependency_names()
            for dep in old_deps:
                self._children[dep].discard(name)
        else:
            self._definition_order.append(name)

        self._definitions[name] = definition

        # Register dependency edges
        for dep in definition.dependency_names():
            self._children[dep].add(name)

        # Resolve this point
        pos = definition.resolve(self._positions)
        self._positions[name] = pos

        # Propagate to all downstream dependents
        self._propagate(name)
        return pos

    def get_position(self, name: str) -> Vec:
        if name not in self._positions:
            raise KeyError(f"Point '{name}' not defined")
        return self._positions[name].copy()

    def get_definition(self, name: str) -> PointDefinition:
        if name not in self._definitions:
            raise KeyError(f"Point '{name}' not defined")
        return self._definitions[name]

    def has_point(self, name: str) -> bool:
        return name in self._definitions

    def set_position_direct(self, name: str, position: Vec):
        """Set position directly (used by transforms). Propagates to dependents."""
        self._positions[name] = position.copy()
        self._propagate(name)

    def get_dependents(self, name: str, recursive: bool = True) -> set[str]:
        """Get all points that depend on this point."""
        if not recursive:
            return set(self._children.get(name, set()))
        visited: set[str] = set()
        stack = list(self._children.get(name, set()))
        while stack:
            child = stack.pop()
            if child not in visited:
                visited.add(child)
                stack.extend(self._children.get(child, set()))
        return visited

    def get_dependencies(self, name: str, recursive: bool = True) -> set[str]:
        """Get all points this point depends on."""
        if name not in self._definitions:
            return set()
        direct = set(self._definitions[name].dependency_names())
        if not recursive:
            return direct
        all_deps: set[str] = set()
        stack = list(direct)
        while stack:
            dep = stack.pop()
            if dep not in all_deps:
                all_deps.add(dep)
                if dep in self._definitions:
                    stack.extend(self._definitions[dep].dependency_names())
        return all_deps

    def chain_depth(self, name: str) -> int:
        """Number of dependency hops from origin to this point."""
        if name == "O":
            return 0
        defn = self._definitions.get(name)
        if defn is None or not defn.anchors:
            return 0
        return 1 + max(self.chain_depth(a) for a in defn.anchors)

    def _propagate(self, name: str):
        """Re-resolve all downstream dependents in topological order."""
        to_update = self._topo_sort_dependents(name)
        for child in to_update:
            defn = self._definitions[child]
            self._positions[child] = defn.resolve(self._positions)

    def _topo_sort_dependents(self, name: str) -> list[str]:
        """Topological sort of all dependents of name."""
        visited: set[str] = set()
        order: list[str] = []

        def dfs(n: str):
            if n in visited:
                return
            visited.add(n)
            for child in self._children.get(n, set()):
                dfs(child)
            order.append(n)

        for child in self._children.get(name, set()):
            dfs(child)

        order.reverse()
        return order

    def to_dict(self) -> dict:
        return {
            "dim": self.dim,
            "definitions": {name: defn.to_dict() for name, defn in self._definitions.items()},
            "definition_order": self._definition_order,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Space:
        space = cls(dim=d["dim"])
        # Re-define points in order to rebuild DAG and positions
        for name in d["definition_order"]:
            if name == "O":
                continue
            defn = PointDefinition.from_dict(d["definitions"][name])
            space.define_point(name, defn)
        return space

    def copy(self) -> Space:
        """Deep copy of the space."""
        return Space.from_dict(self.to_dict())
