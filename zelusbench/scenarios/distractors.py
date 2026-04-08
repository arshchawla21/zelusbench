"""Distractor injection strategies for scenario generation."""

from __future__ import annotations

import random
import string

from ..geometry.point import PointDefinition, cartesian_offset, midpoint, magnitude_direction


def generate_distractor_names(count: int, existing: set[str], rng: random.Random) -> list[str]:
    """Generate unique point names for distractors."""
    # Use Greek letters and subscripted names to make them look real
    pool = list(string.ascii_uppercase)
    pool = [c for c in pool if c not in existing and c != "O"]
    # If we need more, add numbered variants
    extras = [f"{c}{i}" for c in string.ascii_uppercase for i in range(1, 5)]
    extras = [e for e in extras if e not in existing]
    pool.extend(extras)
    rng.shuffle(pool)
    return pool[:count]


def generate_disconnected_distractors(
    count: int,
    dim: int,
    existing_names: set[str],
    rng: random.Random,
    coord_range: tuple[float, float] = (-10.0, 10.0),
) -> list[tuple[str, PointDefinition]]:
    """Generate a disconnected subgraph of distractor points.

    These points reference each other but never connect to the main graph.
    """
    if count == 0:
        return []

    names = generate_distractor_names(count, existing_names, rng)
    results: list[tuple[str, PointDefinition]] = []

    # First distractor is offset from origin (creating a separate cluster)
    offset = [rng.uniform(*coord_range) for _ in range(dim)]
    results.append((names[0], cartesian_offset("O", offset)))

    # Remaining distractors chain off each other
    for i in range(1, len(names)):
        strategy = rng.choice(["offset", "direction", "midpoint"])
        if strategy == "offset" or i < 2:
            anchor = rng.choice(names[:i])
            offset = [rng.uniform(-5, 5) for _ in range(dim)]
            results.append((names[i], cartesian_offset(anchor, offset)))
        elif strategy == "direction":
            anchor = rng.choice(names[:i])
            direction = [rng.gauss(0, 1) for _ in range(dim)]
            mag = rng.uniform(1, 8)
            results.append((names[i], magnitude_direction(anchor, mag, direction)))
        elif strategy == "midpoint" and i >= 2:
            a, b = rng.sample(names[:i], 2)
            results.append((names[i], midpoint(a, b)))

    return results


def generate_branch_distractors(
    count: int,
    dim: int,
    main_chain_names: list[str],
    existing_names: set[str],
    rng: random.Random,
    coord_range: tuple[float, float] = (-10.0, 10.0),
) -> list[tuple[str, PointDefinition]]:
    """Generate distractors that branch off the main chain but are never queried.

    These are on the dependency chain but irrelevant to queries.
    """
    if count == 0:
        return []

    names = generate_distractor_names(count, existing_names, rng)
    results: list[tuple[str, PointDefinition]] = []

    for i, name in enumerate(names):
        # Branch off a random point in the main chain
        anchor = rng.choice(main_chain_names)
        offset = [rng.uniform(-5, 5) for _ in range(dim)]
        results.append((name, cartesian_offset(anchor, offset)))

    return results


