"""Extract coordinates, distances, and boolean answers from model output."""

from __future__ import annotations

import re

import numpy as np

from ..geometry.vectors import Vec


def parse_coordinates(text: str, dim: int) -> Vec | None:
    """Extract a coordinate vector from model output text.

    Uses the LAST match to skip over intermediate reasoning/calculations.

    Looks for patterns like:
    - (1.0, 2.0, 3.0)
    - [1.0, 2.0, 3.0]
    - x=1.0, y=2.0, z=3.0
    - 1.0, 2.0, 3.0
    """
    # Try parenthesized/bracketed tuple — use LAST match to skip reasoning
    pattern = r'[\(\[]\s*(-?[\d]+\.?[\d]*)\s*,\s*(-?[\d]+\.?[\d]*)'
    if dim >= 3:
        pattern += r'\s*,\s*(-?[\d]+\.?[\d]*)'
    pattern += r'\s*[\)\]]'

    matches = list(re.finditer(pattern, text))
    if matches:
        match = matches[-1]  # last match = final answer
        coords = [float(match.group(i + 1)) for i in range(dim)]
        return np.array(coords, dtype=np.float64)

    # Try x= y= z= format (search from end)
    coord_labels = ["x", "y", "z", "w"][:dim]
    values = []
    for label in coord_labels:
        pat = rf'{label}\s*[=:]\s*(-?[\d]+\.?[\d]*)'
        all_m = list(re.finditer(pat, text, re.IGNORECASE))
        if all_m:
            values.append(float(all_m[-1].group(1)))
    if len(values) == dim:
        return np.array(values, dtype=np.float64)

    return None


def parse_distance(text: str) -> float | None:
    """Extract a distance (single number) from model output."""
    # Look for common patterns
    patterns = [
        r'(?:distance|dist|result|answer)\s*(?:is|=|:)\s*(-?[\d]+\.?[\d]*)',
        r'(?:≈|≅|~)\s*(-?[\d]+\.?[\d]*)',
        r'(-?[\d]+\.?[\d]*)\s*(?:units?)',
        r'^\s*(-?[\d]+\.?[\d]*)\s*$',
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
        if m:
            return float(m.group(1))

    # Fallback: find the last number in the text
    numbers = re.findall(r'-?[\d]+\.[\d]+|-?[\d]+', text)
    if numbers:
        return float(numbers[-1])

    return None


def parse_boolean(text: str, option_a: str, option_b: str) -> str | None:
    """Extract a boolean/choice answer from model output."""
    text_lower = text.lower().strip()
    a_lower = option_a.lower()
    b_lower = option_b.lower()

    # Check for exact match first
    if text_lower == a_lower or f"point {a_lower}" in text_lower:
        return option_a
    if text_lower == b_lower or f"point {b_lower}" in text_lower:
        return option_b

    # Check which appears last (the model's final answer)
    last_a = text_lower.rfind(a_lower)
    last_b = text_lower.rfind(b_lower)

    if last_a > last_b:
        return option_a
    if last_b > last_a:
        return option_b

    return None


def _extract_answer_line(text: str, query_id: str) -> str | None:
    """Extract the text after [Answer q_ID] if present."""
    pattern = rf'\[Answer\s+{re.escape(query_id)}\]\s*(.+)'
    m = re.search(pattern, text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None


def parse_model_response(text: str, query: dict) -> dict:
    """Parse a model response for a specific query.

    Tries the structured [Answer q_ID] tag first, then falls back
    to full-text extraction (using last match to skip reasoning).

    Returns dict with 'parsed_value' and 'parse_success'.
    """
    query_type = query["query_type"]
    query_id = query.get("query_id", "")

    # Try structured answer line first
    answer_line = _extract_answer_line(text, query_id)

    match query_type:
        case "POSITION":
            dim = len(query["ground_truth"]) if isinstance(query["ground_truth"], list) else 3
            # Try answer line first, then full text
            val = None
            if answer_line:
                val = parse_coordinates(answer_line, dim)
            if val is None:
                val = parse_coordinates(text, dim)
            return {"parsed_value": val, "parse_success": val is not None}

        case "DISTANCE":
            val = None
            if answer_line:
                val = parse_distance(answer_line)
            if val is None:
                val = parse_distance(text)
            return {"parsed_value": val, "parse_success": val is not None}

        case "BOOLEAN":
            options = query["target_points"][1:]
            val = None
            if answer_line:
                val = parse_boolean(answer_line, options[0], options[1])
            if val is None:
                val = parse_boolean(text, options[0], options[1])
            return {"parsed_value": val, "parse_success": val is not None}

        case _:
            return {"parsed_value": None, "parse_success": False}
