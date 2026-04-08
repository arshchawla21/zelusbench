"""Evaluation: parsing, scoring, and diagnostic reports."""

from .parser import parse_coordinates, parse_distance, parse_boolean, parse_model_response
from .scorer import QueryScore, ScoreTier, score_query, score_scenario
from .reports import DiagnosticProfile, build_diagnostic_profile

__all__ = [
    "parse_coordinates", "parse_distance", "parse_boolean", "parse_model_response",
    "QueryScore", "ScoreTier", "score_query", "score_scenario",
    "DiagnosticProfile", "build_diagnostic_profile",
]
