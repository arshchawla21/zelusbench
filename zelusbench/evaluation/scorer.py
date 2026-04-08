"""Score predicted answers against ground truth."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

from ..geometry.vectors import Vec

EPSILON = 1.0  # Tolerance for near-origin scoring


class ScoreTier(Enum):
    EXACT = auto()       # < 1% relative error -> 1.0
    CLOSE = auto()       # < 5% relative error -> 0.7
    APPROXIMATE = auto() # < 15% relative error -> 0.3
    WRONG = auto()       # >= 15% relative error -> 0.0
    REFUSED = auto()     # Unparseable -> 0.0


TIER_SCORES = {
    ScoreTier.EXACT: 1.0,
    ScoreTier.CLOSE: 0.7,
    ScoreTier.APPROXIMATE: 0.3,
    ScoreTier.WRONG: 0.0,
    ScoreTier.REFUSED: 0.0,
}


@dataclass
class QueryScore:
    query_id: str
    query_type: str
    score: float
    tier: ScoreTier
    relative_error: float | None
    predicted: object
    ground_truth: object
    chain_depth: int
    query_index: int

    def to_dict(self) -> dict:
        pred = self.predicted
        truth = self.ground_truth
        if isinstance(pred, np.ndarray):
            pred = pred.tolist()
        if isinstance(truth, np.ndarray):
            truth = truth.tolist()
        return {
            "query_id": self.query_id,
            "query_type": self.query_type,
            "score": self.score,
            "tier": self.tier.name,
            "relative_error": self.relative_error,
            "predicted": pred,
            "ground_truth": truth,
            "chain_depth": self.chain_depth,
            "query_index": self.query_index,
        }


def relative_error_vec(predicted: Vec, truth: Vec) -> float:
    """Compute relative error for vector predictions."""
    error = float(np.linalg.norm(predicted - truth))
    denom = max(float(np.linalg.norm(truth)), EPSILON)
    return error / denom


def relative_error_scalar(predicted: float, truth: float) -> float:
    """Compute relative error for scalar predictions."""
    error = abs(predicted - truth)
    denom = max(abs(truth), EPSILON)
    return error / denom


def tier_from_error(rel_error: float) -> ScoreTier:
    if rel_error < 0.01:
        return ScoreTier.EXACT
    elif rel_error < 0.05:
        return ScoreTier.CLOSE
    elif rel_error < 0.15:
        return ScoreTier.APPROXIMATE
    else:
        return ScoreTier.WRONG


def score_query(query: dict, parsed: dict) -> QueryScore:
    """Score a single query given parsed model output.

    Args:
        query: Query dict with ground_truth, query_type, etc.
        parsed: Output of parse_model_response with parsed_value, parse_success.
    """
    query_id = query["query_id"]
    query_type = query["query_type"]
    ground_truth = query["ground_truth"]
    chain_depth = query.get("chain_depth", 0)
    query_index = query.get("query_index", 0)

    if not parsed["parse_success"] or parsed["parsed_value"] is None:
        return QueryScore(
            query_id=query_id, query_type=query_type,
            score=0.0, tier=ScoreTier.REFUSED,
            relative_error=None,
            predicted=None, ground_truth=ground_truth,
            chain_depth=chain_depth, query_index=query_index,
        )

    predicted = parsed["parsed_value"]

    match query_type:
        case "POSITION":
            truth_vec = np.array(ground_truth, dtype=np.float64)
            pred_vec = np.array(predicted, dtype=np.float64) if not isinstance(predicted, np.ndarray) else predicted
            rel_err = relative_error_vec(pred_vec, truth_vec)
            tier = tier_from_error(rel_err)
            return QueryScore(
                query_id=query_id, query_type=query_type,
                score=TIER_SCORES[tier], tier=tier,
                relative_error=rel_err,
                predicted=pred_vec, ground_truth=truth_vec,
                chain_depth=chain_depth, query_index=query_index,
            )

        case "DISTANCE":
            truth_val = float(ground_truth)
            pred_val = float(predicted)
            rel_err = relative_error_scalar(pred_val, truth_val)
            tier = tier_from_error(rel_err)
            return QueryScore(
                query_id=query_id, query_type=query_type,
                score=TIER_SCORES[tier], tier=tier,
                relative_error=rel_err,
                predicted=pred_val, ground_truth=truth_val,
                chain_depth=chain_depth, query_index=query_index,
            )

        case "BOOLEAN":
            is_correct = str(predicted).strip().upper() == str(ground_truth).strip().upper()
            return QueryScore(
                query_id=query_id, query_type=query_type,
                score=1.0 if is_correct else 0.0,
                tier=ScoreTier.EXACT if is_correct else ScoreTier.WRONG,
                relative_error=0.0 if is_correct else 1.0,
                predicted=predicted, ground_truth=ground_truth,
                chain_depth=chain_depth, query_index=query_index,
            )

        case _:
            return QueryScore(
                query_id=query_id, query_type=query_type,
                score=0.0, tier=ScoreTier.REFUSED,
                relative_error=None,
                predicted=predicted, ground_truth=ground_truth,
                chain_depth=chain_depth, query_index=query_index,
            )


def score_scenario(queries: list[dict], parsed_responses: list[dict]) -> list[QueryScore]:
    """Score all queries in a scenario."""
    scores = []
    for query, parsed in zip(queries, parsed_responses):
        scores.append(score_query(query, parsed))
    return scores


def score_response(response_text: str, scenario) -> list[QueryScore]:
    """Score a full model response against a scenario's queries.

    Splits the response by [Answer q_ID] tags first (structured format).
    Falls back to [Query q_ID] tags, then treats the entire response as
    one block per query.
    """
    from .parser import parse_model_response

    query_dicts = [q.to_dict() for q in scenario.queries]

    # Try splitting by [Answer q_ID] tags first
    answer_parts = re.split(r'\[Answer\s+q_\d+\]', response_text)
    if len(answer_parts) > 1:
        answer_parts = answer_parts[1:]

    # Fall back to [Query q_ID] tags
    if len(answer_parts) != len(query_dicts):
        answer_parts = re.split(r'\[Query\s+q_\d+\]', response_text)
        if len(answer_parts) > 1:
            answer_parts = answer_parts[1:]

    # If neither split works, give the full text to each parser
    if len(answer_parts) != len(query_dicts):
        answer_parts = [response_text] * len(query_dicts)

    return [score_query(qd, parse_model_response(rp, qd))
            for qd, rp in zip(query_dicts, answer_parts)]
