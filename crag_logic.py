"""Pure-Python CRAG decision logic (no third-party dependencies).

Kept separate from the LangGraph wiring so it can be unit-tested on its own.
"""
import re
from typing import List, Literal, Optional, Sequence

Confidence = Literal["correct", "incorrect", "ambiguous"]


def decide_confidence(scores: Sequence[float], upper: float, lower: float) -> Confidence:
    """Paper's action trigger.

    correct   : at least one document scores above the upper threshold
    incorrect : every document scores below the lower threshold
    ambiguous : everything else
    """
    if not scores:
        return "incorrect"
    if max(scores) > upper:
        return "correct"
    if all(s < lower for s in scores):
        return "incorrect"
    return "ambiguous"


def split_into_strips(text: str, sentences_per_strip: int = 2) -> List[str]:
    """Decompose a document into short 'knowledge strips' (a few sentences each)."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    return [
        " ".join(sentences[i : i + sentences_per_strip])
        for i in range(0, len(sentences), sentences_per_strip)
    ]


def filter_strips(
    strips: Sequence[str],
    scores: Sequence[float],
    threshold: float,
    top_k: Optional[int] = None,
) -> List[str]:
    """Keep strips scoring >= threshold (optionally only the best top_k), in original order."""
    if len(strips) != len(scores):
        raise ValueError("strips and scores must have the same length")
    keep = [i for i, s in enumerate(scores) if s >= threshold]
    if top_k is not None and len(keep) > top_k:
        keep = sorted(sorted(keep, key=lambda i: scores[i], reverse=True)[:top_k])
    return [strips[i] for i in keep]


def recompose(strips: Sequence[str]) -> str:
    """Recompose the surviving strips, in order, into one knowledge passage."""
    return "\n".join(strips)


def parse_score(text: str) -> float:
    """Pull the first number out of an LLM reply and clamp it to [0, 1]. Defaults to 0.0."""
    m = re.search(r"\d*\.?\d+", text or "")
    if not m:
        return 0.0
    return min(1.0, max(0.0, float(m.group())))


def sigmoid(x: float) -> float:
    import math
    return 1.0 / (1.0 + math.exp(-x))
