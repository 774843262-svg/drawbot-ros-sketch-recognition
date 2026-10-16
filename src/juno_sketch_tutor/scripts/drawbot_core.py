#!/usr/bin/env python3
"""ROS-independent command, state and prediction logic for DrawBot."""

from dataclasses import dataclass
import re
from typing import Optional, Sequence

import numpy as np


COMMAND_PATTERNS = {
    "stop": [r"\bstop\b", r"\bquit\b", r"\bexit\b", r"\bend(?: the)? game\b"],
    "ready": [
        r"\bready\b",
        r"\bdone\b",
        r"\bi(?: am|'m) done\b",
        r"\bi finished\b",
        r"\bfinish drawing\b",
        r"\bfinished(?: drawing)?\b",
        r"\brecognize(?: it)?\b",
        r"\bguess(?: my drawing)?\b",
        r"\bwhat did i draw\b",
        r"\blook at my picture\b",
    ],
    "repeat": [r"\brepeat\b", r"\bsay it again\b", r"\bone more time\b", r"\bagain\b"],
    "start": [
        r"\bstart(?: the)? game\b",
        r"\bstart\b",
        r"\bbegin(?: the game)?\b",
        r"\blet'?s play\b",
        r"\bplay game\b",
    ],
}


def parse_voice_command(text: str) -> Optional[str]:
    """Map a transcript to one game command; safety-oriented stop wins ties."""
    if not isinstance(text, str):
        return None
    normalized = text.lower().strip()
    for command in ("stop", "ready", "repeat", "start"):
        if any(re.search(pattern, normalized) for pattern in COMMAND_PATTERNS[command]):
            return command
    return None


def build_prediction_payload(
    probabilities: Sequence[float], classes: Sequence[str], top_k: int = 3
) -> dict:
    """Create a stable top-k JSON-ready payload from model probabilities."""
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 1 or len(values) != len(classes):
        raise ValueError("probabilities must be one-dimensional and match classes")
    if not np.all(np.isfinite(values)):
        raise ValueError("probabilities must be finite")
    if len(classes) == 0:
        raise ValueError("classes must not be empty")
    k = max(1, min(int(top_k), len(classes)))
    indices = values.argsort()[-k:][::-1]
    top = [
        {"label": str(classes[int(index)]), "confidence": round(float(values[index]), 4)}
        for index in indices
    ]
    return {"top1": top[0], "top3": top}


def confidence_message(payload: dict, high: float = 0.75, medium: float = 0.45) -> str:
    """Turn one prediction payload into the robot's uncertainty-aware response."""
    top = payload["top3"]
    label = payload["top1"]["label"]
    confidence = float(payload["top1"]["confidence"])
    if confidence >= high:
        return f"I think this is a {label}."
    if confidence >= medium:
        options = ", ".join(item["label"] for item in top)
        return f"I am not very sure. It may be {options}."
    return "I cannot recognize it clearly. Please draw it again with thicker lines."


@dataclass
class SketchGameState:
    """Pure state machine shared by the ROS node and offline tests."""

    started: bool = False

    def handle(self, command: str) -> tuple[str, bool]:
        normalized = command.strip().lower()
        if normalized == "start":
            self.started = True
            return (
                "Game started. Please draw a simple picture, such as apple or cat. "
                "When you finish, send ready.",
                False,
            )
        if normalized in {"ready", "repeat"}:
            if not self.started:
                return "Please start the game first.", False
            if normalized == "ready":
                return "Okay. I will recognize your drawing now.", True
            return "Please show me your drawing again.", True
        if normalized == "stop":
            self.started = False
            return "Game stopped. Thank you for playing.", False
        return "Unknown command. Please send start, ready, repeat, or stop.", False
