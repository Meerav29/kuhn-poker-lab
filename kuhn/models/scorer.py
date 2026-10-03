"""Scorer interface: the one thing every model family implements.

A scorer takes a state (prompt text) and a typed Question and returns a
probability for each of the question's labels. ModelBot is written only
against this interface.
"""
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Question:
    kind: str        # "choice" (pick a label) or "noul" (yes/no probability)
    instructions: str
    options: tuple   # ((label, action), ...) where action is 'p' or 'b'

    @property
    def labels(self) -> tuple:
        return tuple(label for label, _ in self.options)

    def key(self) -> str:
        return json.dumps([self.kind, self.instructions, [list(o) for o in self.options]])


class UniformScorer:
    scorer_id = "uniform"

    def score(self, state: str, question: Question) -> dict:
        p = 1.0 / len(question.labels)
        return {label: p for label in question.labels}


class LookupScorer:
    """Returns a fixed row per state. Used to push a known strategy
    (e.g. Nash) through the full adapter as a pipeline sanity check."""

    def __init__(self, table: dict, scorer_id: str = "lookup"):
        self._table = table
        self.scorer_id = scorer_id

    def score(self, state: str, question: Question) -> dict:
        return dict(self._table[state])


class CachedScorer:
    """Wraps a scorer with an on-disk JSON cache keyed by (state, question)."""

    def __init__(self, inner, cache_dir):
        self.inner = inner
        self.scorer_id = inner.scorer_id
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", inner.scorer_id)
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        self._path = Path(cache_dir) / f"{safe}.json"
        self._data = json.loads(self._path.read_text(encoding="utf-8")) if self._path.exists() else {}

    def score(self, state: str, question: Question) -> dict:
        key = hashlib.sha256((state + "\x00" + question.key()).encode("utf-8")).hexdigest()
        if key not in self._data:
            self._data[key] = self.inner.score(state, question)
            self._path.write_text(json.dumps(self._data), encoding="utf-8")
        return dict(self._data[key])
