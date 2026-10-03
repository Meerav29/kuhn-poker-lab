"""Scorer for the Laya family (https://huggingface.co/convaiinnovations/laya).

Laya's `predict(state, questions)` takes typed questions and returns an
`answers` dict. Choice answers must carry per-label probabilities; noul
answers carry the probability of "yes". See scripts/probe_laya.py.
"""
import math
from kuhn.models.scorer import Question

_PROB_KEYS = ("probs", "probabilities", "scores")


def extract_probs(answer: dict, question: Question) -> dict:
    if question.kind == "noul":
        if "noul" not in answer:
            raise KeyError(f"noul answer missing 'noul' key; keys present: {sorted(answer)}")
        p = answer["noul"]
        if isinstance(p, bool) or not isinstance(p, (int, float)):
            raise ValueError(f"expected numeric noul probability, got {p!r}")
        p = float(p)
        if math.isnan(p) or not (0.0 <= p <= 1.0):
            raise ValueError(f"noul probability must be in [0, 1] and not NaN, got {p}")
        return {"yes": p, "no": 1.0 - p}

    for key in _PROB_KEYS:
        if key in answer:
            raw = answer[key]
            break
    else:
        raise KeyError(f"choice answer has none of {_PROB_KEYS}: keys={sorted(answer)}")

    # Extract and validate probabilities for each label
    probs = {}
    for label in question.labels:
        if label not in raw:
            raise KeyError(f"choice probability missing label '{label}'; keys in probs: {sorted(raw.keys())}")
        v = float(raw[label])
        if math.isnan(v) or not math.isfinite(v) or v < 0:
            raise ValueError(f"choice probability values must be non-negative and finite; got {raw}")
        probs[label] = v

    total = sum(probs.values())
    if total <= 0:
        raise ValueError(f"choice probability total must be > 0; got {raw}")

    return {label: v / total for label, v in probs.items()}


def _to_laya_question(question: Question) -> dict:
    spec = {"type": question.kind, "instructions": question.instructions}
    if question.kind == "choice":
        spec["criteria"] = list(question.labels)
    return spec


class LayaScorer:
    def __init__(self, repo: str = "convaiinnovations/laya", subfolder: str = None,
                 name: str = "laya-en"):
        import laya  # lazy: keeps the rest of the package importable without laya

        self.scorer_id = name
        self._agent = laya.load(repo, subfolder=subfolder) if subfolder else laya.load(repo)

    def score(self, state: str, question: Question) -> dict:
        result = self._agent.predict(state, {"q": _to_laya_question(question)})
        return extract_probs(result["answers"]["q"], question)
