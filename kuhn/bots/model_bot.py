"""ModelBot: adapts any Scorer to the lab's bot interface.

information set -> prompt text (encoder) + typed Question -> label
probabilities (scorer) -> action distribution over the legal actions.
"""
from kuhn.encoders import get_encoder
from kuhn.models.scorer import Question

LABELSETS = {
    "plain": {"open": ("check", "bet"), "facing": ("fold", "call")},
    "alt": {"open": ("wait", "raise"), "facing": ("give up", "match")},
}


def build_question(infoset: str, qtype: str = "choice", wording: str = "plain") -> Question:
    facing = infoset[1:].endswith("b")
    prefix = "The opponent bet. " if facing else ""
    if qtype == "choice":
        if wording not in LABELSETS:
            raise ValueError(f"unknown wording {wording!r}; choose from {sorted(LABELSETS)}")
        pass_label, bet_label = LABELSETS[wording]["facing" if facing else "open"]
        options = ((pass_label, "p"), (bet_label, "b"))
        instructions = f"{prefix}Do you {pass_label} or {bet_label}?"
    elif qtype == "noul":
        options = (("yes", "b"), ("no", "p"))
        instructions = f"{prefix}Should you {'call' if facing else 'bet'}?"
    else:
        raise ValueError(f"unknown qtype {qtype!r}; choose 'choice' or 'noul'")
    return Question(kind=qtype, instructions=instructions, options=options)


class ModelBot:
    def __init__(self, scorer, encoder: str = "minimal", qtype: str = "choice",
                 wording: str = "plain", name: str = None):
        self.scorer = scorer
        self._encode = get_encoder(encoder)
        self.qtype = qtype
        self.wording = wording
        self.name = name or f"{scorer.scorer_id}|{encoder}|{qtype}|{wording}"
        self._memo = {}  # at most 12 infosets: never query the model twice for the same one

    def _label_probs(self, infoset: str) -> dict:
        if infoset not in self._memo:
            question = build_question(infoset, self.qtype, self.wording)
            probs = self.scorer.score(self._encode(infoset), question)
            self._memo[infoset] = {
                action: max(probs.get(label, 0.0), 0.0) for label, action in question.options
            }
        return self._memo[infoset]

    def action_probs(self, infoset: str, legal: tuple) -> dict:
        by_action = self._label_probs(infoset)
        dist = {a: by_action.get(a, 0.0) for a in legal}
        total = sum(dist.values())
        if total <= 0:
            return {a: 1.0 / len(legal) for a in legal}
        return {a: v / total for a, v in dist.items()}
