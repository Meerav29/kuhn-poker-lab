"""Closed-form Kuhn Poker Nash equilibrium.

Kuhn Poker's equilibria form a one-parameter family, alpha in [0, 1/3]
(Player 1's Jack-bet probability at the first decision). See
docs/2026-08-25-kuhn-poker-lab-design.md for why this project implements
the alpha=0 point. Formulas verified during planning: exploitability is
exactly 0.0 and the game value is exactly -1/18 for every alpha in the
family (recomputed with kuhn.evaluate against this class in Task 4).
"""


class NashBot:
    def __init__(self, alpha: float = 0.0):
        if not (0.0 <= alpha <= 1 / 3):
            raise ValueError(f"alpha must be in [0, 1/3], got {alpha}")
        self.alpha = alpha

    def action_probs(self, infoset: str, legal: tuple) -> dict:
        card = int(infoset[0])
        history = infoset[1:]
        bet_prob = self._bet_probability(card, history)
        return {"p": 1.0 - bet_prob, "b": bet_prob}

    def _bet_probability(self, card: int, history: str) -> float:
        a = self.alpha
        if history == "":  # Player 1's first action
            return {0: a, 1: 0.0, 2: 3 * a}[card]
        if history == "p":  # Player 2, checked to
            return {0: 1 / 3, 1: 0.0, 2: 1.0}[card]
        if history == "b":  # Player 2, facing the opening bet
            return {0: 0.0, 1: 1 / 3, 2: 1.0}[card]
        if history == "pb":  # Player 1, checked then faced a bet
            return {0: 0.0, 1: a + 1 / 3, 2: 1.0}[card]
        raise ValueError(f"no decision at history {history!r}")
