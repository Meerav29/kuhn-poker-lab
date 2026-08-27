"""Hand-coded rule bots -- weak/reference baselines for the comparison."""


class HonestBot:
    """Never bluffs: bets/calls only with the King (card 2)."""

    def action_probs(self, infoset: str, legal: tuple) -> dict:
        card = int(infoset[0])
        target = "b" if card == 2 else "p"
        return {a: (1.0 if a == target else 0.0) for a in legal}


class AlwaysBluffBot:
    """Bets/calls with the Jack at a fixed high rate (a bluff), plays
    honestly with Queen and King."""

    BLUFF_RATE = 0.8

    def action_probs(self, infoset: str, legal: tuple) -> dict:
        card = int(infoset[0])
        if card == 0:
            return {"p": 1 - self.BLUFF_RATE, "b": self.BLUFF_RATE}
        target = "b" if card == 2 else "p"
        return {a: (1.0 if a == target else 0.0) for a in legal}


class RandomBot:
    """Uniformly random legal action -- floor/sanity baseline."""

    def action_probs(self, infoset: str, legal: tuple) -> dict:
        p = 1.0 / len(legal)
        return {a: p for a in legal}
