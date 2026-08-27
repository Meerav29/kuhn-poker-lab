from typing import Protocol


class Bot(Protocol):
    def action_probs(self, infoset: str, legal: tuple) -> dict:
        """Return a probability distribution (action -> probability,
        summing to 1) over the legal actions at this information set."""
        ...
