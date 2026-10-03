"""State encoders: information set -> prompt text.

Pure and deterministic. An encoder only ever sees the infoset key (own
card + public betting history), so it cannot leak the opponent's card.
"""
from kuhn import game

CARD_WORDS = {0: "Jack", 1: "Queen", 2: "King"}

NARRATION = {
    "": "you act first, nothing has happened yet",
    "p": "the opponent checked",
    "b": "the opponent bet 1 chip",
    "pb": "you checked and the opponent then bet 1 chip",
}

POT = {"": 2, "p": 2, "b": 3, "pb": 3}   # chips in the pot when this decision is made
COST = {"": 1, "p": 1, "b": 1, "pb": 1}  # chips needed to bet or call

RULES = (
    "Kuhn Poker. Deck: Jack, Queen, King (King beats Queen beats Jack). "
    "Each player antes 1 chip and gets one private card. One betting round: "
    "check or bet 1 chip; facing a bet you may call 1 chip or fold. "
    "Higher card wins at showdown."
)


def _split(infoset: str):
    return int(infoset[0]), infoset[1:]


def minimal(infoset: str) -> str:
    card, history = _split(infoset)
    return f"card={game.CARD_NAMES[card]} history={history or '-'}"


def natural(infoset: str) -> str:
    card, history = _split(infoset)
    return f"You hold the {CARD_WORDS[card]}. Situation: {NARRATION[history]}."


def rules(infoset: str) -> str:
    return f"{RULES} {natural(infoset)}"


def payoff(infoset: str) -> str:
    _, history = _split(infoset)
    return (
        f"{natural(infoset)} The pot is {POT[history]} chips "
        f"and it costs {COST[history]} chip to continue."
    )


ENCODERS = {"minimal": minimal, "natural": natural, "rules": rules, "payoff": payoff}


def get_encoder(name: str):
    try:
        return ENCODERS[name]
    except KeyError:
        raise ValueError(f"unknown encoder {name!r}; choose from {sorted(ENCODERS)}")
