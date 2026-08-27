"""Kuhn Poker extensive-form game definition.

Cards: 0=Jack, 1=Queen, 2=King (higher wins showdown).
Actions: 'p' (pass -- check if no bet is pending, fold if a bet is
         pending), 'b' (bet -- bet if no bet is pending, call if a
         bet is pending).

History strings are sequences of 'p'/'b'. This is the standard Kuhn
Poker formalism from Zinkevich et al. (2007), "Regret Minimization in
Games with Incomplete Information," also used in Neller & Lanctot's
CFR tutorial.
"""

CARDS = (0, 1, 2)  # Jack, Queen, King
CARD_NAMES = {0: "J", 1: "Q", 2: "K"}
ACTIONS = ("p", "b")

TERMINAL_HISTORIES = {
    "pp": "showdown",
    "bp": "p1_wins",   # P1 bet, P2 folded
    "bb": "showdown",
    "pbp": "p2_wins",  # P1 checked, P2 bet, P1 folded
    "pbb": "showdown",
}


def is_terminal(history: str) -> bool:
    return history in TERMINAL_HISTORIES


def whose_turn(history: str) -> int:
    """0 = player 1's turn, 1 = player 2's turn. Only valid for non-terminal histories."""
    assert not is_terminal(history)
    return len(history) % 2


def legal_actions(history: str) -> tuple:
    assert not is_terminal(history)
    return ACTIONS


def payoff(history: str, card1: int, card2: int) -> int:
    """Net chips won by player 1 (player 2's payoff is the negation)."""
    assert is_terminal(history)
    kind = TERMINAL_HISTORIES[history]
    if kind == "p1_wins":
        return 1
    if kind == "p2_wins":
        return -1
    pot = 2 if history == "pp" else 4
    higher = 1 if card1 > card2 else -1
    return higher * (pot // 2)


def infoset_key(card: int, history: str) -> str:
    """Information set identifier: the acting player's own card plus the
    public betting history so far. Never encodes the opponent's card --
    that's exactly the hidden information the infoset represents."""
    return f"{card}{history}"


def all_infosets() -> list:
    """Every reachable (card, history) pair at which some player must act."""
    infosets = set()

    def walk(history, card1, card2):
        if is_terminal(history):
            return
        player = whose_turn(history)
        card = card1 if player == 0 else card2
        infosets.add(infoset_key(card, history))
        for a in legal_actions(history):
            walk(history + a, card1, card2)

    for c1 in CARDS:
        for c2 in CARDS:
            if c1 != c2:
                walk("", c1, c2)
    return sorted(infosets)


def all_deals() -> list:
    """All 6 ordered (card1, card2) deals, each equally likely."""
    return [(c1, c2) for c1 in CARDS for c2 in CARDS if c1 != c2]
