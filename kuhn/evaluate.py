"""Exact evaluation: best-response value and exploitability.

Exploitability(sigma) = the average, over both player seats, of how much
a perfect best-responder can win against the other seat's part of sigma,
above the (zero-sum) equilibrium value. At a true Nash equilibrium this
is exactly 0.
"""
from kuhn import game


def _probs(bot, card: int, history: str):
    infoset = game.infoset_key(card, history)
    legal = game.legal_actions(history)
    dist = bot.action_probs(infoset, legal)
    return dist.get("p", 0.0), dist.get("b", 0.0)


def best_response_value(bot, br_player: int) -> float:
    """Value to br_player (0 or 1) of best-responding against `bot`
    playing the OTHER seat. `bot` also defines br_player's own seat's
    strategy nowhere in this computation -- br_player is replaced by an
    exact best responder instead.

    Respects information sets: br_player's decisions depend only on its
    own card and the public history, aggregated over every opponent card
    consistent with that history (weighted by the opponent's reach
    probability under `bot`'s strategy) -- never on the opponent's actual
    card for that specific deal.
    """

    def rec(history, own_card, opp_reach):
        # opp_reach: {opponent_card: probability weight}
        if game.is_terminal(history):
            total = 0.0
            for opp_card, w in opp_reach.items():
                c1, c2 = (own_card, opp_card) if br_player == 0 else (opp_card, own_card)
                p1_payoff = game.payoff(history, c1, c2)
                total += w * (p1_payoff if br_player == 0 else -p1_payoff)
            return total

        player = game.whose_turn(history)
        if player == br_player:
            # br_player picks ONE action per infoset -- take the max once,
            # aggregated over the whole opponent-card distribution.
            return max(rec(history + a, own_card, opp_reach) for a in game.legal_actions(history))

        total = 0.0
        for a in game.legal_actions(history):
            new_reach = {}
            for opp_card, w in opp_reach.items():
                p_pass, p_bet = _probs(bot, opp_card, history)
                p_a = p_pass if a == "p" else p_bet
                if p_a > 0:
                    new_reach[opp_card] = w * p_a
            if new_reach:
                total += rec(history + a, own_card, new_reach)
        return total

    total = 0.0
    for own_card in game.CARDS:
        others = [c for c in game.CARDS if c != own_card]
        total += rec("", own_card, {c: 0.5 for c in others})
    return total / len(game.CARDS)


def exploitability(bot) -> float:
    br1 = best_response_value(bot, 0)
    br2 = best_response_value(bot, 1)
    return (br1 + br2) / 2
