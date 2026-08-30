"""Vanilla counterfactual regret minimization (CFR) via random-deal
self-play. Converges to a Nash equilibrium in this two-player zero-sum
game (Zinkevich et al., 2007); which point in the alpha-family it lands
on is not controlled, only that its average strategy's exploitability
goes to 0 as training iterations grow.
"""
import random
from collections import defaultdict

from kuhn import game


class CFRBot:
    """Wraps a fixed strategy dict (as produced by CFRTrainer) behind the
    standard Bot interface."""

    def __init__(self, strategy: dict):
        self._strategy = strategy  # {infoset: {"p": prob, "b": prob}}

    def action_probs(self, infoset: str, legal: tuple) -> dict:
        dist = self._strategy.get(infoset)
        if dist is None:
            p = 1.0 / len(legal)
            return {a: p for a in legal}
        return dist


class CFRTrainer:
    def __init__(self, seed: int = None):
        self._rng = random.Random(seed)
        self._regret_sum = defaultdict(lambda: [0.0, 0.0])
        self._strategy_sum = defaultdict(lambda: [0.0, 0.0])

    def _current_strategy(self, key: str):
        r = self._regret_sum[key]
        pos = [max(x, 0.0) for x in r]
        s = sum(pos)
        return [x / s for x in pos] if s > 0 else [0.5, 0.5]

    def _cfr(self, history: str, card1: int, card2: int, reach1: float, reach2: float) -> float:
        if game.is_terminal(history):
            return game.payoff(history, card1, card2)

        player = game.whose_turn(history)
        card = card1 if player == 0 else card2
        key = game.infoset_key(card, history)
        strat = self._current_strategy(key)

        reach = reach1 if player == 0 else reach2
        ssum = self._strategy_sum[key]
        for i in range(2):
            ssum[i] += reach * strat[i]

        util_p1 = [0.0, 0.0]
        for i, a in enumerate(game.ACTIONS):
            nh = history + a
            if player == 0:
                util_p1[i] = self._cfr(nh, card1, card2, reach1 * strat[i], reach2)
            else:
                util_p1[i] = self._cfr(nh, card1, card2, reach1, reach2 * strat[i])

        node_util_p1 = sum(strat[i] * util_p1[i] for i in range(2))
        if player == 0:
            own_util, own_node_util, opp_reach = util_p1, node_util_p1, reach2
        else:
            own_util = [-u for u in util_p1]
            own_node_util = -node_util_p1
            opp_reach = reach1

        for i in range(2):
            self._regret_sum[key][i] += opp_reach * (own_util[i] - own_node_util)

        return node_util_p1

    def _average_bot(self) -> CFRBot:
        strategy = {}
        for key, counts in self._strategy_sum.items():
            total = sum(counts)
            if total > 0:
                strategy[key] = {"p": counts[0] / total, "b": counts[1] / total}
            else:
                strategy[key] = {"p": 0.5, "b": 0.5}
        return CFRBot(strategy)

    def train(self, iterations: int, log_every: int = None):
        from kuhn.evaluate import exploitability  # local import: avoids a cycle at module load time

        log = []
        for it in range(1, iterations + 1):
            c1, c2 = self._rng.sample(game.CARDS, 2)
            self._cfr("", c1, c2, 1.0, 1.0)
            if log_every and it % log_every == 0:
                log.append((it, exploitability(self._average_bot())))
        return self._average_bot(), log
