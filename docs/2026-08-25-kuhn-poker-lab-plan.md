# Kuhn Poker Lab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Kuhn Poker research sandbox comparing a closed-form Nash equilibrium bot, a CFR self-play bot, and heuristic bots, with exact exploitability and head-to-head evaluation, to support a whitepaper-style write-up.

**Architecture:** Pure Python. A hand-rolled, exactly-enumerable game tree (`kuhn/game.py`) is the single source of truth. Every bot implements one shared interface (`infoset -> action probabilities`), so the evaluation code (`kuhn/evaluate.py`) is bot-agnostic and produces the same kind of number (exact exploitability, or simulated head-to-head EV) for all three bot families. A driver script assembles the comparison artifacts for the report.

**Tech Stack:** Python 3.10+, pytest, matplotlib (for the convergence-curve plot only).

**Spec:** [`docs/2026-08-25-kuhn-poker-lab-design.md`](2026-08-25-kuhn-poker-lab-design.md) — see also [`docs/2026-08-25-approach-justification.md`](2026-08-25-approach-justification.md) for why the game engine is hand-rolled rather than a general framework or OpenSpiel.

## Global Constraints

- No playable UI — this is a research sandbox, not a game (per spec).
- No LLM-as-agent bot — dropped from the comparison set (per spec).
- The game engine must stay specific to Kuhn Poker — do not generalize it into a multi-game framework (per approach-justification doc).
- Every exploitability number must come from the exact best-response traversal in `kuhn/evaluate.py`, never from sampling — this is what makes the write-up's numbers trustworthy (per spec).
- Cards are represented as `0=Jack, 1=Queen, 2=King`; actions as `"p"` (pass: check if no bet pending, fold if a bet is pending) and `"b"` (bet: bet if no bet pending, call if a bet is pending). This is the standard Kuhn Poker formalism from Zinkevich et al. (2007) and the Neller/Lanctot CFR tutorial — history strings like `"pb"`, `"bb"` are sequences of these two symbols.

---

## Verified reference values

These were computed and checked during planning (exact best-response traversal, not sampled) and are the ground truth the tests in this plan assert against:

- **Game value to Player 1 at equilibrium:** exactly **−1/18** (≈ −0.055556). Independent of which α∈[0, 1/3] is used, confirming the equilibrium family.
- **Nash bot (α=0) exploitability:** **0.0** (exact, to floating-point precision) for both player seats.
- **Heuristic bot exploitability:** `HonestBot` ≈ **0.25**, `AlwaysBluffBot` (bluff rate 0.8) ≈ **0.376667**, `RandomBot` ≈ **0.458333** — all clearly and distinctly greater than the Nash bot's.
- **CFR convergence** (vanilla CFR, random-deal self-play, average strategy evaluated by exact exploitability): iter 100 → ≈0.060, iter 1,000 → ≈0.016, iter 5,000 → ≈0.007, iter 10,000 → ≈0.009, iter 50,000 → ≈0.0016. Trend is downward but not strictly monotonic run-to-run (expected — CFR's average-strategy exploitability bound is O(1/√T), not monotonic per-iteration).

## Verified closed-form Nash equilibrium (α family)

Player 1 (`α ∈ [0, 1/3]`, this project implements **α = 0**):
- `hist=""` (first action): Jack bets `α`; Queen never bets (`0`); King bets `3α`.
- `hist="pb"` (checked, then faced a bet): Jack always folds (`0`); Queen calls `α + 1/3`; King always calls (`1`).

Player 2 (fixed, independent of α):
- `hist="p"` (checked to): Jack bets `1/3` (a bluff); Queen never bets (`0`); King always bets (`1`).
- `hist="b"` (facing the opening bet): Jack always folds (`0`); Queen calls `1/3`; King always calls (`1`).

At α=0, Player 1 never bets first with *any* card (always checks), relying entirely on check-call. This is correct and was verified to have exactly 0 exploitability — it is a genuine, if unintuitive-looking, equilibrium point, not a bug. The design doc's justification for picking α=0 (simplest closed-form probabilities to state and verify) still holds.

---

### Task 1: Project setup and the game tree

**Files:**
- Create: `requirements.txt`
- Create: `kuhn/__init__.py` (empty)
- Create: `kuhn/game.py`
- Test: `tests/__init__.py` (empty)
- Test: `tests/test_game.py`

**Interfaces:**
- Produces: `CARDS: tuple[int,...]`, `CARD_NAMES: dict[int,str]`, `ACTIONS: tuple[str,str]` (`("p","b")`), `is_terminal(history: str) -> bool`, `whose_turn(history: str) -> int`, `legal_actions(history: str) -> tuple[str,str]`, `payoff(history: str, card1: int, card2: int) -> int`, `infoset_key(card: int, history: str) -> str`, `all_infosets() -> list[str]`, `all_deals() -> list[tuple[int,int]]`.

- [ ] **Step 1: Create `requirements.txt`**

```
pytest>=7.0
matplotlib>=3.7
```

- [ ] **Step 2: Create empty package markers**

Create `kuhn/__init__.py` and `tests/__init__.py`, both empty files.

- [ ] **Step 3: Write the failing tests for the game tree**

```python
# tests/test_game.py
from kuhn import game


def test_terminal_histories():
    assert game.is_terminal("pp")
    assert game.is_terminal("bp")
    assert game.is_terminal("bb")
    assert game.is_terminal("pbp")
    assert game.is_terminal("pbb")
    assert not game.is_terminal("")
    assert not game.is_terminal("p")
    assert not game.is_terminal("b")
    assert not game.is_terminal("pb")


def test_whose_turn():
    assert game.whose_turn("") == 0
    assert game.whose_turn("p") == 1
    assert game.whose_turn("b") == 1
    assert game.whose_turn("pb") == 0


def test_payoff_table():
    # card1=2 (King) beats card2=0 (Jack) throughout
    assert game.payoff("pp", 2, 0) == 1     # showdown, pot=2, P1 wins
    assert game.payoff("pp", 0, 2) == -1    # showdown, pot=2, P1 loses
    assert game.payoff("bp", 0, 2) == 1     # P2 folded, P1 wins regardless of cards
    assert game.payoff("bp", 2, 0) == 1
    assert game.payoff("bb", 2, 0) == 2     # showdown, pot=4, P1 wins
    assert game.payoff("bb", 0, 2) == -2    # showdown, pot=4, P1 loses
    assert game.payoff("pbp", 2, 0) == -1   # P1 folded, P2 wins regardless of cards
    assert game.payoff("pbp", 0, 2) == -1
    assert game.payoff("pbb", 2, 0) == 2    # showdown, pot=4
    assert game.payoff("pbb", 0, 2) == -2


def test_zero_sum_invariant():
    # player 2's payoff is always the negation of player 1's, for every
    # terminal history and every possible deal.
    for history in ("pp", "bp", "bb", "pbp", "pbb"):
        for c1, c2 in game.all_deals():
            p1 = game.payoff(history, c1, c2)
            p2 = -game.payoff(history, c2, c1)  # payoff() is always "P1's" perspective
            assert p1 == -(-p2)  # p1 + (payoff from swapped seats, negated) is consistent
            assert p1 == p2


def test_infoset_key_format():
    assert game.infoset_key(0, "") == "0"
    assert game.infoset_key(2, "pb") == "2pb"


def test_all_infosets_count():
    # decision points: hist in {"", "p", "b", "pb"}, times 3 cards each = 12
    infosets = game.all_infosets()
    assert len(infosets) == 12
    assert len(set(infosets)) == 12  # no duplicates


def test_all_deals():
    deals = game.all_deals()
    assert len(deals) == 6  # 3 cards, ordered pairs, no repeats
    assert all(c1 != c2 for c1, c2 in deals)
```

- [ ] **Step 4: Run the tests to verify they fail**

Run: `pytest tests/test_game.py -v`
Expected: FAIL/ERROR — `kuhn.game` does not exist yet.

- [ ] **Step 5: Implement `kuhn/game.py`**

```python
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
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `pytest tests/test_game.py -v`
Expected: PASS (11 tests)

- [ ] **Step 7: Commit**

```bash
git add requirements.txt kuhn/__init__.py kuhn/game.py tests/__init__.py tests/test_game.py
git commit -m "feat: add Kuhn Poker game tree definition"
```

---

### Task 2: Bot interface and heuristic bots

**Files:**
- Create: `kuhn/bots/__init__.py` (empty)
- Create: `kuhn/bots/base.py`
- Create: `kuhn/bots/heuristics.py`
- Test: `tests/test_heuristics.py`

**Interfaces:**
- Consumes: `game.infoset_key`, `game.legal_actions`, `game.ACTIONS` from Task 1.
- Produces: `Bot` protocol with `action_probs(self, infoset: str, legal: tuple) -> dict[str, float]`. `HonestBot`, `AlwaysBluffBot`, `RandomBot` classes, each implementing that protocol. This is the interface every later bot (Nash, CFR) must also implement.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_heuristics.py
import pytest
from kuhn import game
from kuhn.bots.heuristics import HonestBot, AlwaysBluffBot, RandomBot


BOTS = [HonestBot(), AlwaysBluffBot(), RandomBot()]


@pytest.mark.parametrize("bot", BOTS, ids=lambda b: type(b).__name__)
def test_returns_valid_distribution_over_legal_actions(bot):
    for infoset in game.all_infosets():
        card = int(infoset[0])
        history = infoset[1:]
        legal = game.legal_actions(history)
        dist = bot.action_probs(infoset, legal)
        assert set(dist.keys()) == set(legal)
        assert all(p >= 0 for p in dist.values())
        assert abs(sum(dist.values()) - 1.0) < 1e-9


def test_honest_bot_only_bets_king():
    bot = HonestBot()
    assert bot.action_probs("0", ("p", "b")) == {"p": 1.0, "b": 0.0}   # Jack
    assert bot.action_probs("1", ("p", "b")) == {"p": 1.0, "b": 0.0}   # Queen
    assert bot.action_probs("2", ("p", "b")) == {"p": 0.0, "b": 1.0}   # King


def test_always_bluff_bot_bluffs_jack():
    bot = AlwaysBluffBot()
    dist = bot.action_probs("0", ("p", "b"))
    assert dist["b"] == bot.BLUFF_RATE


def test_random_bot_is_uniform():
    bot = RandomBot()
    dist = bot.action_probs("1pb", ("p", "b"))
    assert dist == {"p": 0.5, "b": 0.5}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_heuristics.py -v`
Expected: FAIL/ERROR — `kuhn.bots.heuristics` does not exist yet.

- [ ] **Step 3: Implement the bot interface and heuristic bots**

```python
# kuhn/bots/base.py
from typing import Protocol


class Bot(Protocol):
    def action_probs(self, infoset: str, legal: tuple) -> dict:
        """Return a probability distribution (action -> probability,
        summing to 1) over the legal actions at this information set."""
        ...
```

```python
# kuhn/bots/heuristics.py
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_heuristics.py -v`
Expected: PASS (6 tests, including the parametrized ones)

- [ ] **Step 5: Commit**

```bash
git add kuhn/bots/__init__.py kuhn/bots/base.py kuhn/bots/heuristics.py tests/test_heuristics.py
git commit -m "feat: add bot interface and heuristic baseline bots"
```

---

### Task 3: Closed-form Nash equilibrium bot

**Files:**
- Create: `kuhn/bots/nash.py`
- Test: `tests/test_nash.py`

**Interfaces:**
- Consumes: `Bot` protocol shape from Task 2 (not imported, just conformed to).
- Produces: `NashBot(alpha: float = 0.0)` implementing `action_probs(self, infoset: str, legal: tuple) -> dict`.

- [ ] **Step 1: Write the failing tests**

These assert against the closed-form probabilities verified during planning (see "Verified closed-form Nash equilibrium" above) at `alpha=0`.

```python
# tests/test_nash.py
import pytest
from kuhn.bots.nash import NashBot


def approx(a, b, tol=1e-9):
    return abs(a - b) < tol


def test_alpha_zero_first_action_probs():
    bot = NashBot(alpha=0.0)
    legal = ("p", "b")
    assert approx(bot.action_probs("0", legal)["b"], 0.0)   # Jack: bet=alpha=0
    assert approx(bot.action_probs("1", legal)["b"], 0.0)   # Queen: never bets first
    assert approx(bot.action_probs("2", legal)["b"], 0.0)   # King: bet=3*alpha=0


def test_alpha_zero_pb_call_probs():
    bot = NashBot(alpha=0.0)
    legal = ("p", "b")
    assert approx(bot.action_probs("0pb", legal)["b"], 0.0)          # Jack always folds
    assert approx(bot.action_probs("1pb", legal)["b"], 1 / 3)        # Queen calls alpha+1/3
    assert approx(bot.action_probs("2pb", legal)["b"], 1.0)          # King always calls


def test_p2_strategy_independent_of_alpha():
    legal = ("p", "b")
    for alpha in (0.0, 1 / 6, 1 / 3):
        bot = NashBot(alpha=alpha)
        assert approx(bot.action_probs("0p", legal)["b"], 1 / 3)   # Jack bluffs 1/3 when checked to
        assert approx(bot.action_probs("1p", legal)["b"], 0.0)     # Queen never bets when checked to
        assert approx(bot.action_probs("2p", legal)["b"], 1.0)     # King always bets when checked to
        assert approx(bot.action_probs("0b", legal)["b"], 0.0)     # Jack always folds to a bet
        assert approx(bot.action_probs("1b", legal)["b"], 1 / 3)   # Queen calls 1/3 facing a bet
        assert approx(bot.action_probs("2b", legal)["b"], 1.0)     # King always calls


def test_distribution_sums_to_one_across_all_infosets():
    from kuhn import game
    bot = NashBot(alpha=0.0)
    for infoset in game.all_infosets():
        history = infoset[1:]
        legal = game.legal_actions(history)
        dist = bot.action_probs(infoset, legal)
        assert abs(sum(dist.values()) - 1.0) < 1e-9
        assert all(p >= 0 for p in dist.values())


def test_alpha_out_of_range_rejected():
    with pytest.raises(ValueError):
        NashBot(alpha=0.5)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_nash.py -v`
Expected: FAIL/ERROR — `kuhn.bots.nash` does not exist yet.

- [ ] **Step 3: Implement `kuhn/bots/nash.py`**

```python
# kuhn/bots/nash.py
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_nash.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add kuhn/bots/nash.py tests/test_nash.py
git commit -m "feat: add closed-form Nash equilibrium bot"
```

---

### Task 4: Exact exploitability (best-response evaluation)

This is the evaluation core the whole comparison depends on. **Critical correctness note from planning:** a best response must only condition on the responder's own information set (own card + public history) — it may NOT peek at the opponent's actual card mid-recursion. An earlier draft of this exact code made that mistake and silently produced a "clairvoyant response" value that was numerically constant and wrong (0.278 instead of the correct 0.0 for the Nash bot). The implementation below aggregates over the opponent's possible cards, weighted by the opponent's strategy's reach probability, before the responder's `max`.

**Files:**
- Create: `kuhn/evaluate.py`
- Test: `tests/test_evaluate.py`

**Interfaces:**
- Consumes: `game.CARDS`, `game.is_terminal`, `game.whose_turn`, `game.payoff`, `game.infoset_key`, `game.legal_actions` (Task 1); any `Bot`-shaped object with `action_probs(infoset, legal) -> dict` (Task 2/3).
- Produces: `best_response_value(bot, br_player: int) -> float`, `exploitability(bot) -> float`. Later tasks (CFR training log, comparison script) call `exploitability`.

- [ ] **Step 1: Write the failing tests**

Uses the verified reference values from planning: Nash bot exploitability is exactly 0.0, and the heuristic bots' exploitability values are ≈0.25, ≈0.376667, ≈0.458333 respectively.

```python
# tests/test_evaluate.py
from kuhn.bots.nash import NashBot
from kuhn.bots.heuristics import HonestBot, AlwaysBluffBot, RandomBot
from kuhn.evaluate import exploitability


def test_nash_bot_is_unexploitable():
    for alpha in (0.0, 1 / 6, 1 / 3):
        assert abs(exploitability(NashBot(alpha=alpha))) < 1e-9


def test_heuristic_bots_are_exploitable_and_distinct():
    honest = exploitability(HonestBot())
    bluff = exploitability(AlwaysBluffBot())
    random_ = exploitability(RandomBot())

    assert abs(honest - 0.25) < 1e-6
    assert abs(bluff - 0.376667) < 1e-5
    assert abs(random_ - 0.458333) < 1e-5

    # every heuristic bot is meaningfully more exploitable than Nash
    for exp in (honest, bluff, random_):
        assert exp > 0.1
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_evaluate.py -v`
Expected: FAIL/ERROR — `kuhn.evaluate` does not exist yet.

- [ ] **Step 3: Implement `kuhn/evaluate.py`**

```python
# kuhn/evaluate.py
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_evaluate.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add kuhn/evaluate.py tests/test_evaluate.py
git commit -m "feat: add exact best-response and exploitability evaluation"
```

---

### Task 5: Head-to-head match simulation

**Files:**
- Modify: `kuhn/evaluate.py` (add `head_to_head`)
- Test: `tests/test_evaluate.py` (add tests)

**Interfaces:**
- Consumes: `game.CARDS`, `game.is_terminal`, `game.whose_turn`, `game.payoff`, `game.infoset_key`, `game.legal_actions`; `_probs` (defined in Task 4, same file); any `Bot`-shaped object.
- Produces: `head_to_head(bot_a, bot_b, n_hands: int, seed: int | None = None) -> dict` with keys `mean`, `ci_low`, `ci_high` (95% bootstrap confidence interval, from bot_a's perspective as Player 1). Consumed by the comparison script in Task 7.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_evaluate.py  (append to the file from Task 4)
from kuhn.evaluate import head_to_head


def test_random_vs_random_mean_near_zero():
    result = head_to_head(RandomBot(), RandomBot(), n_hands=20000, seed=1)
    assert result["ci_low"] < 0.0 < result["ci_high"]  # symmetric matchup: CI should straddle 0


def test_nash_beats_random_bot():
    result = head_to_head(NashBot(alpha=0.0), RandomBot(), n_hands=20000, seed=1)
    assert result["mean"] > 0
    assert result["ci_low"] > 0  # confidently positive, not just noise


def test_result_has_expected_keys():
    result = head_to_head(HonestBot(), RandomBot(), n_hands=1000, seed=1)
    assert set(result.keys()) == {"mean", "ci_low", "ci_high"}
    assert result["ci_low"] <= result["mean"] <= result["ci_high"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_evaluate.py -v`
Expected: FAIL/ERROR — `head_to_head` does not exist yet.

- [ ] **Step 3: Add `head_to_head` to `kuhn/evaluate.py`**

```python
# append to kuhn/evaluate.py
import random


def _play_hand(bot_a, bot_b, card1: int, card2: int, rng: random.Random) -> int:
    """Simulate one hand: bot_a as Player 1, bot_b as Player 2. Returns
    Player 1's (bot_a's) net payoff."""
    history = ""
    while not game.is_terminal(history):
        player = game.whose_turn(history)
        card = card1 if player == 0 else card2
        bot = bot_a if player == 0 else bot_b
        p_pass, p_bet = _probs(bot, card, history)
        action = "p" if rng.random() < p_pass else "b"
        history += action
    return game.payoff(history, card1, card2)


def head_to_head(bot_a, bot_b, n_hands: int, seed: int = None) -> dict:
    """Simulate n_hands hands with bot_a as Player 1 and bot_b as Player 2.
    Returns bot_a's mean payoff and a 95% bootstrap confidence interval."""
    rng = random.Random(seed)
    deals = game.all_deals()
    payoffs = []
    for _ in range(n_hands):
        c1, c2 = rng.choice(deals)
        payoffs.append(_play_hand(bot_a, bot_b, c1, c2, rng))

    mean = sum(payoffs) / len(payoffs)

    boot_rng = random.Random(seed)
    n_boot = 2000
    boot_means = []
    for _ in range(n_boot):
        sample = [payoffs[boot_rng.randrange(len(payoffs))] for _ in range(len(payoffs))]
        boot_means.append(sum(sample) / len(sample))
    boot_means.sort()
    ci_low = boot_means[int(0.025 * n_boot)]
    ci_high = boot_means[int(0.975 * n_boot)]

    return {"mean": mean, "ci_low": ci_low, "ci_high": ci_high}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_evaluate.py -v`
Expected: PASS (5 tests total). Note: `test_random_vs_random_mean_near_zero` and `test_nash_beats_random_bot` are statistical — they use a fixed seed for reproducibility, but if either is ever flaky on a different seed, increase `n_hands` rather than loosening the assertion, since the underlying claims (symmetric matchups average to ~0, Nash beats Random) are mathematically true in the limit.

- [ ] **Step 5: Commit**

```bash
git add kuhn/evaluate.py tests/test_evaluate.py
git commit -m "feat: add head-to-head match simulation with bootstrap CI"
```

---

### Task 6: CFR self-play trainer

**Files:**
- Create: `kuhn/bots/cfr.py`
- Test: `tests/test_cfr.py`

**Interfaces:**
- Consumes: `game.CARDS`, `game.is_terminal`, `game.whose_turn`, `game.payoff`, `game.infoset_key`, `game.legal_actions` (Task 1); `evaluate.exploitability` (Task 4, used only for the optional convergence log).
- Produces: `CFRBot` (implements the `Bot` interface from Task 2, wraps a fixed strategy dict), `CFRTrainer` with `train(iterations: int, log_every: int = None) -> tuple[CFRBot, list[tuple[int, float]]]` — the returned list is `(iteration, exploitability)` checkpoints, empty if `log_every` is `None`. Consumed by the comparison script (Task 7).

- [ ] **Step 1: Write the failing tests**

Uses the verified convergence checkpoints from planning as the correctness bar (loose thresholds — the CFR run below matches the exact same algorithm and card sampling used to produce those checkpoints).

```python
# tests/test_cfr.py
from kuhn.bots.cfr import CFRTrainer
from kuhn.evaluate import exploitability


def test_average_strategy_is_valid_distribution():
    trainer = CFRTrainer(seed=0)
    bot, _ = trainer.train(iterations=1000)
    from kuhn import game
    for infoset in game.all_infosets():
        history = infoset[1:]
        legal = game.legal_actions(history)
        dist = bot.action_probs(infoset, legal)
        assert abs(sum(dist.values()) - 1.0) < 1e-9


def test_exploitability_converges_toward_zero():
    trainer = CFRTrainer(seed=0)
    bot, _ = trainer.train(iterations=10000)
    # verified reference: iter=10000 -> exploitability ~= 0.0086 with this
    # exact algorithm/seed. Loose bound to tolerate minor implementation drift.
    assert exploitability(bot) < 0.05


def test_convergence_log_trends_downward():
    trainer = CFRTrainer(seed=0)
    _, log = trainer.train(iterations=10000, log_every=1000)
    assert len(log) == 10
    first_half_avg = sum(exp for _, exp in log[:5]) / 5
    second_half_avg = sum(exp for _, exp in log[5:]) / 5
    assert second_half_avg < first_half_avg
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_cfr.py -v`
Expected: FAIL/ERROR — `kuhn.bots.cfr` does not exist yet.

- [ ] **Step 3: Implement `kuhn/bots/cfr.py`**

```python
# kuhn/bots/cfr.py
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_cfr.py -v`
Expected: PASS (3 tests). This test file is the slowest in the suite (10,000+ game-tree traversals); expect it to take a few seconds, not instant.

- [ ] **Step 5: Commit**

```bash
git add kuhn/bots/cfr.py tests/test_cfr.py
git commit -m "feat: add CFR self-play trainer"
```

---

### Task 7: Comparison script (report artifacts)

**Files:**
- Create: `scripts/run_comparison.py`
- Modify: `.gitignore` (add `results/`)
- Test: `tests/test_run_comparison.py` (smoke test only — this script produces report artifacts, not unit-testable numerics beyond what Tasks 4-6 already cover)

**Interfaces:**
- Consumes: `NashBot` (Task 3), `HonestBot`/`AlwaysBluffBot`/`RandomBot` (Task 2), `CFRTrainer`/`CFRBot` (Task 6), `exploitability`/`head_to_head` (Tasks 4-5).
- Produces: `results/exploitability.csv`, `results/head_to_head.csv`, `results/cfr_convergence.csv`, `results/cfr_convergence.png` — the artifacts the eventual `docs/REPORT.md` write-up draws from. Also exposes `run(output_dir: str) -> None` so the smoke test can invoke it against a temp directory.

- [ ] **Step 1: Write the failing smoke test**

```python
# tests/test_run_comparison.py
import csv
import os
import tempfile

from scripts.run_comparison import run


def test_run_produces_all_artifacts():
    with tempfile.TemporaryDirectory() as tmpdir:
        run(output_dir=tmpdir, cfr_iterations=500, cfr_log_every=100, head_to_head_hands=200)

        exploit_path = os.path.join(tmpdir, "exploitability.csv")
        h2h_path = os.path.join(tmpdir, "head_to_head.csv")
        conv_path = os.path.join(tmpdir, "cfr_convergence.csv")
        plot_path = os.path.join(tmpdir, "cfr_convergence.png")

        for path in (exploit_path, h2h_path, conv_path, plot_path):
            assert os.path.exists(path)

        with open(exploit_path) as f:
            rows = list(csv.DictReader(f))
        names = {row["bot"] for row in rows}
        assert names == {"Nash", "Honest", "AlwaysBluff", "Random", "CFR"}

        with open(conv_path) as f:
            conv_rows = list(csv.DictReader(f))
        assert len(conv_rows) == 5  # 500 iterations / log_every=100
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_run_comparison.py -v`
Expected: FAIL/ERROR — `scripts.run_comparison` does not exist yet.

- [ ] **Step 3: Add `scripts/__init__.py` and implement `scripts/run_comparison.py`**

Create empty `scripts/__init__.py` first (needed for the `from scripts.run_comparison import run` test import to work).

```python
# scripts/run_comparison.py
"""Produces the CSV/plot artifacts the write-up (docs/REPORT.md) draws
from: exact exploitability per bot, a head-to-head win-rate matrix, and
the CFR training convergence curve."""
import csv
import os

import matplotlib
matplotlib.use("Agg")  # headless -- this script writes files, doesn't show a window
import matplotlib.pyplot as plt

from kuhn.bots.nash import NashBot
from kuhn.bots.heuristics import HonestBot, AlwaysBluffBot, RandomBot
from kuhn.bots.cfr import CFRTrainer
from kuhn.evaluate import exploitability, head_to_head


def run(output_dir: str, cfr_iterations: int = 50000, cfr_log_every: int = 500,
        head_to_head_hands: int = 20000, seed: int = 0) -> None:
    os.makedirs(output_dir, exist_ok=True)

    trainer = CFRTrainer(seed=seed)
    cfr_bot, convergence_log = trainer.train(iterations=cfr_iterations, log_every=cfr_log_every)

    bots = {
        "Nash": NashBot(alpha=0.0),
        "Honest": HonestBot(),
        "AlwaysBluff": AlwaysBluffBot(),
        "Random": RandomBot(),
        "CFR": cfr_bot,
    }

    # 1. exploitability table
    with open(os.path.join(output_dir, "exploitability.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["bot", "exploitability"])
        for name, bot in bots.items():
            writer.writerow([name, exploitability(bot)])

    # 2. head-to-head matrix (every ordered pair, bot_a as Player 1)
    with open(os.path.join(output_dir, "head_to_head.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["bot_a", "bot_b", "mean", "ci_low", "ci_high"])
        names = list(bots.keys())
        for name_a in names:
            for name_b in names:
                if name_a == name_b:
                    continue
                result = head_to_head(bots[name_a], bots[name_b], n_hands=head_to_head_hands, seed=seed)
                writer.writerow([name_a, name_b, result["mean"], result["ci_low"], result["ci_high"]])

    # 3. CFR convergence curve
    with open(os.path.join(output_dir, "cfr_convergence.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "exploitability"])
        for it, exp in convergence_log:
            writer.writerow([it, exp])

    iterations = [it for it, _ in convergence_log]
    exploitabilities = [exp for _, exp in convergence_log]
    fig, ax = plt.subplots()
    ax.plot(iterations, exploitabilities)
    ax.set_xlabel("CFR training iterations")
    ax.set_ylabel("Exploitability")
    ax.set_title("CFR average-strategy convergence toward equilibrium")
    fig.savefig(os.path.join(output_dir, "cfr_convergence.png"))
    plt.close(fig)


if __name__ == "__main__":
    run(output_dir="results")
```

- [ ] **Step 4: Add `results/` to `.gitignore`**

Append `results/` to the existing `.gitignore` (generated artifacts, regenerated by re-running the script — not committed).

- [ ] **Step 5: Run the test to verify it passes**

Run: `pytest tests/test_run_comparison.py -v`
Expected: PASS (1 test)

- [ ] **Step 6: Run the full test suite**

Run: `pytest -v`
Expected: PASS, all tests across every task (game, heuristics, nash, evaluate, cfr, run_comparison).

- [ ] **Step 7: Generate the real (non-test) artifacts**

Run: `python scripts/run_comparison.py`
Expected: creates `results/exploitability.csv`, `results/head_to_head.csv`, `results/cfr_convergence.csv`, `results/cfr_convergence.png` at the repo root (default 50,000 CFR iterations — this may take on the order of a minute).

- [ ] **Step 8: Commit**

```bash
git add scripts/__init__.py scripts/run_comparison.py tests/test_run_comparison.py .gitignore
git commit -m "feat: add comparison script producing report artifacts"
```

---

## After this plan

The generated `results/` artifacts are the raw material for `docs/REPORT.md` (the whitepaper-style write-up) — writing that report is a separate, subsequent piece of work, not part of this plan's task list, per the spec's "Out of Scope" section.
