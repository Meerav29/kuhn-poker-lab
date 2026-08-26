# Kuhn Poker Lab — Design Spec

## Purpose

A research sandbox comparing three approaches to playing Kuhn Poker — a closed-form Nash equilibrium bot, a CFR (counterfactual regret minimization) self-play bot, and a set of hand-coded heuristic bots — measured rigorously enough to support a whitepaper-style write-up. This is a bot-research project, not a polished playable game; a UI is explicitly out of scope.

## Background: Kuhn Poker

Kuhn Poker is the smallest standard example of an imperfect-information game with a known equilibrium. Two players are each dealt one card from a 3-card deck (Jack, Queen, King), ante 1 chip, and play a single betting round (check/bet, then call/fold on a bet). The full game tree has a small, fixed number of histories and information sets, which is what makes exact (non-sampled) analysis tractable here.

Kuhn Poker's Nash equilibria are not a single strategy but a **one-parameter family**, conventionally indexed by α ∈ [0, 1/3] (the probability Player 1 bets with a Jack). This spec's Nash bot implements the canonical point **α = 0** (never bluff-bet a Jack as the first action) — chosen for having the simplest closed-form action probabilities to state and verify against, not because it's more "correct" than other points in the family. The design doc and report should be explicit that this is one representative equilibrium, not "the" equilibrium.

## Components

- **`kuhn/game.py`** — the extensive-form game definition: card deck, hand-rolled enumeration of every history and information set, legal actions at each decision point, and the terminal zero-sum payoff function. This is the single source of truth every bot and evaluation routine is built against.
- **`kuhn/bots/nash.py`** — closed-form equilibrium strategy at α = 0, implemented directly from the published Kuhn Poker equilibrium formulas.
- **`kuhn/bots/cfr.py`** — vanilla CFR self-play trainer (regret matching per information set). Logs the *average* strategy's exploitability every K iterations, producing the data for a convergence curve.
- **`kuhn/bots/heuristics.py`** — a handful of hand-coded rule bots as weak/reference baselines:
  - `HonestBot` — bets/calls only with the King, checks/folds otherwise (no bluffing).
  - `AlwaysBluffBot` — bets with the Jack at a fixed high rate.
  - `RandomBot` — uniformly random legal action, as a floor/sanity baseline.
- **`kuhn/evaluate.py`**:
  - `exploitability(bot)` — exact best-response value via full game-tree traversal (minimax over the opponent's best response at every reachable information set). No sampling; the number is exact given the hand-rolled tree.
  - `head_to_head(bot_a, bot_b, n_hands)` — simulates N hands between two bots, returns mean payoff with a bootstrap confidence interval.
- **`scripts/`** — driver scripts that produce the artifacts the write-up needs: an exploitability comparison table/bar chart across all bots, a CFR convergence curve, and a head-to-head win-rate matrix.
- **`docs/`** — this spec, `2026-08-25-approach-justification.md` (why Approach A over a general engine or OpenSpiel), and the eventual `REPORT.md` (the whitepaper-style write-up, written after implementation — not part of this spec's deliverables).
- **`tests/`** — pytest suite; see Testing below.

## Bot Interface

All bots share one interface: given an information set, return a probability distribution over that information set's legal actions. `evaluate.py` is written entirely against this interface — it never knows or cares whether it's evaluating the Nash bot, a CFR-trained strategy, or a heuristic. This is what keeps the three-way comparison apples-to-apples: the same exploitability and head-to-head code produces every number in the report.

## Data Flow

1. `game.py` defines the tree once: histories → information sets → legal actions → terminal payoffs.
2. Each bot is a pure function of information set → action distribution, built once (Nash, heuristics) or trained (CFR).
3. `evaluate.exploitability(bot)` traverses the tree exactly, computing the opponent's best-response value against the bot's fixed strategy.
4. `evaluate.head_to_head(bot_a, bot_b, n)` samples N random deals and betting sequences, playing both bots against the tree's transition rules, and aggregates payoffs.
5. `bots/cfr.py`'s training loop calls `exploitability()` on its own current average strategy every K iterations to log the convergence curve — reusing the same exact evaluator used for the final comparison, not a separate approximate metric.

## Testing

Correctness here rests on tests, not runtime error handling — this is research code whose only job is to produce numbers that belong in a report:

- **Game tree structure**: correct count of information sets and terminal histories; every terminal history has a well-defined zero-sum payoff.
- **Zero-sum invariant**: for any random trajectory through the tree, Player 1's payoff + Player 2's payoff = 0.
- **Nash bot self-check**: the α=0 bot's own exploitability, computed by `evaluate.exploitability`, should be at or extremely near the known theoretical value (sanity check that the closed-form implementation matches the published equilibrium formulas).
- **CFR convergence trend**: exploitability of the CFR average strategy should trend downward over training iterations (not strictly monotonic run-to-run, but a clear downward trend over a fixed iteration budget).
- **Heuristic bots are exploitable**: sanity check that `HonestBot`, `AlwaysBluffBot`, and `RandomBot` all have exploitability meaningfully greater than the Nash bot's — i.e., the evaluator is actually discriminating, not returning a constant.

## Out of Scope

- Any playable UI (web or otherwise) — this is explicitly a research sandbox, per the "bot research, UI secondary" decision.
- An LLM-as-agent bot — considered and dropped from the comparison set.
- Generalizing the game engine beyond Kuhn Poker (see `2026-08-25-approach-justification.md` for why this was rejected as Approach B).
- The `REPORT.md` write-up itself is a downstream deliverable, not part of this implementation's acceptance criteria — the spec's job is to produce the code and data the report will draw from.
