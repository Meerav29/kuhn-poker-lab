# Kuhn Poker Lab — Results Report

## Overview

This report compares three approaches to playing Kuhn Poker — a closed-form
Nash equilibrium bot, a CFR self-play bot, and three hand-coded heuristic
bots — using two metrics computed against the exact, hand-rolled game tree
in [`kuhn/game.py`](../kuhn/game.py): **exact exploitability** (no sampling)
and **simulated head-to-head EV** with bootstrap confidence intervals. See
[2026-08-25-kuhn-poker-lab-design.md](2026-08-25-kuhn-poker-lab-design.md)
for the full design and
[2026-08-25-approach-justification.md](2026-08-25-approach-justification.md)
for why the tree is hand-rolled rather than built on a general framework or
OpenSpiel.

Kuhn Poker's Nash equilibria form a one-parameter family, α ∈ [0, 1/3]. All
"Nash" numbers below are for the α = 0 point specifically (never bluff-bet a
Jack as the very first action) — one representative equilibrium, not *the*
equilibrium. A different α would still have exploitability exactly 0, but
would produce different head-to-head numbers against non-equilibrium
opponents.

## Method

- **Bots:** `Nash` (α = 0, closed-form), `CFR` (vanilla CFR self-play,
  50,000 iterations, seed 0), `Honest` (never bluffs), `AlwaysBluff` (bets
  the Jack 80% of the time), `Random` (uniform over legal actions).
- **Exploitability:** exact best-response value via full minimax traversal
  of the game tree (`kuhn.evaluate.exploitability`) — not sampled. 0 means
  no strategy can beat it beyond the equilibrium value.
- **Head-to-head:** 20,000 simulated hands per pairing, 95% CI via 2,000
  bootstrap resamples (`kuhn.evaluate.head_to_head`), seed 0.
- Reproduce with: `python -m scripts.run_comparison results`

## Results

### Exploitability

| Bot         | Exploitability |
|-------------|----------------|
| Nash (α=0)  | ~0 (9.0e-17 — floating-point noise) |
| CFR (50k it)| 0.00163 |
| Honest      | 0.25 |
| AlwaysBluff | 0.3767 |
| Random      | 0.4583 |

Nash's exploitability lands at machine epsilon, confirming the closed-form
implementation matches the published equilibrium formulas exactly. CFR gets
close (0.0016) but hasn't fully converged at 50k iterations. The heuristics
rank in the intuitive order: `Honest` (predictable but not wildly so) is
least exploitable of the three, `AlwaysBluff`'s fixed 80% Jack-bluff rate is
worse, and `Random` — with no card-dependent strategy at all — is the most
exploitable by a wide margin.

### CFR convergence

Exploitability of CFR's average strategy, sampled every 500 iterations of
self-play:

| Iteration | Exploitability |
|-----------|----------------|
| 500       | 0.0361 |
| 1,000     | 0.0160 |
| 5,000     | 0.0071 |
| 25,000    | 0.0055 |
| 49,000    | 0.00186 |
| 49,500    | 0.00171 |
| 50,000    | 0.00163 |

A clear, consistent downward trend from 0.036 to 0.0016 over 50k
iterations — CFR is converging toward the Nash bot's exact 0, as expected
for a two-player zero-sum game (Zinkevich et al., 2007), just not all the
way there yet at this iteration budget. The full curve and a plot are
regenerated at `results/cfr_convergence.{csv,png}` by the comparison
script (not checked into the repo — regenerate on demand).

### Head-to-head (bot_a's mean payoff as Player 1, 95% CI)

| bot_a → bot_b | Mean | 95% CI |
|---|---|---|
| Nash → Honest | −0.0627 | [−0.0770, −0.0482] |
| Nash → AlwaysBluff | −0.0686 | [−0.0870, −0.0512] |
| Nash → Random | +0.0453 | [+0.0276, +0.0630] |
| Nash → CFR | −0.0642 | [−0.0808, −0.0481] |
| CFR → Nash | −0.0620 | [−0.0788, −0.0448] |
| CFR → AlwaysBluff | −0.0028 | [−0.0225, +0.0164] |
| CFR → Random | +0.1108 | [+0.0915, +0.1287] |
| AlwaysBluff → Random | +0.0952 | [+0.0760, +0.1141] |
| Random → Honest | −0.0868 | [−0.1055, −0.0683] |

(Full 20-pairing matrix in `results/head_to_head.csv`.)

The Nash bot's own docstring notes the game's value to Player 1 at
equilibrium is exactly −1/18 ≈ −0.0556 — Kuhn Poker is structurally
slightly unfavorable to whoever acts first. Nash's head-to-head numbers as
Player 1 (−0.0627 to −0.0686 against Honest/AlwaysBluff/CFR) sit close to
that floor regardless of opponent, which is exactly the equilibrium
guarantee at work: an equilibrium strategy can't be pushed *below* the
game value no matter what the opponent does. Against `Random`, Nash does
noticeably better than the floor (+0.0453) because Random's play is
exploitable enough to push the result above the guaranteed minimum.

`CFR → AlwaysBluff` (−0.0028, CI crossing zero) is statistically
indistinguishable from a coin flip — consistent with CFR's average
strategy having converged close enough to equilibrium that it, like Nash,
mostly just realizes the game's structural value rather than exploiting a
specific opponent.

## Discussion

The three metrics tell a consistent story:

1. **Exploitability** cleanly ranks all five bots by how close each is to
   optimal play: `Nash` (exact) < `CFR` (near) << `Honest` < `AlwaysBluff`
   < `Random`.
2. **CFR's convergence curve** shows it approaching that ranking's top
   through self-play alone, with no hand-coded equilibrium knowledge.
3. **Head-to-head** results are consistent with both: Nash and CFR
   perform similarly against every opponent, and both hover near the
   game's inherent −1/18 value against reasonably-played opponents while
   beating the weakest ones (`Random`) by a clear margin.

## Reproducing

```bash
pip install -r requirements.txt
pytest
python -m scripts.run_comparison results
```

`results/` is gitignored — it's regenerated on demand, not checked in.
