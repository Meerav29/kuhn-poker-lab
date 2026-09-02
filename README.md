# Kuhn Poker Lab

A research sandbox comparing three approaches to playing Kuhn Poker — a closed-form Nash equilibrium bot, a CFR self-play bot, and hand-coded heuristic bots — evaluated by exact exploitability and head-to-head simulation, in support of a whitepaper-style write-up.

See [`docs/2026-08-25-kuhn-poker-lab-design.md`](docs/2026-08-25-kuhn-poker-lab-design.md) for the full design spec, and [`docs/2026-08-25-approach-justification.md`](docs/2026-08-25-approach-justification.md) for why the game engine is hand-rolled rather than built on a general framework or OpenSpiel.

Status: implemented — game engine, all three bots (Nash, CFR, heuristics), exploitability evaluation, and head-to-head simulation are built and tested. The whitepaper write-up (`docs/REPORT.md`) is the remaining piece.

## Running

```bash
pip install -r requirements.txt
pytest                          # run the test suite
python -m scripts.run_comparison <output_dir>   # generate exploitability.csv, head_to_head.csv, and the CFR convergence plot
```
