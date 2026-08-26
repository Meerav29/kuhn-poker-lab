# Why Approach A (hand-rolled exact game tree)

Three approaches were considered for how to build the game/evaluation core. This records why Approach A was chosen, for future reference if the project's scope ever grows past Kuhn Poker.

## The options

**A — Hand-rolled tiny game tree.** Kuhn Poker's full extensive-form tree is small: 3 cards, one betting round, a handful of information sets per player. Enumerate it explicitly in `game.py`; every bot and evaluation routine is built directly against that enumeration.

**B — General extensive-form game engine.** Build a reusable information-set/CFR framework that could support other games later, not just Kuhn Poker.

**C — OpenSpiel (DeepMind's game-theory library).** Use OpenSpiel's built-in Kuhn Poker implementation and CFR solver; write only the heuristic bots and the plotting/report code ourselves.

## Why not B

Nothing in the brief calls for a multi-game framework — the goal is a deep, correct treatment of *this* game, not a reusable engine. Building B means designing abstractions (generic infoset representations, pluggable game rules) whose only consumer, for now, is a game small enough not to need them. That's speculative generality against a one-game brief, and it would dilute effort that's better spent on the bots and the evaluation rigor the write-up actually depends on.

## Why not C

OpenSpiel is correct and battle-tested, and would have been the fastest path to working CFR. Two things ruled it out:

1. **Dependency weight.** OpenSpiel is a C++ library with Python bindings (pybind11), and installation friction on Windows is a known pain point — exactly the kind of setup cost this project's "ship in a day or two, from a Windows machine" pattern (see `daily-log.md`) is meant to avoid.
2. **Auditability for the write-up.** The deliverable is a whitepaper-style report making specific numerical claims (exploitability values, convergence behavior). If the game tree and best-response computation live inside a library, those claims rest on "trust OpenSpiel's implementation" rather than "here is the exact code that computed this number, read it end to end." For a report whose whole point is rigor, a hand-rolled, inspectable implementation is a stronger foundation than a faster one.

## Why A

Kuhn Poker's tree is genuinely small enough that hand-rolling it is *less* total work than integrating OpenSpiel would have been on this machine, and it produces exact — not sampled or library-mediated — exploitability numbers. Every number in the eventual report traces to code in this repo that can be read, tested, and explained line by line. That directly serves the "deep and good" bar this project is aiming for, more than either a premature framework (B) or an opaque dependency (C) would.
