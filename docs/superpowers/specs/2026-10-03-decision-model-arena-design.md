# Decision-Model Poker Arena — Design Spec

## Purpose

Use Kuhn Poker as a game-based eval for open-weight "typed decision models" (Jev-class models tracked by JevBench, and the Laya family). Wrap each model as a bot, play it against the classical Nash/CFR bots already in this repo, and measure how close it gets to equilibrium play. Then run an experiment ladder (zero-shot, state-shape variants, fine-tuning) to see what moves the numbers. The output is a blog-style whitepaper, `docs/REPORT_MODELS.md`.

This extends `kuhn-poker-lab` rather than starting a new repo. Every number comes from the same `kuhn/evaluate.py` used for the original Nash/CFR/heuristic comparison, so the two reports are directly comparable.

## Background

- **Typed decision models** take a state plus a typed question and return one of a fixed set of answers with calibrated probabilities in a single forward pass. They do not generate text. JevBench ranks 70+ open Jev-class models on accuracy, cost, latency and calibration.
- **Laya** is an open-weight (Apache 2.0) ModernBERT-large (421M) decision model. `laya-mlx` runs it on Apple Silicon only; this project runs everything through Hugging Face `transformers` on a Windows machine, so `laya-mlx` is out of scope.
- A model's output is a probability distribution over labels. That is the same shape as this repo's bot interface (information set to action distribution), so a model can play a mixed strategy, and its calibration affects play directly (bluff frequency is a probability).

## Components

New code lives beside the existing lab; existing modules are not changed except where noted.

- **`kuhn/bots/model_bot.py`** — `ModelBot` adapter. For an information set it (1) builds text with a state encoder, (2) pairs it with a typed question and fixed labels, (3) asks the model for label probabilities, (4) restricts to legal actions and renormalizes, (5) returns the action distribution. Labels are `check`/`bet` when no bet is faced and `fold`/`call` when a bet is faced.
- **`kuhn/encoders.py`** — state encoders: `info set -> prompt text`. Deterministic; distinct information sets must produce distinct prompts. Variants include minimal (card + history), rules-stated, pot/payoff context, natural-language history, and alternate label wording.
- **`kuhn/models/registry.py`** + **`models.yaml`** — manifest of candidate models: Hugging Face ID, scoring method, size, license, and notes. Excluded candidates are listed with the reason (API-only, no open weights, input format doesn't fit a typed question).
- **`kuhn/models/loaders.py`** — loads a manifest entry via `transformers` and exposes `score(prompt, labels) -> probabilities`. One code path for all models. Outputs are cached on disk by (model, prompt).
- **`kuhn/evaluate.py` (extended)** — adds `distance_from_nash(bot)`: per-information-set distance (L1 and KL) between the bot's action distribution and the Nash bot's, over all 12 information sets.
- **`experiments/`** — experiment configs (model × encoder × label wording × optional fine-tune checkpoint) and the rung definitions.
- **`kuhn/finetune.py`** — fine-tuning on (state text, CFR average-strategy distribution) pairs with a soft-label loss.
- **`scripts/run_models.py`** — runs configs, writes one CSV row per config with exploitability, head-to-head vs Nash and vs CFR (mean payoff + bootstrap CI), and distance from Nash.
- **`scripts/make_report_assets.py`** — generates the leaderboard table, the ladder plots, and the model × information-set heatmap from the CSVs. No hand-made figures.
- **`docs/REPORT_MODELS.md`** — the write-up (see Write-up).

## Experiment Ladder

Every rung is a config on the same harness, so tables line up.

- **Rung 0** — zero-shot, minimal encoder.
- **Rung 1** — zero-shot, state-shape and wording variants (rules stated, pot/payoff context, natural-language history, label wording).
- **Rung 2** — fine-tune a few top rung-0/1 models on CFR self-play data at 100 / 1k / 10k examples. The target is the soft equilibrium distribution, not a hard label. Also run one held-out variant (held-out card permutations or held-out state wording) to check for memorization.
- **Rung 3 (stretch)** — further tweaks, e.g. opponent tendencies in the state.

## Model Selection

- Start from the JevBench leaderboard and the Laya family. Keep open-weight models that run through `transformers`; drop API-only entries.
- Record every exclusion and its reason in `models.yaml` (also useful post material).
- Include non-Jev baselines for context: a uniform-label "model", and a generic small encoder with an untrained classification head, to show whether typed-decision training matters.
- A model whose input format does not fit a typed question is skipped, not forced.

## Data Flow

1. `models.yaml` lists candidates; experiment configs choose model × encoder × labels × checkpoint.
2. The runner builds a `ModelBot` per config.
3. It calls `exploitability()`, `head_to_head()` against Nash and CFR, and `distance_from_nash()`.
4. Results go to CSV, one row per config.
5. `make_report_assets.py` turns CSVs into figures and tables.

A model's full strategy is 12 queries (one per information set), so exploitability and distance from Nash are exact. Only head-to-head is sampled, using the existing bootstrap CI.

## Testing

- **Adapter validity**: `ModelBot` always returns a valid distribution over legal actions only, never assigning probability to an illegal action.
- **Encoder determinism and injectivity**: same information set gives the same prompt; different information sets never give the same prompt.
- **Pipeline sanity bot**: a fake model returning the Nash probabilities through the adapter must score exploitability ≈ 0 and distance from Nash ≈ 0.
- **Cache**: reruns return identical outputs.
- **Fine-tune data**: generated targets match the CFR average strategy for each information set.
- **Regression**: the existing 27 lab tests still pass.

## Write-up

`docs/REPORT_MODELS.md`, structured as a blog post:

- Headline leaderboard: distance from Nash and exploitability per model, plus head-to-head results against the classical bots.
- The ladder story: where zero-shot fails, which state tweaks help, how far fine-tuning closes the gap.
- "Where do they go wrong": model × information-set heatmap (e.g. never bluffing with a Jack, over-calling with a Queen).
- Honest caveats: Kuhn has only 12 information sets, so a fine-tuned model can nearly memorize the equilibrium; the held-out variant addresses this only partly. Leduc is the natural follow-up.

## Constraints and Assumptions

- Runs on a Windows machine via PyTorch/`transformers` (CPU, or NVIDIA GPU if present). One code path for all models.
- Fine-tuning targets models up to ~400M parameters and is expected to run locally. If a model is too heavy, the user is consulted before any cloud GPU spend.
- The "old-school bot" opponents are the existing Nash (α = 0) and CFR bots.
- Model availability and licenses are checked at implementation time; the candidate list is not fixed in this spec.

## Out of Scope

- Leduc or other games, and a general multi-game arena (a follow-up).
- Generative LLMs (chat or API).
- Any UI.
- Cloud GPU use without explicit approval.
- `laya-mlx` / Apple Silicon-specific runs.
- Changing the existing Nash, CFR or heuristic bots or their reports.
