# Decision-Model Poker Arena Implementation Plan (rungs 0–1 + rung-2 feasibility)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wrap open-weight typed decision models (Laya, plus a local-LLM logprob baseline) as Kuhn Poker bots, evaluate them with the lab's exact exploitability and head-to-head code against the Nash/CFR bots, and run the zero-shot and prompt-variant rungs of the experiment ladder.

**Architecture:** A `Scorer` interface (`score(state, question) -> {label: prob}`) hides each model family's API. `ModelBot` turns an information set into prompt text (encoder) plus a typed question, asks the scorer, and maps label probabilities back to the lab's `action_probs(infoset, legal)` bot interface. A runner expands experiment configs, evaluates every bot, and writes tidy CSVs that a report-assets script turns into a leaderboard and heatmap.

**Tech Stack:** Python 3.12, pytest, PyYAML, matplotlib (existing), `torch` + `transformers` (already installed: torch 2.13 ROCm, transformers 5.17, GPU visible), `laya` (installed in Task 5).

**Spec:** `docs/superpowers/specs/2026-10-03-decision-model-arena-design.md` (read its **Amendments** section; it supersedes earlier text).

**Scope note:** Rung 2 (fine-tuning) is **not** implemented here. Task 10 is a feasibility probe whose result decides the separate rung-2 plan.

## Global Constraints

- Runs on a Windows machine via PyTorch/`transformers` (CPU, or NVIDIA GPU if present). Shell commands in this plan use bash syntax (Git Bash); run from the repo root `C:\Users\meera\Github-Projects\kuhn-poker-lab`.
- The existing bot interface is `bot.action_probs(infoset: str, legal: tuple) -> dict` returning `{"p": prob, "b": prob}`. Actions: `"p"` = check/fold, `"b"` = bet/call. An infoset key is `f"{card}{history}"` with card 0/1/2 = Jack/Queen/King and history in `{"", "p", "b", "pb"}` (12 infosets total).
- The "old-school bot" opponents are the existing Nash (α = 0) and CFR bots.
- Fine-tuning targets models up to ~400M parameters and is expected to run locally. If a model is too heavy, consult the user before any cloud GPU spend.
- The existing 27 lab tests must keep passing after every task (`python -m pytest -q`).
- Out of scope: Leduc/other games, generative LLMs as contestants, any UI, cloud GPU use without approval, `laya-mlx`/Apple Silicon runs, changes to existing bots or `docs/REPORT.md`.
- Commit messages end with: `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.

## File Structure

| File | Responsibility |
|---|---|
| `kuhn/encoders.py` | Pure `infoset -> prompt text` encoders (4 variants) |
| `kuhn/models/__init__.py` | Package marker |
| `kuhn/models/scorer.py` | `Question`, `UniformScorer`, `LookupScorer`, `CachedScorer` |
| `kuhn/bots/model_bot.py` | `build_question`, `ModelBot` adapter |
| `kuhn/models/sanity.py` | `nash_sanity_bot`: Nash probabilities pushed through the full adapter |
| `kuhn/evaluate.py` (modify) | add `distance_from_nash` |
| `kuhn/models/laya_scorer.py` | `extract_probs`, `LayaScorer` |
| `kuhn/models/logprob_scorer.py` | `probs_from_logprobs`, `LogprobScorer` |
| `kuhn/models/registry.py` + `models.yaml` | manifest loading, `build_scorer` |
| `scripts/run_models.py` | config expansion, evaluation, CSV output |
| `experiments/rung0.yaml`, `experiments/rung1.yaml` | experiment configs |
| `kuhn/finetune_data.py` | dataset builder for the later rung-2 plan |
| `scripts/probe_laya.py`, `scripts/probe_laya_training.py` | API and training-feasibility probes |
| `scripts/make_report_assets.py` | leaderboard table + heatmap |
| `pytest.ini` | registers the `slow` marker |

---

### Task 1: State encoders

**Files:**
- Create: `kuhn/encoders.py`
- Test: `tests/test_encoders.py`

**Interfaces:**
- Consumes: `kuhn.game.CARD_NAMES`, `kuhn.game.all_infosets()`
- Produces: `ENCODERS: dict[str, Callable[[str], str]]`, `get_encoder(name: str) -> Callable[[str], str]` (raises `ValueError` for unknown names). Encoder names: `"minimal"`, `"natural"`, `"rules"`, `"payoff"`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_encoders.py
import pytest

from kuhn import game
from kuhn.encoders import ENCODERS, get_encoder


@pytest.mark.parametrize("name", sorted(ENCODERS))
def test_encoder_is_injective_over_all_infosets(name):
    enc = get_encoder(name)
    prompts = [enc(i) for i in game.all_infosets()]
    assert len(prompts) == 12
    assert len(set(prompts)) == 12


@pytest.mark.parametrize("name", sorted(ENCODERS))
def test_encoder_is_deterministic(name):
    enc = get_encoder(name)
    for infoset in game.all_infosets():
        assert enc(infoset) == enc(infoset)


def test_minimal_exact_text():
    assert get_encoder("minimal")("2pb") == "card=K history=pb"
    assert get_encoder("minimal")("0") == "card=J history=-"


def test_natural_exact_text():
    assert get_encoder("natural")("0b") == "You hold the Jack. Situation: the opponent bet 1 chip."


def test_rules_and_payoff_extend_natural():
    nat = get_encoder("natural")("1p")
    assert nat in get_encoder("rules")("1p")
    assert nat in get_encoder("payoff")("1p")
    assert "pot is 2 chips" in get_encoder("payoff")("1p")
    assert "pot is 3 chips" in get_encoder("payoff")("1b")


def test_unknown_encoder_raises():
    with pytest.raises(ValueError):
        get_encoder("nope")
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_encoders.py -q`
Expected: FAIL (`ModuleNotFoundError: No module named 'kuhn.encoders'`)

- [ ] **Step 3: Write the implementation**

```python
# kuhn/encoders.py
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
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_encoders.py -q`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add kuhn/encoders.py tests/test_encoders.py
git commit -m "feat: add infoset-to-prompt state encoders

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Scorer interface, baselines, and disk cache

**Files:**
- Create: `kuhn/models/__init__.py` (empty), `kuhn/models/scorer.py`
- Test: `tests/test_scorer.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `Question(kind: str, instructions: str, options: tuple)` frozen dataclass; `options` is `((label, action), ...)` with action `"p"` or `"b"`; properties/methods `labels -> tuple[str, ...]`, `key() -> str`.
  - Scorer protocol: attribute `scorer_id: str`; method `score(state: str, question: Question) -> dict[str, float]` (probabilities over `question.labels`).
  - `UniformScorer()`, `LookupScorer(table: dict[str, dict[str, float]], scorer_id: str = "lookup")`, `CachedScorer(inner, cache_dir)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_scorer.py
import pytest

from kuhn.models.scorer import CachedScorer, LookupScorer, Question, UniformScorer

Q = Question(kind="choice", instructions="Check or bet?", options=(("check", "p"), ("bet", "b")))


def test_question_labels_and_key():
    assert Q.labels == ("check", "bet")
    other = Question(kind="choice", instructions="Check or bet?", options=(("wait", "p"), ("raise", "b")))
    assert Q.key() != other.key()
    assert Q.key() == Q.key()


def test_uniform_scorer():
    assert UniformScorer().score("anything", Q) == {"check": 0.5, "bet": 0.5}


def test_lookup_scorer_returns_table_row_and_raises_on_missing():
    s = LookupScorer({"s1": {"check": 0.9, "bet": 0.1}})
    assert s.score("s1", Q) == {"check": 0.9, "bet": 0.1}
    with pytest.raises(KeyError):
        s.score("unknown", Q)


class CountingScorer:
    scorer_id = "counting/model:1"

    def __init__(self):
        self.calls = 0

    def score(self, state, question):
        self.calls += 1
        return {"check": 0.25, "bet": 0.75}


def test_cached_scorer_hits_cache_in_memory_and_across_instances(tmp_path):
    inner = CountingScorer()
    cached = CachedScorer(inner, tmp_path)
    assert cached.score("s", Q) == {"check": 0.25, "bet": 0.75}
    assert cached.score("s", Q) == {"check": 0.25, "bet": 0.75}
    assert inner.calls == 1

    inner2 = CountingScorer()
    again = CachedScorer(inner2, tmp_path)
    assert again.score("s", Q) == {"check": 0.25, "bet": 0.75}
    assert inner2.calls == 0

    again.score("different state", Q)
    assert inner2.calls == 1
    assert cached.scorer_id == "counting/model:1"
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_scorer.py -q`
Expected: FAIL (`ModuleNotFoundError: No module named 'kuhn.models'`)

- [ ] **Step 3: Write the implementation**

`kuhn/models/__init__.py` is an empty file. Then:

```python
# kuhn/models/scorer.py
"""Scorer interface: the one thing every model family implements.

A scorer takes a state (prompt text) and a typed Question and returns a
probability for each of the question's labels. ModelBot is written only
against this interface.
"""
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Question:
    kind: str        # "choice" (pick a label) or "noul" (yes/no probability)
    instructions: str
    options: tuple   # ((label, action), ...) where action is 'p' or 'b'

    @property
    def labels(self) -> tuple:
        return tuple(label for label, _ in self.options)

    def key(self) -> str:
        return json.dumps([self.kind, self.instructions, [list(o) for o in self.options]])


class UniformScorer:
    scorer_id = "uniform"

    def score(self, state: str, question: Question) -> dict:
        p = 1.0 / len(question.labels)
        return {label: p for label in question.labels}


class LookupScorer:
    """Returns a fixed row per state. Used to push a known strategy
    (e.g. Nash) through the full adapter as a pipeline sanity check."""

    def __init__(self, table: dict, scorer_id: str = "lookup"):
        self._table = table
        self.scorer_id = scorer_id

    def score(self, state: str, question: Question) -> dict:
        return dict(self._table[state])


class CachedScorer:
    """Wraps a scorer with an on-disk JSON cache keyed by (state, question)."""

    def __init__(self, inner, cache_dir):
        self.inner = inner
        self.scorer_id = inner.scorer_id
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", inner.scorer_id)
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        self._path = Path(cache_dir) / f"{safe}.json"
        self._data = json.loads(self._path.read_text(encoding="utf-8")) if self._path.exists() else {}

    def score(self, state: str, question: Question) -> dict:
        key = hashlib.sha256((state + "\x00" + question.key()).encode("utf-8")).hexdigest()
        if key not in self._data:
            self._data[key] = self.inner.score(state, question)
            self._path.write_text(json.dumps(self._data), encoding="utf-8")
        return dict(self._data[key])
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_scorer.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add kuhn/models/__init__.py kuhn/models/scorer.py tests/test_scorer.py
git commit -m "feat: add Scorer interface with uniform, lookup, and disk-cached scorers

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 3: ModelBot adapter and the Nash pipeline sanity bot

**Files:**
- Create: `kuhn/bots/model_bot.py`, `kuhn/models/sanity.py`
- Test: `tests/test_model_bot.py`

**Interfaces:**
- Consumes: `get_encoder` (Task 1); `Question`, `UniformScorer`, `LookupScorer` (Task 2); `kuhn.bots.nash.NashBot`; `kuhn.evaluate.exploitability`; `kuhn.bots.heuristics.RandomBot`.
- Produces:
  - `build_question(infoset: str, qtype: str = "choice", wording: str = "plain") -> Question`. `qtype` in `{"choice", "noul"}`; `wording` in `{"plain", "alt"}` (only affects `choice`).
  - `ModelBot(scorer, encoder: str = "minimal", qtype: str = "choice", wording: str = "plain", name: str | None = None)` with `.action_probs(infoset, legal) -> dict` over exactly the `legal` actions, and `.name`.
  - `nash_sanity_bot(encoder="minimal", qtype="choice", wording="plain", alpha=0.0) -> ModelBot`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_model_bot.py
import pytest

from kuhn import game
from kuhn.bots.heuristics import RandomBot
from kuhn.bots.model_bot import ModelBot, build_question
from kuhn.evaluate import exploitability
from kuhn.models.sanity import nash_sanity_bot
from kuhn.models.scorer import UniformScorer

ENCODERS = ["minimal", "natural", "rules", "payoff"]
SETUPS = [("choice", "plain"), ("choice", "alt"), ("noul", "plain")]


def test_build_question_choice_plain():
    q = build_question("1", "choice", "plain")
    assert q.options == (("check", "p"), ("bet", "b"))
    assert q.instructions == "Do you check or bet?"
    q = build_question("2pb", "choice", "plain")
    assert q.options == (("fold", "p"), ("call", "b"))
    assert q.instructions == "The opponent bet. Do you fold or call?"


def test_build_question_alt_and_noul():
    assert build_question("0b", "choice", "alt").options == (("give up", "p"), ("match", "b"))
    q = build_question("0p", "noul")
    assert q.options == (("yes", "b"), ("no", "p"))
    assert q.instructions == "Should you bet?"
    assert build_question("1b", "noul").instructions == "The opponent bet. Should you call?"


def test_build_question_rejects_bad_args():
    with pytest.raises(ValueError):
        build_question("0", "score")
    with pytest.raises(ValueError):
        build_question("0", "choice", "nope")


@pytest.mark.parametrize("qtype,wording", SETUPS)
def test_uniform_model_matches_random_bot(qtype, wording):
    bot = ModelBot(UniformScorer(), qtype=qtype, wording=wording)
    for infoset in game.all_infosets():
        assert bot.action_probs(infoset, game.ACTIONS) == {"p": 0.5, "b": 0.5}
    assert abs(exploitability(bot) - exploitability(RandomBot())) < 1e-9


def test_action_probs_only_covers_legal_actions():
    class AlwaysBet:
        scorer_id = "always-bet"

        def score(self, state, question):
            return {label: (1.0 if action == "b" else 0.0) for label, action in question.options}

    bot = ModelBot(AlwaysBet())
    assert bot.action_probs("0", ("b",)) == {"b": 1.0}
    assert "p" not in bot.action_probs("0", ("b",))


def test_all_zero_scorer_output_falls_back_to_uniform():
    class AllZero:
        scorer_id = "zero"

        def score(self, state, question):
            return {label: 0.0 for label in question.labels}

    assert ModelBot(AllZero()).action_probs("1", game.ACTIONS) == {"p": 0.5, "b": 0.5}


def test_probabilities_are_renormalized_and_memoized():
    class Counting:
        scorer_id = "counting"
        calls = 0

        def score(self, state, question):
            Counting.calls += 1
            return {question.labels[0]: 2.0, question.labels[1]: 6.0}

    bot = ModelBot(Counting())
    dist = bot.action_probs("0", game.ACTIONS)
    assert dist == {"p": 0.25, "b": 0.75}
    bot.action_probs("0", game.ACTIONS)
    assert Counting.calls == 1


@pytest.mark.parametrize("encoder", ENCODERS)
@pytest.mark.parametrize("qtype,wording", SETUPS)
def test_nash_sanity_bot_is_unexploitable(encoder, qtype, wording):
    bot = nash_sanity_bot(encoder=encoder, qtype=qtype, wording=wording)
    assert exploitability(bot) < 1e-9
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_model_bot.py -q`
Expected: FAIL (`ModuleNotFoundError: No module named 'kuhn.bots.model_bot'`)

- [ ] **Step 3: Write the implementation**

```python
# kuhn/bots/model_bot.py
"""ModelBot: adapts any Scorer to the lab's bot interface.

information set -> prompt text (encoder) + typed Question -> label
probabilities (scorer) -> action distribution over the legal actions.
"""
from kuhn.encoders import get_encoder
from kuhn.models.scorer import Question

LABELSETS = {
    "plain": {"open": ("check", "bet"), "facing": ("fold", "call")},
    "alt": {"open": ("wait", "raise"), "facing": ("give up", "match")},
}


def build_question(infoset: str, qtype: str = "choice", wording: str = "plain") -> Question:
    facing = infoset[1:].endswith("b")
    prefix = "The opponent bet. " if facing else ""
    if qtype == "choice":
        if wording not in LABELSETS:
            raise ValueError(f"unknown wording {wording!r}; choose from {sorted(LABELSETS)}")
        pass_label, bet_label = LABELSETS[wording]["facing" if facing else "open"]
        options = ((pass_label, "p"), (bet_label, "b"))
        instructions = f"{prefix}Do you {pass_label} or {bet_label}?"
    elif qtype == "noul":
        options = (("yes", "b"), ("no", "p"))
        instructions = f"{prefix}Should you {'call' if facing else 'bet'}?"
    else:
        raise ValueError(f"unknown qtype {qtype!r}; choose 'choice' or 'noul'")
    return Question(kind=qtype, instructions=instructions, options=options)


class ModelBot:
    def __init__(self, scorer, encoder: str = "minimal", qtype: str = "choice",
                 wording: str = "plain", name: str = None):
        self.scorer = scorer
        self._encode = get_encoder(encoder)
        self.qtype = qtype
        self.wording = wording
        self.name = name or f"{scorer.scorer_id}|{encoder}|{qtype}|{wording}"
        self._memo = {}  # at most 12 infosets: never query the model twice for the same one

    def _label_probs(self, infoset: str) -> dict:
        if infoset not in self._memo:
            question = build_question(infoset, self.qtype, self.wording)
            probs = self.scorer.score(self._encode(infoset), question)
            self._memo[infoset] = {
                action: max(probs.get(label, 0.0), 0.0) for label, action in question.options
            }
        return self._memo[infoset]

    def action_probs(self, infoset: str, legal: tuple) -> dict:
        by_action = self._label_probs(infoset)
        dist = {a: by_action.get(a, 0.0) for a in legal}
        total = sum(dist.values())
        if total <= 0:
            return {a: 1.0 / len(legal) for a in legal}
        return {a: v / total for a, v in dist.items()}
```

```python
# kuhn/models/sanity.py
"""Pipeline sanity check: push the Nash strategy through the full
encoder -> Question -> scorer -> ModelBot path. Its exploitability must
be ~0; if it isn't, the adapter (not any model) is broken."""
from kuhn import game
from kuhn.bots.model_bot import ModelBot, build_question
from kuhn.bots.nash import NashBot
from kuhn.encoders import get_encoder
from kuhn.models.scorer import LookupScorer


def nash_sanity_bot(encoder: str = "minimal", qtype: str = "choice",
                    wording: str = "plain", alpha: float = 0.0) -> ModelBot:
    nash = NashBot(alpha=alpha)
    enc = get_encoder(encoder)
    table = {}
    for infoset in game.all_infosets():
        question = build_question(infoset, qtype, wording)
        dist = nash.action_probs(infoset, game.ACTIONS)
        table[enc(infoset)] = {label: dist[action] for label, action in question.options}
    return ModelBot(LookupScorer(table, "sanity-nash"), encoder=encoder, qtype=qtype,
                    wording=wording, name="sanity-nash")
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_model_bot.py -q`
Expected: PASS. Then run the whole suite: `python -m pytest -q`, expecting all existing tests plus the new ones to pass.

- [ ] **Step 5: Commit**

```bash
git add kuhn/bots/model_bot.py kuhn/models/sanity.py tests/test_model_bot.py
git commit -m "feat: add ModelBot adapter and Nash pipeline sanity bot

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 4: `distance_from_nash`

**Files:**
- Modify: `kuhn/evaluate.py` (add imports at the top and a function at the end)
- Test: `tests/test_distance.py`

**Interfaces:**
- Consumes: `NashBot`, `game.all_infosets`, `game.legal_actions`.
- Produces: `distance_from_nash(bot, alpha: float | None = None, grid: int = 34) -> dict` with keys `"alpha"` (the equilibrium used), `"total_l1"` (sum over infosets), `"per_infoset"` (`{infoset: {"l1": float, "kl": float}}`). `alpha=None` picks the α in `[0, 1/3]` (grid of `grid + 1` points) minimizing `total_l1`; ties go to the smaller α.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_distance.py
from kuhn import game
from kuhn.bots.heuristics import HonestBot, RandomBot
from kuhn.bots.nash import NashBot
from kuhn.evaluate import distance_from_nash


def test_nash_bot_has_zero_distance_at_its_own_alpha():
    d = distance_from_nash(NashBot(alpha=0.0), alpha=0.0)
    assert d["total_l1"] < 1e-12
    assert all(r["l1"] < 1e-12 and r["kl"] < 1e-9 for r in d["per_infoset"].values())


def test_nearest_alpha_recovers_other_family_members():
    d = distance_from_nash(NashBot(alpha=1 / 6))
    assert abs(d["alpha"] - 1 / 6) < 1e-9
    assert d["total_l1"] < 1e-9


def test_per_infoset_covers_all_twelve():
    d = distance_from_nash(RandomBot())
    assert set(d["per_infoset"]) == set(game.all_infosets())


def test_weak_bots_are_far_from_nash():
    assert distance_from_nash(RandomBot())["total_l1"] > 1.0
    assert distance_from_nash(HonestBot())["total_l1"] > 1.0


def test_kl_is_finite_when_bot_never_plays_a_nash_action():
    d = distance_from_nash(HonestBot(), alpha=0.0)
    assert all(r["kl"] == r["kl"] and r["kl"] < 1e6 for r in d["per_infoset"].values())
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_distance.py -q`
Expected: FAIL (`ImportError: cannot import name 'distance_from_nash'`)

- [ ] **Step 3: Write the implementation**

Edit `kuhn/evaluate.py`: change the imports at the top from

```python
import random

from kuhn import game
```

to

```python
import math
import random

from kuhn import game
from kuhn.bots.nash import NashBot
```

and append to the end of the file:

```python
def _kl(nash_probs, bot_probs, eps: float = 1e-6) -> float:
    """KL(nash || bot). The bot's probabilities are smoothed by eps so a bot
    that never takes an action Nash sometimes takes is penalised heavily
    but finitely."""
    return sum(q * math.log(q / (p + eps)) for q, p in zip(nash_probs, bot_probs) if q > 0)


def distance_from_nash(bot, alpha: float = None, grid: int = 34) -> dict:
    """Per-infoset distance (L1 and KL) between `bot` and a Nash bot.

    Kuhn's equilibria form a family indexed by alpha in [0, 1/3], so by
    default the *nearest* family member (smallest total L1) is used and
    reported. Pass `alpha` to measure against one fixed member."""
    infosets = game.all_infosets()
    bot_dists = {}
    for key in infosets:
        d = bot.action_probs(key, game.legal_actions(key[1:]))
        bot_dists[key] = (d.get("p", 0.0), d.get("b", 0.0))

    def rows_for(a: float) -> dict:
        nash = NashBot(alpha=a)
        rows = {}
        for key in infosets:
            nd = nash.action_probs(key, game.ACTIONS)
            n = (nd["p"], nd["b"])
            b = bot_dists[key]
            rows[key] = {"l1": abs(n[0] - b[0]) + abs(n[1] - b[1]), "kl": _kl(n, b)}
        return rows

    candidates = [alpha] if alpha is not None else [(1 / 3) * i / grid for i in range(grid + 1)]
    best = None
    for a in candidates:
        rows = rows_for(a)
        total = sum(r["l1"] for r in rows.values())
        if best is None or total < best["total_l1"] - 1e-12:
            best = {"alpha": a, "total_l1": total, "per_infoset": rows}
    return best
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_distance.py -q` then `python -m pytest -q`
Expected: PASS; the whole suite passes (no circular import: `nash.py` imports nothing from `kuhn`).

- [ ] **Step 5: Commit**

```bash
git add kuhn/evaluate.py tests/test_distance.py
git commit -m "feat: add per-infoset distance_from_nash with nearest-alpha search

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Laya API probe, then `LayaScorer`

This task has a **gate**: the Laya model card documents the `choice` output only as "Selected label", not per-label probabilities. The probe checks what the real output contains before `LayaScorer` depends on it.

**Files:**
- Create: `scripts/probe_laya.py`, `kuhn/models/laya_scorer.py`, `tests/fixtures/laya_raw.json` (written by the probe)
- Modify: `requirements.txt`
- Test: `tests/test_laya_scorer.py`

**Interfaces:**
- Consumes: `Question` (Task 2).
- Produces: `extract_probs(answer: dict, question: Question) -> dict[str, float]`; `LayaScorer(repo: str = "convaiinnovations/laya", subfolder: str | None = None, name: str = "laya-en")` with `.scorer_id` and `.score(state, question)`.

- [ ] **Step 1: Install and confirm the download with the user**

`pip install laya` and the first `laya.load` download the English checkpoint (~808 MB from Hugging Face). Ask the user to confirm before running the install and probe, and state the package, source and size.

Run: `pip install laya` then `python -c "import laya; print(laya.__file__)"`
Expected: prints a path. If `pip install laya` fails or breaks the existing torch install (`python -c "import torch; print(torch.cuda.is_available())"` must still print `True`), STOP and report to the user.

- [ ] **Step 2: Write and run the probe**

```python
# scripts/probe_laya.py
"""Prints (and saves) Laya's raw predict() output for one choice and one
noul question, so LayaScorer is written against reality, not the docs."""
import json
import os

import laya

agent = laya.load("convaiinnovations/laya")
state = "card=K history=b"
questions = {
    "act": {"type": "choice", "instructions": "The opponent bet. Do you fold or call?",
            "criteria": ["fold", "call"]},
    "call_yes": {"type": "noul", "instructions": "The opponent bet. Should you call?"},
}
result = agent.predict(state, questions)
text = json.dumps(result, indent=2, default=str)
print(text)

os.makedirs("tests/fixtures", exist_ok=True)
with open("tests/fixtures/laya_raw.json", "w", encoding="utf-8") as f:
    f.write(text)
```

Run: `python -m scripts.probe_laya`
Expected: prints a JSON structure with `answers` -> `act` and `call_yes`.

- [ ] **Step 3: Gate on the real output shape**

Open `tests/fixtures/laya_raw.json` and check two things:
1. `answers["act"]` holds **per-label probabilities** (under a key such as `probs`, `probabilities` or `scores`) and not only the selected label.
2. `answers["call_yes"]["noul"]` is a plain number in [0, 1].

If (1) holds under a key not in `("probs", "probabilities", "scores")`, add that key to `_PROB_KEYS` in Step 5. If (2) is a dict or the choice answer has no probabilities at all, **STOP and report to the user**: the "calibrated probabilities" the benchmark idea relies on aren't exposed, and the design needs a decision (for example, falling back to `noul`-only, or reading logits another way).

- [ ] **Step 4: Write the failing tests**

```python
# tests/test_laya_scorer.py
import pytest

from kuhn.models.laya_scorer import extract_probs
from kuhn.models.scorer import Question

CHOICE = Question("choice", "Do you fold or call?", (("fold", "p"), ("call", "b")))
NOUL = Question("noul", "Should you call?", (("yes", "b"), ("no", "p")))


def test_extract_choice_probs_normalizes():
    out = extract_probs({"choice": "call", "probs": {"fold": 0.2, "call": 0.6}}, CHOICE)
    assert out == pytest.approx({"fold": 0.25, "call": 0.75})


def test_extract_choice_accepts_alternate_prob_keys():
    out = extract_probs({"choice": "fold", "probabilities": {"fold": 0.9, "call": 0.1}}, CHOICE)
    assert out == pytest.approx({"fold": 0.9, "call": 0.1})


def test_extract_choice_without_probabilities_raises():
    with pytest.raises(KeyError):
        extract_probs({"choice": "call"}, CHOICE)


def test_extract_noul():
    out = extract_probs({"noul": 0.7}, NOUL)
    assert out == pytest.approx({"yes": 0.7, "no": 0.3})


def test_extract_noul_non_numeric_raises():
    with pytest.raises(ValueError):
        extract_probs({"noul": {"yes": 0.7, "no": 0.2, "unknown": 0.1}}, NOUL)
```

- [ ] **Step 5: Run to verify it fails, then implement**

Run: `python -m pytest tests/test_laya_scorer.py -q`
Expected: FAIL (`ModuleNotFoundError: No module named 'kuhn.models.laya_scorer'`)

```python
# kuhn/models/laya_scorer.py
"""Scorer for the Laya family (https://huggingface.co/convaiinnovations/laya).

Laya's `predict(state, questions)` takes typed questions and returns an
`answers` dict. Choice answers must carry per-label probabilities; noul
answers carry the probability of "yes". See scripts/probe_laya.py.
"""
from kuhn.models.scorer import Question

_PROB_KEYS = ("probs", "probabilities", "scores")


def extract_probs(answer: dict, question: Question) -> dict:
    if question.kind == "noul":
        p = answer["noul"]
        if isinstance(p, bool) or not isinstance(p, (int, float)):
            raise ValueError(f"expected numeric noul probability, got {p!r}")
        return {"yes": float(p), "no": 1.0 - float(p)}
    for key in _PROB_KEYS:
        if key in answer:
            raw = answer[key]
            break
    else:
        raise KeyError(f"choice answer has none of {_PROB_KEYS}: keys={sorted(answer)}")
    probs = {label: float(raw[label]) for label in question.labels}
    total = sum(probs.values())
    return {label: v / total for label, v in probs.items()}


def _to_laya_question(question: Question) -> dict:
    spec = {"type": question.kind, "instructions": question.instructions}
    if question.kind == "choice":
        spec["criteria"] = list(question.labels)
    return spec


class LayaScorer:
    def __init__(self, repo: str = "convaiinnovations/laya", subfolder: str = None,
                 name: str = "laya-en"):
        import laya  # lazy: keeps the rest of the package importable without laya

        self.scorer_id = name
        self._agent = laya.load(repo, subfolder=subfolder) if subfolder else laya.load(repo)

    def score(self, state: str, question: Question) -> dict:
        result = self._agent.predict(state, {"q": _to_laya_question(question)})
        return extract_probs(result["answers"]["q"], question)
```

Append `laya` to `requirements.txt` (one line), and add `pyyaml`, `torch`, `transformers` if they are not already listed.

Run: `python -m pytest tests/test_laya_scorer.py -q`
Expected: PASS

- [ ] **Step 6: Smoke-test LayaScorer on the real model**

Run:
```bash
python -c "
from kuhn.models.laya_scorer import LayaScorer
from kuhn.bots.model_bot import ModelBot
bot = ModelBot(LayaScorer(), encoder='natural')
from kuhn import game
for i in game.all_infosets(): print(i, bot.action_probs(i, game.ACTIONS))
"
```
Expected: 12 lines, each a valid `{"p": ..., "b": ...}` summing to 1. If it raises, fix per the Step 3 gate rules.

- [ ] **Step 7: Commit**

```bash
git add scripts/probe_laya.py kuhn/models/laya_scorer.py tests/test_laya_scorer.py tests/fixtures/laya_raw.json requirements.txt
git commit -m "feat: add Laya scorer with API probe and fixture

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Local-LLM logprob baseline scorer

**Files:**
- Create: `kuhn/models/logprob_scorer.py`, `pytest.ini`
- Test: `tests/test_logprob_scorer.py`

**Interfaces:**
- Consumes: `Question` (Task 2).
- Produces: `probs_from_logprobs(logprobs: dict[str, float]) -> dict[str, float]` (softmax); `LogprobScorer(model_id: str = "Qwen/Qwen2.5-0.5B-Instruct", name: str | None = None)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_logprob_scorer.py
import math
import os

import pytest

from kuhn.models.logprob_scorer import LogprobScorer, probs_from_logprobs
from kuhn.models.scorer import Question


def test_probs_from_logprobs_is_a_softmax():
    out = probs_from_logprobs({"a": math.log(1.0), "b": math.log(3.0)})
    assert out == pytest.approx({"a": 0.25, "b": 0.75})


def test_probs_from_logprobs_is_stable_for_very_negative_values():
    out = probs_from_logprobs({"a": -1000.0, "b": -1001.0})
    assert sum(out.values()) == pytest.approx(1.0)
    assert out["a"] > out["b"]


@pytest.mark.slow
@pytest.mark.skipif(not os.environ.get("RUN_SLOW"), reason="downloads a model; set RUN_SLOW=1")
def test_logprob_scorer_returns_a_distribution_over_labels():
    q = Question("choice", "Do you check or bet?", (("check", "p"), ("bet", "b")))
    out = LogprobScorer().score("card=K history=-", q)
    assert set(out) == {"check", "bet"}
    assert sum(out.values()) == pytest.approx(1.0)
```

`pytest.ini`:

```ini
[pytest]
markers =
    slow: downloads or runs a real model (set RUN_SLOW=1 to enable)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_logprob_scorer.py -q`
Expected: FAIL (`ModuleNotFoundError: No module named 'kuhn.models.logprob_scorer'`)

- [ ] **Step 3: Write the implementation**

```python
# kuhn/models/logprob_scorer.py
"""Generic non-Jev baseline: read the next-token logprobs of a small local
causal LM over the answer labels (the same idea as the XavierJev entry on
the Jev catalogue). Shows what an off-the-shelf LM does with no
decision-model training."""
import math

from kuhn.models.scorer import Question


def probs_from_logprobs(logprobs: dict) -> dict:
    m = max(logprobs.values())
    exp = {k: math.exp(v - m) for k, v in logprobs.items()}
    z = sum(exp.values())
    return {k: v / z for k, v in exp.items()}


class LogprobScorer:
    def __init__(self, model_id: str = "Qwen/Qwen2.5-0.5B-Instruct", name: str = None):
        import torch  # lazy imports keep unit tests and the rest of the package light
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.scorer_id = name or f"logprob-{model_id.split('/')[-1]}"
        self._torch = torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tok = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModelForCausalLM.from_pretrained(model_id).to(self._device).eval()

    @staticmethod
    def _prompt(state: str, question: Question) -> str:
        options = ", ".join(question.labels)
        return f"{state}\n{question.instructions} Answer with exactly one of: {options}.\nAnswer:"

    def score(self, state: str, question: Question) -> dict:
        ids = {label: self._tok(" " + label, add_special_tokens=False).input_ids[0]
               for label in question.labels}
        if len(set(ids.values())) != len(ids):
            raise ValueError(f"labels share a first token, cannot score them apart: {question.labels}")
        enc = self._tok(self._prompt(state, question), return_tensors="pt").to(self._device)
        with self._torch.no_grad():
            logits = self._model(**enc).logits[0, -1]
        logp = self._torch.log_softmax(logits.float(), dim=-1)
        return probs_from_logprobs({label: logp[i].item() for label, i in ids.items()})
```

- [ ] **Step 4: Run to verify it passes, then run the slow test once**

Run: `python -m pytest tests/test_logprob_scorer.py -q`
Expected: PASS (slow test skipped)

Confirm with the user before the model download (Qwen2.5-0.5B-Instruct from Hugging Face, ~1 GB), then run:
`RUN_SLOW=1 python -m pytest tests/test_logprob_scorer.py -q`
Expected: PASS (3 tests). If label first-token collisions occur for some wording, the error message names the labels; change the `alt` wording in `kuhn/bots/model_bot.py` `LABELSETS` to avoid shared first tokens and re-run `python -m pytest -q`.

- [ ] **Step 5: Commit**

```bash
git add kuhn/models/logprob_scorer.py tests/test_logprob_scorer.py pytest.ini
git commit -m "feat: add local-LLM label-logprob baseline scorer

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Model manifest and registry

**Files:**
- Create: `models.yaml`, `kuhn/models/registry.py`
- Test: `tests/test_registry.py`

**Interfaces:**
- Consumes: `UniformScorer`, `CachedScorer` (Task 2); `LayaScorer` (Task 5); `LogprobScorer` (Task 6).
- Produces: `load_manifest(path: str = "models.yaml") -> dict` (keys `models`, `excluded`); `get_entry(manifest: dict, model_id: str) -> dict` (raises `ValueError`); `build_scorer(entry: dict, cache_dir: str | None = None)`; `KNOWN_KINDS = ("uniform", "laya", "logprob")`.

- [ ] **Step 1: Write the manifest**

```yaml
# models.yaml
# Every candidate is either runnable (has a loader kind) or excluded with a reason.
models:
  - id: uniform
    kind: uniform
    notes: "Baseline: 50/50 over the answer labels (equivalent to RandomBot)."
  - id: laya-en
    kind: laya
    repo: convaiinnovations/laya
    params: 421M
    license: Apache-2.0
    notes: "ModernBERT-large typed decision model, English. Released 2026-09-18."
  - id: laya-multilingual
    kind: laya
    repo: convaiinnovations/laya
    subfolder: multilingual
    params: 322M
    license: Apache-2.0
    notes: "Multilingual Laya checkpoint (100+ languages)."
  - id: qwen2.5-0.5b-logprob
    kind: logprob
    repo: Qwen/Qwen2.5-0.5B-Instruct
    params: 0.5B
    license: Apache-2.0
    notes: "Non-Jev baseline: next-token label logprobs from a small generic causal LM."

excluded:
  - id: jev
    reason: "TypeSafe AI's Jev is a paid closed API with no published weights; it cannot run locally."
  - id: laya-mlx
    reason: "Apple Silicon-only MLX port of the same weights as laya-en; this project runs on Windows."
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_registry.py
from pathlib import Path

import pytest

from kuhn.models.registry import KNOWN_KINDS, build_scorer, get_entry, load_manifest
from kuhn.models.scorer import UniformScorer

MANIFEST = Path(__file__).parent.parent / "models.yaml"


def test_manifest_entries_are_well_formed():
    manifest = load_manifest(MANIFEST)
    ids = [e["id"] for e in manifest["models"]]
    assert len(ids) == len(set(ids))
    for entry in manifest["models"]:
        assert entry["kind"] in KNOWN_KINDS
        if entry["kind"] in ("laya", "logprob"):
            assert entry["repo"]
    for entry in manifest["excluded"]:
        assert entry["id"] and entry["reason"]
    assert not set(ids) & {e["id"] for e in manifest["excluded"]}


def test_get_entry_and_unknown_id():
    manifest = load_manifest(MANIFEST)
    assert get_entry(manifest, "uniform")["kind"] == "uniform"
    with pytest.raises(ValueError):
        get_entry(manifest, "does-not-exist")


def test_build_scorer_uniform_and_unknown_kind():
    assert isinstance(build_scorer({"id": "uniform", "kind": "uniform"}), UniformScorer)
    with pytest.raises(ValueError):
        build_scorer({"id": "x", "kind": "mystery"})
```

- [ ] **Step 3: Run to verify it fails**

Run: `python -m pytest tests/test_registry.py -q`
Expected: FAIL (`ModuleNotFoundError: No module named 'kuhn.models.registry'`)

- [ ] **Step 4: Write the implementation**

```python
# kuhn/models/registry.py
"""Reads models.yaml and builds scorers for its runnable entries."""
import yaml

from kuhn.models.scorer import CachedScorer, UniformScorer

KNOWN_KINDS = ("uniform", "laya", "logprob")


def load_manifest(path: str = "models.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_entry(manifest: dict, model_id: str) -> dict:
    for entry in manifest["models"]:
        if entry["id"] == model_id:
            return entry
    raise ValueError(f"model {model_id!r} is not a runnable manifest entry")


def build_scorer(entry: dict, cache_dir: str = None):
    kind = entry["kind"]
    if kind == "uniform":
        return UniformScorer()
    if kind == "laya":
        from kuhn.models.laya_scorer import LayaScorer
        scorer = LayaScorer(entry["repo"], entry.get("subfolder"), name=entry["id"])
    elif kind == "logprob":
        from kuhn.models.logprob_scorer import LogprobScorer
        scorer = LogprobScorer(entry["repo"], name=entry["id"])
    else:
        raise ValueError(f"unknown model kind {kind!r}; choose from {KNOWN_KINDS}")
    return CachedScorer(scorer, cache_dir) if cache_dir else scorer
```

- [ ] **Step 5: Run to verify it passes**

Run: `python -m pytest tests/test_registry.py -q` then `python -m pytest -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add models.yaml kuhn/models/registry.py tests/test_registry.py
git commit -m "feat: add model manifest and scorer registry

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Experiment runner and configs

**Files:**
- Create: `scripts/run_models.py`, `experiments/rung0.yaml`, `experiments/rung1.yaml`
- Test: `tests/test_run_models.py`

**Interfaces:**
- Consumes: `ModelBot` (Task 3), `nash_sanity_bot` (Task 3), `distance_from_nash` (Task 4), `load_manifest`/`get_entry`/`build_scorer` (Task 7), existing `CFRTrainer`, `NashBot`, `exploitability`, `head_to_head`.
- Produces:
  - `expand_configs(cfg: dict) -> list[dict]` (each dict has `model`, `encoder`, `qtype`, `wording`)
  - `config_id(c: dict) -> str` = `"model|encoder|qtype|wording"`
  - `run_experiments(config_path, manifest_path, output_dir, cache_dir) -> None` writing `summary.csv` (`config_id,model,encoder,qtype,wording,exploitability,total_l1,alpha`), `head_to_head.csv` (`config_id,opponent,seat,mean,ci_low,ci_high`; `seat` is `P1` or `P2`, and `mean` is always the contender's payoff), `infoset_distance.csv` (`config_id,infoset,l1,kl`).
  - Every run appends a `sanity-nash|minimal|choice|plain` contender whose exploitability must be ~0.

- [ ] **Step 1: Write the configs**

```yaml
# experiments/rung0.yaml  -- zero-shot, minimal state
seed: 0
cfr_iterations: 50000
head_to_head_hands: 20000
grid:
  models: [uniform, laya-en, laya-multilingual, qwen2.5-0.5b-logprob]
  encoders: [minimal]
  qtypes: [choice]
  wordings: [plain]
```

```yaml
# experiments/rung1.yaml  -- zero-shot, prompt/question variants
seed: 0
cfr_iterations: 50000
head_to_head_hands: 20000
grid:
  models: [uniform, laya-en, qwen2.5-0.5b-logprob]
  encoders: [minimal, natural, rules, payoff]
  qtypes: [choice, noul]
  wordings: [plain, alt]
```

(For `noul` only the first wording is used, since wording only changes `choice` labels.)

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_run_models.py
import csv
from pathlib import Path

import yaml

from kuhn.bots.heuristics import RandomBot
from kuhn.evaluate import exploitability
from scripts.run_models import config_id, expand_configs, run_experiments

MANIFEST = str(Path(__file__).parent.parent / "models.yaml")


def test_expand_configs_grid_skips_redundant_noul_wordings():
    cfg = {"grid": {"models": ["m"], "encoders": ["minimal", "natural"],
                    "qtypes": ["choice", "noul"], "wordings": ["plain", "alt"]}}
    configs = expand_configs(cfg)
    # per encoder: choice x 2 wordings + noul x 1 wording = 3
    assert len(configs) == 6
    assert {"model": "m", "encoder": "minimal", "qtype": "noul", "wording": "plain"} in configs
    assert {"model": "m", "encoder": "minimal", "qtype": "noul", "wording": "alt"} not in configs
    assert config_id(configs[0]) == "m|minimal|choice|plain"


def test_run_experiments_writes_tidy_csvs_and_sanity_row(tmp_path):
    cfg = {"seed": 0, "cfr_iterations": 300, "head_to_head_hands": 200,
           "grid": {"models": ["uniform"], "encoders": ["minimal"],
                    "qtypes": ["choice"], "wordings": ["plain"]}}
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    out = tmp_path / "out"

    run_experiments(str(cfg_path), MANIFEST, str(out), str(tmp_path / "cache"))

    with open(out / "summary.csv", newline="", encoding="utf-8") as f:
        summary = {r["config_id"]: r for r in csv.DictReader(f)}
    assert set(summary) == {"uniform|minimal|choice|plain", "sanity-nash|minimal|choice|plain"}
    assert float(summary["sanity-nash|minimal|choice|plain"]["exploitability"]) < 1e-9
    uniform_expl = float(summary["uniform|minimal|choice|plain"]["exploitability"])
    assert abs(uniform_expl - exploitability(RandomBot())) < 1e-9

    with open(out / "head_to_head.csv", newline="", encoding="utf-8") as f:
        h2h = list(csv.DictReader(f))
    # 2 contenders x 2 opponents x 2 seats
    assert len(h2h) == 8
    assert {r["seat"] for r in h2h} == {"P1", "P2"}
    assert {r["opponent"] for r in h2h} == {"Nash", "CFR"}

    with open(out / "infoset_distance.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2 * 12
```

- [ ] **Step 3: Run to verify it fails**

Run: `python -m pytest tests/test_run_models.py -q`
Expected: FAIL (`ModuleNotFoundError: No module named 'scripts.run_models'`)

- [ ] **Step 4: Write the implementation**

```python
# scripts/run_models.py
"""Evaluate model bots against the classical bots and write tidy CSVs.

Usage: python -m scripts.run_models experiments/rung0.yaml results/rung0
"""
import csv
import os
import sys

import yaml

from kuhn.bots.cfr import CFRTrainer
from kuhn.bots.model_bot import ModelBot
from kuhn.bots.nash import NashBot
from kuhn.evaluate import distance_from_nash, exploitability, head_to_head
from kuhn.models.registry import build_scorer, get_entry, load_manifest
from kuhn.models.sanity import nash_sanity_bot


def config_id(c: dict) -> str:
    return f"{c['model']}|{c['encoder']}|{c['qtype']}|{c['wording']}"


def expand_configs(cfg: dict) -> list:
    configs = list(cfg.get("configs", []))
    grid = cfg.get("grid")
    if grid:
        for model in grid["models"]:
            for encoder in grid["encoders"]:
                for qtype in grid["qtypes"]:
                    for wording in grid["wordings"]:
                        if qtype == "noul" and wording != grid["wordings"][0]:
                            continue  # wording only changes `choice` labels
                        configs.append({"model": model, "encoder": encoder,
                                        "qtype": qtype, "wording": wording})
    return configs


def _seat_results(bot, opponent, n_hands: int, seed: int) -> dict:
    """Contender's payoff in each seat. bot=P1 as-is; bot=P2 is the negation
    of the opponent-as-P1 result."""
    as_p1 = head_to_head(bot, opponent, n_hands=n_hands, seed=seed)
    flipped = head_to_head(opponent, bot, n_hands=n_hands, seed=seed)
    as_p2 = {"mean": -flipped["mean"], "ci_low": -flipped["ci_high"], "ci_high": -flipped["ci_low"]}
    return {"P1": as_p1, "P2": as_p2}


def run_experiments(config_path: str, manifest_path: str, output_dir: str, cache_dir: str) -> None:
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    manifest = load_manifest(manifest_path)
    seed = cfg.get("seed", 0)
    n_hands = cfg["head_to_head_hands"]
    os.makedirs(output_dir, exist_ok=True)

    cfr_bot, _ = CFRTrainer(seed=seed).train(iterations=cfg["cfr_iterations"])
    opponents = {"Nash": NashBot(alpha=0.0), "CFR": cfr_bot}

    scorers = {}
    contenders = []  # (config dict, bot)
    for c in expand_configs(cfg):
        if c["model"] not in scorers:
            scorers[c["model"]] = build_scorer(get_entry(manifest, c["model"]), cache_dir)
        bot = ModelBot(scorers[c["model"]], encoder=c["encoder"], qtype=c["qtype"],
                       wording=c["wording"])
        contenders.append((c, bot))
    sanity = {"model": "sanity-nash", "encoder": "minimal", "qtype": "choice", "wording": "plain"}
    contenders.append((sanity, nash_sanity_bot()))

    summary_rows, h2h_rows, infoset_rows = [], [], []
    for c, bot in contenders:
        cid = config_id(c)
        dist = distance_from_nash(bot)
        summary_rows.append({**c, "config_id": cid, "exploitability": exploitability(bot),
                             "total_l1": dist["total_l1"], "alpha": dist["alpha"]})
        for key, row in dist["per_infoset"].items():
            infoset_rows.append({"config_id": cid, "infoset": key, "l1": row["l1"], "kl": row["kl"]})
        for opp_name, opp in opponents.items():
            for seat, res in _seat_results(bot, opp, n_hands, seed).items():
                h2h_rows.append({"config_id": cid, "opponent": opp_name, "seat": seat,
                                 "mean": res["mean"], "ci_low": res["ci_low"], "ci_high": res["ci_high"]})

    def write(name, fields, rows):
        with open(os.path.join(output_dir, name), "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    write("summary.csv", ["config_id", "model", "encoder", "qtype", "wording",
                          "exploitability", "total_l1", "alpha"], summary_rows)
    write("head_to_head.csv", ["config_id", "opponent", "seat", "mean", "ci_low", "ci_high"], h2h_rows)
    write("infoset_distance.csv", ["config_id", "infoset", "l1", "kl"], infoset_rows)


if __name__ == "__main__":
    run_experiments(sys.argv[1], "models.yaml", sys.argv[2], "results/cache")
```

- [ ] **Step 5: Run to verify it passes**

Run: `python -m pytest tests/test_run_models.py -q` then `python -m pytest -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add scripts/run_models.py experiments/rung0.yaml experiments/rung1.yaml tests/test_run_models.py
git commit -m "feat: add experiment runner and rung 0/1 configs

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Fine-tuning dataset builder (for the later rung-2 plan)

Kuhn has only 12 distinct training examples per prompt variant, so the dataset is the full set of infoset × variant examples, not a sampled stream. Scaling comes from variant coverage, and held-out variants are the memorization check.

**Files:**
- Create: `kuhn/finetune_data.py`
- Test: `tests/test_finetune_data.py`

**Interfaces:**
- Consumes: `get_encoder`, `build_question`, `game.all_infosets`.
- Produces: `build_dataset(strategy_bot, encoders=("minimal",), qtypes=("choice",), wordings=("plain",)) -> list[dict]` where each example is `{"infoset", "encoder", "qtype", "wording", "state": str, "question": Question, "target": {label: prob}}`; `split_held_out(examples, held_out_encoders) -> (train, held_out)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_finetune_data.py
import pytest

from kuhn import game
from kuhn.bots.nash import NashBot
from kuhn.finetune_data import build_dataset, split_held_out


def test_dataset_has_one_example_per_infoset_and_variant():
    ex = build_dataset(NashBot(), encoders=("minimal", "natural"), qtypes=("choice", "noul"),
                       wordings=("plain",))
    assert len(ex) == 12 * 2 * 2
    assert {e["infoset"] for e in ex} == set(game.all_infosets())


def test_targets_match_the_strategy_and_sum_to_one():
    bot = NashBot(alpha=1 / 6)
    for e in build_dataset(bot):
        assert sum(e["target"].values()) == pytest.approx(1.0)
        dist = bot.action_probs(e["infoset"], game.ACTIONS)
        for label, action in e["question"].options:
            assert e["target"][label] == pytest.approx(dist[action])


def test_split_held_out_by_encoder():
    ex = build_dataset(NashBot(), encoders=("minimal", "natural", "rules"))
    train, held = split_held_out(ex, ("rules",))
    assert {e["encoder"] for e in train} == {"minimal", "natural"}
    assert {e["encoder"] for e in held} == {"rules"}
    assert len(train) + len(held) == len(ex)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_finetune_data.py -q`
Expected: FAIL (`ModuleNotFoundError: No module named 'kuhn.finetune_data'`)

- [ ] **Step 3: Write the implementation**

```python
# kuhn/finetune_data.py
"""Fine-tuning data: (prompt, typed question, soft target) for every
infoset x prompt variant. Targets are the strategy bot's action
distribution (e.g. the CFR average strategy), not hard labels."""
from kuhn import game
from kuhn.bots.model_bot import build_question
from kuhn.encoders import get_encoder


def build_dataset(strategy_bot, encoders=("minimal",), qtypes=("choice",), wordings=("plain",)) -> list:
    examples = []
    for encoder in encoders:
        enc = get_encoder(encoder)
        for qtype in qtypes:
            for wording in wordings:
                if qtype == "noul" and wording != wordings[0]:
                    continue  # wording only changes `choice` labels
                for infoset in game.all_infosets():
                    question = build_question(infoset, qtype, wording)
                    dist = strategy_bot.action_probs(infoset, game.ACTIONS)
                    examples.append({
                        "infoset": infoset, "encoder": encoder, "qtype": qtype, "wording": wording,
                        "state": enc(infoset), "question": question,
                        "target": {label: dist.get(action, 0.0) for label, action in question.options},
                    })
    return examples


def split_held_out(examples: list, held_out_encoders) -> tuple:
    held = set(held_out_encoders)
    train = [e for e in examples if e["encoder"] not in held]
    held_out = [e for e in examples if e["encoder"] in held]
    return train, held_out
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_finetune_data.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add kuhn/finetune_data.py tests/test_finetune_data.py
git commit -m "feat: add fine-tuning dataset builder with held-out split

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 10: Rung-2 feasibility probe (decides the next plan)

The spec's fine-tuning rung assumes Laya can be fine-tuned. Nothing found so far says it can. This task finds out and records the answer. It does **not** build a training loop.

**Files:**
- Create: `scripts/probe_laya_training.py`, `docs/superpowers/notes/2026-10-03-laya-training-feasibility.md`

**Interfaces:**
- Consumes: an installed `laya` package (Task 5).
- Produces: a written go / no-go note plus, if go, the exact training entry points.

- [ ] **Step 1: Write and run the probe**

```python
# scripts/probe_laya_training.py
"""Looks inside the installed `laya` package for anything training-related."""
import importlib.metadata
import pkgutil

import laya

print("file:", laya.__file__)
print("public names:", [n for n in dir(laya) if not n.startswith("_")])
print("submodules:")
for m in pkgutil.walk_packages(laya.__path__, "laya."):
    print("  ", m.name)

meta = importlib.metadata.metadata("laya")
print("summary:", meta.get("Summary"))
print("project urls:", meta.get_all("Project-URL"))
hits = [n for n in dir(laya) if any(k in n.lower() for k in ("train", "fit", "finetune", "loss", "optim"))]
print("training-looking names:", hits)
```

Run: `python -m scripts.probe_laya_training`
Expected: prints the package layout.

- [ ] **Step 2: Check the model card and repo for training support**

Fetch `https://huggingface.co/convaiinnovations/laya` and the project links printed above; look for a training, fine-tuning or "heads" section, and note whether the base encoder (ModernBERT-large) can be loaded and trained with the typed heads or only used frozen.

- [ ] **Step 3: Write the note**

Create `docs/superpowers/notes/2026-10-03-laya-training-feasibility.md` with: what the package exposes, whether fine-tuning is supported (yes / no / unclear), the exact entry points if yes, and a one-line recommendation: **go** (write the rung-2 plan using those entry points), **no-go for Laya** (fine-tune only the `LogprobScorer` baseline's LM, or fine-tune a ModernBERT classifier head directly), or **unclear** (ask the user).

- [ ] **Step 4: Commit and report**

```bash
git add scripts/probe_laya_training.py docs/superpowers/notes/2026-10-03-laya-training-feasibility.md
git commit -m "docs: record Laya fine-tuning feasibility probe

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

Tell the user the go/no-go result before planning rung 2.

---

### Task 11: Report assets (leaderboard and heatmap)

**Files:**
- Create: `scripts/make_report_assets.py`
- Test: `tests/test_report_assets.py`

**Interfaces:**
- Consumes: the CSV files written by `run_experiments` (Task 8).
- Produces: `make_assets(results_dir: str, out_dir: str) -> None` writing `leaderboard.md` (configs ranked by ascending exploitability) and `heatmap.png` (rows = configs in leaderboard order, columns = 12 infosets, cell = L1 distance from Nash).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_report_assets.py
import csv

from kuhn import game
from scripts.make_report_assets import make_assets


def _write(path, fields, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def test_make_assets_ranks_by_exploitability_and_writes_heatmap(tmp_path):
    res = tmp_path / "res"
    res.mkdir()
    fields = ["config_id", "model", "encoder", "qtype", "wording", "exploitability", "total_l1", "alpha"]
    _write(res / "summary.csv", fields, [
        {"config_id": "bad|m|c|p", "model": "bad", "encoder": "m", "qtype": "c", "wording": "p",
         "exploitability": 0.5, "total_l1": 6.0, "alpha": 0.0},
        {"config_id": "good|m|c|p", "model": "good", "encoder": "m", "qtype": "c", "wording": "p",
         "exploitability": 0.01, "total_l1": 0.3, "alpha": 0.1},
    ])
    _write(res / "infoset_distance.csv", ["config_id", "infoset", "l1", "kl"], [
        {"config_id": cid, "infoset": i, "l1": 0.1, "kl": 0.1}
        for cid in ("bad|m|c|p", "good|m|c|p") for i in game.all_infosets()
    ])
    out = tmp_path / "out"

    make_assets(str(res), str(out))

    lines = (out / "leaderboard.md").read_text(encoding="utf-8").splitlines()
    assert lines[2].startswith("| 1 | good|m|c|p")
    assert lines[3].startswith("| 2 | bad|m|c|p")
    assert (out / "heatmap.png").stat().st_size > 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_report_assets.py -q`
Expected: FAIL (`ModuleNotFoundError: No module named 'scripts.make_report_assets'`)

- [ ] **Step 3: Write the implementation**

```python
# scripts/make_report_assets.py
"""Turns run_models CSVs into the leaderboard table and infoset heatmap.

Usage: python -m scripts.make_report_assets results/rung0 results/rung0/assets
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kuhn import game


def _read(path: str) -> list:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _infoset_label(key: str) -> str:
    return f"{game.CARD_NAMES[int(key[0])]} {key[1:] or '-'}"


def make_assets(results_dir: str, out_dir: str) -> None:
    summary = _read(os.path.join(results_dir, "summary.csv"))
    rows = _read(os.path.join(results_dir, "infoset_distance.csv"))
    os.makedirs(out_dir, exist_ok=True)

    ranked = sorted(summary, key=lambda r: float(r["exploitability"]))
    with open(os.path.join(out_dir, "leaderboard.md"), "w", encoding="utf-8") as f:
        f.write("| rank | config | exploitability | total L1 from Nash | nearest alpha |\n")
        f.write("|---|---|---|---|---|\n")
        for i, r in enumerate(ranked, 1):
            f.write(f"| {i} | {r['config_id']} | {float(r['exploitability']):.4f} "
                    f"| {float(r['total_l1']):.3f} | {float(r['alpha']):.3f} |\n")

    ids = [r["config_id"] for r in ranked]
    infosets = sorted({r["infoset"] for r in rows})
    value = {(r["config_id"], r["infoset"]): float(r["l1"]) for r in rows}
    grid = [[value[(cid, k)] for k in infosets] for cid in ids]

    fig, ax = plt.subplots(figsize=(2 + 0.7 * len(infosets), 1.5 + 0.4 * len(ids)))
    im = ax.imshow(grid, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(infosets)))
    ax.set_xticklabels([_infoset_label(k) for k in infosets], rotation=45, ha="right")
    ax.set_yticks(range(len(ids)))
    ax.set_yticklabels(ids, fontsize=7)
    ax.set_title("L1 distance from Nash, per information set (card, history)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "heatmap.png"))
    plt.close(fig)


if __name__ == "__main__":
    make_assets(sys.argv[1], sys.argv[2])
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_report_assets.py -q` then `python -m pytest -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/make_report_assets.py tests/test_report_assets.py
git commit -m "feat: add leaderboard and heatmap report-asset generator

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 12: Run rungs 0–1 and write the first report

**Files:**
- Create: `docs/REPORT_MODELS.md`
- Modify: `README.md` (add the new run commands and link the spec and report)

**Interfaces:**
- Consumes: everything above.
- Produces: `results/rung0/`, `results/rung1/` (gitignored) and `docs/REPORT_MODELS.md`.

- [ ] **Step 1: Run the full test suite**

Run: `python -m pytest -q`
Expected: all tests pass (27 original plus the new ones), 1 skipped (slow).

- [ ] **Step 2: Run rung 0**

Run: `python -m scripts.run_models experiments/rung0.yaml results/rung0 && python -m scripts.make_report_assets results/rung0 results/rung0/assets`
Expected: `summary.csv` has 5 rows (4 models + `sanity-nash`); the `sanity-nash` exploitability prints as ~0. If it is not ~0, stop: the adapter is broken, and no model numbers can be trusted.

- [ ] **Step 3: Run rung 1**

Run: `python -m scripts.run_models experiments/rung1.yaml results/rung1 && python -m scripts.make_report_assets results/rung1 results/rung1/assets`
Expected: 3 models × 4 encoders × 3 question setups = 36 rows plus the sanity row. Laya and Qwen outputs are cached under `results/cache`, so a rerun is fast.

- [ ] **Step 4: Read the results before writing anything**

Open both `leaderboard.md` files and both `heatmap.png` files. Note: which model is closest to Nash, whether any beats `uniform`, which encoder or question type helps, and which infosets are consistently wrong. Do not write claims the numbers don't support.

- [ ] **Step 5: Write `docs/REPORT_MODELS.md`**

Invoke the `anthropic-skills:meerav-blog-writer` skill for voice and structure. Required content: the leaderboard table, the zero-shot vs prompt-variant story, the heatmap with a "where do they go wrong" reading, the head-to-head results against Nash and CFR (from `head_to_head.csv`, both seats), the exclusion list from `models.yaml` with reasons, and the caveats (Kuhn has 12 information sets; rung 2 is not done; results are for the specific prompts and checkpoints listed). Copy the final figures and tables from `results/` into the report; do not retype numbers by hand.

- [ ] **Step 6: Update the README and commit**

Add to `README.md`'s Running section:

```bash
python -m scripts.run_models experiments/rung0.yaml results/rung0
python -m scripts.make_report_assets results/rung0 results/rung0/assets
```

and a line linking `docs/REPORT_MODELS.md` and the spec.

```bash
git add docs/REPORT_MODELS.md README.md
git commit -m "docs: add decision-model arena report for rungs 0-1

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```
