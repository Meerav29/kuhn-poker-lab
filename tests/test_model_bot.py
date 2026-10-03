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
