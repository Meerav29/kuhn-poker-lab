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
