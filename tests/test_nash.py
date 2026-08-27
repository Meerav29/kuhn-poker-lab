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
