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
