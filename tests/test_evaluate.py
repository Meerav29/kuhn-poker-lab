from kuhn.bots.nash import NashBot
from kuhn.bots.heuristics import HonestBot, AlwaysBluffBot, RandomBot
from kuhn.evaluate import exploitability


def test_nash_bot_is_unexploitable():
    for alpha in (0.0, 1 / 6, 1 / 3):
        assert abs(exploitability(NashBot(alpha=alpha))) < 1e-9


def test_heuristic_bots_are_exploitable_and_distinct():
    honest = exploitability(HonestBot())
    bluff = exploitability(AlwaysBluffBot())
    random_ = exploitability(RandomBot())

    assert abs(honest - 0.25) < 1e-6
    assert abs(bluff - 0.376667) < 1e-5
    assert abs(random_ - 0.458333) < 1e-5

    # every heuristic bot is meaningfully more exploitable than Nash
    for exp in (honest, bluff, random_):
        assert exp > 0.1
