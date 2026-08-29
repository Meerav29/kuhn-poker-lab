from kuhn.bots.nash import NashBot
from kuhn.bots.heuristics import HonestBot, AlwaysBluffBot, RandomBot
from kuhn.evaluate import exploitability
from kuhn.evaluate import head_to_head


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


def test_random_vs_random_mean_matches_exact_value():
    # Kuhn Poker's two seats are NOT symmetric under uniform-random play --
    # Player 1 acts first (check/bet) while Player 2 responds (call/fold),
    # so RandomBot vs RandomBot does not average to 0. Exact enumeration
    # (verified independently via three separate traversal implementations
    # during Task 5 implementation) gives Player 1's true expected payoff
    # as exactly +0.125. The simulated CI should capture that value.
    result = head_to_head(RandomBot(), RandomBot(), n_hands=20000, seed=1)
    assert result["ci_low"] < 0.125 < result["ci_high"]


def test_nash_beats_random_bot():
    result = head_to_head(NashBot(alpha=0.0), RandomBot(), n_hands=20000, seed=1)
    assert result["mean"] > 0
    assert result["ci_low"] > 0  # confidently positive, not just noise


def test_result_has_expected_keys():
    result = head_to_head(HonestBot(), RandomBot(), n_hands=1000, seed=1)
    assert set(result.keys()) == {"mean", "ci_low", "ci_high"}
    assert result["ci_low"] <= result["mean"] <= result["ci_high"]
