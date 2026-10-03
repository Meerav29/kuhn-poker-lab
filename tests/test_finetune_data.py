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
