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
