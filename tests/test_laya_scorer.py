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
