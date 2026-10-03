import json
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


# Tests for error handling and fixture coverage

def test_extract_fixture_choice():
    """Test against the real laya_raw.json fixture for choice question."""
    with open("tests/fixtures/laya_raw.json", encoding="utf-8") as f:
        fixture = json.load(f)
    out = extract_probs(fixture["answers"]["act"], CHOICE)
    # Fixture has fold: 0.3819, call: 0.6181 (already normalized)
    assert out == pytest.approx({"fold": 0.3819, "call": 0.6181})
    # Verify it sums to 1
    assert sum(out.values()) == pytest.approx(1.0)


def test_extract_fixture_noul():
    """Test against the real laya_raw.json fixture for noul question."""
    with open("tests/fixtures/laya_raw.json", encoding="utf-8") as f:
        fixture = json.load(f)
    out = extract_probs(fixture["answers"]["call_yes"], NOUL)
    # Fixture has noul: 0.1875, so yes should be 0.1875, no should be 0.8125
    assert out == pytest.approx({"yes": 0.1875, "no": 0.8125})
    # Verify it sums to 1
    assert sum(out.values()) == pytest.approx(1.0)


def test_extract_choice_missing_label_raises():
    """Test that a missing label in the probability dict raises KeyError with message."""
    with pytest.raises(KeyError) as exc_info:
        extract_probs({"choice": "call", "probs": {"fold": 0.5}}, CHOICE)
    # Should mention the missing label and the keys present
    assert "call" in str(exc_info.value) or "missing" in str(exc_info.value).lower()


def test_extract_choice_zero_total_raises():
    """Test that probabilities summing to <= 0 raise ValueError."""
    with pytest.raises(ValueError):
        extract_probs({"choice": "call", "probs": {"fold": 0.0, "call": 0.0}}, CHOICE)


def test_extract_choice_negative_prob_raises():
    """Test that negative probabilities raise ValueError."""
    with pytest.raises(ValueError):
        extract_probs({"choice": "call", "probs": {"fold": -0.1, "call": 0.6}}, CHOICE)


def test_extract_choice_non_finite_prob_raises():
    """Test that non-finite (inf, nan) probabilities raise ValueError."""
    with pytest.raises(ValueError):
        extract_probs({"choice": "call", "probs": {"fold": float("inf"), "call": 0.1}}, CHOICE)
    with pytest.raises(ValueError):
        extract_probs({"choice": "call", "probs": {"fold": float("nan"), "call": 0.5}}, CHOICE)


def test_extract_noul_missing_key_raises():
    """Test that missing 'noul' key raises KeyError with keys present."""
    with pytest.raises(KeyError) as exc_info:
        extract_probs({"confidence": 0.8}, NOUL)
    # Should list the keys present
    assert "confidence" in str(exc_info.value) or "noul" in str(exc_info.value)


def test_extract_noul_outside_range_raises():
    """Test that noul value outside [0, 1] raises ValueError."""
    with pytest.raises(ValueError):
        extract_probs({"noul": 1.5}, NOUL)
    with pytest.raises(ValueError):
        extract_probs({"noul": -0.1}, NOUL)


def test_extract_noul_nan_raises():
    """Test that noul value of NaN raises ValueError."""
    with pytest.raises(ValueError):
        extract_probs({"noul": float("nan")}, NOUL)
