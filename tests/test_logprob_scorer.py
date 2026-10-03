import math
import os

import pytest

from kuhn.models.logprob_scorer import LogprobScorer, probs_from_logprobs
from kuhn.models.scorer import Question


def test_probs_from_logprobs_is_a_softmax():
    out = probs_from_logprobs({"a": math.log(1.0), "b": math.log(3.0)})
    assert out == pytest.approx({"a": 0.25, "b": 0.75})


def test_probs_from_logprobs_is_stable_for_very_negative_values():
    out = probs_from_logprobs({"a": -1000.0, "b": -1001.0})
    assert sum(out.values()) == pytest.approx(1.0)
    assert out["a"] > out["b"]


@pytest.mark.slow
@pytest.mark.skipif(not os.environ.get("RUN_SLOW"), reason="downloads a model; set RUN_SLOW=1")
def test_logprob_scorer_returns_a_distribution_over_labels():
    q = Question("choice", "Do you check or bet?", (("check", "p"), ("bet", "b")))
    out = LogprobScorer().score("card=K history=-", q)
    assert set(out) == {"check", "bet"}
    assert sum(out.values()) == pytest.approx(1.0)
