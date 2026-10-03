import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from kuhn.models.logprob_scorer import probs_from_logprobs


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
    script = """
import json
from kuhn.models.logprob_scorer import LogprobScorer
from kuhn.models.scorer import Question

q = Question("choice", "Do you check or bet?", (("check", "p"), ("bet", "b")))
out = LogprobScorer().score("card=K history=-", q)
print(json.dumps(out))
"""
    repo_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=900,
        cwd=repo_root
    )
    assert result.returncode == 0, f"subprocess failed: stderr={result.stderr[-2000:]}, stdout={result.stdout[-500:]}"
    try:
        out = json.loads(result.stdout.strip().split('\n')[-1])
    except (json.JSONDecodeError, ValueError, IndexError) as e:
        pytest.fail(f"failed to parse JSON from stdout: {e}, stdout={result.stdout[-500:]}")
    assert set(out) == {"check", "bet"}
    assert sum(out.values()) == pytest.approx(1.0)
