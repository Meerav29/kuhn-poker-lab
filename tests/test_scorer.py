import pytest

from kuhn.models.scorer import CachedScorer, LookupScorer, Question, UniformScorer

Q = Question(kind="choice", instructions="Check or bet?", options=(("check", "p"), ("bet", "b")))


def test_question_labels_and_key():
    assert Q.labels == ("check", "bet")
    other = Question(kind="choice", instructions="Check or bet?", options=(("wait", "p"), ("raise", "b")))
    assert Q.key() != other.key()
    assert Q.key() == Q.key()


def test_uniform_scorer():
    assert UniformScorer().score("anything", Q) == {"check": 0.5, "bet": 0.5}


def test_lookup_scorer_returns_table_row_and_raises_on_missing():
    s = LookupScorer({"s1": {"check": 0.9, "bet": 0.1}})
    assert s.score("s1", Q) == {"check": 0.9, "bet": 0.1}
    with pytest.raises(KeyError):
        s.score("unknown", Q)


class CountingScorer:
    scorer_id = "counting/model:1"

    def __init__(self):
        self.calls = 0

    def score(self, state, question):
        self.calls += 1
        return {"check": 0.25, "bet": 0.75}


def test_cached_scorer_hits_cache_in_memory_and_across_instances(tmp_path):
    inner = CountingScorer()
    cached = CachedScorer(inner, tmp_path)
    assert cached.score("s", Q) == {"check": 0.25, "bet": 0.75}
    assert cached.score("s", Q) == {"check": 0.25, "bet": 0.75}
    assert inner.calls == 1

    inner2 = CountingScorer()
    again = CachedScorer(inner2, tmp_path)
    assert again.score("s", Q) == {"check": 0.25, "bet": 0.75}
    assert inner2.calls == 0

    again.score("different state", Q)
    assert inner2.calls == 1
    assert cached.scorer_id == "counting/model:1"
