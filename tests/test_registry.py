from pathlib import Path

import pytest

from kuhn.models.registry import KNOWN_KINDS, build_scorer, get_entry, load_manifest
from kuhn.models.scorer import UniformScorer

MANIFEST = Path(__file__).parent.parent / "models.yaml"


def test_manifest_entries_are_well_formed():
    manifest = load_manifest(MANIFEST)
    ids = [e["id"] for e in manifest["models"]]
    assert len(ids) == len(set(ids))
    for entry in manifest["models"]:
        assert entry["kind"] in KNOWN_KINDS
        if entry["kind"] in ("laya", "logprob"):
            assert entry["repo"]
    for entry in manifest["excluded"]:
        assert entry["id"] and entry["reason"]
    assert not set(ids) & {e["id"] for e in manifest["excluded"]}


def test_get_entry_and_unknown_id():
    manifest = load_manifest(MANIFEST)
    assert get_entry(manifest, "uniform")["kind"] == "uniform"
    with pytest.raises(ValueError):
        get_entry(manifest, "does-not-exist")


def test_build_scorer_uniform_and_unknown_kind():
    assert isinstance(build_scorer({"id": "uniform", "kind": "uniform"}), UniformScorer)
    with pytest.raises(ValueError):
        build_scorer({"id": "x", "kind": "mystery"})
