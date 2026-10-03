"""Reads models.yaml and builds scorers for its runnable entries."""
import yaml

from kuhn.models.scorer import CachedScorer, UniformScorer

KNOWN_KINDS = ("uniform", "laya", "logprob")


def load_manifest(path: str = "models.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_entry(manifest: dict, model_id: str) -> dict:
    for entry in manifest["models"]:
        if entry["id"] == model_id:
            return entry
    raise ValueError(f"model {model_id!r} is not a runnable manifest entry")


def build_scorer(entry: dict, cache_dir: str = None):
    kind = entry["kind"]
    if kind == "uniform":
        return UniformScorer()
    if kind == "laya":
        from kuhn.models.laya_scorer import LayaScorer
        scorer = LayaScorer(entry["repo"], entry.get("subfolder"), name=entry["id"])
    elif kind == "logprob":
        from kuhn.models.logprob_scorer import LogprobScorer
        scorer = LogprobScorer(entry["repo"], name=entry["id"])
    else:
        raise ValueError(f"unknown model kind {kind!r}; choose from {KNOWN_KINDS}")
    return CachedScorer(scorer, cache_dir) if cache_dir else scorer
