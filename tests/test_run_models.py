import csv
from pathlib import Path

import yaml

from kuhn.bots.heuristics import RandomBot
from kuhn.evaluate import exploitability
from scripts.run_models import config_id, expand_configs, run_experiments

MANIFEST = str(Path(__file__).parent.parent / "models.yaml")


def test_expand_configs_grid_skips_redundant_noul_wordings():
    cfg = {"grid": {"models": ["m"], "encoders": ["minimal", "natural"],
                    "qtypes": ["choice", "noul"], "wordings": ["plain", "alt"]}}
    configs = expand_configs(cfg)
    # per encoder: choice x 2 wordings + noul x 1 wording = 3
    assert len(configs) == 6
    assert {"model": "m", "encoder": "minimal", "qtype": "noul", "wording": "plain"} in configs
    assert {"model": "m", "encoder": "minimal", "qtype": "noul", "wording": "alt"} not in configs
    assert config_id(configs[0]) == "m|minimal|choice|plain"


def test_run_experiments_writes_tidy_csvs_and_sanity_row(tmp_path):
    cfg = {"seed": 0, "cfr_iterations": 300, "head_to_head_hands": 200,
           "grid": {"models": ["uniform"], "encoders": ["minimal"],
                    "qtypes": ["choice"], "wordings": ["plain"]}}
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    out = tmp_path / "out"

    run_experiments(str(cfg_path), MANIFEST, str(out), str(tmp_path / "cache"))

    with open(out / "summary.csv", newline="", encoding="utf-8") as f:
        summary = {r["config_id"]: r for r in csv.DictReader(f)}
    assert set(summary) == {"uniform|minimal|choice|plain", "sanity-nash|minimal|choice|plain"}
    assert float(summary["sanity-nash|minimal|choice|plain"]["exploitability"]) < 1e-9
    uniform_expl = float(summary["uniform|minimal|choice|plain"]["exploitability"])
    assert abs(uniform_expl - exploitability(RandomBot())) < 1e-9

    with open(out / "head_to_head.csv", newline="", encoding="utf-8") as f:
        h2h = list(csv.DictReader(f))
    # 2 contenders x 2 opponents x 2 seats
    assert len(h2h) == 8
    assert {r["seat"] for r in h2h} == {"P1", "P2"}
    assert {r["opponent"] for r in h2h} == {"Nash", "CFR"}

    with open(out / "infoset_distance.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2 * 12
