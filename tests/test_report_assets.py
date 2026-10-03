import csv

from kuhn import game
from scripts.make_report_assets import make_assets


def _write(path, fields, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def test_make_assets_ranks_by_exploitability_and_writes_heatmap(tmp_path):
    res = tmp_path / "res"
    res.mkdir()
    fields = ["config_id", "model", "encoder", "qtype", "wording", "exploitability", "total_l1", "alpha"]
    _write(res / "summary.csv", fields, [
        {"config_id": "bad|m|c|p", "model": "bad", "encoder": "m", "qtype": "c", "wording": "p",
         "exploitability": 0.5, "total_l1": 6.0, "alpha": 0.0},
        {"config_id": "good|m|c|p", "model": "good", "encoder": "m", "qtype": "c", "wording": "p",
         "exploitability": 0.01, "total_l1": 0.3, "alpha": 0.1},
    ])
    _write(res / "infoset_distance.csv", ["config_id", "infoset", "l1", "kl"], [
        {"config_id": cid, "infoset": i, "l1": 0.1, "kl": 0.1}
        for cid in ("bad|m|c|p", "good|m|c|p") for i in game.all_infosets()
    ])
    out = tmp_path / "out"

    make_assets(str(res), str(out))

    lines = (out / "leaderboard.md").read_text(encoding="utf-8").splitlines()
    assert lines[2].startswith("| 1 | good|m|c|p")
    assert lines[3].startswith("| 2 | bad|m|c|p")
    assert (out / "heatmap.png").stat().st_size > 0
