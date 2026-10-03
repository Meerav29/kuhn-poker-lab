import csv
import re

import pytest

from kuhn import game
from scripts.make_report_assets import make_assets, _build_grid


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
    # Check for escaped pipes in config_id cells
    assert lines[2].startswith("| 1 | good\\|m\\|c\\|p |")
    assert lines[3].startswith("| 2 | bad\\|m\\|c\\|p |")

    # Verify markdown table structure: each data row should have same number of cells as header
    header_cells = re.split(r"(?<!\\)\|", lines[1].strip())
    num_header_cells = len([c for c in header_cells if c.strip()])

    for i, line in enumerate(lines[2:4], start=2):
        cells = re.split(r"(?<!\\)\|", line.strip())
        num_data_cells = len([c for c in cells if c.strip()])
        assert num_data_cells == num_header_cells, f"Row {i} has {num_data_cells} cells, expected {num_header_cells}"

    assert (out / "heatmap.png").stat().st_size > 0


def test_build_grid_ranks_by_exploitability_and_orders_infosets():
    """Test _build_grid with distinct l1 values to verify row/column ordering and cell values."""
    infosets = sorted(game.all_infosets())
    assert len(infosets) >= 12, "Need at least 12 infosets"

    summary = [
        {"config_id": "config_0", "exploitability": "0.1", "total_l1": "1.0", "alpha": "0.0"},
        {"config_id": "config_1", "exploitability": "0.05", "total_l1": "0.5", "alpha": "0.1"},
    ]

    # Create distinct l1 values: 0.01*config_index + 0.001*infoset_index
    rows = [
        {"config_id": f"config_{c}", "infoset": infosets[i], "l1": str(0.01*c + 0.001*i), "kl": "0.0"}
        for c in range(2) for i in range(len(infosets))
    ]

    ids, ret_infosets, grid = _build_grid(summary, rows)

    # Row order: best exploitability first (config_1 with 0.05, then config_0 with 0.1)
    assert ids == ["config_1", "config_0"]

    # Column order: sorted infosets
    assert ret_infosets == infosets

    # Grid dimensions
    assert len(grid) == 2
    assert len(grid[0]) == len(infosets)

    # Check exact cell values
    # grid[0] is config_1 (index 1): 0.01*1 + 0.001*i
    assert abs(grid[0][0] - (0.01*1 + 0.001*0)) < 1e-9
    assert abs(grid[0][5] - (0.01*1 + 0.001*5)) < 1e-9

    # grid[1] is config_0 (index 0): 0.01*0 + 0.001*i
    assert abs(grid[1][0] - (0.01*0 + 0.001*0)) < 1e-9
    assert abs(grid[1][5] - (0.01*0 + 0.001*5)) < 1e-9


def test_make_assets_raises_on_missing_infoset_for_config(tmp_path):
    """Test that missing (config_id, infoset) pair raises ValueError."""
    res = tmp_path / "res"
    res.mkdir()
    fields = ["config_id", "model", "encoder", "qtype", "wording", "exploitability", "total_l1", "alpha"]

    # Only write data for first infoset for cfg0, but all infosets for cfg1
    all_infosets = sorted(game.all_infosets())
    _write(res / "summary.csv", fields, [
        {"config_id": "cfg0", "model": "m", "encoder": "e", "qtype": "q", "wording": "w",
         "exploitability": 0.1, "total_l1": 1.0, "alpha": 0.0},
        {"config_id": "cfg1", "model": "m", "encoder": "e", "qtype": "q", "wording": "w",
         "exploitability": 0.05, "total_l1": 0.5, "alpha": 0.1},
    ])
    _write(res / "infoset_distance.csv", ["config_id", "infoset", "l1", "kl"], [
        {"config_id": "cfg0", "infoset": all_infosets[0], "l1": 0.1, "kl": 0.1}
    ] + [
        {"config_id": "cfg1", "infoset": i, "l1": 0.1, "kl": 0.1}
        for i in all_infosets
    ])
    out = tmp_path / "out"

    with pytest.raises(ValueError, match=r"cfg0.*infoset"):
        make_assets(str(res), str(out))


def test_make_assets_raises_on_empty_summary(tmp_path):
    """Test that empty summary.csv raises ValueError."""
    res = tmp_path / "res"
    res.mkdir()
    fields = ["config_id", "model", "encoder", "qtype", "wording", "exploitability", "total_l1", "alpha"]
    _write(res / "summary.csv", fields, [])
    _write(res / "infoset_distance.csv", ["config_id", "infoset", "l1", "kl"], [])
    out = tmp_path / "out"

    with pytest.raises(ValueError, match="summary.csv has no rows"):
        make_assets(str(res), str(out))
