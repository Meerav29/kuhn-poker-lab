"""Turns run_models CSVs into the leaderboard table and infoset heatmap.

Usage: python -m scripts.make_report_assets results/rung0 results/rung0/assets
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kuhn import game


def _read(path: str) -> list:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _infoset_label(key: str) -> str:
    return f"{game.CARD_NAMES[int(key[0])]} {key[1:] or '-'}"


def _build_grid(summary: list, rows: list) -> tuple:
    """Build grid from summary and infoset_distance data.

    Returns:
        (ids, infosets, grid) where:
        - ids: config_ids in ascending exploitability order
        - infosets: sorted infoset keys
        - grid[i][j]: L1 distance for ids[i] x infosets[j]
    """
    ranked = sorted(summary, key=lambda r: float(r["exploitability"]))
    ids = [r["config_id"] for r in ranked]

    # Use all infosets from the game, or from data if data is empty
    if rows:
        infosets = sorted({r["infoset"] for r in rows})
    else:
        infosets = []

    # Build value lookup
    value = {}
    for r in rows:
        value[(r["config_id"], r["infoset"])] = float(r["l1"])

    # Find all infosets that appear in rows for any config
    all_infosets_in_data = {r["infoset"] for r in rows}

    # Validate that each config has all infosets that appear in data
    for cid in ids:
        for k in all_infosets_in_data:
            if (cid, k) not in value:
                raise ValueError(f"{cid}: missing infoset {k}")

    # Construct grid
    grid = []
    for cid in ids:
        row = []
        for k in infosets:
            row.append(value[(cid, k)])
        grid.append(row)

    return ids, infosets, grid


def make_assets(results_dir: str, out_dir: str) -> None:
    summary = _read(os.path.join(results_dir, "summary.csv"))
    rows = _read(os.path.join(results_dir, "infoset_distance.csv"))

    if not summary:
        raise ValueError("summary.csv has no rows")

    os.makedirs(out_dir, exist_ok=True)

    ids, infosets, grid = _build_grid(summary, rows)

    # Write leaderboard with escaped pipes in config_id
    ranked = sorted(summary, key=lambda r: float(r["exploitability"]))
    with open(os.path.join(out_dir, "leaderboard.md"), "w", encoding="utf-8") as f:
        f.write("| rank | config | exploitability | total L1 from Nash | nearest alpha |\n")
        f.write("|---|---|---|---|---|\n")
        for i, r in enumerate(ranked, 1):
            escaped_config = r['config_id'].replace("|", "\\|")
            f.write(f"| {i} | {escaped_config} | {float(r['exploitability']):.4f} "
                    f"| {float(r['total_l1']):.3f} | {float(r['alpha']):.3f} |\n")

    fig, ax = plt.subplots(figsize=(2 + 0.7 * len(infosets), 1.5 + 0.4 * len(ids)))
    im = ax.imshow(grid, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(infosets)))
    ax.set_xticklabels([_infoset_label(k) for k in infosets], rotation=45, ha="right")
    ax.set_yticks(range(len(ids)))
    ax.set_yticklabels(ids, fontsize=7)
    ax.set_title("L1 distance from Nash, per information set (card, history)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "heatmap.png"))
    plt.close(fig)


if __name__ == "__main__":
    make_assets(sys.argv[1], sys.argv[2])
