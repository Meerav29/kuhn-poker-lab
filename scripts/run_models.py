"""Evaluate model bots against the classical bots and write tidy CSVs.

Usage: python -m scripts.run_models experiments/rung0.yaml results/rung0
"""
import csv
import os
import sys

import yaml

from kuhn.bots.cfr import CFRTrainer
from kuhn.bots.model_bot import ModelBot
from kuhn.bots.nash import NashBot
from kuhn.evaluate import distance_from_nash, exploitability, head_to_head
from kuhn.models.registry import build_scorer, get_entry, load_manifest
from kuhn.models.sanity import nash_sanity_bot


def config_id(c: dict) -> str:
    return f"{c['model']}|{c['encoder']}|{c['qtype']}|{c['wording']}"


def expand_configs(cfg: dict) -> list:
    configs = list(cfg.get("configs", []))
    grid = cfg.get("grid")
    if grid:
        for model in grid["models"]:
            for encoder in grid["encoders"]:
                for qtype in grid["qtypes"]:
                    for wording in grid["wordings"]:
                        if qtype == "noul" and wording != grid["wordings"][0]:
                            continue  # wording only changes `choice` labels
                        configs.append({"model": model, "encoder": encoder,
                                        "qtype": qtype, "wording": wording})
    return configs


def _seat_results(bot, opponent, n_hands: int, seed: int) -> dict:
    """Contender's payoff in each seat. bot=P1 as-is; bot=P2 is the negation
    of the opponent-as-P1 result."""
    as_p1 = head_to_head(bot, opponent, n_hands=n_hands, seed=seed)
    flipped = head_to_head(opponent, bot, n_hands=n_hands, seed=seed)
    as_p2 = {"mean": -flipped["mean"], "ci_low": -flipped["ci_high"], "ci_high": -flipped["ci_low"]}
    return {"P1": as_p1, "P2": as_p2}


def run_experiments(config_path: str, manifest_path: str, output_dir: str, cache_dir: str) -> None:
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    manifest = load_manifest(manifest_path)
    seed = cfg.get("seed", 0)
    n_hands = cfg["head_to_head_hands"]
    os.makedirs(output_dir, exist_ok=True)

    cfr_bot, _ = CFRTrainer(seed=seed).train(iterations=cfg["cfr_iterations"])
    opponents = {"Nash": NashBot(alpha=0.0), "CFR": cfr_bot}

    scorers = {}
    contenders = []  # (config dict, bot)
    for c in expand_configs(cfg):
        if c["model"] not in scorers:
            scorers[c["model"]] = build_scorer(get_entry(manifest, c["model"]), cache_dir)
        bot = ModelBot(scorers[c["model"]], encoder=c["encoder"], qtype=c["qtype"],
                       wording=c["wording"])
        contenders.append((c, bot))
    sanity = {"model": "sanity-nash", "encoder": "minimal", "qtype": "choice", "wording": "plain"}
    contenders.append((sanity, nash_sanity_bot()))

    summary_rows, h2h_rows, infoset_rows = [], [], []
    for c, bot in contenders:
        cid = config_id(c)
        dist = distance_from_nash(bot)
        summary_rows.append({**c, "config_id": cid, "exploitability": exploitability(bot),
                             "total_l1": dist["total_l1"], "alpha": dist["alpha"]})
        for key, row in dist["per_infoset"].items():
            infoset_rows.append({"config_id": cid, "infoset": key, "l1": row["l1"], "kl": row["kl"]})
        for opp_name, opp in opponents.items():
            for seat, res in _seat_results(bot, opp, n_hands, seed).items():
                h2h_rows.append({"config_id": cid, "opponent": opp_name, "seat": seat,
                                 "mean": res["mean"], "ci_low": res["ci_low"], "ci_high": res["ci_high"]})

    def write(name, fields, rows):
        with open(os.path.join(output_dir, name), "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    write("summary.csv", ["config_id", "model", "encoder", "qtype", "wording",
                          "exploitability", "total_l1", "alpha"], summary_rows)
    write("head_to_head.csv", ["config_id", "opponent", "seat", "mean", "ci_low", "ci_high"], h2h_rows)
    write("infoset_distance.csv", ["config_id", "infoset", "l1", "kl"], infoset_rows)


if __name__ == "__main__":
    run_experiments(sys.argv[1], "models.yaml", sys.argv[2], "results/cache")
