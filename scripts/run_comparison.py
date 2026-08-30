"""Produces the CSV/plot artifacts the write-up (docs/REPORT.md) draws
from: exact exploitability per bot, a head-to-head win-rate matrix, and
the CFR training convergence curve."""
import csv
import os

import matplotlib
matplotlib.use("Agg")  # headless -- this script writes files, doesn't show a window
import matplotlib.pyplot as plt

from kuhn.bots.nash import NashBot
from kuhn.bots.heuristics import HonestBot, AlwaysBluffBot, RandomBot
from kuhn.bots.cfr import CFRTrainer
from kuhn.evaluate import exploitability, head_to_head


def run(output_dir: str, cfr_iterations: int = 50000, cfr_log_every: int = 500,
        head_to_head_hands: int = 20000, seed: int = 0) -> None:
    os.makedirs(output_dir, exist_ok=True)

    trainer = CFRTrainer(seed=seed)
    cfr_bot, convergence_log = trainer.train(iterations=cfr_iterations, log_every=cfr_log_every)

    bots = {
        "Nash": NashBot(alpha=0.0),
        "Honest": HonestBot(),
        "AlwaysBluff": AlwaysBluffBot(),
        "Random": RandomBot(),
        "CFR": cfr_bot,
    }

    # 1. exploitability table
    with open(os.path.join(output_dir, "exploitability.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["bot", "exploitability"])
        for name, bot in bots.items():
            writer.writerow([name, exploitability(bot)])

    # 2. head-to-head matrix (every ordered pair, bot_a as Player 1)
    with open(os.path.join(output_dir, "head_to_head.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["bot_a", "bot_b", "mean", "ci_low", "ci_high"])
        names = list(bots.keys())
        for name_a in names:
            for name_b in names:
                if name_a == name_b:
                    continue
                result = head_to_head(bots[name_a], bots[name_b], n_hands=head_to_head_hands, seed=seed)
                writer.writerow([name_a, name_b, result["mean"], result["ci_low"], result["ci_high"]])

    # 3. CFR convergence curve
    with open(os.path.join(output_dir, "cfr_convergence.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "exploitability"])
        for it, exp in convergence_log:
            writer.writerow([it, exp])

    iterations = [it for it, _ in convergence_log]
    exploitabilities = [exp for _, exp in convergence_log]
    fig, ax = plt.subplots()
    ax.plot(iterations, exploitabilities)
    ax.set_xlabel("CFR training iterations")
    ax.set_ylabel("Exploitability")
    ax.set_title("CFR average-strategy convergence toward equilibrium")
    fig.savefig(os.path.join(output_dir, "cfr_convergence.png"))
    plt.close(fig)


if __name__ == "__main__":
    run(output_dir="results")
