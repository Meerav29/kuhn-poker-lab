"""One-off check: does Laya's noul P(yes) track its choice P(bet)? Also shows how much
Laya's bet probability depends on the card. Run as plain python (not pytest):
    python -m scripts.check_laya_choice_vs_noul > results/rung1/choice_vs_noul.txt
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from kuhn import game
from kuhn.bots.model_bot import build_question
from kuhn.bots.nash import NashBot
from kuhn.encoders import get_encoder
from kuhn.models.laya_scorer import LayaScorer


def main() -> None:
    scorer = LayaScorer()
    nash = NashBot(0.0)
    for encoder in ("minimal", "natural"):
        enc = get_encoder(encoder)
        print(f"--- {encoder}: choice P(bet/call) vs noul P(yes) vs Nash P(bet/call)")
        same_side = 0
        infosets = game.all_infosets()
        for infoset in infosets:
            q_choice = build_question(infoset, "choice", "plain")
            q_noul = build_question(infoset, "noul")
            p_choice = scorer.score(enc(infoset), q_choice)
            p_noul = scorer.score(enc(infoset), q_noul)
            bet_label = [label for label, action in q_choice.options if action == "b"][0]
            p_bet, p_yes = p_choice[bet_label], p_noul["yes"]
            nash_bet = nash.action_probs(infoset, game.ACTIONS)["b"]
            same_side += (p_bet > 0.5) == (p_yes > 0.5)
            print(f"{infoset:4} choice P(b)={p_bet:.2f}  noul P(yes)={p_yes:.2f}  nash P(b)={nash_bet:.2f}")
        print(f"same side of 0.5: {same_side} / {len(infosets)}")


if __name__ == "__main__":
    main()
