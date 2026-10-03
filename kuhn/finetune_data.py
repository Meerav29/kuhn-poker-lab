"""Fine-tuning data: (prompt, typed question, soft target) for every
infoset x prompt variant. Targets are the strategy bot's action
distribution (e.g. the CFR average strategy), not hard labels."""
from kuhn import game
from kuhn.bots.model_bot import build_question
from kuhn.encoders import get_encoder


def build_dataset(strategy_bot, encoders=("minimal",), qtypes=("choice",), wordings=("plain",)) -> list:
    examples = []
    for encoder in encoders:
        enc = get_encoder(encoder)
        for qtype in qtypes:
            for wording in wordings:
                if qtype == "noul" and wording != wordings[0]:
                    continue  # wording only changes `choice` labels
                for infoset in game.all_infosets():
                    question = build_question(infoset, qtype, wording)
                    dist = strategy_bot.action_probs(infoset, game.ACTIONS)
                    examples.append({
                        "infoset": infoset, "encoder": encoder, "qtype": qtype, "wording": wording,
                        "state": enc(infoset), "question": question,
                        "target": {label: dist.get(action, 0.0) for label, action in question.options},
                    })
    return examples


def split_held_out(examples: list, held_out_encoders) -> tuple:
    held = set(held_out_encoders)
    train = [e for e in examples if e["encoder"] not in held]
    held_out = [e for e in examples if e["encoder"] in held]
    return train, held_out
