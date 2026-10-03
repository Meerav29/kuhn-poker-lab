"""Pipeline sanity check: push the Nash strategy through the full
encoder -> Question -> scorer -> ModelBot path. Its exploitability must
be ~0; if it isn't, the adapter (not any model) is broken."""
from kuhn import game
from kuhn.bots.model_bot import ModelBot, build_question
from kuhn.bots.nash import NashBot
from kuhn.encoders import get_encoder
from kuhn.models.scorer import LookupScorer


def nash_sanity_bot(encoder: str = "minimal", qtype: str = "choice",
                    wording: str = "plain", alpha: float = 0.0) -> ModelBot:
    nash = NashBot(alpha=alpha)
    enc = get_encoder(encoder)
    table = {}
    for infoset in game.all_infosets():
        question = build_question(infoset, qtype, wording)
        dist = nash.action_probs(infoset, game.ACTIONS)
        table[enc(infoset)] = {label: dist[action] for label, action in question.options}
    return ModelBot(LookupScorer(table, "sanity-nash"), encoder=encoder, qtype=qtype,
                    wording=wording, name="sanity-nash")
