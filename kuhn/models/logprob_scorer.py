"""Generic non-Jev baseline: read the next-token logprobs of a small local
causal LM over the answer labels (the same idea as the XavierJev entry on
the Jev catalogue). Shows what an off-the-shelf LM does with no
decision-model training."""
import math

from kuhn.models.scorer import Question


def probs_from_logprobs(logprobs: dict) -> dict:
    m = max(logprobs.values())
    exp = {k: math.exp(v - m) for k, v in logprobs.items()}
    z = sum(exp.values())
    return {k: v / z for k, v in exp.items()}


class LogprobScorer:
    def __init__(self, model_id: str = "Qwen/Qwen2.5-0.5B-Instruct", name: str = None):
        import torch  # lazy imports keep unit tests and the rest of the package light
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.scorer_id = name or f"logprob-{model_id.split('/')[-1]}"
        self._torch = torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tok = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModelForCausalLM.from_pretrained(model_id).to(self._device).eval()

    @staticmethod
    def _prompt(state: str, question: Question) -> str:
        options = ", ".join(question.labels)
        return f"{state}\n{question.instructions} Answer with exactly one of: {options}.\nAnswer:"

    def score(self, state: str, question: Question) -> dict:
        ids = {label: self._tok(" " + label, add_special_tokens=False).input_ids[0]
               for label in question.labels}
        if len(set(ids.values())) != len(ids):
            raise ValueError(f"labels share a first token, cannot score them apart: {question.labels}")
        enc = self._tok(self._prompt(state, question), return_tensors="pt").to(self._device)
        with self._torch.no_grad():
            logits = self._model(**enc).logits[0, -1]
        logp = self._torch.log_softmax(logits.float(), dim=-1)
        return probs_from_logprobs({label: logp[i].item() for label, i in ids.items()})
