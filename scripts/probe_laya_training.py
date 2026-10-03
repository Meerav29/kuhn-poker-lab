"""Looks inside the installed `laya` package for anything training-related."""
import importlib.metadata
import pkgutil
import warnings

warnings.filterwarnings("ignore")

import laya

print("file:", laya.__file__)
print("public names:", [n for n in dir(laya) if not n.startswith("_")])
print("submodules:")
for m in pkgutil.walk_packages(laya.__path__, "laya."):
    print("  ", m.name)

meta = importlib.metadata.metadata("laya")
print("summary:", meta.get("Summary"))
print("project urls:", meta.get_all("Project-URL"))
print("requires:", importlib.metadata.requires("laya"))
hits = [n for n in dir(laya) if any(k in n.lower() for k in ("train", "fit", "finetune", "loss", "optim"))]
print("training-looking names:", hits)

# ---- Part 2: is the loaded model a plain, differentiable torch module? ----
# One forward + backward on a single soft-target example. No optimizer, no loop.
import torch
from laya.common import QTYPES, build_sequence, collate_items, proper_reward

agent = laya.load("convaiinnovations/laya")
model = agent.model
print("model class:", type(model).__module__ + "." + type(model).__name__)
print("is nn.Module:", isinstance(model, torch.nn.Module), "| device:", agent.device)
groups = {"encoder": model.encoder, "head": model.head, "type_emb": model.type_emb,
          "scorer": model.scorer, "act_head": model.act_head}
total = 0
for name, mod in groups.items():
    n = sum(p.numel() for p in mod.parameters())
    total += n
    print(f"  params {name}: {n/1e6:.1f}M")
print(f"  params total: {total/1e6:.1f}M | buffers: {[n for n, _ in model.named_buffers() if n == 'temperature']}")
print("requires_grad on load:", all(p.requires_grad for p in model.parameters()))
print("train flag after load:", model.training, "| has detach_encoder/head_checkpointing:",
      hasattr(model, "head_checkpointing"))

# Internal question form used by laya.agent: {"t": type, "ins": text, "crit": {label: description}}.
q = {"t": "choice", "ins": "The opponent bet. Do you fold or call?", "crit": {"fold": None, "call": None}}
ids, markers = build_sequence(agent.tok, "card=K history=b", q)
item = {"ids": ids, "markers": markers, "qtype": QTYPES["choice"], "target": [0.3, 0.7]}
batch = collate_items([[item]], agent.tok.pad_token_id)
dev = agent.device
b = {k: v.to(dev) for k, v in batch.items() if hasattr(v, "to")}

model.train()
if dev.type == "cuda":
    torch.cuda.reset_peak_memory_stats()
logits, act_logits = model(b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"])
logp = torch.log_softmax(logits.float(), -1)
soft_ce = -(b["target"] * logp.masked_fill(~b["marker_mask"], 0)).sum(-1).mean()
reward = proper_reward(torch.softmax(logits.float(), -1), b["target"], b["qtype"], b["marker_mask"])
print("logits:", logits.detach().cpu().tolist(), "soft_ce:", float(soft_ce), "proper_reward:", reward.tolist())
soft_ce.backward()
n_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
n_tot = sum(1 for p in model.parameters())
print(f"params with nonzero grad after soft-CE backward: {n_grad}/{n_tot}")
print("encoder grad present:", any(p.grad is not None for p in model.encoder.parameters()),
      "| scorer grad present:", any(p.grad is not None for p in model.scorer.parameters()),
      "| act_head grad (expected none for scorer-only loss):",
      any(p.grad is not None for p in model.act_head.parameters()))
if dev.type == "cuda":
    print(f"peak GPU mem, fp32 weights + fwd/bwd of 1 short example: {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")
    print("device:", torch.cuda.get_device_name(0), "| total GiB:", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1))
