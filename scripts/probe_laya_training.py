"""Probes whether the installed `laya` package can be fine-tuned locally.

Part 1 (inference-shape check): walks the installed package layout, metadata and
training-looking names, then loads the cached checkpoint and reports the module
tree / parameter counts and the observed signatures the rung-2 plan would call.

Part 2 (forward/backward/step): builds a realistic small soft-target batch from
kuhn.finetune_data.build_dataset(NashBot()), runs forward+backward with
detach_encoder=True (head-only), reports peak GPU memory and which parameter
groups got gradients, takes ONE AdamW step over the trainable parameters, then
saves the weights to a temp dir, reloads via laya.load(<dir>) and compares
predictions with the in-memory agent. No training loop is built.

Run as a plain script: `python -m scripts.probe_laya_training` (never via pytest).
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # nothing may be downloaded

import importlib.metadata
import inspect
import pkgutil
import shutil
import tempfile
import warnings

warnings.filterwarnings("ignore")

import torch

import laya
from laya.common import QTYPES, build_model, build_sequence, collate_items, proper_reward

print("file:", laya.__file__)
print("version:", importlib.metadata.version("laya"))
print("public names:", [n for n in dir(laya) if not n.startswith("_")])
print("submodules:")
for m in pkgutil.walk_packages(laya.__path__, "laya."):
    print("  ", m.name)

meta = importlib.metadata.metadata("laya")
print("summary:", meta.get("Summary"))
print("project urls:", meta.get_all("Project-URL"))
hits = [n for n in dir(laya) if any(k in n.lower() for k in ("train", "fit", "finetune", "loss", "optim"))]
print("training-looking names:", hits)

# ---- Part 1: observed signatures and the loaded module ----
print("\n== signatures ==")
agent = laya.load("convaiinnovations/laya")
model = agent.model
for name, fn in (("build_model", build_model), ("build_sequence", build_sequence),
                 ("collate_items", collate_items), ("proper_reward", proper_reward),
                 ("DecisionModel.forward", model.forward)):
    print(f"{name}{inspect.signature(fn)}")
print("detach_encoder in forward signature:", "detach_encoder" in inspect.signature(model.forward).parameters)
print("head_checkpointing attribute:", hasattr(model, "head_checkpointing"), "(attribute only, not a forward arg)")
print("agent.tok:", type(agent.tok).__name__, "| pad_token_id:", agent.tok.pad_token_id,
      "| agent.device:", agent.device)
print("agent.cfg keys:", sorted(agent.cfg))
print("QTYPES:", QTYPES)

print("\n== module ==")
print("model class:", type(model).__module__ + "." + type(model).__name__,
      "| nn.Module:", isinstance(model, torch.nn.Module))
groups = {"encoder": model.encoder, "head": model.head, "type_emb": model.type_emb,
          "scorer": model.scorer, "act_head": model.act_head}
total = 0
for name, mod in groups.items():
    n = sum(p.numel() for p in mod.parameters())
    total += n
    print(f"  params {name}: {n/1e6:.1f}M ({len(list(mod.parameters()))} tensors)")
print(f"  params total: {total/1e6:.1f}M | dtype: {next(model.parameters()).dtype} | "
      f"requires_grad all: {all(p.requires_grad for p in model.parameters())} | training flag: {model.training}")
dev = agent.device
if dev.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0), "| reported total GiB:",
          round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2))
    torch.cuda.synchronize()
    print(f"GiB allocated after load: {torch.cuda.memory_allocated()/2**30:.2f}")

# ---- Part 2: realistic head-only batch, one AdamW step ----
print("\n== head-only forward/backward/step ==")
from kuhn.bots.nash import NashBot
from kuhn.finetune_data import build_dataset

examples = build_dataset(NashBot(), encoders=("natural",), qtypes=("choice",), wordings=("plain",))
items = []
for ex in examples:
    qn = ex["question"]
    labels = [lab for lab, _ in qn.options]
    q = {"t": "choice", "ins": qn.instructions, "crit": {lab: None for lab in labels}}
    ids, markers = build_sequence(agent.tok, ex["state"], q)
    items.append({"ids": ids, "markers": markers, "qtype": QTYPES["choice"],
                  "target": [ex["target"][lab] for lab in labels]})
print("examples:", len(items), "| options per item:", sorted({len(i["markers"]) for i in items}),
      "| seq len min/max:", min(len(i["ids"]) for i in items), max(len(i["ids"]) for i in items))
batch = collate_items([[it] for it in items], agent.tok.pad_token_id)
print("collate_items keys:", sorted(batch))
b = {k: v.to(dev) for k, v in batch.items() if hasattr(v, "to")}

model.train()
if dev.type == "cuda":
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
logits, act_logits = model(b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"],
                           b["qtype"], detach_encoder=True)
# forward() already fills padded option slots with -1e4 (masked_fill on ~marker_mask) before returning logits;
# a batched loop must keep that masking before softmax. Targets are zero in padded slots (collate_items).
logp = torch.log_softmax(logits.float(), -1)
soft_ce = -(b["target"] * logp.masked_fill(~b["marker_mask"], 0.0)).sum(-1).mean()
reward = proper_reward(torch.softmax(logits.float(), -1), b["target"], b["qtype"], b["marker_mask"])
print(f"soft_ce: {float(soft_ce):.4f} | proper_reward mean: {float(reward.mean()):.4f} (higher is better)")
soft_ce.backward()
if dev.type == "cuda":
    torch.cuda.synchronize()
    print(f"peak GiB after fwd+bwd (head-only, batch {len(items)}): {torch.cuda.max_memory_allocated()/2**30:.2f}")
print("tensors with a gradient / total, per group:")
no_grad_names = []
for name, mod in groups.items():
    ps = list(mod.named_parameters())
    got = [n for n, p in ps if p.grad is not None]
    nz = [n for n, p in ps if p.grad is not None and p.grad.abs().sum() > 0]
    miss = [n for n, p in ps if p.grad is None or p.grad.abs().sum() == 0]
    no_grad_names += [f"{name}.{n}" for n in miss]
    print(f"  {name}: grad {len(got)}/{len(ps)}, nonzero {len(nz)}/{len(ps)}")
print("tensors without gradient / zero gradient:", no_grad_names)

trainable = [p for p in model.parameters() if p.grad is not None]
opt = torch.optim.AdamW(trainable, lr=1e-4, weight_decay=0.01)
if dev.type == "cuda":
    torch.cuda.reset_peak_memory_stats()
before = [p.detach().clone() for p in trainable[:3]]
opt.step()
opt.zero_grad(set_to_none=True)
if dev.type == "cuda":
    torch.cuda.synchronize()
    print(f"AdamW step over {len(trainable)} tensors ({sum(p.numel() for p in trainable)/1e6:.1f}M params): "
          f"peak GiB during step {torch.cuda.max_memory_allocated()/2**30:.2f}, "
          f"allocated after {torch.cuda.memory_allocated()/2**30:.2f}")
print("weights changed by step:", any(not torch.equal(a, p) for a, p in zip(before, trainable[:3])))

# ---- Part 2b: save / reload round trip ----
print("\n== save/reload round trip ==")
model.eval()
state = "card=K history=b"
questions = {"act": {"type": "choice", "instructions": "The opponent bet. Do you fold or call?",
                     "criteria": ["fold", "call"]}}
mem_pred = agent.predict(state, questions)["answers"]["act"]["probabilities"]
print("in-memory (post-step) probs:", mem_pred)

import glob

from huggingface_hub import constants

# The cache holds only the 5 file groups laya fetches (config, weights, tokenizer/, encoder/), so
# snapshot_download(local_files_only=True) reports it incomplete; read the snapshot dir directly.
snaps = glob.glob(os.path.join(constants.HF_HUB_CACHE, "models--convaiinnovations--laya", "snapshots", "*"))
src_dir = snaps[0]
print("cached model dir contents:", sorted(os.listdir(src_dir)))
tmp = tempfile.mkdtemp(prefix="laya_roundtrip_")
try:
    out = os.path.join(tmp, "model")
    shutil.copytree(src_dir, out, symlinks=False)
    from safetensors.torch import save_file

    wpath = os.path.join(out, "model.safetensors")
    if os.path.islink(wpath):
        os.remove(wpath)
    save_file({k: v.detach().contiguous().cpu() for k, v in model.state_dict().items()}, wpath)
    print(f"saved model.safetensors: {os.path.getsize(wpath)/2**30:.2f} GiB (state_dict as held in memory)")
    agent2 = laya.load(out)
    re_pred = agent2.predict(state, questions)["answers"]["act"]["probabilities"]
    print("reloaded probs:             ", re_pred)
    diff = max(abs(mem_pred[k] - re_pred[k]) for k in mem_pred)
    print(f"max |prob diff| (predict rounds to 4 dp): {diff:.6f} | within 1e-4: {diff <= 1e-4}")
    s1, s2 = agent.model.state_dict(), agent2.model.state_dict()
    max_w = max(float((s1[k].float().cpu() - s2[k].float().cpu()).abs().max()) for k in s1)
    print("max |weight diff|, in-memory vs reloaded:", max_w)
except Exception as e:  # report exactly what failed
    import traceback

    traceback.print_exc()
    print("ROUNDTRIP FAILED:", type(e).__name__, e)
finally:
    shutil.rmtree(tmp, ignore_errors=True)
    print("temp dir removed:", not os.path.exists(tmp))
