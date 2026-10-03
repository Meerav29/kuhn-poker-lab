# Laya fine-tuning feasibility (Task 10 probe)

Date: 2026-10-03. Package: `laya` 0.3.24 (installed), checkpoint `convaiinnovations/laya` (cached).
Probe: `python -m scripts.probe_laya_training` (plain python, not pytest).
Tags: **[V]** verified by running code or reading installed source; **[R]** read on the model card/GitHub via WebFetch (a summarising fetcher, so treat details as secondary until checked against the notebook itself); **[I]** my inference.

## What the package exposes

- [V] No trainer ships in the pip package. `dir(laya)` has no `train`/`Trainer`/`finetune`; the only "fit" names are calibration helpers (`fit_temperatures`, `fit_binning_map`, `fit_abstention_thresholds`, which fit post-hoc temperatures and thresholds, not weights). The only `backward()` in the package is inside calibration (LBFGS on a temperature).
- [V] `agent.model` is a plain `torch.nn.Module`: `laya.common.DecisionModel`, built by `laya.common.build_model(cfg, encoder_dir, pretrained=False)` and filled with `load_state_dict(safetensors, strict=True)` (see `Agent.__init__`). Inference-only wrappers (`@torch.no_grad`, `_InferenceGate`, optional `compile`/`fast`/ONNX paths) sit around it in `Agent`; the module itself is stock eager PyTorch.
- [V] Parameters (421.3M): encoder (ModernBERT-large) 394.8M, 2-layer transformer head 25.2M, scorer MLP 1.1M, act head 0.3M, type embedding ~0, plus a `temperature` buffer. All have `requires_grad=True` after load; the model is in eval mode.
- [V] The typed heads: `forward(input_ids, attention_mask, marker_pos, marker_mask, qtype, detach_encoder=False)` returns `(logits [rows, options], act_logits)`. `choice`/`noul`/`score` are NOT separate heads; they share one scorer, with a `[MASK]` marker token per option, `type_emb` for the question type (`QTYPES = {choice:0, score:1, noul:2}`), and a softmax over option marker logits (then per-type temperature at inference). Soft targets are therefore natural: a distribution over the option markers.
- [V] Training-oriented hooks exist in the library code: `detach_encoder` flag (train head only), `model.head_checkpointing`, `proper_reward(q, target, qtype, mask, w_sph, w_rps)` (log + spherical + ranked-probability score, accepts soft `target`), `td_lambda_targets`, `collate_items` (accepts per-item `target` lists, builds the batch), `build_sequence`/`build_head` (text -> ids + marker positions), and `build_model(..., pretrained=True)` whose comment says "Training-time Hub load of the base encoder".
- [V] One forward + backward on a single soft-target example (`[0.3, 0.7]`, no optimizer, no loop) worked on the GPU: soft cross-entropy 0.611, `proper_reward` -0.230, nonzero gradients on 201 of 205 parameter tensors, including the encoder and scorer (act head gets none under a scorer-only loss, as expected). So the heads are differentiable and soft-target loss is straightforward.

## Training recipe upstream

- [R] The model card says it was trained with "RLCD" (exploration noise on logits, strictly proper scoring-rule reward) and recommends domain fine-tuning (laya-typed-decisions 0.766 vs 0.362 base on one benchmark). It points to `notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb` (and an MPS script `laya_finetune_typed_decisions_mps.py`) in `github.com/NandhaKishorM/laya`. The notebook, per the fetch summary, imports `build_model, proper_reward, QTYPES` from `laya.common`, uses AdamW (encoder lr 2.5e-5, head lr 1e-4), fp16 autocast, encoder gradient checkpointing plus `head_checkpointing`, a loss of the RL term plus a full soft cross-entropy term, and saves `model.safetensors` via `safetensors.save_file`. Not verified by me; the notebook should be read directly when the rung-2 plan is written.
- [I] A soft-target-only fine-tune (the plan's use of `kuhn/finetune_data.py`) can skip the RL term and use soft cross-entropy on the marker logits; the probe shows that loss trains the encoder and scorer.

## Hardware

- [V] GPU: AMD Radeon (ROCm torch), reported total 6.2 GiB. Forward+backward of one short example at fp32 peaked at 3.39 GiB (weights 1.7 GiB plus grads and activations).
- [I] Full fine-tune with fp32 AdamW needs about 4 x 421M x 4 B = 6.7 GiB for weights, grads and two Adam states before activations, so it will not fit in 6.2 GiB as is. Workable options: freeze the encoder (`detach_encoder=True`, ~27M trainable params, optimizer state negligible), or train head + top encoder layers only, or bf16/8-bit optimizer states (untested on this ROCm build). Not tested; also the ROCm training stability of ModernBERT here is unverified.

## Verdict

Fine-tuning supported: **yes, with caveats**. Supported at the model level (differentiable `nn.Module`, soft-target-compatible loss helpers, upstream notebook), but there is no packaged `train`/`fit` entry point; you must write the loop yourself on the internals.

Entry points for the rung-2 plan (all [V] to exist in the installed package):
- `laya.load(...)` then `agent.model` (or `laya.common.build_model(cfg, encoder_dir=..., pretrained=False)` + `safetensors.torch.load_file` + `load_state_dict(strict=True)`)
- `laya.common.build_sequence(tok, state, {"t": "choice", "ins": ..., "crit": {label: desc}})` -> `(ids, markers)`
- `laya.common.collate_items([[item, ...]], pad_id)` with `item = {"ids", "markers", "qtype", "target"}`
- `model(input_ids, attention_mask, marker_pos, marker_mask, qtype, detach_encoder=...)` -> `(logits, act_logits)`
- Loss: soft CE on `logits` masked by `marker_mask`, optionally `laya.common.proper_reward`
- Save: `safetensors.torch.save_file(model.state_dict())` into a copy of the model dir; reload with `laya.load(path)`; temperatures (`model.temperature`, `laya.calibrate`) should be refit after training [I].

**Recommendation: GO for Laya, head-only first** (freeze encoder via `detach_encoder=True` to fit 6.2 GiB; escalate to partial encoder unfreezing if accuracy lags). Fallback if the loop proves unstable on ROCm: fine-tune a ModernBERT classifier head directly, or only the `LogprobScorer` baseline's LM. Read the upstream notebook before writing the plan.
