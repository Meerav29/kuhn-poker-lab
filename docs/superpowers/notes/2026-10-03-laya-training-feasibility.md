# Laya fine-tuning feasibility (Task 10 probe)

Date: 2026-10-03. Package: `laya` 0.3.24 (installed), checkpoint `convaiinnovations/laya` (cached; probe runs with `HF_HUB_OFFLINE=1`).
Probe: `python -m scripts.probe_laya_training` (plain python, not pytest). All "run" references below are that script.

Tags: **[V]** verified by running code (the run is named); **[R]** read in installed source or on the model card/GitHub (via WebFetch, a summarising fetcher) but not exercised; **[I]** inference, not tested.

## What the package exposes

- [V] (package walk) No trainer ships in the pip package: no `train`/`Trainer`/`finetune` names in `dir(laya)`. The only "fit" names (`fit_temperatures`, `fit_binning_map`, `fit_abstention_thresholds`) are post-hoc calibration. [R] The only `backward()` in the package is in `calibrate.py` (LBFGS on a temperature).
- [V] `agent.model` is a plain `torch.nn.Module`, `laya.common.DecisionModel`, fp32 on load, all parameters `requires_grad=True`, eval mode. 421.3M params: encoder (ModernBERT-large) 394.8M in 170 tensors, 2-layer transformer head 25.2M (24 tensors), scorer MLP 1.1M (6), act head 0.3M (4), type embedding (1). [R] `Agent` wraps it with `@torch.no_grad` inference helpers and optional compile/fast/ONNX paths; the module itself is eager PyTorch.
- [R] `choice`/`noul`/`score` are not separate heads. One shared scorer reads the `[MASK]` marker token of each option; `type_emb` (QTYPES `{choice:0, score:1, noul:2}`, [V]) tags the question type; softmax over option markers gives the distribution. Soft targets are therefore a distribution over option markers.

## Entry points for a rung-2 plan (each tagged individually)

- [V] (run, signatures section) `laya.load("convaiinnovations/laya")` returns an `Agent`; `agent.model` is the trainable module; `agent.tok` is a `TokenizersBackend` tokenizer with `pad_token_id` 50283; `agent.device` is `cuda` here; `agent.cfg` keys: act_costs, amp_dtype, cost_wrong_act, encoder, head_layers, head_max_len, max_len, max_prefixes, model_name, temperature, temperature_by_options, training.
- [V] (run) `laya.common.build_sequence(tok, state, q, max_len=512, head_max_len=192, option_order=None, truncate_left=False, state_ids=None, return_stats=False, return_truncation_stats=False)` returns `(ids, markers)`. The question dict form that worked is the internal one, `{"t": "choice", "ins": <instructions>, "crit": {label: None, ...}}` (note: not the public `type/instructions/criteria` form that `agent.predict` takes; [R] `Agent` converts between them).
- [V] (run) `laya.common.collate_items(batch, pad_id)` where `batch` is a list of lists of items `{"ids", "markers", "qtype", "target": [probs]}`. Returned keys: `attention_mask, input_ids, label, marker_mask, marker_pos, meta, qtype, target`. `meta` is a list of dicts (not a tensor); the rest move to the device with `.to`.
- [V] (run) `DecisionModel.forward(input_ids, attention_mask, marker_pos, marker_mask, qtype, detach_encoder: bool = False)` returns `(logits [rows, options], act_logits)`. `detach_encoder` is confirmed in the signature. `head_checkpointing` exists as a plain attribute (not a forward argument); its effect was not exercised ([R]).
- [R] `build_model(cfg, encoder_dir=None, pretrained=True, revision=None)`: signature observed [V]; I did not call it, `laya.load` does so internally with `pretrained=False` plus `load_state_dict(strict=True)`.
- [V] (run, head-only step) AdamW over only the parameters that received gradients (31 tensors, 26.2M params) stepped and changed weights. [R] Upstream uses encoder lr 2.5e-5 / head lr 1e-4 (notebook, via fetch summary, unverified).
- [V] (run, round trip) Save/reload works: copy the cached model dir (it contains `encoder/`, `model.safetensors`, `rl_agent_config.json`, `tokenizer/`) to a temp dir, overwrite `model.safetensors` with `safetensors.torch.save_file(model.state_dict())` (fp32, 1.57 GiB, tensors made `.contiguous().cpu()`), then `laya.load(<that dir>)`. After one AdamW step, the in-memory agent and the reloaded agent gave identical `predict` probabilities (`fold 0.3887, call 0.6113` both; max diff 0.000000, within 1e-4) and `max |weight diff| = 0.0`. Temp dir deleted afterwards. Caveat: `predict` rounds to 4 dp, but the exact weight comparison backs it up. `snapshot_download(local_files_only=True)` reports the cache "incomplete" (laya fetches only 5 file groups), so the probe reads the snapshot directory directly.

## Loss, precisely

- [V] The probe's soft cross-entropy is `-(target * log_softmax(logits)).sum(-1).mean()`, applied to the marker logits returned by `forward`. `forward` already fills padded option slots with `-1e4` (`masked_fill(~marker_mask, ...)` on the logits, [R]), so padding is masked **before** the softmax; `collate_items` also zero-pads `target`. The probe additionally `masked_fill`s `logp` to 0 at padded slots. Any batched loop must keep padded slots at -inf/-1e4 before softmax and must not feed padded targets. In this probe all 12 items had 2 options, so padding was not actually exercised [I for mixed option counts].
- [V] `proper_reward(q, target, qtype, mask, w_sph=0.5, w_rps=1.0, log_floor=-9.21)` expects `q` as **probabilities** (it takes `log(q)` itself and `q.norm`), not logits; `mask` is the boolean option mask. [R] It returns `target·log q + w_sph * spherical score` (minus a ranked-probability term for `score` questions). It is a **reward: higher is better**; negate it to use as a loss. The run computed it on `softmax(logits)`: mean -1.0095 on the 12-example batch (soft CE 1.3217). Its gradients were not used for training in the probe.

## Gradients (head-only run, `detach_encoder=True`, batch of 12)

- [V] encoder 0/170 tensors with gradient (detached, as intended); head 24/24; type_emb 1/1; scorer 6/6; act_head 0/4 (all nonzero where present). Trainable via AdamW: head + type_emb + scorer = 31 tensors, 26.2M params.
- [V] In the earlier full-backward run (encoder attached, scorer-only loss on one example), 201 of 205 tensors had nonzero gradients. The missing 4 are the act head's 4 tensors [I: matches the 0/4 act_head result above; the act head feeds only `act_logits`, which the scorer-only loss never touches]. Training the act head would need its own loss.

## Hardware (units: GiB throughout)

- [V] GPU: AMD Radeon(TM) Graphics via ROCm torch; reports **6.22 GiB** total. Usable memory is lower (driver/other processes, and this may be a shared-memory device); I did not measure the usable ceiling. Weights alone (fp32) were 1.57 GiB allocated after load.
- [V] Head-only, batch 12, seq len 29 to 39 tokens: peak **2.59 GiB** after forward+backward; AdamW step over 26.2M params peaked at 2.04 GiB (allocated after step 1.84 GiB). Roughly 3.6 GiB of headroom against the reported 6.22 GiB, at this tiny sequence length.
- [I] Full fine-tune with fp32 AdamW: 421.3M params x 16 B (weights, grads, two Adam states) is about 6.28 GiB before activations, above the reported 6.22 GiB. It cannot fit as-is. Options (all untested): partial encoder unfreezing, lower-precision optimizer states, gradient checkpointing.
- [I] ROCm stability over many steps is untested. Only one forward/backward and one optimizer step were run.

## Verdict

Fine-tuning supported: **yes at the model level, no packaged trainer.** The module is differentiable, soft-target loss works, a head-only step fits and a saved checkpoint reloads through `laya.load` with identical predictions. The training loop would be hand-written on the internals above.

**Recommendation: GO (head-only).** Basis, all [V] from the run: 26.2M trainable params, peak 2.59 GiB (fwd+bwd, batch 12) and 2.04 GiB (AdamW step) against a reported 6.22 GiB GPU, plus a working save/reload round trip. Still unverified and requiring a user decision before the rung-2 plan commits to them: full-model or partial-encoder fine-tuning, larger batches or longer sequences, and any cloud GPU. The model is 421M parameters, slightly over the plan's roughly 400M local-fine-tune constraint, so anything beyond head-only means consulting the user before spending on cloud. ROCm training stability over many steps is [I]/untested. Fallback if head-only proves unstable or too weak: fine-tune a ModernBERT classifier head directly, or only the `LogprobScorer` baseline's LM. The upstream notebook (`laya_finetune_typed_decisions_2xT4_kaggle.ipynb`) is [R] via a fetch summary only; read it directly before writing the plan.
