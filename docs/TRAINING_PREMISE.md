# IQA-T1 Training Premise Brief (R2)

Date: 2026-09-06  
Owner: Machine Learning Engineer  
Sources: [arXiv:2607.12375](https://arxiv.org/abs/2607.12375), repo `README` / SFT+GRPO configs, local MPS inference work (R1–I3)

**Status:** training still frozen until this brief is accepted and a training Outcome is opened.

Also intended path on Mac: `~/Projects/IQA-T1/scratch/TRAINING_PREMISE.md` (copy when host is reachable).

---

## 1. What we already proved (inference)

| Item | Result |
| --- | --- |
| Stock Mac run | Blocked (`flash_attention_2`, no `model.to` off-CUDA, default internal SFT path) |
| MPS path | Works: SDPA + `model.to(mps)` + public `zibuyu-02/IQA-T1` |
| Smoke | `826373.jpg` → **Score 4.22** (`MAX_TOOL_CALLS=6`) |
| Tools | Local OpenCV/numpy generators; no CUDA |
| Entry | `~/Projects/IQA-T1/./run_mps.sh` + `.venv` + fork [peter0749/IQA-T1](https://github.com/peter0749/IQA-T1) |

Inference ≠ training. Tool loop and scoring format are understood; **we have not reproduced paper PLCC/SRCC or any train run**.

---

## 2. Paper thesis (training-relevant)

IQA-T1 = Qwen3-VL + **tool-generated visual evidence** in the reasoning chain.

Problem: MLLM visual tokens are semantically biased; low-level degradations (noise/blur/freq) are under-represented.

Solution:
1. 15 deterministic perceptual tools → structured maps/histograms.
2. Model learns to **call tools**, **read evidence**, **emit score** in `[1,5]`.
3. Two-stage training:
   - **SFT:** how to talk / call / ground on evidence (Q-Tool chains).
   - **GRPO:** when to call, not too many, score closer to GT.

Ablations that matter for our plan:
- Evidence images matter a lot (SFT without real maps is much weaker).
- Dynamic tool policy beats “always call all 15” or fixed-K.
- Reward pieces: format + score + tool-count (prefer ≤4) + no-repeat.

Paper headline: avg PLCC/SRCC **0.795 / 0.784** on 7 benchmarks (train on KonIQ, eval OOD).

---

## 3. Data: Q-Tool

| Fact | Detail |
| --- | --- |
| HF | `zibuyu-02/Q-Tool` (~14.5 GB) |
| Scale | ~10.3k KonIQ images × 15 precomputed tool maps; ~11.3k reasoning rows |
| Layout (repo expectation) | `dataset/KONIQ/{koniq,tools,metas}` |
| SFT | LlamaFactory dataset `koniq_iqa`; `media_dir` → KONIQ; `mask_mm_tokens: true` (don’t CE-train on evidence image tokens) |
| GRPO | EasyR1 JSON (`train_koniq_7k_rl.json` etc.); `image_dir: YOUR_KONIQ_DATA_DIR` |
| Local inference note | Runtime can regenerate tools on the fly; **training recipes assume precomputed maps** for speed/consistency |

**Premise:** do not start train until Q-Tool is downloaded, layout-checked, and a few samples spot-checked against `scripts/tools.py` outputs (determinism claim).

---

## 4. Stage I — SFT (LlamaFactory)

From paper + `configs/qwen3vl_4b_iqa_full_sft.yaml` / `train_qwen3vl_iqa_sft-4b.sh`:

| Knob | Value |
| --- | --- |
| Base | `Qwen3-VL-4B-Instruct` (download separately) |
| Type | **Full** LM+projector; **freeze vision tower** |
| Optim | lr `1e-5`, warmup 0.05, 2 epochs |
| Batch | per-device 1 × accum 16 |
| Special tokens | `<answer_start/end>` + 15 tool name tokens; `resize_vocab` |
| Parallelism | script: `CUDA_VISIBLE_DEVICES=0,1,2,3`, DeepSpeed ZeRO-3 offload |
| Env | `environment_sft.yaml` (Linux/CUDA; not Homebrew MPS stack) |
| Out | `saves/qwen3-vl-4B-Instruct-all-SFT` |

**Hardware floor (stock recipe):** multi-GPU CUDA (≥4× consumer 24GB class is what the launcher assumes).  
**M2 Ultra:** not a drop-in target for this recipe (no DeepSpeed/CUDA stack; full 4B SFT + multimodal context is a different engineering project).

---

## 5. Stage II — GRPO (EasyR1 / verl + vLLM)

From paper + `qwen3_vl_4b_iqa_agent_grpo-v1.yaml`:

| Knob | Value |
| --- | --- |
| Init | SFT checkpoint |
| GPUs | **6×4090 (24GB)** (`n_gpus_per_node: 6`) |
| FSDP | `enable_full_shard: true` (~31GB model+optim+grad → must shard) |
| Rollout | vLLM, `tensor_parallel_size: 2`, `n=8` samples/prompt |
| Agent | multi-turn tools, `max_turns: 6`, stop on `<answer_end>` |
| LR | `1e-6` (config comments also mention `5e-7` variants) |
| KL | `1e-2` |
| Reward | `examples/reward_function/iqa.py:compute_score` |
| Env | `environment_grpo.yaml` (torch+cu, vLLM, Ray) |

**Hardware floor (stock recipe):** 6×24GB CUDA + large host RAM/disk.  
**M2 Ultra:** GRPO path is CUDA/vLLM-shaped; treat as **cloud-only** unless we redesign (out of scope for a “faithful recipe” first train).

---

## 6. M2 Ultra vs cloud — honest split

| Work | M2 Ultra (192GB) | Cloud CUDA |
| --- | --- | --- |
| Inference / smoke / tool debug | ✅ done | optional fidelity check |
| Download & audit Q-Tool + base Qwen3-VL | ✅ good (disk/CPU) | ✅ |
| Rewrite SFT to MLX/LoRA | possible R&D | not paper-faithful |
| Stock SFT (full + DS ZeRO-3) | ❌ | ✅ (≈4×24GB+) |
| Stock GRPO (vLLM+FSDP agent) | ❌ | ✅ (≈6×24GB) |
| Eval PLCC/SRCC on 7 sets | heavy; possible offline later | easier with GPU batch infer |

**Recommendation order for a training program:**
1. **Data readiness** on Mac (download, layout, spot-check).
2. **Cloud SFT smoke** (short steps) before full 2 epochs — verify loss/format/special tokens.
3. **Cloud GRPO** only after SFT checkpoint looks sane.
4. Keep Mac as **infer + analysis** machine; do not promise paper-faithful train on MPS.

Cost note (from R1 CODEX report, still UNVERIFIED wall-clock): SFT hours on 4×4090 and GRPO many hours on 6×4090 — run a **20-step CUDA benchmark** before locking budget.

---

## 7. Gaps / risks before any train card

1. **Faithfulness of our Score 4.22** vs paper / CUDA — unknown (single image, no PLCC).
2. **SFT default path vs public GRPO weights** — already bit us in inference; train must track checkpoint lineage carefully.
3. **Two conda envs** — SFT and GRPO are incompatible; don’t mix with `.venv` MPS infer env.
4. **Q-Tool precomputed maps vs live tools** — training assumes caches; verify hash/size before long jobs.
5. **Upstream vs our fork** — train patches should live on `peter0749/IQA-T1` (or a train branch), not pushed to `zibuyu-02`.

---

## 8. Proposed next Outcomes (for PM — do not open until user picks)

| ID | Outcome | Depends |
| --- | --- | --- |
| D1 | Q-Tool + Qwen3-VL-4B-Instruct landed & audited under `~/Projects/IQA-T1` | R2 accept |
| T1 | Cloud SFT smoke (N steps) with stock recipe | D1 + rent/approve CUDA |
| T2 | Full SFT → publish SFT ckpt path | T1 |
| T3 | Cloud GRPO from SFT | T2 |
| E1 | Mini eval (KonIQ holdout or 1–2 OOD sets) PLCC/SRCC | T2 or public ckpt |

**Out of scope for first train wave:** MLX full train, MPS GRPO, changing the 15-tool library, unfreezing without D1.

---

## 9. One-line premise

> IQA-T1 training is a **CUDA two-stage** problem (SFT then GRPO) on **Q-Tool**; the M2 Ultra is already a solid **inference/debug** box. Next concrete step is **data readiness (D1)**, then an explicit go on **cloud SFT smoke (T1)** — not local full training.
