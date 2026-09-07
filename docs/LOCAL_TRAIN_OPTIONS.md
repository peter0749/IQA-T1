# IQA-T1 Local Training Options (R3)

Date: 2026-09-07  
Owner: Machine Learning Engineer  
Premise docs: [`TRAINING_PREMISE.md`](./TRAINING_PREMISE.md), Mac `~/Projects/IQA-T1/scratch/D1_STATUS.md`  
Constraint from user: **no cloud rental** — train on M2 Ultra only.

Also mirrored on Mac as `~/Projects/IQA-T1/scratch/LOCAL_TRAIN_OPTIONS.md`.

---

## 1. Hard constraints

| Constraint | Implication |
| --- | --- |
| No cloud CUDA | Paper-faithful SFT (DeepSpeed ZeRO-3, 4×GPU) and GRPO (vLLM + 6×4090) **will not run as shipped** |
| M2 Ultra 192GB | Strong unified memory; good for **inference + LoRA-scale** work; not a CUDA stack |
| D1 ready | Base `repo/Qwen3-VL-4B-Instruct/` (8.3G) + `repo/dataset/KONIQ/` (16G, 10375×15 tools) verified |
| Infer already works | MPS path + Score 4.22 smoke — keep that env separate from any train env |
| Paper data synth used GPT-4o | Local substitute available: **Qwen 3.8 Flash Next** via `~/mtplx_setup` (see §4) |

Anything labeled “local train” below is **non-paper-faithful** unless we later prove parity.

---

## 2. Option matrix

### A — Local LoRA / QLoRA SFT smoke (**recommended first → L1**)

**Goal:** prove we can fine-tune Qwen3-VL-4B on a **tiny** Q-Tool slice on Mac, emit valid `<think>…</think><answer_start>{"score":…}<answer_end>` + tool tags, without DeepSpeed/vLLM.

| Piece | Proposal |
| --- | --- |
| Backend | Prefer **MLX** (or PyTorch MPS if tooling is easier); **not** LlamaFactory+DeepSpeed stock script |
| Params | LoRA on language / projector; **freeze vision tower** (matches paper SFT intent) |
| Data | 50–500 Q-Tool conversations first (not full 11k) |
| Batch | 1 + grad accum; short max steps (e.g. 20–100) |
| Success | Loss decreases; sample generations parse; no OOM; artifact path recorded |
| Non-goals | Paper PLCC/SRCC; full 2-epoch SFT; GRPO |

**Pros:** fits “本機訓練”; uses D1 assets; cheap to fail.  
**Cons:** recipe rewrite; score quality unknown.

### B — Local “data factory” with Flash Next (parallel / prep)

**Goal:** replace paper’s **GPT-4o reasoning-chain synthesis** with local **Qwen 3.8 Flash Next** for any *new* or *augmented* Q-Tool-style chains (not required to unblock L1 if we train on existing Q-Tool).

Launch (from Mac):

```bash
# Official daily driver on :8000 — do not dual-serve large models
~/mtplx_setup/serve_flash_next.sh
# or managed switcher
cd ~/mtplx_setup && python3 mtplx_local.py switch flash-next
```

| Detail | Value |
| --- | --- |
| Script | `~/mtplx_setup/serve_flash_next.sh` |
| Model pack | `~/.mtplx/models/Youssofal--Qwen3.8-Flash-Next-MTPLX-Optimized-Speed` |
| API | `http://127.0.0.1:8000/v1` |
| Model id | `mtplx-flash-next-optimized-speed` (managed Hermes alias may be `mtplx-local`) |
| Note | Stop other large residents first; cold rollback `serve_abliterated.sh` |

**Pros:** keeps data gen offline; user-preferred substitute for GPT-4o.  
**Cons:** quality vs GPT-4o **UNVERIFIED**; vision capability for evidence-grounded chain writing needs a small bake-off (`flash-next-vision-smoke` exists under mtplx qualification outputs).  
**Does not replace SFT weights training** — it only helps *build* supervision text/chains.

### C — Attempt stock LlamaFactory on MPS (not recommended)

Port `environment_sft.yaml` / DeepSpeed off — high chance of dead ends (CUDA ops, FA2, DS). Time sink; skip unless A fails and we need a specific HF Trainer path.

### D — Local GRPO / RL (defer)

Needs rollout engine + multi-turn tool agent + reward. On Mac this is a **separate research project** (no vLLM CUDA). Do **not** put on L1 critical path.

---

## 3. Recommended sequence

1. **R3** (this doc) — accept options.  
2. **L1** — Option A: local LoRA SFT smoke on a Q-Tool subset.  
3. Optional **D2** — Option B bake-off: Flash Next vs a few gold Q-Tool chains (format + tool tags + score sanity), *before* regenerating large data.  
4. Only later: scale LoRA data / steps; still no paper GRPO locally without a new design.

Keep paper CUDA SFT/GRPO as a **documented reference**, not the execution plan, while “no cloud” stands.

---

## 4. GPT-4o → Flash Next (user aside, captured)

Paper §3.2 uses GPT-4o under human templates to synthesize multimodal reasoning chains. User direction: use **local Qwen 3.8 Flash Next** (`~/mtplx_setup`) instead for that role.

When we open a data-gen card, acceptance should include:
- serve via `serve_flash_next.sh` / `mtplx_local.py switch flash-next`
- same tool-library + template constraints as paper
- verification rubric (tool appropriateness / evidence grounding / logical consistency) — possibly lighter than paper’s full pipeline for smoke
- explicit label: **synthetic-local**, not GPT-4o

Existing released Q-Tool on disk remains valid for L1; Flash Next is for *future* regen/augmentation.

---

## 5. Env hygiene

| Env | Use |
| --- | --- |
| `~/Projects/IQA-T1/.venv` | MPS **inference** only (`./run_mps.sh`) |
| New train venv (e.g. `.venv-train-mlx`) | L1 LoRA — do not mix with infer pins |
| `~/mtplx_setup` + `~/.mtplx/venv-2.11.2` | Flash Next **serve** for data gen — not the SFT trainer |

---

## 6. Decision ask (for PM / user)

After R3: open **L1** as “本機 LoRA SFT smoke（非 paper-faithful）” with Option A defaults?  
Optionally schedule **D2** Flash Next chain-quality bake-off in parallel.

**Out of scope until reopened:** cloud T1, stock GRPO, pushing train artifacts to `zibuyu-02` upstream.
