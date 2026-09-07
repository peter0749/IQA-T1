# IQA-T1 Study — Phase handoff (2026-09-07)

Quota wrap. Local-first path only; no cloud rental; no blind 11k regen.

## Board

GitHub Project #4 (owner `peter0749`): IQA-T1 Study.

## What shipped (Done)

| Card | Outcome (one line) | Where |
|------|--------------------|--------|
| R1 | Codex feasibility verified; don't run paper stack as-is on M2 | `CODEX_REPORT.md` (Mac scratch) |
| I1–I3 | MPS inference works; launcher + venv | `./run_mps.sh`, `MPS_INFER.md`, `requirements-mps.txt` |
| R2 | Training premise (CUDA SFT→GRPO + Q-Tool; M2 = infer/debug) | `docs/TRAINING_PREMISE.md` |
| D1 | Q-Tool + Qwen3-VL-4B base downloaded/verified | Mac `~/Projects/IQA-T1/` + scratch notes |
| R3 | Local train options | `docs/LOCAL_TRAIN_OPTIONS.md` |
| L1 | Local LoRA/QLoRA SFT smoke (not paper-faithful) | Mac `scratch/l1_lora_out/SUMMARY.json` |
| D2 | Flash Next bake-off N=8 mean 2.5/4; draft-only, no blind full regen | Mac `scratch/D2_BAKEOFF.md` |
| T | Schema gate required before any train-json ingest | validator on Mac; board Done |
| P1 | Tight prompt alone → gate 0/8 (answer_start only) | Mac `scratch/D2_P1_RETEST.md` |
| P2 / P2b | After session-bank clear + real `reasoning_effort`: low 1.00 / auto 0.875 / medium 1.00 | Mac `scratch/D2_P2_RETEST.md`, `scratch/D2_P2b_RETEST.md` |
| L3 | Prompt-controlled local LoRA validation: baseline tool contract 3/8; explicit tool contract 0/8 and format 0/8, so do not use that prompt as a fix | Mac `scratch/L3_PROMPT_VALIDATION.md`, `scratch/l3_prompt_validation/run1/` |

## Standing rules (locked)

1. **No blind 11k regen** until gate + quality bar hold at scale.
2. **Schema gate mandatory** before rows enter train json: `<think>` / `<answer_start>` / unique tools ≤4 / 15-tool whitelist / score ∈ [1,5]. Failures drop.
3. **Default Flash Next synth effort = `low` or `medium`** (auto ok but slightly weaker; avoid `high` until 507 understood).
4. **No cloud** for this study phase unless user explicitly reopens cost.
5. **L1 ≠ paper**: local LoRA smoke is not SFT/GRPO reproduction; chat_template may flatten assistant images.

## Backlog (not opened)

- **R4**: Larger VLM + harness ≠ paper PLCC/SRCC / stable tool policy without Q-Tool-level SFT (+ ideally GRPO). Research note only unless prioritized.

## Mac layout (execution host)

- Host: `kuang-yujeng@172.26.12.82` (KuangYuacStudio, M2 Ultra)
- Root: `~/Projects/IQA-T1/`
- Repo clone / fork work: under that tree + GitHub `peter0749/IQA-T1`
- Flash Next serve: `~/mtplx_setup/serve_flash_next.sh` on `:8000` (stop when idle)
- Scratch reports: `~/Projects/IQA-T1/scratch/`
  - Bake-off writeups: `D2_BAKEOFF.md`, `D2_P1_RETEST.md`, `D2_P2_RETEST.md`, `D2_P2b_RETEST.md`
  - L1: `l1_lora_out/SUMMARY.json`
  - L3: `L3_PROMPT_VALIDATION.md`, `l3_prompt_validation/run1/{manifest,baseline,contract,comparison}.json`
  - Gate/run artifacts (SSE confirmed): `synth_gate/gate_n8_p2_*.json`, `d2_bakeoff/results_p2.json`, `request_proof_p2_{low,auto,medium}.json`, `run_p2_bakeoff.py`

## How to resume next session

1. Read this file + `docs/CHECKPOINT.md`.
2. Confirm board #4: L3 is Done; no open This-quarter cards; R4 still Backlog.
3. Optional next (needs explicit user priority):
   - Small gated synth batch (not 11k) with low/medium + gate
   - Longer local LoRA / apply adapter to MPS infer
   - R4 one-pager only
   - Do not treat L3's explicit 1–4-tool contract as an inference remedy; it caused immediate score-only outputs on all 8 held-out images.
4. STOs: SSE = serve/wiring/scripts; MLE = rubrics/accept; PM = board/Notion/handoff

## Displace (what we deliberately did not do)

- Paper-faithful multi-CUDA SFT/GRPO
- Full Q-Tool / GPT-4o-scale regeneration with Flash Next
- Opening R4 implementation
