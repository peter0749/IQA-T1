# IQA-T1 — Checkpoint confirmations (2026-09-07)

Use this list to re-verify after a pause. Board mirrors these as **C-** cards.

E1 protocol v1 has been executed. C7–C9 are pilots. C10 G0 PASS. C11 D16 mechanism. C12 H32 quality result is inconclusive; do not treat it as evidence benefit or as “no use of evidence.”

## Must confirm before claiming path still green

- [ ] **C1 MPS**: `./run_mps.sh` still scores a sample image (expect ~4.x class result; exact score may drift).
- [ ] **C2 L1 smoke**: `scratch/l1_lora_out/SUMMARY.json` present; adapter under `scratch/l1_lora_out/`.
- [ ] **C3 Synth gate**: validator rejects missing `<think>` / tool>4 / bad score; see T Done.
- [ ] **C4 Flash Next effort**: with session-bank cleared, N=8 gate ≥0.75 on **low** and **medium** (see `scratch/D2_P2b_RETEST.md`).
- [ ] **C5 No 11k**: no full regen job started; any new synth is small-N + gate.
- [ ] **C6 Assets**: Q-Tool + Qwen3-VL-4B base paths still valid on Mac (D1).
- [ ] **C7 L3 prompt validation**: `scratch/L3_PROMPT_VALIDATION.md` and `scratch/l3_prompt_validation/run1/` are present; retain the result that the tested explicit tool contract was counterproductive (0/8 complete tool contract).
- [ ] **C8 L4 alignment audit**: `scratch/L4_PROMPT_ALIGNMENT_AUDIT.md` and `scratch/l4_prompt_alignment_audit/run2/audit.json` are present; retain the original short user prompt and do not add tool-library or output-format instructions to it.
- [ ] **C9 L5 tool-first SFT**: `scratch/L5_TOOL_FIRST_SFT_VALIDATION.md` and `scratch/l5_tool_first_sft/run1/` are present; retain the distinction between 8/8 tool execution and 0/8 exact held-out evidence-list match.
- [ ] **C10 E1 G0**: `docs/E1_G0_STATUS.md` and `scratch/e1/g0/run2/decision.json` classification PASS; 12 rollouts, unique adapters, no H32 access. Not a quality result.
- [ ] **C11 E1 D**: `docs/E1_D_STATUS.md` and `scratch/e1/d/run1/analysis.json`; D16 only. L5 first tool invariant; L5 first-evidence score change CI excludes 0; D16 error-diff CI includes 0.
- [ ] **C12 E1 H**: `docs/E1_H_STATUS.md` and `scratch/e1/h/run1/analysis.json` classification **inconclusive**. L5 primary −0.044 [−0.122, 0.025]; did not meet the predeclared 0.05 / CI-upper<0 rule.
- [ ] **C13 Paper D16 audit**: `docs/PAPER_D16_AUDIT.md`; official IQA-T1 on D16 first-tool 14/16 Histogram, 8 sequences, MAE 0.159 descriptive. Not PLCC.

## Do not resume without user go

- Cloud GPU rental
- Blind full dataset regeneration
- Claiming paper PLCC/SRCC from local LoRA alone
