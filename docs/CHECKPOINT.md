# IQA-T1 — Checkpoint confirmations (2026-09-07)

Use this list to re-verify after a pause. Board mirrors these as **C-** cards.

## Must confirm before claiming path still green

- [ ] **C1 MPS**: `./run_mps.sh` still scores a sample image (expect ~4.x class result; exact score may drift).
- [ ] **C2 L1 smoke**: `scratch/l1_lora_out/SUMMARY.json` present; adapter under `scratch/l1_lora_out/`.
- [ ] **C3 Synth gate**: validator rejects missing `<think>` / tool>4 / bad score; see T Done.
- [ ] **C4 Flash Next effort**: with session-bank cleared, N=8 gate ≥0.75 on **low** and **medium** (see `scratch/D2_P2b_RETEST.md`).
- [ ] **C5 No 11k**: no full regen job started; any new synth is small-N + gate.
- [ ] **C6 Assets**: Q-Tool + Qwen3-VL-4B base paths still valid on Mac (D1).
- [ ] **C7 L3 prompt validation**: `scratch/L3_PROMPT_VALIDATION.md` and `scratch/l3_prompt_validation/run1/` are present; retain the result that the tested explicit tool contract was counterproductive (0/8 complete tool contract).

## Do not resume without user go

- Cloud GPU rental
- Blind full dataset regeneration
- Claiming paper PLCC/SRCC from local LoRA alone
