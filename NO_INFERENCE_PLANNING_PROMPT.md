# No-Inference Planning Prompt

Use this prompt when you want planning and repo analysis while keeping inference fully off.

## Copy/Paste Prompt

```text
Work in /Volumes/VIXinSSD/wayfinder.

Hard constraint: do not run inference, benchmark, or finetuning commands in this session.
You may inspect files/logs and edit planning docs only.

Tasks:
1. Analyze recent failures and blockers from notes/LAB_NOTEBOOK.md and notes/experiments.ndjson.
2. Produce a prioritized next-run queue (max 6 items) aligned to notes/GLM_POST_REBOOT_FULL_BENCH_PROTOCOL.md.
3. For each queued item, provide:
   - experiment id
   - question
   - hypothesis
   - exact command to run later
   - controls
   - metrics to capture
   - stop-gate criteria
4. Append PRERUN entries for the selected queue to:
   - notes/LAB_NOTEBOOK.md
   - notes/experiments.ndjson
5. If asked, update AGENTS.md with planning-mode rules only (no benchmark execution).

Output requirements:
- Findings first, ordered by severity.
- Cite evidence with file:line references.
- Distinguish facts from assumptions.
- Do not claim measured performance changes unless already present in artifacts.
```

## Session Checklist

- Confirm no-inference constraint at start.
- Confirm no model-running commands were executed.
- Ensure every queued experiment has a PRERUN entry in both notebook files.
- Keep retro/backfill inference off by default in all prepared commands.
