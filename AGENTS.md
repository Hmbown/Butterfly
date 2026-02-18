# AGENTS

This file defines execution protocol for all future coding/research turns in this repo.

## Mandatory Sub-Agent Policy

- For every non-trivial task, spawn at least one sub-agent before editing code.
- Minimum pattern:
  - `explorer` for code/architecture reconnaissance.
  - `worker` for bounded execution tasks (benchmarks, sweeps, scripted validation).
- Only skip sub-agents for trivial single-command tasks.
- If sub-agent output conflicts, run a tie-break experiment and record both hypotheses.

## No-Inference Planning Mode

- Trigger this mode when the user asks for planning/analysis without inference.
- Do not run model inference, benchmark, or finetuning commands in this mode.
- Allowed actions: inspect logs/code, synthesize failures, draft run queues, prepare PRERUN entries, and author runbook/prompt docs.
- Plans must be evidence-backed (`file:line`) and must separate facts from assumptions.
- If experiments are queued, prepare exact commands and stop-gate criteria before any run.

## Bell Labs Notebook Policy

- Treat each benchmark/ablation as an experiment with a written record.
- Record experiments in two places:
  - Human-readable: `notes/LAB_NOTEBOOK.md`
  - Machine-readable: `notes/experiments.ndjson`
- No benchmark run without a prior hypothesis entry.
- No code change lands without a post-run result entry.

## Required Experiment Entry Fields

Every experiment must include:

- `id`: stable run id (timestamp + short slug)
- `question`: what we are trying to answer
- `hypothesis`: expected outcome and why
- `change_set`: code/config changes applied
- `command`: exact command executed
- `controls`: what is held fixed
- `metrics`: key outputs (throughput, latency, peak memory, quality metric if any)
- `decision`: keep / revert / follow-up
- `next_action`: immediate next step

## Comparison Discipline

- Compare new runs against a named baseline run path.
- Explicitly state:
  - absolute metric
  - delta vs baseline
  - percentage delta vs baseline
- For memory, always include sign convention:
  - reduction % = `100 * (1 - wayfinder/dense)`

## Retrocausal Safety Rule

- Retrocausal/backfill mechanisms must default to off for inference.
- Any retro feature must expose explicit toggles in config and benchmark CLI.
- Causality tests must pass with retro disabled.

## Current Strategic Sequence

1. Reproduce and optimize on the original non-Qwen path first.
2. Lock best schedule/config there (including retro experiments).
3. Port only proven ideas to Qwen integration.
4. Re-benchmark Qwen and update README with measured (not aspirational) numbers.

## Active Program Prompt

- Current long-running handoff prompt for the HCSA/Wayfinder legitimacy program:
  - `notes/prompting/HCSA_LONGRUN_PROMPT.md`
- Use that prompt as the default starting contract for next-agent execution on this campaign.
