# agy-orchestrator

Windows-native PowerShell 7 multi-model router, live stream monitor, and autonomous repository-wide GitHub backlog execution system over the official `agy -p --model ... --output-format stream-json` CLI.

Never modifies Antigravity or Maestro internals. Maestro plans (`docs/maestro/plans/*.md`) and all open GitHub issues are discovered and referenced cleanly by relative forward-slash paths.

## Components (`tools/agy-orchestrator/`)

- **`policy.json`** — Model mapping, quota-aware `routeFallbacks`, task classifier rules, lifecycle phase definitions, `backlog` discovery & external action guards, protected path patterns, weekly (25%) and 5-hour (10%) quota reserves, and per-group token/call budgets.
- **`AgyOrchestrator.ps1`** — Core routing with quota-aware fallback, paginated/incremental GitHub backlog sync (`gh api repos/{owner}/{repo}/issues?state=all&per_page=100&page=N&since=...`), dependency & priority graph solver, code reconciliation (`ISSUE_STATUS: ALREADY_COMPLETED` / `ISSUE_STATUS: BLOCKED`), UTF-8/path-clean handoff builder, real-time NDJSON `stream-json` parser, workspace file-diff guard, conversation resumption, and quota accounting.
- **`Invoke-Agy.ps1`** — CLI entrypoint supporting `-AllOpenIssues` (autonomous repository-wide backlog execution), `-MaxIssues <N>`, single `-Issue <N>`, `-Lifecycle` execution, `-Live` console streaming, `-WorkerTimeoutMinutes` (default `20`), and `-ResumeLatest` / `-ResumeConversationId`.
- **`Watch-Agy.ps1`** — Terminal dashboard and backlog queue controller (`-Follow` for live updates, `-Json` for structured status, `-Pause`, `-Resume`, `-Stop`, `-Cancel`, and `-RetryIssue <N>`).
- **`Quota.ps1`** — Weekly & 5-hour quota viewer, manual setter (`-Set <group> -Percent <w> -FiveHourPercent <5h>`), and zero-token CLI sync (`-SyncFromCli` via `agy -p "/usage" --output-format json`).

## Core Commands

### 1. Autonomous Repository-Wide Backlog Execution (`-AllOpenIssues`)

```powershell
# Repository-wide dry-run preview (0 tokens; discovers all open issues, solves dependencies/priorities, and previews model routing):
pwsh -NoProfile -File .\tools\agy-orchestrator\Invoke-Agy.ps1 -AllOpenIssues -Lifecycle -AllowWrites -YoloWorker -Live -DryRun

# Launch autonomous execution across all actionable open issues in the repository:
pwsh -NoProfile -File .\tools\agy-orchestrator\Invoke-Agy.ps1 -AllOpenIssues -Lifecycle -AllowWrites -YoloWorker -Live

# Process only the next N actionable issues (e.g. 1 issue) from the backlog queue:
pwsh -NoProfile -File .\tools\agy-orchestrator\Invoke-Agy.ps1 -AllOpenIssues -MaxIssues 1 -Lifecycle -AllowWrites -YoloWorker -Live
```

### 2. Live Dashboard & Backlog Controls (`Watch-Agy.ps1`)

```powershell
# Live-updating terminal dashboard in a second PowerShell window (shows backlog queue, active worker, tool calls, and quota):
pwsh -NoProfile -File .\tools\agy-orchestrator\Watch-Agy.ps1 -Follow

# Pause backlog runner after the current issue finishes:
pwsh -NoProfile -File .\tools\agy-orchestrator\Watch-Agy.ps1 -Pause

# Clear pause/stop/cancel signals before resuming:
pwsh -NoProfile -File .\tools\agy-orchestrator\Watch-Agy.ps1 -Resume

# Gracefully stop the backlog runner after the current phase/issue:
pwsh -NoProfile -File .\tools\agy-orchestrator\Watch-Agy.ps1 -Stop

# Cancel the active worker subprocess immediately:
pwsh -NoProfile -File .\tools\agy-orchestrator\Watch-Agy.ps1 -Cancel

# Reset a FAILED, INCOMPLETE, or BLOCKED issue back to PENDING for retry:
pwsh -NoProfile -File .\tools\agy-orchestrator\Watch-Agy.ps1 -RetryIssue 33
```

### 3. Single-Issue & Resumption Commands

```powershell
# Execute the full 6-phase lifecycle for a single issue:
pwsh -NoProfile -File .\tools\agy-orchestrator\Invoke-Agy.ps1 -Issue 33 -Lifecycle -AllowWrites -YoloWorker -Live -PlanFile 'docs/maestro/plans/*impl-plan.md'

# Resume a PARTIAL issue (e.g. Issue #37 paused by internal per-run token cap) — runs ONLY remaining phases (test, review, verify):
pwsh -NoProfile -File .\tools\agy-orchestrator\Invoke-Agy.ps1 -Issue 37 -Lifecycle -AllowWrites -YoloWorker -Live -PlanFile 'docs/maestro/plans/*impl-plan.md'

# Resume the latest timed-out (INCOMPLETE) or interrupted conversation without repeating prior steps:
pwsh -NoProfile -File .\tools\agy-orchestrator\Invoke-Agy.ps1 -ResumeLatest -YoloWorker -Live -PlanFile 'docs/maestro/plans/*impl-plan.md'
```

### 4. Local Recovery & Idempotent GitHub Delivery (`-RecoverLocalState` / `-SyncGitHub`)

```powershell
# Reconcile local completed (#33-#36) and partially completed (#37) runs into backlog-queue.json without invoking any model workers:
pwsh -NoProfile -File .\tools\agy-orchestrator\Invoke-Agy.ps1 -RecoverLocalState

# Dry-run GitHub delivery synchronization (preview isolated branches, commits, Draft PRs, comments, and labels):
pwsh -NoProfile -File .\tools\agy-orchestrator\Invoke-Agy.ps1 -SyncGitHub -EnableGitHubComments -EnableGitHubLabels -EnableBranchCommit -EnablePush -EnableDraftPr -DryRun

# Execute idempotent GitHub progress comments, labels, isolated branch commits, pushes, and Draft PRs (never auto-merges or auto-closes):
pwsh -NoProfile -File .\tools\agy-orchestrator\Invoke-Agy.ps1 -SyncGitHub -EnableGitHubComments -EnableGitHubLabels -EnableBranchCommit -EnablePush -EnableDraftPr
```
