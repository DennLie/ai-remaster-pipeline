#requires -Version 7
<#
.SYNOPSIS  Route tasks or full GitHub issue lifecycles to model-specific `agy -p` workers with live monitoring & quota guards.
.EXAMPLE   ./Invoke-Agy.ps1 -Issue 33 -PlanFile 'docs/maestro/plans/*impl-plan.md' -YoloWorker -Live -DryRun
.EXAMPLE   ./Invoke-Agy.ps1 -Issue 33 -PlanFile 'docs/maestro/plans/*impl-plan.md' -YoloWorker -Live
.EXAMPLE   ./Invoke-Agy.ps1 -Issue 33 -Lifecycle -AllowWrites -YoloWorker -Live -PlanFile 'docs/maestro/plans/*impl-plan.md'
.EXAMPLE   ./Invoke-Agy.ps1 -Issue 33 -ResumeLatest -YoloWorker -Live
Exit codes: 0 ok, 1 failed/incomplete/cancelled/budget, 2 blocked (authorization, quota reserve, limits).
#>
[CmdletBinding()]
param(
    [string[]]$Task,
    [int[]]$Issue,
    [switch]$AllOpenIssues,              # discover & execute all actionable open GitHub issues autonomously
    [int]$MaxIssues = 0,                 # optional limit when running -AllOpenIssues (0 = all actionable issues)
    [string[]]$PlanFile = @(),           # paths or wildcards, referenced cleanly in the handoff
    [string[]]$ContextFile = @(),
    [string]$Type,                       # explicit task type, overrides classifier
    [switch]$Lifecycle,                  # run full issue lifecycle: investigation -> implement -> test -> review -> corrections -> verify
    [string[]]$Phases = @(),             # optional subset of lifecycle phases
    [string]$HandoffFrom,                # 'latest' or a summary.json / result-N.json from an earlier run
    [string]$ResumeConversationId,       # resume an interrupted/timed-out worker conversation ID via --conversation
    [switch]$ResumeLatest,               # automatically find & resume the last INCOMPLETE/FAILED worker conversation
    [int]$WorkerTimeoutMinutes = 0,      # defaults to policy.budgets.workerTimeoutMinutes (20 minutes)
    [ValidateSet('json', 'stream-json')][string]$OutputFormat = 'stream-json',
    [switch]$Live,                       # stream real-time worker tool calls, file ops, errors & token usage to console
    [string]$PolicyPath = (Join-Path $PSScriptRoot 'policy.json'),
    [string]$AgyCommand = 'agy',
    [string]$GhCommand = 'gh',
    [string]$GitCommand = 'git',
    [string]$RunRoot = (Join-Path (Get-Location).Path '.cache/agy-orch'),
    [string]$StatePath,
    [string]$QueuePath,
    [switch]$RecoverLocalState,          # reconcile local runs (#33-#37) into backlog-queue.json without running workers
    [switch]$SyncGitHub,                 # run idempotent GitHub delivery sync (comments, labels, branches, Draft PRs)
    [switch]$EnableGitHubComments,       # enable idempotent progress/completion comments on GitHub issues
    [switch]$EnableGitHubLabels,         # enable idempotent status label updates on GitHub issues
    [switch]$EnableBranchCommit,         # enable isolated per-issue branch & commit creation (with strict ownership guard)
    [switch]$EnablePush,                 # enable pushing isolated issue branches to origin
    [switch]$EnableDraftPr,              # enable creating idempotent Draft PRs for pushed issue branches
    [switch]$Parallel,
    [switch]$DryRun,
    [switch]$AllowWrites,                # required for write phases (implement/refactor/test/corrections) to use --mode accept-edits
    [switch]$RequireWriteAuthorization,  # block write-type tasks at plan time if -AllowWrites was not passed
    [switch]$YoloWorker,                 # explicit opt-in: forwards --dangerously-skip-permissions to workers
    [switch]$AuthorizeCostly,            # explicit approval: Opus
    [switch]$AuthorizeArgon,             # explicit approval: Argon
    [switch]$AuthorizeDestructive,
    [switch]$OverrideReserve,            # dispatch even when quota is below the weekly or 5-hour reserve
    [switch]$DeterministicVerify         # prefer deterministic local test runner & file ownership check for verify phase (0 LLM tokens)
)
. (Join-Path $PSScriptRoot 'AgyOrchestrator.ps1')
$policy = Get-AgyPolicy $PolicyPath
if (-not $StatePath) { $StatePath = Join-Path $RunRoot 'state.json' }
if (-not $QueuePath) { $QueuePath = Join-Path $RunRoot 'backlog-queue.json' }
$auth = @{ AuthorizeCostly = [bool]$AuthorizeCostly; AuthorizeArgon = [bool]$AuthorizeArgon; AuthorizeDestructive = [bool]$AuthorizeDestructive }

if ($RecoverLocalState -and -not $SyncGitHub -and -not $AllOpenIssues -and -not $Issue -and -not $Task) {
    $rep = Repair-AgyBacklogQueue -QueuePath $QueuePath -RunRoot $RunRoot -Policy $policy -DryRun:$DryRun
    $rep | ConvertTo-Json -Depth 10
    exit 0
}

if ($SyncGitHub) {
    $syncOut = Sync-AgyGitHubDelivery -QueuePath $QueuePath -IssueNumbers $Issue -Policy $policy `
        -GhCommand $GhCommand -GitCommand $GitCommand `
        -EnableComments:$EnableGitHubComments -EnableLabels:$EnableGitHubLabels `
        -EnableBranchCommit:$EnableBranchCommit -EnablePush:$EnablePush -EnableDraftPr:$EnableDraftPr `
        -DryRun:$DryRun
    $syncOut | ConvertTo-Json -Depth 10
    exit 0
}

# Auto-discover Maestro implementation plans if none were explicitly specified on a backlog run
if ($AllOpenIssues -and $PlanFile.Count -eq 0 -and (Test-Path 'docs/maestro/plans')) {
    $autoPlans = @(Get-ChildItem -Path 'docs/maestro/plans/*impl-plan.md' -File -ErrorAction SilentlyContinue)
    if ($autoPlans.Count -gt 0) {
        $PlanFile = @('docs/maestro/plans/*impl-plan.md')
    }
}
$ctx = @(Resolve-AgyPlanFiles (@($PlanFile) + @($ContextFile)))

if ($AllOpenIssues) {
    if ($Lifecycle -and -not $AllowWrites -and -not $DryRun) {
        Write-Error "Autonomous -AllOpenIssues -Lifecycle requires explicit -AllowWrites authorization."; exit 2
    }
    $bOut = Invoke-AgyBacklogRunner -Policy $policy -AgyCommand $AgyCommand -GhCommand $GhCommand `
        -RunRoot $RunRoot -StatePath $StatePath -QueuePath $QueuePath -Auth $auth `
        -AllowWrites ([bool]$AllowWrites) -Lifecycle:$Lifecycle -Phases $Phases `
        -YoloWorker:$YoloWorker -Live:$Live -DryRun:$DryRun -OverrideReserve:$OverrideReserve `
        -WorkerTimeoutMinutes $WorkerTimeoutMinutes -MaxIssuesToProcess $MaxIssues `
        -OutputFormat $OutputFormat -ContextFiles $ctx -DeterministicVerify:$DeterministicVerify
    $bOut | ConvertTo-Json -Depth 10
    switch ($bOut.status) { 'OK' { exit 0 } 'BLOCKED' { exit 2 } default { exit 1 } }
}

$prior = ''
$priorMods = @()
if ($ResumeLatest -and -not $ResumeConversationId) {
    $resumable = Find-AgyResumableWorker -RunRoot $RunRoot
    $ResumeConversationId = [string]$resumable.conversationId
    if (-not $Type -and $resumable.PSObject.Properties['taskType'] -and $resumable.taskType) {
        $Type = [string]$resumable.taskType
    }
    if ($resumable.PSObject.Properties['modifiedFiles'] -and $resumable.modifiedFiles) {
        $priorMods = @($resumable.modifiedFiles)
    }
    if ($resumable.PSObject.Properties['response'] -and $resumable.response) {
        $prior = [string]$resumable.response
    }
}

if ($HandoffFrom) {
    $hf = if ($HandoffFrom -eq 'latest') { Join-Path $RunRoot 'latest.json' } else { $HandoffFrom }
    if (-not (Test-Path -LiteralPath $hf)) { Write-Error "Handoff source not found: $hf"; exit 2 }
    $j = Get-Content -LiteralPath $hf -Raw -Encoding utf8 | ConvertFrom-Json
    $prior = if ($j.PSObject.Properties['results'] -and @($j.results).Count) { [string](@($j.results)[-1].response) } elseif ($j.PSObject.Properties['response']) { [string]$j.response } else { '' }
    if ($j.PSObject.Properties['results'] -and @($j.results).Count) {
        $priorMods = @(@($j.results) | ForEach-Object { if ($_.PSObject.Properties['modifiedFiles'] -and $_.modifiedFiles) { @($_.modifiedFiles) } } | Select-Object -Unique)
    }
}

$tasks = @()
$enforceWriteAuth = [bool]($RequireWriteAuthorization -or $Lifecycle -or ($Type -and ($policy.writeTypes -contains $Type) -and $Issue))
foreach ($n in $Issue) {
    $issueText = Get-AgyIssueText -Number $n -GhCommand $GhCommand -MaxChars ([int]$policy.budgets.issueMaxChars)
    if ($Lifecycle) {
        $effectivePhases = $Phases
        if ((-not $effectivePhases -or $effectivePhases.Count -eq 0) -and (Test-Path -LiteralPath $QueuePath)) {
            $qCheck = Get-AgyBacklogQueue -QueuePath $QueuePath
            $nk = [string]$n
            if ($qCheck.issues.ContainsKey($nk) -and $qCheck.issues[$nk].status -eq 'PARTIAL') {
                $doneP = @($qCheck.issues[$nk].completedPhases | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) })
                if ($doneP.Count -gt 0) {
                    $effectivePhases = Get-AgyRemainingLifecyclePhases -CompletedPhases $doneP -Status 'PARTIAL' -Policy $policy
                    if (-not $prior) {
                        $cp = Get-AgyIssueLastWorkerSummary -RunRoot $RunRoot -IssueEntry $qCheck.issues[$nk]
                        $prior = [string]$cp.priorSummary
                        if ($cp.modifiedFiles.Count -gt 0) { $priorMods = @($priorMods + @($cp.modifiedFiles) | Select-Object -Unique) }
                    }
                }
            }
        }
        $built = @(New-AgyIssueLifecycleTasks -IssueNumber $n -IssueText $issueText -Policy $policy -AllowWrites ([bool]$AllowWrites) -Phases $effectivePhases)
        if ($priorMods.Count -gt 0 -and $built.Count -gt 0) { $built[0].PriorModifiedFiles = $priorMods }
        $tasks += $built
    } else {
        $t = if ($Type) { $Type } else { 'investigation' }
        $tasks += @{
            Text                 = $issueText
            Type                 = $t
            AllowWrites          = [bool]$AllowWrites
            ResumeConversationId = $ResumeConversationId
            PriorModifiedFiles   = $priorMods
        }
    }
}
foreach ($t in $Task) {
    $tasks += @{
        Text                 = $t
        Type                 = $Type
        AllowWrites          = [bool]$AllowWrites
        ResumeConversationId = $(if ($tasks.Count -eq 0) { $ResumeConversationId } else { '' })
        PriorModifiedFiles   = $priorMods
    }
}
if (-not $tasks.Count) { Write-Error 'Provide -Task, -Issue, or -AllOpenIssues.'; exit 2 }

$s = Invoke-AgyOrchestration -Tasks $tasks -Policy $policy -AgyCommand $AgyCommand -Auth $auth -RunRoot $RunRoot -StatePath $StatePath `
    -AllowWrites ([bool]$AllowWrites) -EnforceWriteAuth $enforceWriteAuth -Parallel:$Parallel -DryRun:$DryRun `
    -OverrideReserve:$OverrideReserve -YoloWorker:$YoloWorker -Live:$Live -WorkerTimeoutMinutes $WorkerTimeoutMinutes `
    -OutputFormat $OutputFormat -ResumeConversationId $ResumeConversationId -ContextFiles $ctx -PriorSummary $prior `
    -DeterministicVerify:$DeterministicVerify
$s | ConvertTo-Json -Depth 8
switch ($s.status) { 'OK' { exit 0 } 'BLOCKED' { exit 2 } default { exit 1 } }
