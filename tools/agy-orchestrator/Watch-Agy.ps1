#requires -Version 7
<#
.SYNOPSIS  Live dashboard, backlog inspector, and pause/stop/resume/retry/cancel controller for `tools/agy-orchestrator`.
.EXAMPLE   ./Watch-Agy.ps1                 # inspect backlog queue, latest run, active worker, quota, and recent events
.EXAMPLE   ./Watch-Agy.ps1 -Follow         # live-updating terminal dashboard
.EXAMPLE   ./Watch-Agy.ps1 -Pause          # pause the backlog runner after the current worker finishes
.EXAMPLE   ./Watch-Agy.ps1 -Resume         # clear pause/stop/cancel signals so backlog execution can continue
.EXAMPLE   ./Watch-Agy.ps1 -Stop           # request graceful stop of the backlog runner
.EXAMPLE   ./Watch-Agy.ps1 -Cancel         # request immediate cancellation of the active worker
.EXAMPLE   ./Watch-Agy.ps1 -RetryIssue 33  # reset FAILED/INCOMPLETE issue #33 to PENDING for retry
#>
[CmdletBinding()]
param(
    [string]$RunRoot = (Join-Path (Get-Location).Path '.cache/agy-orch'),
    [string]$RunId,
    [string]$PolicyPath = (Join-Path $PSScriptRoot 'policy.json'),
    [switch]$Follow,
    [int]$RefreshSeconds = 2,
    [int]$MaxIterations = 0,
    [switch]$Pause,
    [switch]$Resume,
    [switch]$Stop,
    [switch]$Cancel,
    [int]$RetryIssue = 0,
    [switch]$Json
)
. (Join-Path $PSScriptRoot 'AgyOrchestrator.ps1')
$policy = Get-AgyPolicy $PolicyPath
$statePath = Join-Path $RunRoot 'state.json'
$queuePath = Join-Path $RunRoot 'backlog-queue.json'
$liveStatusPath = Join-Path $RunRoot 'live-status.json'
$pausePath  = Join-Path $RunRoot 'pause.signal'
$stopPath   = Join-Path $RunRoot 'stop.signal'
$cancelPath = Join-Path $RunRoot 'cancel.signal'

if ($Pause) {
    Write-AgyUtf8File -Path $pausePath -Value ("paused at " + (Get-Date).ToString('o'))
    Write-Host "Pause signal written to $pausePath" -ForegroundColor Yellow
    exit 0
}
if ($Stop) {
    Write-AgyUtf8File -Path $stopPath -Value ("stopped at " + (Get-Date).ToString('o'))
    Write-Host "Stop signal written to $stopPath" -ForegroundColor Yellow
    exit 0
}
if ($Cancel) {
    Write-AgyUtf8File -Path $cancelPath -Value ("cancelled at " + (Get-Date).ToString('o'))
    Write-Host "Cancellation signal written to $cancelPath" -ForegroundColor Yellow
    exit 0
}
if ($Resume) {
    foreach ($sig in @($pausePath, $stopPath, $cancelPath)) {
        if (Test-Path -LiteralPath $sig) { Remove-Item -LiteralPath $sig -Force -ErrorAction SilentlyContinue }
    }
    if (Test-Path -LiteralPath $queuePath) {
        $q = Get-AgyBacklogQueue -QueuePath $queuePath
        if ($q.runnerState -in @('PAUSED', 'STOPPED', 'PAUSED_INCOMPLETE', 'PAUSED_RUN_BUDGET', 'BLOCKED_QUOTA_OR_AUTH', 'BLOCKED_PROVIDER_QUOTA')) {
            $q.runnerState = 'IDLE'
        }
        Save-AgyBacklogQueue -Queue $q -QueuePath $queuePath
    }
    Write-Host "Cleared pause/stop/cancel signals; ready to resume." -ForegroundColor Green
    exit 0
}
if ($RetryIssue -gt 0) {
    $q = Get-AgyBacklogQueue -QueuePath $queuePath
    $k = [string]$RetryIssue
    if (-not $q.issues.ContainsKey($k)) { throw "Issue #$RetryIssue not found in backlog queue at $queuePath" }
    $doneCount = @($q.issues[$k].completedPhases | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) }).Count
    $q.issues[$k].status = if ($doneCount -gt 0) { 'PARTIAL' } else { 'PENDING' }
    $q.issues[$k].blockReason = $null
    Save-AgyBacklogQueue -Queue $q -QueuePath $queuePath
    Write-Host "Issue #$RetryIssue reset to $($q.issues[$k].status) (preserving completedPhases=$((@($q.issues[$k].completedPhases)) -join ','))." -ForegroundColor Green
    exit 0
}

function Get-AgyDashboardSnapshot {
    $runDir = if ($RunId) {
        Join-Path $RunRoot $RunId
    } elseif (Test-Path -LiteralPath (Join-Path $RunRoot 'latest.json')) {
        $latest = Get-Content -LiteralPath (Join-Path $RunRoot 'latest.json') -Raw -Encoding utf8 | ConvertFrom-Json
        $cand = if ($latest.PSObject.Properties['runDir'] -and $latest.runDir) {
            if ([System.IO.Path]::IsPathRooted($latest.runDir)) { $latest.runDir } else { Join-Path (Get-Location).Path $latest.runDir }
        } else { $null }
        if ($cand -and (Test-Path -LiteralPath $cand)) {
            $cand
        } elseif ($latest.PSObject.Properties['runId'] -and (Test-Path -LiteralPath (Join-Path $RunRoot $latest.runId))) {
            Join-Path $RunRoot $latest.runId
        } else { $cand }
    } else {
        $lastDir = Get-ChildItem -LiteralPath $RunRoot -Directory -ErrorAction SilentlyContinue | Sort-Object Name | Select-Object -Last 1
        if ($lastDir) { $lastDir.FullName } else { $null }
    }

    $live = if (Test-Path -LiteralPath $liveStatusPath) {
        try { Get-Content -LiteralPath $liveStatusPath -Raw -Encoding utf8 | ConvertFrom-Json } catch { $null }
    } else { $null }

    $summary = if ($runDir -and (Test-Path -LiteralPath (Join-Path $runDir 'summary.json'))) {
        try { Get-Content -LiteralPath (Join-Path $runDir 'summary.json') -Raw -Encoding utf8 | ConvertFrom-Json } catch { $null }
    } elseif (Test-Path -LiteralPath (Join-Path $RunRoot 'latest.json')) {
        try { Get-Content -LiteralPath (Join-Path $RunRoot 'latest.json') -Raw -Encoding utf8 | ConvertFrom-Json } catch { $null }
    } else { $null }

    $recentEvents = [System.Collections.Generic.List[object]]::new()
    if ($runDir -and (Test-Path -LiteralPath $runDir)) {
        $evFiles = @(Get-ChildItem -LiteralPath $runDir -Filter 'worker-*.events.jsonl' -File -ErrorAction SilentlyContinue | Sort-Object Name)
        foreach ($ef in $evFiles) {
            $lines = @(Get-Content -LiteralPath $ef.FullName -Tail 8 -Encoding utf8 -ErrorAction SilentlyContinue)
            foreach ($ln in $lines) {
                $ev = Format-AgyStreamEvent -Line $ln
                if ($ev) {
                    $recentEvents.Add([pscustomobject]@{
                        file    = $ef.Name
                        kind    = $ev.kind
                        message = $ev.message
                    })
                }
            }
        }
    }

    $backlogStats = $null
    $backlogItems = @()
    if (Test-Path -LiteralPath $queuePath) {
        $q = Get-AgyBacklogQueue -QueuePath $queuePath
        $backlogStats = Get-AgyBacklogStats -Queue $q
        $backlogItems = @($q.issues.Values | Where-Object { $_.state -eq 'open' } | Sort-Object { [int]$_.priority }, { if ($null -ne $_.phaseNumber) { [int]$_.phaseNumber } else { 999 } }, { [int]$_.number } | Select-Object -First 15 | ForEach-Object {
            $doneP = @($_.completedPhases | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) })
            $remP  = if ($_.ContainsKey('remainingPhases') -and $_.remainingPhases) {
                @($_.remainingPhases)
            } else {
                Get-AgyRemainingLifecyclePhases -CompletedPhases $doneP -Status ([string]$_.status) -Policy $policy
            }
            $own = Test-AgyIssueFileOwnership -IssueNumber ([int]$_.number) -IssueEntry $_ -Policy $policy
            [pscustomobject]@{
                number                 = [int]$_.number
                title                  = [string]$_.title
                status                 = [string]$_.status
                verificationState      = [string](Get-AgyPropValue $_ 'verificationState' $_.status)
                implementationComplete = [bool](Get-AgyPropValue $_ 'implementationComplete' $false)
                verificationComplete   = [bool](Get-AgyPropValue $_ 'verificationComplete' $false)
                humanReviewPending     = [bool](Get-AgyPropValue $_ 'humanReviewPending' $false)
                priority               = [int]$_.priority
                phase                  = $_.phaseNumber
                currentPhase           = $_.currentPhase
                completedPhases        = $doneP
                remainingPhases        = $remP
                dependencies           = @($_.dependencies)
                blockReason            = $_.blockReason
                tokensUsed             = [int]$_.tokensUsed
                delivery               = $(if ($_.ContainsKey('delivery') -and $_.delivery) { $_.delivery } else { $null })
                ownership              = $own
            }
        })
    }

    $quota = @(Get-AgyQuotaReport -Policy $policy -StatePath $statePath)
    $budgetKind = if ($summary -and $summary.PSObject.Properties['budgetKind'] -and $summary.budgetKind) {
        [string]$summary.budgetKind
    } elseif ($summary -and $summary.status -eq 'BUDGET_EXHAUSTED') {
        $notesJoined = ($summary.notes -join ' ')
        if ($notesJoined -match 'per-run token budget|token budget reached') { 'INTERNAL_RUN_BUDGET' } else { 'PROVIDER_QUOTA_RESERVE' }
    } else { $null }

    [pscustomobject]@{
        timestamp    = (Get-Date).ToString('o')
        runDir       = $(if ($runDir) { Format-AgyCleanPath -Path $runDir } else { $null })
        backlog      = $backlogStats
        backlogQueue = $backlogItems
        liveWorker   = $live
        summary      = $summary
        budgetKind   = $budgetKind
        recentEvents = @($recentEvents | Select-Object -Last 10)
        quota        = $quota
    }
}

$iter = 0
do {
    $iter++
    $snap = Get-AgyDashboardSnapshot
    if ($Json) {
        $snap | ConvertTo-Json -Depth 8
        break
    }
    if ($Follow -and $iter -gt 1) { Clear-Host }
    Write-Host "=== Antigravity Orchestrator Dashboard ($($snap.timestamp)) ===" -ForegroundColor Cyan
    if ($snap.backlog) {
        $b = $snap.backlog
        $partialCnt = if ($b.PSObject.Properties['partial']) { $b.partial } else { 0 }
        $implCnt    = if ($b.PSObject.Properties['implementationComplete']) { $b.implementationComplete } else { 0 }
        $humanCnt   = if ($b.PSObject.Properties['humanReviewPending']) { $b.humanReviewPending } else { 0 }
        Write-Host ("Backlog Queue : state={0} activeIssue={1} | open={2} (pending={3} partial={4} running={5} completed={6} implOnly={7} humanReview={8} blocked={9} failed={10} incomplete={11}) | lastSync={12}" -f `
            $b.runnerState, $(if ($b.activeIssue) { "#$($b.activeIssue)" } else { 'none' }), $b.openTotal, $b.pending, $partialCnt, $b.running, $b.completed, $implCnt, $humanCnt, $b.blocked, $b.failed, $b.incomplete, $b.lastSyncAt) -ForegroundColor Magenta
        foreach ($bi in $snap.backlogQueue) {
            $bCol = switch ($bi.status) { 'COMPLETED' { 'Green' } 'ALREADY_COMPLETED' { 'Green' } 'IMPLEMENTATION_COMPLETE' { 'Cyan' } 'HUMAN_REVIEW_PENDING' { 'Yellow' } 'PARTIAL' { 'Yellow' } 'RUNNING' { 'Yellow' } 'PENDING' { 'Cyan' } 'BLOCKED' { 'DarkYellow' } default { 'Red' } }
            $depStr = if ($bi.dependencies.Count) { "deps=#$(($bi.dependencies) -join ',#')" } else { 'deps=none' }
            $phaseInfo = if ($bi.status -eq 'PARTIAL') {
                " [done=$($bi.completedPhases -join ',') -> next=$($bi.remainingPhases -join ',')]"
            } elseif ($bi.blockReason) {
                " [$($bi.blockReason)]"
            } else { '' }
            Write-Host ("  #{0,-4} [{1,-23}] ver={2,-23} pri={3,3} phase={4,-2} {5,-16} tokens={6,-7} {7}{8}" -f `
                $bi.number, $bi.status, $bi.verificationState, $bi.priority, $(if ($null -ne $bi.phase) { $bi.phase } else { '-' }), $depStr, $bi.tokensUsed, $bi.title.Substring(0, [Math]::Min(48, $bi.title.Length)), $phaseInfo) -ForegroundColor $bCol
            if ($bi.delivery -or ($bi.ownership -and $bi.ownership.existingOwnedFiles.Count -gt 0)) {
                $brStr = if ($bi.delivery -and $bi.delivery.branch) { $bi.delivery.branch } else { 'no-branch' }
                $cmStr = if ($bi.delivery -and $bi.delivery.commitSha) { $bi.delivery.commitSha.Substring(0, [Math]::Min(8, $bi.delivery.commitSha.Length)) } else { '-' }
                $prStr = if ($bi.delivery -and $bi.delivery.prUrl) { $bi.delivery.prUrl } else { 'no-PR' }
                $ownStr = if ($bi.ownership.canCommitSafely) { "clean ($($bi.ownership.existingOwnedFiles.Count) files)" } elseif ($bi.ownership.sharedDirtyConflicts.Count -gt 0) { "shared-dirty ($($bi.ownership.sharedDirtyConflicts -join ','))" } else { $bi.ownership.reason }
                Write-Host ("         delivery: branch={0} commit={1} pr={2} | ownership={3}" -f $brStr, $cmStr, $prStr, $ownStr) -ForegroundColor Gray
            }
        }
    }
    if ($snap.runDir) { Write-Host "`nRun Directory : $($snap.runDir)" }
    if ($snap.liveWorker) {
        $lw = $snap.liveWorker
        Write-Host ("Active Worker : #{0} [{1}] model={2} mode={3} state={4} attempt={5}/{6} tools={7} tokens={8} conv={9}" -f `
            $lw.workerIndex, $lw.taskType, $lw.model, $lw.mode, $lw.state, $lw.attempt, $lw.maxAttempts, $lw.toolCalls, $lw.tokens, $lw.conversationId) -ForegroundColor Yellow
        Write-Host ("Last Activity : {0}" -f $lw.lastActivity)
    }
    if ($snap.summary) {
        $s = $snap.summary
        $col = switch ($s.status) { 'OK' { 'Green' } 'BLOCKED' { 'Yellow' } 'BUDGET_EXHAUSTED' { 'Yellow' } default { 'Red' } }
        $bkStr = if ($snap.budgetKind) { " budgetKind=$($snap.budgetKind)" } else { '' }
        Write-Host ("Run Summary   : runId={0} status={1}{2} totalTokens={3}" -f $s.runId, $s.status, $bkStr, $s.totalTokens) -ForegroundColor $col
        if ($snap.budgetKind -eq 'INTERNAL_RUN_BUDGET') {
            Write-Host "  budget note : Stopped by INTERNAL per-run token cap (provider quotas remain above reserve; safe to resume)." -ForegroundColor Cyan
        } elseif ($snap.budgetKind -eq 'PROVIDER_QUOTA_RESERVE') {
            Write-Host "  budget note : Stopped by PROVIDER quota reserve threshold." -ForegroundColor Red
        }
        if ($s.notes -and @($s.notes).Count) {
            foreach ($n in $s.notes) { Write-Host "  note: $n" -ForegroundColor DarkYellow }
        }
        if ($s.results -and @($s.results).Count) {
            Write-Host "`nWorker Results:" -ForegroundColor Cyan
            foreach ($r in $s.results) {
                $rCol = switch ($r.status) { 'SUCCESS' { 'Green' } 'INCOMPLETE' { 'Yellow' } default { 'Red' } }
                $filesCount = if ($r.PSObject.Properties['modifiedFiles'] -and $r.modifiedFiles) { @($r.modifiedFiles).Count } else { 0 }
                Write-Host ("  #{0} type={1} model={2} status={3} attempts={4} tokens={5} filesChanged={6} conv={7}" -f `
                    $r.index, $r.taskType, $r.requestedModel, $r.status, $r.attempts, $r.tokensAllAttempts, $filesCount, $r.conversationId) -ForegroundColor $rCol
                if ($r.error) {
                    $shortErr = if ($r.error.Length -gt 200) { $r.error.Substring(0, 200) + '...' } else { $r.error }
                    Write-Host "     error: $shortErr" -ForegroundColor Red
                }
            }
        }
    }
    if ($snap.recentEvents -and @($snap.recentEvents).Count) {
        Write-Host "`nRecent Stream Events:" -ForegroundColor Cyan
        foreach ($ev in $snap.recentEvents) {
            Write-Host ("  [{0}] ({1}) {2}" -f $ev.file, $ev.kind, $ev.message) -ForegroundColor Gray
        }
    }
    Write-Host "`nProvider Quota vs. Internal Per-Run Budgets:" -ForegroundColor Cyan
    foreach ($q in $snap.quota) {
        $qCol = if ($q.belowReserve) { 'Red' } else { 'Green' }
        $5hStr = if ($null -ne $q.fiveHourRemainingPercent) { "$($q.fiveHourRemainingPercent)% (res $($q.fiveHourReservePercent)%)" } else { 'n/a' }
        $runCap = if ($policy.groups.PSObject.Properties[$q.group]) { $policy.groups.($q.group).perRunTokenBudget } else { 'n/a' }
        Write-Host ("  {0,-18} weekly={1,5}% (res {2}%) | 5h={3,-16} | kind={4,-8} | belowReserve={5,-5} | 7dTokens={6,-8} | perRunCap={7}" -f `
            $q.group, $q.remainingPercent, $q.reservePercent, $5hStr, $q.kind, $q.belowReserve, $q.localTokensUsed7d, $runCap) -ForegroundColor $qCol
    }
    if (-not $Follow) { break }
    if ($MaxIterations -gt 0 -and $iter -ge $MaxIterations) { break }
    if ($snap.liveWorker -and $snap.liveWorker.state -in @('SUCCESS', 'FAILED', 'INCOMPLETE', 'CANCELLED') -and $snap.summary -and (-not $snap.backlog -or $snap.backlog.runnerState -ne 'RUNNING')) { break }
    Start-Sleep -Seconds $RefreshSeconds
} while ($true)
