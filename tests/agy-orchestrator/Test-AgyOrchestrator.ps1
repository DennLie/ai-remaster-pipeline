#requires -Version 7
# Integration & regression tests for tools/agy-orchestrator using a mocked agy. No live model calls.
$ErrorActionPreference = 'Stop'
$here = $PSScriptRoot
. (Join-Path $here '..\..\tools\agy-orchestrator\AgyOrchestrator.ps1')
$policy = Get-AgyPolicy (Join-Path $here '..\..\tools\agy-orchestrator\policy.json')
$mock = Join-Path $here 'agy.cmd'
$tmp = Join-Path ([IO.Path]::GetTempPath()) ('agy-orch-test-' + [guid]::NewGuid().ToString('N'))
$null = New-Item -ItemType Directory $tmp
$script:fail = 0
function Check($name, [bool]$cond) { if ($cond) { Write-Host "PASS $name" } else { Write-Host "FAIL $name"; $script:fail++ } }
function ResolveTestPath([string]$p) { if ([System.IO.Path]::IsPathRooted($p)) { $p } else { Join-Path $tmp $p } }
function Run($tasks, $auth = @{}, [bool]$writes = $false, [switch]$Par, [switch]$Dry, $env = 'ok') {
    $env:MOCK_AGY_MODE = $env
    $env:MOCK_AGY_LOG = Join-Path $tmp ([guid]::NewGuid().ToString('N') + '.log')
    $s = Invoke-AgyOrchestration -Tasks $tasks -Policy $policy -AgyCommand $mock -RunRoot $tmp -Auth $auth -AllowWrites $writes -Parallel:$Par -DryRun:$Dry -WorkDir $tmp
    [pscustomobject]@{ S = $s; Log = $(if (Test-Path $env:MOCK_AGY_LOG) { Get-Content $env:MOCK_AGY_LOG -Raw } else { '' }) }
}

# Classification
$cases = @{
    'Summarize the README and update docs' = 'gemini-3.8-flash-medium'
    'Update the GitHub issue labels'       = 'gemini-3.8-flash-medium'
    'Investigate and trace the call chain across the repo' = 'gemini-3.8-flash-high'
    'Implement a new helper function'      = 'gemini-3.8-flash-high'
    'Refactor the server module'           = 'gemini-3.8-flash-high'
    'Write unit tests for dust'            = 'gemini-3.8-flash-high'
    'Debug the intermittent crash'         = 'claude-opus-5-5-high'
    'Architecture trade-offs for the pipeline' = 'claude-opus-5-5-high'
}
foreach ($k in $cases.Keys) { Check "route '$k'" ((Resolve-AgyRoute -Text $k -Policy $policy).Model -eq $cases[$k]) }
Check 'explicit type overrides' ((Resolve-AgyRoute -Text 'Summarize' -Type 'implement' -Policy $policy).Model -eq 'gemini-3.8-flash-high')

# Worker runs with requested model, mode and structured result
$r = Run @('Implement a helper function')
$res = $r.S.results[0]
Check 'flashHigh worker success' ($r.S.status -eq 'OK' -and $res.status -eq 'SUCCESS')
Check 'argv has --model flashHigh' ($r.Log -match '\[--model\] \[gemini-3.8-flash-high\]')
Check 'argv has -p and stream-json output by default' ($r.Log -match '^\[-p\]' -and $r.Log -match '\[--output-format\] \[stream-json\]')
Check 'default timeout is 20m (1200s)' ($r.Log -match '\[--print-timeout\] \[1200s\]' -and $res.timeoutSeconds -eq 1200)
Check 'no write mode by default' ($r.Log -notmatch 'accept-edits' -and $r.Log -notmatch '\[--mode\]')
Check 'never skips permissions' ($r.Log -notmatch 'dangerously')
Check 'usage captured' ($res.usage.total_tokens -eq 120 -and $r.S.totalTokens -eq 120)
Check 'self-report captured' ($res.modelSelfReport -eq 'gemini-3.8-flash-high')
Check 'handoff file written' (Test-Path (Join-Path $tmp $res.handoff))
$w = Run @('Implement a helper function') -writes $true
Check 'AllowWrites enables accept-edits' ($w.Log -match '\[--mode\] \[accept-edits\]')
$w = Run @('Summarize docs') -writes $true
Check 'AllowWrites does not affect flash tasks' ($w.Log -notmatch 'accept-edits')

# Authorization gates
$b = Run @('Debug the crash')
Check 'opus blocked without authorization' ($b.S.status -eq 'BLOCKED' -and $b.Log -eq '')
$b = Run @('Debug the crash') @{ AuthorizeCostly = $true }
Check 'opus allowed with AuthorizeCostly' ($b.S.status -eq 'OK' -and $b.Log -match 'claude-opus-5-5-high')
$b = Run @(@{ Text = 'Think hard'; Type = 'exceptional' }) @{ AuthorizeCostly = $true }
Check 'argon blocked with only AuthorizeCostly' ($b.S.status -eq 'BLOCKED')
$b = Run @(@{ Text = 'Think hard'; Type = 'exceptional' }) @{ AuthorizeArgon = $true }
Check 'argon allowed with AuthorizeArgon' ($b.S.status -eq 'OK' -and $b.Log -match 'gemini-4-argon-high')
$b = Run @('Delete the old output folder')
Check 'destructive text blocked' ($b.S.status -eq 'BLOCKED' -and $b.Log -eq '')
$b = Run @('Delete the old output folder') @{ AuthorizeDestructive = $true }
Check 'destructive allowed when authorized' ($b.S.status -eq 'OK')

# Failure handling: retry same model, no escalation
$f = Run @('Implement a helper function') -env 'fail'
$calls = @($f.Log -split "`n" | Where-Object { $_ }).Count
Check 'failure reported' ($f.S.status -eq 'FAILED' -and $f.S.results[0].error -match 'quota')
Check 'retried once on same model only' ($calls -eq 2 -and $f.Log -notmatch 'opus|argon')
$f = Run @('Implement a helper function') -env 'flaky'
Check 'flaky recovers via same-model retry' ($f.S.status -eq 'OK' -and $f.S.results[0].attempts -eq 2)
$f = Run @('Implement a helper function') -env 'badjson'
Check 'bad JSON is a failure' ($f.S.status -eq 'FAILED')
$f = Run @('Implement a helper function') -env 'empty'
Check 'empty response is a failure' ($f.S.status -eq 'FAILED' -and $f.S.results[0].error -match 'empty')
$f = Run @('Implement a helper', 'Summarize docs') -env 'fail'
Check 'stops sequence after failure' ($f.S.results.Count -eq 1)

# Sequence, handoff chaining, budgets
$s = Run @('Summarize docs', 'Implement a helper function')
Check 'sequential two tasks' ($s.S.results.Count -eq 2 -and $s.S.totalTokens -eq 240)
Check 'handoff chains prior summary' ((Get-Content (Join-Path $tmp $s.S.results[1].handoff) -Raw) -match 'Previous worker summary')
$tight = $policy | ConvertTo-Json -Depth 8 | ConvertFrom-Json; $tight.budgets.maxTotalTokens = 100
$env:MOCK_AGY_MODE = 'ok'; $env:MOCK_AGY_LOG = Join-Path $tmp 'b.log'
$s = Invoke-AgyOrchestration -Tasks @('Summarize docs', 'Summarize more docs') -Policy $tight -AgyCommand $mock -RunRoot $tmp -WorkDir $tmp
Check 'token budget stops run' ($s.status -eq 'BUDGET_EXHAUSTED' -and $s.results.Count -eq 1)
$s = Run @('a', 'b', 'c', 'd', 'e', 'f', 'g')
Check 'maxWorkersPerRun enforced' ($s.S.status -eq 'BLOCKED')

# Parallel
$p = Run @('Summarize docs', 'Implement a helper function') -Par
Check 'parallel runs 2 workers' ($p.S.results.Count -eq 2 -and $p.S.status -eq 'OK' -and $p.S.notes -match 'parallel')
$p = Run @('Summarize a', 'Summarize b', 'Summarize c') -Par
Check 'parallel of 3 falls back to sequential' ($p.S.results.Count -eq 3 -and ($p.S.notes -match 'limited'))

# Dry run, issue text, plan context
$d = Run @('Implement a helper function') -Dry
Check 'dry run makes no calls' ($d.Log -eq '' -and $d.S.plan[0].model -eq 'gemini-3.8-flash-high')
Check 'issue text via gh' ((Get-AgyIssueText -Number 33 -GhCommand (Join-Path $here 'gh.cmd')) -match 'Mock issue')
$env:MOCK_AGY_MODE = 'ok'; $env:MOCK_AGY_LOG = Join-Path $tmp 'c.log'
$s = Invoke-AgyOrchestration -Tasks @('Summarize docs') -Policy $policy -AgyCommand $mock -RunRoot $tmp -WorkDir $tmp -ContextFiles @('docs/maestro/plans/x.md')
Check 'plan file referenced in handoff' ((Get-Content (Join-Path $tmp $s.results[0].handoff) -Raw) -match 'docs/maestro/plans/x.md')

# ---------------------------------------------------------------- quota guards ----
function RunQ($tasks, $state, $pol = $policy, [hashtable]$auth = @{}, [switch]$Override, [switch]$Dry) {
    $env:MOCK_AGY_MODE = 'ok'; $env:MOCK_AGY_LOG = Join-Path $tmp ([guid]::NewGuid().ToString('N') + '.log')
    $s = Invoke-AgyOrchestration -Tasks $tasks -Policy $pol -AgyCommand $mock -RunRoot $tmp -StatePath $state -Auth $auth -OverrideReserve:$Override -DryRun:$Dry -WorkDir $tmp
    [pscustomobject]@{ S = $s; Log = $(if (Test-Path $env:MOCK_AGY_LOG) { Get-Content $env:MOCK_AGY_LOG -Raw } else { '' }) }
}
function NewState { Join-Path $tmp ('state-' + [guid]::NewGuid().ToString('N') + '.json') }
$FLASH = 'gemini-flash-pro'; $CL = 'claude-gpt'; $ARG = 'argon'

Check 'model->group mapping' ((Get-AgyModelGroup 'gemini-3.8-flash-high' $policy) -eq $FLASH -and (Get-AgyModelGroup 'claude-opus-5-5-high' $policy) -eq $CL -and (Get-AgyModelGroup 'gpt-oss-120b-medium' $policy) -eq $CL -and (Get-AgyModelGroup 'gemini-4-argon-high' $policy) -eq $ARG)
Check 'default reserve is 25%' ($policy.quota.weeklyReservePercent -eq 25)
Check 'default 5h reserve is 10%' ($policy.quota.fiveHourReservePercent -eq 10)

$st = NewState
$q = Get-AgyGroupStatus -Group $FLASH -Policy $policy -State (Get-AgyState $st)
Check 'no reading + cap => labelled ESTIMATE (local ledger)' ($q.kind -eq 'ESTIMATE' -and $q.remainingPercent -eq 100 -and $q.basis -match 'not a Google reading')
$noCap = $policy | ConvertTo-Json -Depth 8 | ConvertFrom-Json; $noCap.groups.$FLASH.localWeeklyTokenCap = $null
$q = Get-AgyGroupStatus -Group $FLASH -Policy $noCap -State (Get-AgyState $st)
Check 'no reading + no cap => UNKNOWN, not blocked' ($q.kind -eq 'UNKNOWN' -and $null -eq $q.remainingPercent -and -not $q.belowReserve)

Set-AgyQuotaReading -Group $CL -RemainingPercent 20 -StatePath $st -Policy $policy
$q = Get-AgyGroupStatus -Group $CL -Policy $policy -State (Get-AgyState $st)
Check 'manual reading is ESTIMATE, never ACTUAL' ($q.kind -eq 'ESTIMATE' -and $q.basis -match 'manual')
# Flash High is preferred for implement:
$st = NewState
Set-AgyQuotaReading -Group $FLASH -RemainingPercent 96 -FiveHourRemainingPercent 94 -StatePath $st -Policy $policy
Set-AgyQuotaReading -Group $CL    -RemainingPercent 10 -FiveHourRemainingPercent 1  -StatePath $st -Policy $policy
Set-AgyQuotaReading -Group $ARG   -RemainingPercent 0  -FiveHourRemainingPercent 0  -StatePath $st -Policy $policy
$bPref = RunQ @('Implement a helper function') $st
Check 'preferred model for implement is gemini-3.8-flash-high even when claude and argon are low' (
    $bPref.S.status -eq 'OK' -and $bPref.S.results[0].requestedModel -eq 'gemini-3.8-flash-high'
)

# When flash-pro is below reserve and task is genuinely complex, falls back to claude-sonnet ONLY IF claude has quota:
$stFb = NewState
Set-AgyQuotaReading -Group $FLASH -RemainingPercent 15 -FiveHourRemainingPercent 8  -StatePath $stFb -Policy $policy
Set-AgyQuotaReading -Group $CL    -RemainingPercent 60 -FiveHourRemainingPercent 50 -StatePath $stFb -Policy $policy
Set-AgyQuotaReading -Group $ARG   -RemainingPercent 80 -FiveHourRemainingPercent 80 -StatePath $stFb -Policy $policy
$bFb = RunQ @('Implement a complex algorithm with intricate edge cases') $stFb
Check 'when flash-pro < 25% reserve and task is complex, falls back to claude-sonnet-5-5-medium' (
    $bFb.S.status -eq 'OK' -and $bFb.S.results[0].requestedModel -eq 'claude-sonnet-5-5-medium' -and
    $bFb.S.plan[0].primaryModel -eq 'gemini-3.8-flash-high' -and $bFb.S.plan[0].fallbackFrom -eq 'gemini-3.8-flash-high'
)

# When argon has autoDispatch: false, it is NEVER automatically dispatched even if healthy:
Check 'argon is never automatically dispatched (autoDispatch: false)' (
    $policy.groups.argon.autoDispatch -eq $false
)

# When both flash-pro and claude-gpt are below reserve, dispatch is BLOCKED (no call and never calls argon):
$stEx = NewState
Set-AgyQuotaReading -Group $FLASH -RemainingPercent 10 -FiveHourRemainingPercent 5  -StatePath $stEx -Policy $policy
Set-AgyQuotaReading -Group $CL    -RemainingPercent 10 -FiveHourRemainingPercent 2  -StatePath $stEx -Policy $policy
Set-AgyQuotaReading -Group $ARG   -RemainingPercent 90 -FiveHourRemainingPercent 90 -StatePath $stEx -Policy $policy
$bEx = RunQ @('Implement a helper function') $stEx
Check 'when flash-pro and claude-gpt are < reserve, dispatch is BLOCKED and never falls back to argon' (
    $bEx.S.status -eq 'BLOCKED' -and $bEx.Log -eq '' -and ($bEx.S.blocked -match 'below the 25% reserve')
)

$b = RunQ @('Summarize docs') $st
Check 'other tasks on flash succeed when flash quota is healthy' ($b.S.status -eq 'OK')
$b = RunQ @('Implement a helper function') $stEx -Override
Check '-OverrideReserve dispatches primary model with note' ($b.S.status -eq 'OK' -and ($b.S.notes -match 'OVERRIDDEN'))
$b = RunQ @('Implement a helper function') $stEx -Dry
Check 'dry run shows quota block when quota exhausted, no call' ($b.S.status -eq 'BLOCKED' -and $b.Log -eq '' -and $b.S.quota[0].group -eq $FLASH)

# 5-hour reserve enforcement
$st5 = NewState
Set-AgyQuotaReading -Group $FLASH -RemainingPercent 60 -FiveHourRemainingPercent 5 -StatePath $st5 -Policy $policy
Set-AgyQuotaReading -Group $CL    -RemainingPercent 10 -FiveHourRemainingPercent 2 -StatePath $st5 -Policy $policy
$q5 = Get-AgyGroupStatus -Group $FLASH -Policy $policy -State (Get-AgyState $st5)
Check '5h reading 5% < 10% 5h reserve triggers belowFiveHourReserve' ($q5.belowFiveHourReserve -and $q5.belowReserve -and -not $q5.belowWeeklyReserve)
$b5 = RunQ @('Implement a helper function') $st5
Check '5h reserve blocks dispatch when quota exhausted and claude also below reserve' ($b5.S.status -eq 'BLOCKED' -and ($b5.S.blocked -match '5-hour remaining'))
$b5o = RunQ @('Implement a helper function') $st5 -Override
Check '-OverrideReserve overrides 5h reserve with note' ($b5o.S.status -eq 'OK' -and ($b5o.S.notes -match '5h reserve OVERRIDDEN'))

# Sync-AgyQuotaFromCli with mocked /usage JSON
$stSync = NewState
$syncRep = @(Sync-AgyQuotaFromCli -StatePath $stSync -AgyCommand $mock -Policy $policy)
$clSync = $syncRep | Where-Object group -eq $CL
$flSync = $syncRep | Where-Object group -eq $FLASH
Check 'Sync-AgyQuotaFromCli parses weekly and 5h buckets as ACTUAL' ($flSync.kind -eq 'ACTUAL' -and $flSync.remainingPercent -eq 96 -and $flSync.fiveHourRemainingPercent -eq 92 -and $clSync.remainingPercent -eq 48 -and $clSync.fiveHourRemainingPercent -eq 5 -and $clSync.belowFiveHourReserve)

$st = NewState; Set-AgyQuotaReading -Group $FLASH -RemainingPercent 80 -StatePath $st -Policy $policy
$b = RunQ @('Implement a helper function') $st
Check 'manual 80% allows dispatch' ($b.S.status -eq 'OK')
$led = (Get-AgyState $st).ledger
Check 'ledger records tokens per group' ($led.Count -eq 1 -and $led[0].group -eq $FLASH -and $led[0].tokens -eq 120)
Check 'summary carries post-run quota status' ($b.S.quota[0].kind -eq 'ESTIMATE')

# local usage is deducted from a manual reading
$small = $policy | ConvertTo-Json -Depth 8 | ConvertFrom-Json; $small.groups.$FLASH.localWeeklyTokenCap = 1000
$st = NewState; Set-AgyQuotaReading -Group $FLASH -RemainingPercent 30 -StatePath $st -Policy $small
$s0 = Get-AgyState $st; Add-AgyUsage -State $s0 -Group $FLASH -Model 'gemini-3.8-flash-high' -Tokens 100 -RunId 't'; Save-AgyState $s0 $st
$q = Get-AgyGroupStatus -Group $FLASH -Policy $small -State (Get-AgyState $st)
Check 'usage since reading deducted (30% - 10% = 20%)' ($q.remainingPercent -eq 20 -and $q.belowReserve)

# ledger-only local cap (with fallback also exhausted so it blocks)
$st = NewState; $s0 = Get-AgyState $st; Add-AgyUsage -State $s0 -Group $FLASH -Model 'm' -Tokens 800 -RunId 't'; Save-AgyState $s0 $st
Set-AgyQuotaReading -Group $CL -RemainingPercent 10 -StatePath $st -Policy $small
$b = RunQ @('Implement a helper function') $st $small
Check 'local cap 80% used => 20% left => blocked when fallback also below reserve' ($b.S.status -eq 'BLOCKED' -and $b.S.quota[0].kind -eq 'ESTIMATE')
$st = NewState; $s0 = Get-AgyState $st; Add-AgyUsage -State $s0 -Group $FLASH -Model 'm' -Tokens 800 -RunId 't' -Now ([datetime]::UtcNow.AddDays(-8)); Save-AgyState $s0 $st
$b = RunQ @('Implement a helper function') $st $small
Check 'usage older than 7 days ignored' ($b.S.status -eq 'OK')

# official reading: fresh = ACTUAL, stale = ESTIMATE
$st = NewState; Set-AgyQuotaReading -Group $FLASH -RemainingPercent 90 -Source official -StatePath $st -Policy $policy
Check 'fresh official reading is ACTUAL' ((Get-AgyGroupStatus -Group $FLASH -Policy $policy -State (Get-AgyState $st)).kind -eq 'ACTUAL')
Set-AgyQuotaReading -Group $FLASH -RemainingPercent 90 -Source official -StatePath $st -Policy $policy -AsOf ([datetime]::UtcNow.AddHours(-30))
Check 'stale official reading downgraded to ESTIMATE' ((Get-AgyGroupStatus -Group $FLASH -Policy $policy -State (Get-AgyState $st)).kind -eq 'ESTIMATE')
$thrown = $false; try { Set-AgyQuotaReading -Group 'nope' -RemainingPercent 5 -StatePath $st -Policy $policy } catch { $thrown = $true }
Check 'unknown group rejected' $thrown

# per-group reserve override
$ov = $policy | ConvertTo-Json -Depth 8 | ConvertFrom-Json
$ov.groups.$FLASH | Add-Member -NotePropertyName weeklyReservePercent -NotePropertyValue 60 -Force
$st = NewState; Set-AgyQuotaReading -Group $FLASH -RemainingPercent 50 -StatePath $st -Policy $policy
$b = RunQ @('Summarize docs') $st $ov
Check 'per-group reserve override (60%) blocks at 50%' ($b.S.status -eq 'BLOCKED')

# per-run group budgets
$pb = $policy | ConvertTo-Json -Depth 8 | ConvertFrom-Json; $pb.groups.$FLASH.perRunTokenBudget = 100
$b = RunQ @('Implement a helper function', 'Refactor the server module') (NewState) $pb
Check 'per-run group token budget stops after first call' ($b.S.status -eq 'BUDGET_EXHAUSTED' -and $b.S.results.Count -eq 1)
$pc = $policy | ConvertTo-Json -Depth 8 | ConvertFrom-Json; $pc.groups.$FLASH.maxCallsPerRun = 1
$b = RunQ @('Implement a helper function', 'Refactor the server module') (NewState) $pc
Check 'maxCallsPerRun enforced at plan time' ($b.S.status -eq 'BLOCKED' -and $b.Log -eq '')
$pc = $policy | ConvertTo-Json -Depth 8 | ConvertFrom-Json; $pc.groups.$ARG.maxCallsPerRun = 1
$b = RunQ @(@{ Text = 'x'; Type = 'exceptional' }, @{ Text = 'y'; Type = 'exceptional' }) (NewState) $pc @{ AuthorizeArgon = $true }
Check 'argon maxCallsPerRun limit enforced' ($b.S.status -eq 'BLOCKED')

# approvals still required with healthy quota
$st = NewState; Set-AgyQuotaReading -Group $CL -RemainingPercent 100 -StatePath $st -Policy $policy
Check 'opus needs approval even with 100% quota' ((RunQ @('Debug the crash') $st).S.status -eq 'BLOCKED')
Check 'argon needs approval even with healthy quota' ((RunQ @(@{ Text = 'x'; Type = 'exceptional' }) $st).S.status -eq 'BLOCKED')
$b = RunQ @('Debug the crash') $st $policy @{ AuthorizeCostly = $true }
Check 'opus runs when approved and above reserve' ($b.S.status -eq 'OK')

# failures still never escalate and still count tokens
$env:MOCK_AGY_MODE = 'fail'; $env:MOCK_AGY_LOG = Join-Path $tmp 'f.log'
$st = NewState
$s = Invoke-AgyOrchestration -Tasks @('Summarize docs') -Policy $policy -AgyCommand $mock -RunRoot $tmp -StatePath $st -WorkDir $tmp
Check 'failed run reports FAILED, no escalation' ($s.status -eq 'FAILED' -and (Get-Content $env:MOCK_AGY_LOG -Raw) -notmatch 'opus|argon')

# ---------------------------------------------------------------- CLI end-to-end (Maestro plan + issues) ----
$cli = Join-Path $here '..\..\tools\agy-orchestrator\Invoke-Agy.ps1'
$ghm = Join-Path $here 'gh.cmd'
$planDir = Join-Path $tmp 'plans'; $null = New-Item -ItemType Directory $planDir
Set-Content (Join-Path $planDir '2026-01-01-demo-impl-plan.md') '# demo plan'
$rr = Join-Path $tmp 'e2e'; $stE = Join-Path $rr 'state.json'
$env:MOCK_AGY_MODE = 'ok'; $env:MOCK_AGY_LOG = Join-Path $tmp 'e2e.log'
$common = @('-AgyCommand', $mock, '-GhCommand', $ghm, '-RunRoot', $rr, '-PlanFile', (Join-Path $planDir '*impl-plan.md'))

$o = pwsh -NoProfile -File $cli -Issue 33 @common -DryRun | ConvertFrom-Json
Check 'CLI dry-run issue #33 => investigation on flash-high' ($LASTEXITCODE -eq 0 -and $o.plan[0].model -eq 'gemini-3.8-flash-high' -and -not (Test-Path $env:MOCK_AGY_LOG))
$o = pwsh -NoProfile -File $cli -Issue 33 @common | ConvertFrom-Json
Check 'CLI issue #33 analysis runs on flash-high via mock' ($LASTEXITCODE -eq 0 -and $o.results[0].requestedModel -eq 'gemini-3.8-flash-high' -and (Get-Content $env:MOCK_AGY_LOG -Raw) -match 'gemini-3.8-flash-high')
$h = Get-Content (ResolveTestPath $o.results[0].handoff) -Raw -Encoding utf8
Check 'handoff has issue text and Maestro plan path' ($h -match 'Mock issue' -and $h -match 'demo-impl-plan.md')
Check 'handoff enforces read-only investigation boundaries' ($h -match 'INVESTIGATION ONLY \(READ-ONLY\)' -and $h -match 'DO NOT\*\* implement features')
Check 'latest.json written' (Test-Path (Join-Path $rr 'latest.json'))
$o2 = pwsh -NoProfile -File $cli -Issue 33 -Type implement -AllowWrites -HandoffFrom latest @common | ConvertFrom-Json
Check 'CLI implementation step uses flash-high + accept-edits' ($LASTEXITCODE -eq 0 -and $o2.results[0].requestedModel -eq 'gemini-3.8-flash-high' -and $o2.results[0].mode -eq 'accept-edits')
Check 'implementation handoff carries analysis summary' ((Get-Content (ResolveTestPath $o2.results[0].handoff) -Raw) -match 'Previous worker summary' -and (Get-Content (ResolveTestPath $o2.results[0].handoff) -Raw) -match 'MODEL: gemini-3.8-flash-high')
$null = pwsh -NoProfile -File $cli -Issue 33 -Type implement @common -DryRun
Check 'Issue implementation without -AllowWrites is BLOCKED (exit 2)' ($LASTEXITCODE -eq 2)
$null = pwsh -NoProfile -Command "& '$cli' -Issue 33,34,35,36,37,38,39 -AgyCommand '$mock' -GhCommand '$ghm' -RunRoot '$rr' -DryRun; exit `$LASTEXITCODE"
Check 'seven issues in one run blocked (max 6 workers)' ($LASTEXITCODE -eq 2)
foreach ($n in 33..37) { $null = pwsh -NoProfile -File $cli -Issue $n @common -DryRun; Check "issue #$n dry-run accepted" ($LASTEXITCODE -eq 0) }
$null = pwsh -NoProfile -File $cli -Issue 33 -Type debug @common -DryRun
Check 'CLI exit code 2 when opus not approved' ($LASTEXITCODE -eq 2)
Set-AgyQuotaReading -Group $FLASH -RemainingPercent 10 -StatePath $stE -Policy $policy
$null = pwsh -NoProfile -File $cli -Issue 33 @common -DryRun
Check 'CLI exit code 2 when quota below reserve' ($LASTEXITCODE -eq 2)
$null = pwsh -NoProfile -File $cli -Issue 33 @common -DryRun -OverrideReserve
Check 'CLI -OverrideReserve proceeds' ($LASTEXITCODE -eq 0)
$qo = pwsh -NoProfile -File (Join-Path $here '..\..\tools\agy-orchestrator\Quota.ps1') -StatePath $stE | Out-String
Check 'Quota.ps1 report prints labels' ($qo -match 'ESTIMATE' -and $qo -match 'claude-gpt')

# ---------------------------------------------------------------- headless permission handling ----
function RunY([string]$mode, [switch]$Yolo, [bool]$writes = $false, $tasks = @('Implement a helper function'), [switch]$Par) {
    $env:MOCK_AGY_MODE = $mode; $env:MOCK_AGY_LOG = Join-Path $tmp ([guid]::NewGuid().ToString('N') + '.log')
    $st = Join-Path $tmp ('state-' + [guid]::NewGuid().ToString('N') + '.json')
    $s = Invoke-AgyOrchestration -Tasks $tasks -Policy $policy -AgyCommand $mock -RunRoot $tmp -StatePath $st -AllowWrites $writes -YoloWorker:$Yolo -Parallel:$Par -WorkDir $tmp
    [pscustomobject]@{ S = $s; St = $st; Log = $(if (Test-Path $env:MOCK_AGY_LOG) { Get-Content $env:MOCK_AGY_LOG -Raw } else { '' }) }
}
$y = RunY 'ok'
Check 'default: no --dangerously-skip-permissions' ($y.Log -notmatch 'dangerously' -and $y.S.results[0].permissionBypass -eq $false)
$y = RunY 'ok' -Yolo
Check '-YoloWorker forwards --dangerously-skip-permissions' ($y.Log -match '\[--dangerously-skip-permissions\]' -and $y.S.results[0].permissionBypass -eq $true)
Check '-YoloWorker alone does not enable writes' ($y.Log -notmatch 'accept-edits' -and $y.S.results[0].mode -eq '')
Check '-YoloWorker keeps model routing' ($y.Log -match '\[--model\] \[gemini-3.8-flash-high\]')
Check '-YoloWorker prints a warning note' (($y.S.notes -join ' ') -match 'YoloWorker')
$y = RunY 'ok' -writes $true
Check '-AllowWrites alone does not bypass permissions' ($y.Log -match 'accept-edits' -and $y.Log -notmatch 'dangerously')
$y = RunY 'ok' -Yolo -writes $true
Check 'both flags together are independent' ($y.Log -match 'accept-edits' -and $y.Log -match 'dangerously')
$y = RunY 'ok' -Yolo -Par -tasks @('Summarize docs', 'Implement a helper function')
Check '-YoloWorker reaches parallel workers' (@($y.Log -split "`n" | Where-Object { $_ -match 'dangerously' }).Count -eq 2)

foreach ($yo in @($false, $true)) {
    $y = RunY 'denied' -Yolo:$yo
    $r0 = $y.S.results[0]
    Check "permission denial with exit 0 is FAILED (yolo=$yo)" ($y.S.status -eq 'FAILED' -and $r0.status -eq 'FAILED' -and $r0.exitCode -eq 0 -and $r0.error -match 'permission denied in headless')
    Check "denial not retried (yolo=$yo)" ($r0.attempts -eq 1 -and @($y.Log -split "`n" | Where-Object { $_ }).Count -eq 1)
    Check "denial tokens still accounted (yolo=$yo)" ($r0.tokensAllAttempts -eq 28325 -and $y.S.totalTokens -eq 28325 -and (Get-AgyState $y.St).ledger[0].tokens -eq 28325)
}
$y = RunY 'denied' -tasks @('Implement a helper function', 'Summarize docs')
Check 'denial stops the sequence' ($y.S.results.Count -eq 1)
$y = RunY 'empty' -Yolo
Check 'empty response still FAILED with -YoloWorker' ($y.S.status -eq 'FAILED' -and $y.S.results[0].error -match 'empty')

# CLI: switch is wired through Invoke-Agy.ps1
$cliY = Join-Path $here '..\..\tools\agy-orchestrator\Invoke-Agy.ps1'
$rrY = Join-Path $tmp 'yolo'; $env:MOCK_AGY_MODE = 'ok'; $env:MOCK_AGY_LOG = Join-Path $tmp 'cliy.log'
$o = pwsh -NoProfile -File $cliY -Task 'Implement a helper function' -AgyCommand $mock -RunRoot $rrY | ConvertFrom-Json
Check 'CLI default has no bypass' ($LASTEXITCODE -eq 0 -and (Get-Content $env:MOCK_AGY_LOG -Raw) -notmatch 'dangerously')
Remove-Item $env:MOCK_AGY_LOG
$o = pwsh -NoProfile -File $cliY -Task 'Implement a helper function' -AgyCommand $mock -RunRoot $rrY -YoloWorker | ConvertFrom-Json
Check 'CLI -YoloWorker forwards flag' ($LASTEXITCODE -eq 0 -and (Get-Content $env:MOCK_AGY_LOG -Raw) -match 'dangerously' -and $o.results[0].permissionBypass)
$env:MOCK_AGY_MODE = 'denied'
$null = pwsh -NoProfile -File $cliY -Task 'Implement a helper function' -AgyCommand $mock -RunRoot $rrY
Check 'CLI exit code 1 on permission denial' ($LASTEXITCODE -eq 1)

# ---------------------------------------------------------------- streaming, timeout, resume, UTF-8/paths, lifecycle, dashboard ----

# 1. Stream-json parsing & raw event persistence
$env:MOCK_AGY_MODE = 'stream'; $env:MOCK_AGY_LOG = Join-Path $tmp 'stream.log'
$rrS = Join-Path $tmp 'stream-run'
$sStream = Invoke-AgyOrchestration -Tasks @('Implement a helper function') -Policy $policy -AgyCommand $mock -RunRoot $rrS -AllowWrites $true -Live -WorkerTimeoutMinutes 35 -WorkDir $tmp
$rStream = $sStream.results[0]
Check 'configurable WorkerTimeoutMinutes (35m = 2100s) passed' ((Get-Content $env:MOCK_AGY_LOG -Raw) -match '\[--print-timeout\] \[2100s\]' -and $rStream.timeoutSeconds -eq 2100)
Check 'stream-json captures tool calls' ($rStream.toolCallsCount -eq 3 -and $rStream.toolCalls[0].tool -eq 'read_file' -and $rStream.toolCalls[1].commandLine -match 'unittest')
Check 'stream-json persists raw events to worker-1.events.jsonl' (Test-Path (Join-Path $tmp $rStream.eventsFile))
$evLines = @(Get-Content (Join-Path $tmp $rStream.eventsFile))
Check 'events file contains all 5 NDJSON lines' ($evLines.Count -eq 5)
Check 'stream-json captures token usage (240)' ($rStream.tokensAllAttempts -eq 240 -and $sStream.totalTokens -eq 240)

# 2. Print timeout handling: NEVER auto-retry, status INCOMPLETE, partial output & conversationId preserved
$env:MOCK_AGY_MODE = 'timeout'; $env:MOCK_AGY_LOG = Join-Path $tmp 'timeout.log'
$rrT = Join-Path $tmp 'timeout-run'
$sTo = Invoke-AgyOrchestration -Tasks @('Investigate the call chain', 'Implement a helper function') -Policy $policy -AgyCommand $mock -RunRoot $rrT -WorkDir $tmp
$rTo = $sTo.results[0]
$toCalls = @((Get-Content $env:MOCK_AGY_LOG) | Where-Object { $_ }).Count
Check 'timeout NEVER auto-retries (only 1 attempt)' ($rTo.attempts -eq 1 -and $toCalls -eq 1)
Check 'timeout marks worker and run as INCOMPLETE' ($rTo.status -eq 'INCOMPLETE' -and $sTo.status -eq 'INCOMPLETE' -and $rTo.finishedReason -eq 'print_timeout')
Check 'timeout preserves conversationId (conv-timeout-42)' ($rTo.conversationId -eq 'conv-timeout-42')
Check 'timeout preserves partial output file' ($null -ne $rTo.partialOutputFile -and (Get-Content (Join-Path $tmp $rTo.partialOutputFile) -Raw) -match 'Partial investigation notes')
Check 'timeout stops subsequent tasks in sequence' ($sTo.results.Count -eq 1)
Check 'timeout still accounts consumed tokens (52000)' ($rTo.tokensAllAttempts -eq 52000 -and $sTo.totalTokens -eq 52000)

# 3. Safe resumption via -ResumeLatest and -ResumeConversationId
$env:MOCK_AGY_MODE = 'stream'; $env:MOCK_AGY_LOG = Join-Path $tmp 'resume.log'
$oRes = pwsh -NoProfile -File $cliY -Issue 33 -ResumeLatest -AgyCommand $mock -GhCommand $ghm -RunRoot $rrT | ConvertFrom-Json
$resLog = Get-Content $env:MOCK_AGY_LOG -Raw
Check '-ResumeLatest finds conv-timeout-42 and passes --conversation' ($LASTEXITCODE -eq 0 -and $resLog -match '\[--conversation\] \[conv-timeout-42\]' -and $oRes.results[0].resumedFrom -eq 'conv-timeout-42')
$resHandoff = Get-Content (ResolveTestPath $oRes.results[0].handoff) -Raw -Encoding utf8
Check 'resumed handoff records Resumed conversation and prior partial summary' ($resHandoff -match 'Resumed conversation: conv-timeout-42' -and $resHandoff -match 'Partial investigation notes')

# 4. Cancellation / interrupted handling
$env:MOCK_AGY_MODE = 'interrupted'; $env:MOCK_AGY_LOG = Join-Path $tmp 'int.log'
$sInt = Invoke-AgyOrchestration -Tasks @('Summarize docs') -Policy $policy -AgyCommand $mock -RunRoot $tmp -WorkDir $tmp
Check 'interrupted worker marked CANCELLED without retry' ($sInt.status -eq 'CANCELLED' -and $sInt.results[0].status -eq 'CANCELLED' -and $sInt.results[0].attempts -eq 1 -and $sInt.results[0].conversationId -eq 'conv-int-7')

# 5. Clean UTF-8 and Markdown path normalization
$dirtyText = "Fix \scripts/benchmark_defect_masks.py\ and \`ttests/test_defect_mask_benchmark.py\ with range 1–8px — ok"
$cleaned = Format-AgyCleanMarkdown $dirtyText
Check 'Format-AgyCleanMarkdown repairs \path\ and \ttests corruption and preserves UTF-8 en/em-dashes' ($cleaned -match '`scripts/benchmark_defect_masks.py`' -and $cleaned -match '`tests/test_defect_mask_benchmark.py`' -and $cleaned -match '1–8px — ok')
$utfFile = Join-Path $tmp 'utf8-check.md'
Write-AgyUtf8File -Path $utfFile -Value $cleaned
Check 'Write-AgyUtf8File round-trips UTF-8 without mojibake' ((Get-Content $utfFile -Raw -Encoding utf8) -match '1–8px — ok')

# 6. Full autonomous issue lifecycle: investigation -> implement -> test -> review -> (skip corrections when APPROVED) -> verify
$env:MOCK_AGY_MODE = 'stream'; $env:MOCK_AGY_LOG = Join-Path $tmp 'lc-ok.log'
$rrLC = Join-Path $tmp 'lc-ok'
$null = pwsh -NoProfile -File $cliY -Issue 33 -Lifecycle -AgyCommand $mock -GhCommand $ghm -RunRoot $rrLC -DryRun
Check 'Lifecycle without -AllowWrites is BLOCKED at plan time' ($LASTEXITCODE -eq 2)
$oLC = pwsh -NoProfile -File $cliY -Issue 33 -Lifecycle -AllowWrites -YoloWorker -AgyCommand $mock -GhCommand $ghm -RunRoot $rrLC | ConvertFrom-Json
Check 'Lifecycle with -AllowWrites succeeds' ($LASTEXITCODE -eq 0 -and $oLC.status -eq 'OK')
Check 'Lifecycle skips corrections when review is APPROVED (5 phases run)' ($oLC.results.Count -eq 5 -and ($oLC.notes -join ' ') -match 'skipped task 5')
Check 'Lifecycle routes phases to gemini-3.8-flash-high by default for efficiency and tracks independent review' (
    $oLC.results[0].taskType -eq 'investigation' -and $oLC.results[0].requestedModel -eq 'gemini-3.8-flash-high' -and $oLC.results[0].mode -eq '' -and
    $oLC.results[1].taskType -eq 'implement'     -and $oLC.results[1].requestedModel -eq 'gemini-3.8-flash-high' -and $oLC.results[1].mode -eq 'accept-edits' -and
    $oLC.results[2].taskType -eq 'test'          -and $oLC.results[2].requestedModel -eq 'gemini-3.8-flash-high' -and $oLC.results[2].mode -eq 'accept-edits' -and
    $oLC.results[3].taskType -eq 'review'        -and $oLC.results[3].requestedModel -eq 'gemini-3.8-flash-high' -and $oLC.results[3].mode -eq '' -and
    $oLC.results[3].isIndependentReview -eq $true -and $oLC.isIndependentReview -eq $true -and
    $oLC.results[4].taskType -eq 'verify'        -and $oLC.results[4].requestedModel -eq 'gemini-3.8-flash-high' -and $oLC.results[4].mode -eq ''
)

# Lifecycle when review requests changes -> runs corrections phase (all 6 phases run)
$env:MOCK_AGY_MODE = 'review_changes'; $env:MOCK_AGY_LOG = Join-Path $tmp 'lc-chg.log'
$rrLC2 = Join-Path $tmp 'lc-chg'
$oLC2 = pwsh -NoProfile -File $cliY -Issue 33 -Lifecycle -AllowWrites -AgyCommand $mock -GhCommand $ghm -RunRoot $rrLC2 | ConvertFrom-Json
Check 'Lifecycle runs corrections phase when review says CHANGES_REQUIRED (6 phases run on flash-high)' ($LASTEXITCODE -eq 0 -and $oLC2.results.Count -eq 6 -and $oLC2.results[4].taskType -eq 'corrections' -and $oLC2.results[4].requestedModel -eq 'gemini-3.8-flash-high')

# Lifecycle when claude-gpt 5h quota is low (1%) and argon autoDispatch is disabled:
# All phases remain on gemini-3.8-flash-high with ZERO calls to argon:
$env:MOCK_AGY_MODE = 'review_changes'; $env:MOCK_AGY_LOG = Join-Path $tmp 'lc-fb.log'
$rrLC3 = Join-Path $tmp 'lc-fb'
$stLC3 = Join-Path $rrLC3 'state.json'
Set-AgyQuotaReading -Group $FLASH -RemainingPercent 96 -FiveHourRemainingPercent 94 -Source official -StatePath $stLC3 -Policy $policy
Set-AgyQuotaReading -Group $CL    -RemainingPercent 49 -FiveHourRemainingPercent 1  -Source official -StatePath $stLC3 -Policy $policy
Set-AgyQuotaReading -Group $ARG   -RemainingPercent 85 -Source official -StatePath $stLC3 -Policy $policy
$oLC3 = pwsh -NoProfile -File $cliY -Issue 33 -Lifecycle -AllowWrites -YoloWorker -AgyCommand $mock -GhCommand $ghm -RunRoot $rrLC3 -StatePath $stLC3 | ConvertFrom-Json
Check 'Lifecycle routes all phases to gemini-3.8-flash-high without dispatching argon' (
    $LASTEXITCODE -eq 0 -and $oLC3.status -eq 'OK' -and $oLC3.results.Count -eq 6 -and
    $oLC3.results[0].requestedModel -eq 'gemini-3.8-flash-high'   -and
    $oLC3.results[1].requestedModel -eq 'gemini-3.8-flash-high'   -and
    $oLC3.results[2].requestedModel -eq 'gemini-3.8-flash-high'   -and
    $oLC3.results[3].requestedModel -eq 'gemini-3.8-flash-high'   -and
    $oLC3.results[4].requestedModel -eq 'gemini-3.8-flash-high'   -and
    $oLC3.results[5].requestedModel -eq 'gemini-3.8-flash-high'
)

# Deterministic local verification via -DeterministicVerify:
$env:MOCK_AGY_MODE = 'stream'; $env:MOCK_AGY_LOG = Join-Path $tmp 'lc-det.log'
$rrDet = Join-Path $tmp 'lc-det'
$oDet = pwsh -NoProfile -File $cliY -Issue 33 -Lifecycle -AllowWrites -DeterministicVerify -YoloWorker -AgyCommand $mock -GhCommand $ghm -RunRoot $rrDet | ConvertFrom-Json
Check 'Deterministic verification uses local runner with 0 tokens and passes' (
    $LASTEXITCODE -eq 0 -and $oDet.status -eq 'OK' -and $oDet.results.Count -eq 5 -and
    $oDet.results[4].taskType -eq 'verify' -and
    $oDet.results[4].requestedModel -eq 'local-deterministic' -and
    $oDet.results[4].tokensAllAttempts -eq 0 -and
    $oDet.results[4].response -match 'VERIFICATION: PASSED'
)

# 7. Watch-Agy.ps1 dashboard & cancellation signal
$watch = Join-Path $here '..\..\tools\agy-orchestrator\Watch-Agy.ps1'
$wSnap = pwsh -NoProfile -File $watch -RunRoot $rrS -Json | ConvertFrom-Json
Check 'Watch-Agy.ps1 -Json returns run summary, liveWorker, recentEvents, and quota' ($wSnap.summary.status -eq 'OK' -and @($wSnap.recentEvents).Count -gt 0 -and @($wSnap.quota).Count -eq 3)
$null = pwsh -NoProfile -File $watch -RunRoot $rrS -Cancel
Check 'Watch-Agy.ps1 -Cancel writes cancel.signal' (Test-Path (Join-Path $rrS 'cancel.signal'))

# 8. Repository-wide backlog discovery, dependency prioritization, scale pagination, reconciliation & pause/resume
$env:MOCK_GH_MODE = 'default'
$env:MOCK_AGY_MODE = 'stream'
$rrB = Join-Path $tmp 'backlog-e2e'
$qpB = Join-Path $rrB 'backlog-queue.json'
$stB = Join-Path $rrB 'state.json'

# Dry run on repository backlog
$bDry = pwsh -NoProfile -File $cliY -AllOpenIssues -Lifecycle -AllowWrites -YoloWorker -AgyCommand $mock -GhCommand $ghm -RunRoot $rrB -DryRun | ConvertFrom-Json
Check 'Backlog dry-run ignores PRs (4 issues in queue, not 5)' ($LASTEXITCODE -eq 0 -and $bDry.stats.totalDiscovered -eq 4)
Check 'Backlog dry-run detects already-completed issue #32 from checked acceptance criteria' ($bDry.stats.completed -eq 1)
Check 'Backlog dry-run blocks #34 (ready-for-human) and initially blocks #40 (Blocked by #33)' ($bDry.stats.blocked -eq 2)
Check 'Backlog dry-run orders #33 first, then unblocks #40 second' ($bDry.executionOrder.Count -eq 2 -and $bDry.executionOrder[0].number -eq 33 -and $bDry.executionOrder[1].number -eq 40)

# Paginated scale discovery (120 issues across 3 pages of 50)
$env:MOCK_GH_MODE = 'scale120'
$smallPagePol = $policy | ConvertTo-Json -Depth 10 | ConvertFrom-Json
$smallPagePol.backlog.pageSize = 50
$qpScale = Join-Path $tmp 'scale-queue.json'
$syncScale = Sync-AgyGitHubBacklog -QueuePath $qpScale -GhCommand $ghm -Policy $smallPagePol -FullSync
Check 'Paginated discovery fetches 120 issues across 3 pages' ($syncScale.totalInQueue -eq 120 -and $syncScale.pagesFetched -eq 3)

# Live mocked execution of backlog queue (#33 -> #40), plus pause/resume/retry controls
$env:MOCK_GH_MODE = 'default'
$bRun = pwsh -NoProfile -File $cliY -AllOpenIssues -Lifecycle -AllowWrites -YoloWorker -AgyCommand $mock -GhCommand $ghm -RunRoot $rrB | ConvertFrom-Json
Check 'Autonomous backlog runner executes #33 and then #40 to completion, leaving #34 blocked' (
    $LASTEXITCODE -eq 0 -and $bRun.status -eq 'OK' -and $bRun.stopReason -eq 'QUEUE_EXHAUSTED' -and
    $bRun.processedIssues.Count -eq 2 -and $bRun.processedIssues[0].number -eq 33 -and $bRun.processedIssues[1].number -eq 40 -and
    $bRun.stats.completed -eq 3 -and $bRun.stats.blocked -eq 1
)

# Pause & Resume controls via Watch-Agy.ps1
$null = pwsh -NoProfile -File $watch -RunRoot $rrB -Pause
Check 'Watch-Agy.ps1 -Pause creates pause.signal' (Test-Path (Join-Path $rrB 'pause.signal'))
$bPaused = pwsh -NoProfile -File $cliY -AllOpenIssues -Lifecycle -AllowWrites -AgyCommand $mock -GhCommand $ghm -RunRoot $rrB | ConvertFrom-Json
Check 'Backlog runner respects pause.signal immediately' ($bPaused.stopReason -eq 'PAUSED' -and $bPaused.stats.runnerState -eq 'PAUSED')
$null = pwsh -NoProfile -File $watch -RunRoot $rrB -Resume
Check 'Watch-Agy.ps1 -Resume clears pause.signal' (-not (Test-Path (Join-Path $rrB 'pause.signal')))

# ---------------------------------------------------------------- 9. GitHub delivery, file ownership, local recovery & #37 resumption ----
$delivTmp = Join-Path $tmp 'delivery-repo'
$null = New-Item -ItemType Directory -Force $delivTmp
$null = git -C $delivTmp init -b main 2>$null
$null = git -C $delivTmp config user.email "test@example.com"
$null = git -C $delivTmp config user.name "Test Orchestrator"
$null = git -C $delivTmp config commit.gpgsign false
$null = git -C $delivTmp config core.autocrlf false
$null = New-Item -ItemType Directory -Force (Join-Path $delivTmp 'scripts')
$null = New-Item -ItemType Directory -Force (Join-Path $delivTmp 'tests')
$null = New-Item -ItemType Directory -Force (Join-Path $delivTmp 'docs')
$null = New-Item -ItemType Directory -Force (Join-Path $delivTmp 'ai_remaster_gui')
Write-AgyUtf8File -Path (Join-Path $delivTmp 'README.md') -Value "# Test Repo`n"
Write-AgyUtf8File -Path (Join-Path $delivTmp 'ai_remaster_gui/dust.py') -Value "# base dust.py`n"
$null = git -C $delivTmp add README.md ai_remaster_gui/dust.py 2>$null
$null = git -C $delivTmp commit -m "chore: initial main commit" 2>$null
$null = git -C $delivTmp checkout -b feature/dirty-worktree 2>$null

# Simulate pre-existing dirty change on ai_remaster_gui/dust.py + clean issue-owned files for #33, #34, #35, #37
Write-AgyUtf8File -Path (Join-Path $delivTmp 'ai_remaster_gui/dust.py') -Value "# base dust.py`n# pre-existing unrelated dirty edit + issue 37 edit`n"
Write-AgyUtf8File -Path (Join-Path $delivTmp 'scripts/benchmark_defect_masks.py') -Value "# issue 33 benchmark`n"
Write-AgyUtf8File -Path (Join-Path $delivTmp 'tests/test_defect_mask_benchmark.py') -Value "# issue 33 test`n"
Write-AgyUtf8File -Path (Join-Path $delivTmp 'docs/archival-ground-truth-protocol.md') -Value "# issue 34 protocol`n"
Write-AgyUtf8File -Path (Join-Path $delivTmp 'scripts/dataset_adapters.py') -Value "# issue 34 adapters`n"
Write-AgyUtf8File -Path (Join-Path $delivTmp 'scripts/temporal_consistency_scorer.py') -Value "# issue 34 scorer`n"
Write-AgyUtf8File -Path (Join-Path $delivTmp 'tests/test_dataset_adapters.py') -Value "# issue 34 adapter tests`n"
Write-AgyUtf8File -Path (Join-Path $delivTmp 'tests/test_temporal_consistency.py') -Value "# issue 34 temporal tests`n"
Write-AgyUtf8File -Path (Join-Path $delivTmp 'scripts/candidate_detector_eval.py') -Value "# issue 35 eval`n"
Write-AgyUtf8File -Path (Join-Path $delivTmp 'docs/defect-detector-empirical-comparison.md') -Value "# issue 35 report`n"
Write-AgyUtf8File -Path (Join-Path $delivTmp 'tests/test_candidate_detectors.py') -Value "# issue 35 test`n"
Write-AgyUtf8File -Path (Join-Path $delivTmp 'scripts/film_defect_detectors.py') -Value "# issue 37 registry`n"
Write-AgyUtf8File -Path (Join-Path $delivTmp 'tests/test_defect_detector_registry.py') -Value "# issue 37 test`n"
Write-AgyUtf8File -Path (Join-Path $delivTmp 'scripts/cleanup_video.py') -Value "# issue 37 cleanup`n"
Write-AgyUtf8File -Path (Join-Path $delivTmp 'scripts/artifact_ids.py') -Value "# issue 37 artifact ids`n"

# 9a. File ownership checks
$own33 = Test-AgyIssueFileOwnership -IssueNumber 33 -Policy $policy -WorkDir $delivTmp
Check 'File ownership allows clean isolated files for #33' ($own33.canCommitSafely -and $own33.cleanOwnedFiles.Count -eq 2 -and $own33.sharedDirtyConflicts.Count -eq 0)
$own34 = Test-AgyIssueFileOwnership -IssueNumber 34 -Policy $policy -WorkDir $delivTmp
Check 'File ownership allows clean isolated files for #34 (5 files)' ($own34.canCommitSafely -and $own34.cleanOwnedFiles.Count -eq 5 -and $own34.sharedDirtyConflicts.Count -eq 0)
$own37 = Test-AgyIssueFileOwnership -IssueNumber 37 -Policy $policy -WorkDir $delivTmp
Check 'File ownership blocks #37 commit due to shared pre-existing dirty file ai_remaster_gui/dust.py' (-not $own37.canCommitSafely -and ($own37.sharedDirtyConflicts -contains 'ai_remaster_gui/dust.py'))

# 9b. Non-destructive isolated branch commit (GIT_INDEX_FILE plumbing)
$bc33_1 = New-AgyIssueBranchCommit -IssueNumber 33 -IssueTitle 'Foundation benchmark harness' -Policy $policy -WorkDir $delivTmp
Check 'New-AgyIssueBranchCommit creates isolated branch agy/issue-33 without switching active branch' (
    $bc33_1.status -eq 'COMMITTED' -and $bc33_1.branch -eq 'agy/issue-33' -and $bc33_1.commitSha -match '^[0-9a-f]{40}$' -and
    ((git -C $delivTmp rev-parse --abbrev-ref HEAD).Trim() -eq 'feature/dirty-worktree')
)
$committedFiles33 = @(git -C $delivTmp diff-tree --no-commit-id --name-only -r $bc33_1.commitSha)
Check 'Isolated commit on agy/issue-33 contains ONLY #33 files and never unrelated dirty files' (
    $committedFiles33.Count -eq 2 -and ($committedFiles33 -contains 'scripts/benchmark_defect_masks.py') -and ($committedFiles33 -notcontains 'ai_remaster_gui/dust.py')
)
$bc33_2 = New-AgyIssueBranchCommit -IssueNumber 33 -IssueTitle 'Foundation benchmark harness' -Policy $policy -WorkDir $delivTmp
Check 'New-AgyIssueBranchCommit is idempotent (UNCHANGED on repeat call with same tree)' ($bc33_2.status -eq 'UNCHANGED' -and $bc33_2.commitSha -eq $bc33_1.commitSha)
$bc37 = New-AgyIssueBranchCommit -IssueNumber 37 -IssueTitle 'Production detector integration' -Policy $policy -WorkDir $delivTmp
Check 'New-AgyIssueBranchCommit refuses #37 with BLOCKED_OWNERSHIP' ($bc37.status -eq 'BLOCKED_OWNERSHIP' -and ($bc37.sharedDirtyConflicts -contains 'ai_remaster_gui/dust.py'))

# 9c. Local completion recovery (Repair-AgyBacklogQueue):
#     - Reconstructs completedPhases strictly from persisted summary.json evidence
#     - Never infers test/review/verify from file existence
#     - Distinguishes IMPLEMENTATION_COMPLETE, VERIFICATION_COMPLETE, and HUMAN_REVIEW_PENDING for externally completed work
#     - Strips fabricated completedPhases when persisted evidence is missing (incomplete evidence)
#     - Checkpoints PARTIAL #37 with exact remaining phases and PAUSED_RUN_BUDGET
$rrRec = Join-Path $delivTmp '.cache/agy-orch'
$qpRec = Join-Path $rrRec 'backlog-queue.json'

# Persisted summary.json for #33 (full 5-phase lifecycle completed)
$run33Dir = Join-Path $rrRec 'run-33'
$null = New-Item -ItemType Directory -Force $run33Dir
Write-AgyUtf8File -Path (Join-Path $run33Dir 'summary.json') -Value (@{
    runId       = 'run-33'
    status      = 'OK'
    totalTokens = 381095
    notes       = @("review phase returned VERDICT: APPROVED; skipped task 5 (corrections)")
    results     = @(
        @{ index = 1; taskType = 'investigation'; status = 'SUCCESS'; conversationId = 'conv-33-1'; response = 'ISSUE_STATUS: ACTIONABLE'; modifiedFiles = @() },
        @{ index = 2; taskType = 'implement';     status = 'SUCCESS'; conversationId = 'conv-33-2'; response = 'Implemented benchmark harness.'; modifiedFiles = @('scripts/benchmark_defect_masks.py') },
        @{ index = 3; taskType = 'test';          status = 'SUCCESS'; conversationId = 'conv-33-3'; response = 'All benchmark tests passed.'; modifiedFiles = @('tests/test_defect_mask_benchmark.py') },
        @{ index = 4; taskType = 'review';        status = 'SUCCESS'; conversationId = 'conv-33-4'; response = 'VERDICT: APPROVED'; modifiedFiles = @() },
        @{ index = 6; taskType = 'verify';        status = 'SUCCESS'; conversationId = 'conv-33-6'; response = 'VERIFICATION: PASSED'; modifiedFiles = @() }
    )
} | ConvertTo-Json -Depth 8)

# Persisted summary.json for #35 (full 5-phase lifecycle completed)
$run35Dir = Join-Path $rrRec 'run-35'
$null = New-Item -ItemType Directory -Force $run35Dir
Write-AgyUtf8File -Path (Join-Path $run35Dir 'summary.json') -Value (@{
    runId       = 'run-35'
    status      = 'OK'
    totalTokens = 521434
    notes       = @("review phase returned VERDICT: APPROVED; skipped task 5 (corrections)")
    results     = @(
        @{ index = 1; taskType = 'investigation'; status = 'SUCCESS'; conversationId = 'conv-35-1'; response = 'ISSUE_STATUS: ACTIONABLE'; modifiedFiles = @() },
        @{ index = 2; taskType = 'implement';     status = 'SUCCESS'; conversationId = 'conv-35-2'; response = 'Implemented candidate detector eval.'; modifiedFiles = @('scripts/candidate_detector_eval.py', 'docs/defect-detector-empirical-comparison.md') },
        @{ index = 3; taskType = 'test';          status = 'SUCCESS'; conversationId = 'conv-35-3'; response = 'All candidate detector tests passed.'; modifiedFiles = @('tests/test_candidate_detectors.py') },
        @{ index = 4; taskType = 'review';        status = 'SUCCESS'; conversationId = 'conv-35-4'; response = 'VERDICT: APPROVED'; modifiedFiles = @() },
        @{ index = 6; taskType = 'verify';        status = 'SUCCESS'; conversationId = 'conv-35-6'; response = 'VERIFICATION: PASSED'; modifiedFiles = @() }
    )
} | ConvertTo-Json -Depth 8)

# Persisted summary.json for #36 (reconciled ALREADY_COMPLETED by investigation phase)
$run36Dir = Join-Path $rrRec 'run-36'
$null = New-Item -ItemType Directory -Force $run36Dir
Write-AgyUtf8File -Path (Join-Path $run36Dir 'summary.json') -Value (@{
    runId       = 'run-36'
    status      = 'ALREADY_COMPLETED'
    totalTokens = 169220
    notes       = @("task 1 (investigation) reconciled issue as ALREADY_COMPLETED")
    results     = @(
        @{ index = 1; taskType = 'investigation'; status = 'SUCCESS'; conversationId = 'conv-36-1'; response = 'ISSUE_STATUS: ALREADY_COMPLETED'; modifiedFiles = @() }
    )
} | ConvertTo-Json -Depth 8)

# Persisted summary.json for #37 (partial lifecycle: investigation + implement succeeded, stopped by internal run budget)
# Note: tests/test_defect_detector_registry.py ALREADY exists on disk in $delivTmp, so we verify that `test` is NEVER inferred from file existence!
$run37Dir = Join-Path $rrRec '20261109-185511-763b1a'
$null = New-Item -ItemType Directory -Force $run37Dir
Write-AgyUtf8File -Path (Join-Path $run37Dir 'summary.json') -Value (@{
    runId       = '20261109-185511-763b1a'
    status      = 'BUDGET_EXHAUSTED'
    budgetKind  = 'INTERNAL_RUN_BUDGET'
    totalTokens = 445636
    notes       = @("group 'argon' per-run token budget reached before task 3")
    results     = @(
        @{ index = 1; taskType = 'investigation'; status = 'SUCCESS'; conversationId = 'conv-37-inv'; response = 'ISSUE_STATUS: ACTIONABLE'; modifiedFiles = @() },
        @{ index = 2; taskType = 'implement';     status = 'SUCCESS'; conversationId = 'c6be0f3d-d6bc-4787-87da-031e7c02a419'; response = 'Implemented scripts/film_defect_detectors.py and registry.'; modifiedFiles = @('scripts/film_defect_detectors.py', 'tests/test_defect_detector_registry.py') }
    )
} | ConvertTo-Json -Depth 8)

# Include:
# - #34: externally implemented (5 owned files on disk, no summary.json, no missing target files) -> IMPLEMENTATION_COMPLETE
# - #38: externally implemented on disk, but gated by missing human-curated target artifact (.cache/dust-real-annotation-20261106-01/manifest.json) -> HUMAN_REVIEW_PENDING
# - #39: incomplete/fabricated evidence (claims status=COMPLETED and 5 completedPhases in queue JSON, but has NO summary.json and no owned files on disk) -> downgraded to PENDING (INCOMPLETE_EVIDENCE)
Write-AgyUtf8File -Path $qpRec -Value (@{
    lastSyncAt      = '2026-11-09T18:55:08Z'
    runnerState     = 'BLOCKED_QUOTA_OR_AUTH'
    activeIssue     = $null
    totalDiscovered = 7
    issues          = @{
        '33' = @{ number = 33; title = 'Benchmark harness'; state = 'open'; labels = @('ready-for-agent'); priority = 75; phaseNumber = 1; dependencies = @(); targetFiles = @(); status = 'COMPLETED'; currentPhase = 'done'; lastRunId = 'run-33'; completedPhases = @('investigation','implement','test','review','verify'); tokensUsed = 381095; modifiedFiles = @('scripts/benchmark_defect_masks.py','tests/test_defect_mask_benchmark.py') }
        '34' = @{ number = 34; title = 'Ground truth dataset'; state = 'open'; labels = @('ready-for-agent'); priority = 80; phaseNumber = 2; dependencies = @(33); targetFiles = @(); status = 'COMPLETED'; currentPhase = 'done'; lastRunId = $null; completedPhases = @('investigation','implement','test','review','verify'); tokensUsed = 0; modifiedFiles = @() }
        '35' = @{ number = 35; title = 'Candidate comparison'; state = 'open'; labels = @('ready-for-agent'); priority = 85; phaseNumber = 3; dependencies = @(33); targetFiles = @(); status = 'COMPLETED'; currentPhase = 'done'; lastRunId = 'run-35'; completedPhases = @('investigation','implement','test','review','verify'); tokensUsed = 521434; modifiedFiles = @('scripts/dataset_adapters.py','scripts/candidate_detector_eval.py','docs/maestro/plans/x.md') }
        '36' = @{ number = 36; title = 'License audit'; state = 'open'; labels = @('ready-for-agent'); priority = 90; phaseNumber = 4; dependencies = @(35); targetFiles = @(); status = 'ALREADY_COMPLETED'; currentPhase = 'reconciled'; lastRunId = 'run-36'; completedPhases = @('investigation'); tokensUsed = 169220; modifiedFiles = @() }
        '37' = @{ number = 37; title = 'Production detector integration'; state = 'open'; labels = @('ready-for-agent'); priority = 95; phaseNumber = 5; dependencies = @(36); targetFiles = @(); status = 'PENDING'; currentPhase = $null; lastRunId = '20261109-185511-763b1a'; completedPhases = @('investigation','implement','test','review','verify'); tokensUsed = 445636; modifiedFiles = @('') }
        '38' = @{ number = 38; title = 'Human-curated pilot signoff'; state = 'open'; labels = @('ready-for-agent'); priority = 96; phaseNumber = 2; dependencies = @(33); targetFiles = @('docs/archival-ground-truth-protocol.md', '.cache/dust-real-annotation-20261106-01/manifest.json'); status = 'PENDING'; currentPhase = $null; lastRunId = $null; completedPhases = @(); tokensUsed = 0; modifiedFiles = @('docs/archival-ground-truth-protocol.md') }
        '39' = @{ number = 39; title = 'Fabricated completion without evidence'; state = 'open'; labels = @('ready-for-agent'); priority = 98; phaseNumber = 6; dependencies = @(); targetFiles = @('scripts/nonexistent_module.py'); status = 'COMPLETED'; currentPhase = 'done'; lastRunId = 'missing-run-39'; completedPhases = @('investigation','implement','test','review','verify'); tokensUsed = 12345; modifiedFiles = @('scripts/nonexistent_module.py') }
    }
} | ConvertTo-Json -Depth 10)

# Verify -DryRun does NOT write any file
$repDry = Repair-AgyBacklogQueue -QueuePath $qpRec -RunRoot $rrRec -Policy $policy -WorkDir $delivTmp -DryRun
Check 'Repair-AgyBacklogQueue -DryRun writes zero files' ($repDry.dryRun -and @($repDry.writtenFiles).Count -eq 0)

# Real local recovery writes ONLY .cache/agy-orch/backlog-queue.json
$repOut = Repair-AgyBacklogQueue -QueuePath $qpRec -RunRoot $rrRec -Policy $policy -WorkDir $delivTmp
$qRep = Get-AgyBacklogQueue -QueuePath $qpRec
Check 'Repair-AgyBacklogQueue writes ONLY .cache/agy-orch/backlog-queue.json' (
    @($repOut.writtenFiles).Count -eq 1 -and $repOut.writtenFiles[0] -eq '.cache/agy-orch/backlog-queue.json'
)
Check 'Repair-AgyBacklogQueue marks externally implemented #34 as IMPLEMENTATION_COMPLETE and strips fabricated completedPhases to @()' (
    $qRep.issues['34'].status -eq 'IMPLEMENTATION_COMPLETE' -and
    $qRep.issues['34'].verificationState -eq 'IMPLEMENTATION_COMPLETE' -and
    $qRep.issues['34'].implementationComplete -eq $true -and
    $qRep.issues['34'].verificationComplete -eq $false -and
    $qRep.issues['34'].humanReviewPending -eq $false -and
    @($qRep.issues['34'].completedPhases).Count -eq 0 -and
    $qRep.issues['34'].modifiedFiles.Count -eq 5
)
Check 'Repair-AgyBacklogQueue marks #38 (externally implemented on disk but missing human-curation target artifact) as HUMAN_REVIEW_PENDING with empty completedPhases' (
    $qRep.issues['38'].status -eq 'HUMAN_REVIEW_PENDING' -and
    $qRep.issues['38'].verificationState -eq 'HUMAN_REVIEW_PENDING' -and
    $qRep.issues['38'].implementationComplete -eq $true -and
    $qRep.issues['38'].verificationComplete -eq $false -and
    $qRep.issues['38'].humanReviewPending -eq $true -and
    @($qRep.issues['38'].completedPhases).Count -eq 0
)
Check 'Repair-AgyBacklogQueue strips fabricated completedPhases on #39 (incomplete evidence) and downgrades status from COMPLETED to PENDING' (
    $qRep.issues['39'].status -eq 'PENDING' -and
    $qRep.issues['39'].verificationState -eq 'INCOMPLETE_EVIDENCE' -and
    $qRep.issues['39'].implementationComplete -eq $false -and
    $qRep.issues['39'].verificationComplete -eq $false -and
    @($qRep.issues['39'].completedPhases).Count -eq 0 -and
    @($qRep.issues['39'].remainingPhases).Count -eq 6
)
Check 'Repair-AgyBacklogQueue cleans #35 modifiedFiles to strictly owned #35 files and verifies VERIFICATION_COMPLETE from summary.json' (
    $qRep.issues['35'].status -eq 'COMPLETED' -and
    $qRep.issues['35'].verificationState -eq 'VERIFICATION_COMPLETE' -and
    $qRep.issues['35'].verificationComplete -eq $true -and
    $qRep.issues['35'].modifiedFiles.Count -eq 3 -and
    ($qRep.issues['35'].modifiedFiles -notcontains 'scripts/dataset_adapters.py')
)
Check 'Repair-AgyBacklogQueue never infers test/review/verify from test file existence on #37; checkpoints #37 as PARTIAL with completed=(investigation,implement) and remaining=(test,review,corrections,verify)' (
    $qRep.issues['37'].status -eq 'PARTIAL' -and
    $qRep.issues['37'].verificationState -eq 'PARTIAL_LIFECYCLE' -and
    $qRep.issues['37'].implementationComplete -eq $true -and
    $qRep.issues['37'].verificationComplete -eq $false -and
    (($qRep.issues['37'].completedPhases -join ',') -eq 'investigation,implement') -and
    (($qRep.issues['37'].remainingPhases -join ',') -eq 'test,review,corrections,verify') -and
    $qRep.issues['37'].conversationId -eq 'c6be0f3d-d6bc-4787-87da-031e7c02a419'
)
Check 'Repair-AgyBacklogQueue distinguishes internal run token budget (runnerState = PAUSED_RUN_BUDGET)' ($qRep.runnerState -eq 'PAUSED_RUN_BUDGET')

# Verify external verification manifest ($RunRoot/external-evidence.json) upgrades #34 from IMPLEMENTATION_COMPLETE to VERIFICATION_COMPLETE (COMPLETED) while keeping completedPhases = @()
Write-AgyUtf8File -Path (Join-Path $rrRec 'external-evidence.json') -Value (@{
    '34' = @{
        testsPassed        = $true
        reviewApproved     = $true
        verificationPassed = $true
        humanReviewPending = $false
    }
} | ConvertTo-Json -Depth 6)
$repExt = Repair-AgyBacklogQueue -QueuePath $qpRec -RunRoot $rrRec -Policy $policy -WorkDir $delivTmp
$qRepExt = Get-AgyBacklogQueue -QueuePath $qpRec
Check 'Persisted external-evidence.json upgrades externally completed #34 to VERIFICATION_COMPLETE (COMPLETED) without fabricating orchestrator completedPhases' (
    $qRepExt.issues['34'].status -eq 'COMPLETED' -and
    $qRepExt.issues['34'].verificationState -eq 'VERIFICATION_COMPLETE' -and
    $qRepExt.issues['34'].implementationComplete -eq $true -and
    $qRepExt.issues['34'].verificationComplete -eq $true -and
    $qRepExt.issues['34'].humanReviewPending -eq $false -and
    @($qRepExt.issues['34'].completedPhases).Count -eq 0 -and
    @($qRepExt.issues['34'].remainingPhases).Count -eq 0
)

# 9d. Idempotent GitHub delivery synchronization (comments, labels, branches, Draft PRs)
$ghStateFile = Join-Path $tmp 'mock-gh-state.json'
$env:MOCK_GH_STATE_FILE = $ghStateFile
$sync1 = Sync-AgyGitHubDelivery -QueuePath $qpRec -IssueNumbers @(33, 34, 35, 37) -Policy $policy -GhCommand $ghm -WorkDir $delivTmp `
    -EnableComments -EnableLabels -EnableBranchCommit -EnableDraftPr
$item33_1 = $sync1.items | Where-Object issueNumber -eq 33
$item37_1 = $sync1.items | Where-Object issueNumber -eq 37
Check 'First GitHub sync commits branch agy/issue-33, creates Draft PR, posts comment, and adds label' (
    $item33_1.branchAction -in @('COMMITTED', 'UNCHANGED') -and
    $item33_1.prAction -eq 'CREATED_DRAFT_PR' -and
    $item33_1.prUrl -match '/pull/201$' -and
    $item33_1.commentAction -eq 'POSTED' -and
    $item33_1.labelAction -match '^ADDED:'
)
Check 'First GitHub sync posts PARTIAL progress comment on #37 but skips branch/PR creation' (
    $item37_1.localStatus -eq 'PARTIAL' -and
    $item37_1.branchAction -eq 'SKIPPED_STATUS_PARTIAL' -and
    $item37_1.prAction -eq 'SKIPPED' -and
    $item37_1.commentAction -eq 'POSTED' -and
    $item37_1.labelAction -eq 'ADDED:partial-progress'
)
Check 'GitHub delivery sync never allows auto-merge or auto-close' (-not $sync1.autoMergeAllowed -and -not $sync1.autoCloseAllowed)

# Run Sync-AgyGitHubDelivery a second time -> must be 100% idempotent (0 duplicate comments, 0 duplicate PRs, 0 duplicate commits)
$sync2 = Sync-AgyGitHubDelivery -QueuePath $qpRec -IssueNumbers @(33, 34, 35, 37) -Policy $policy -GhCommand $ghm -WorkDir $delivTmp `
    -EnableComments -EnableLabels -EnableBranchCommit -EnableDraftPr
$item33_2 = $sync2.items | Where-Object issueNumber -eq 33
$item37_2 = $sync2.items | Where-Object issueNumber -eq 37
$ghState = Get-Content $ghStateFile -Raw -Encoding utf8 | ConvertFrom-Json -AsHashtable
Check 'Second GitHub sync skips duplicate comments, PRs, commits, and labels' (
    $item33_2.branchAction -eq 'UNCHANGED' -and
    $item33_2.prAction -eq 'SKIPPED_EXISTING_PR' -and
    $item33_2.commentAction -eq 'SKIPPED_DUPLICATE' -and
    $item33_2.labelAction -eq 'SKIPPED_UNCHANGED' -and
    $item37_2.commentAction -eq 'SKIPPED_DUPLICATE' -and
    @($ghState.comments['33']).Count -eq 1 -and
    @($ghState.comments['37']).Count -eq 1 -and
    @($ghState.prs).Count -eq 3
)

# 9e. Resuming PARTIAL Issue #37 executes ONLY remaining phases (test, review, verify) and chains prior implement summary
$env:MOCK_AGY_MODE = 'stream'; $env:MOCK_AGY_LOG = Join-Path $tmp 'resume-37.log'
$st37 = Join-Path $rrRec 'state.json'
$res37 = pwsh -NoProfile -File $cliY -Issue 37 -Lifecycle -AllowWrites -YoloWorker -AgyCommand $mock -GhCommand $ghm -RunRoot $rrRec -QueuePath $qpRec -StatePath $st37 | ConvertFrom-Json
Check 'Resuming PARTIAL #37 executes only remaining phases (test, review, verify; corrections skipped on APPROVED) and not investigation/implement' (
    $LASTEXITCODE -eq 0 -and $res37.status -eq 'OK' -and $res37.results.Count -eq 3 -and
    $res37.results[0].taskType -eq 'test' -and
    $res37.results[1].taskType -eq 'review' -and
    $res37.results[2].taskType -eq 'verify'
)
$handoff37Test = Get-Content (ResolveTestPath $res37.results[0].handoff) -Raw -Encoding utf8
Check 'Resumed #37 test phase handoff carries prior implement phase summary and modifiedFiles' (
    $handoff37Test -match 'Implemented scripts/film_defect_detectors\.py' -and
    $handoff37Test -match 'scripts/film_defect_detectors\.py'
)
Remove-Item Env:\MOCK_GH_STATE_FILE -ErrorAction SilentlyContinue

Remove-Item -Recurse -Force $tmp
Write-Host "`nFailures: $script:fail"
exit $script:fail
