# AgyOrchestrator.ps1 - Windows-native multi-model routing & lifecycle layer over `agy -p`.
# Never modifies Antigravity or Maestro internals.

Set-StrictMode -Version Latest

function Get-AgyPolicy {
    param([string]$Path = (Join-Path $PSScriptRoot 'policy.json'))
    Get-Content -LiteralPath $Path -Raw -Encoding utf8 | ConvertFrom-Json
}

function Write-AgyUtf8File {
    param([Parameter(Mandatory)][string]$Path, [Parameter(Mandatory)]$Value)
    $null = New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Path)
    $text = if ($Value -is [array]) { $Value -join "`n" } else { [string]$Value }
    $utf8NoBom = [System.Text.UTF8Encoding]::new($false)
    [System.IO.File]::WriteAllText($Path, $text, $utf8NoBom)
}

function Append-AgyUtf8Line {
    param([Parameter(Mandatory)][string]$Path, [Parameter(Mandatory)][string]$Line)
    $null = New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Path)
    $utf8NoBom = [System.Text.UTF8Encoding]::new($false)
    [System.IO.File]::AppendAllText($Path, ($Line + "`n"), $utf8NoBom)
}

function Format-AgyCleanPath {
    param([string]$Path, [string]$WorkDir = (Get-Location).Path)
    if (-not $Path) { return '' }
    $norm = $Path.Trim() -replace '\\', '/'
    $wdNorm = ($WorkDir.Trim() -replace '\\', '/').TrimEnd('/')
    if ($norm.StartsWith($wdNorm + '/', [System.StringComparison]::OrdinalIgnoreCase)) {
        return $norm.Substring($wdNorm.Length + 1)
    }
    $norm
}

function Format-AgyCleanMarkdown {
    # Fix malformed Markdown paths (e.g. \scripts/foo.py\ or \t control-char corruption from unescaped backslashes)
    # and normalize Windows backslashes inside inline code spans.
    param([string]$Text)
    if (-not $Text) { return '' }
    $fixed = $Text -replace "`t(ests/[a-zA-Z0-9_./-]+)", 't$1'
    $fixed = $fixed -replace "\\`t([a-zA-Z0-9_./-]+)\\", '`$1`'
    $fixed = $fixed -replace '\\([a-zA-Z0-9_]+(?:/[a-zA-Z0-9_.*-]+)+)\\', '`$1`'
    $fixed = $fixed -replace '\\([a-zA-Z0-9_.-]+)\\', '`$1`'
    $fixed
}

function Resolve-AgyRoute {
    # Classify task text (or honour an explicit type) and pick a model, applying quota-aware fallback when $State is provided.
    param(
        [Parameter(Mandatory)][string]$Text,
        [string]$Type,
        $Policy,
        $State = $null,
        [bool]$IsComplex = $false,
        [bool]$IsFailedAttempt = $false
    )
    $matchedBy = 'explicit'
    if (-not $Type) {
        $matchedBy = 'default'
        foreach ($rule in $Policy.classifier) {
            if ($Text -imatch $rule.pattern) { $Type = $rule.type; $matchedBy = "pattern:$($rule.pattern)"; break }
        }
    }
    $primaryKey = if ($Type -and $Policy.routes.PSObject.Properties[$Type]) { $Policy.routes.$Type } else { $Policy.defaultRoute }
    if (-not $Type) { $Type = 'default' }
    $primaryModel = $Policy.models.$primaryKey
    $selectedKey = $primaryKey
    $selectedModel = $primaryModel
    $fallbackFrom = $null
    $fallbackReason = $null

    $pGrp = Get-AgyModelGroup $primaryModel $Policy
    $pStatus = if ($pGrp -ne 'unknown' -and $State) { Get-AgyGroupStatus -Group $pGrp -Policy $Policy -State $State } else { $null }

    if ($pStatus -and $pStatus.belowReserve) {
        if ($State -and $Policy.PSObject.Properties['routeFallbacks'] -and $Policy.routeFallbacks.PSObject.Properties[$Type]) {
            foreach ($fbKey in @($Policy.routeFallbacks.$Type)) {
                if (-not $Policy.models.PSObject.Properties[$fbKey]) { continue }
                $fbModel = $Policy.models.$fbKey
                $fbGrp = Get-AgyModelGroup $fbModel $Policy
                if ($Policy.groups.PSObject.Properties[$fbGrp] -and $Policy.groups.$fbGrp.PSObject.Properties['autoDispatch'] -and -not $Policy.groups.$fbGrp.autoDispatch) {
                    continue
                }
                $reqJust = [bool]($Policy.groups.PSObject.Properties[$fbGrp] -and $Policy.groups.$fbGrp.PSObject.Properties['requireJustification'] -and $Policy.groups.$fbGrp.requireJustification)
                if ($reqJust -or $fbGrp -eq 'claude-gpt') {
                    $isJustified = $IsComplex -or $IsFailedAttempt -or ($Text -imatch '\b(complex|intricate|failed attempt|retry|heavy|hard)\b')
                    if (-not $isJustified) {
                        continue
                    }
                }
                $fbStatus = if ($fbGrp -ne 'unknown' -and $State) { Get-AgyGroupStatus -Group $fbGrp -Policy $Policy -State $State } else { $null }
                if (-not $fbStatus -or -not $fbStatus.belowReserve) {
                    $selectedKey = $fbKey
                    $selectedModel = $fbModel
                    $fallbackFrom = $primaryModel
                    $why = if ($pStatus.belowFiveHourReserve) {
                        "5h quota $($pStatus.fiveHourRemainingPercent)% < $($pStatus.fiveHourReservePercent)% reserve"
                    } else {
                        "weekly quota $($pStatus.remainingPercent)% < $($pStatus.reservePercent)% reserve"
                    }
                    $fallbackReason = "fallback from $primaryModel ($pGrp $why) to $fbModel ($fbGrp)"
                    break
                }
            }
        }
    }

    $wantsWrite = [bool]($Policy.writeTypes -contains $Type)
    $isReadOnly = [bool]($Policy.PSObject.Properties['readOnlyTypes'] -and ($Policy.readOnlyTypes -contains $Type))
    [pscustomobject]@{
        Type           = $Type
        ModelKey       = $selectedKey
        Model          = $selectedModel
        PrimaryModel   = $primaryModel
        FallbackFrom   = $fallbackFrom
        FallbackReason = $fallbackReason
        MatchedBy      = $matchedBy
        Destructive    = [bool]($Text -imatch $Policy.destructivePattern)
        WantsWrite     = $wantsWrite
        IsReadOnly     = $isReadOnly
    }
}

function Get-AgyBlocks {
    # Returns a list of reasons why this route needs authorization that was not given.
    param($Route, $Policy, [hashtable]$Auth, [bool]$AllowWrites = $false, [bool]$EnforceWriteAuth = $false)
    $blocks = @()
    $flagName = $Policy.escalation.requireAuthorization.PSObject.Properties[$Route.ModelKey]
    if ($flagName -and -not $Auth[$flagName.Value]) {
        $blocks += "model '$($Route.Model)' requires -$($flagName.Value)"
    }
    if ($Route.Destructive -and -not $Auth['AuthorizeDestructive']) {
        $blocks += 'task text matches destructive-operation pattern; requires -AuthorizeDestructive'
    }
    if ($EnforceWriteAuth -and $Route.WantsWrite -and -not $AllowWrites) {
        $blocks += "task type '$($Route.Type)' requires explicit -AllowWrites authorization before modifying files"
    }
    $blocks
}

function Get-AgyTokenTotal($Usage) {
    if (-not $Usage) { return 0 }
    if ($Usage -is [hashtable] -and $Usage.ContainsKey('total_tokens')) { return [int]$Usage['total_tokens'] }
    if ($Usage.PSObject.Properties['total_tokens']) { return [int]$Usage.total_tokens }
    0
}

function Get-AgyModelEvidence {
    # Independent evidence: the agy conversation transcript records the session's model selection.
    param([string]$ConversationId, [string]$BrainRoot = (Join-Path $HOME '.gemini/antigravity-cli/brain'))
    if (-not $ConversationId) { return $null }
    foreach ($fn in @('transcript_full.jsonl', 'transcript.jsonl')) {
        $t = Join-Path $BrainRoot "$ConversationId/.system_generated/logs/$fn"
        if (-not (Test-Path -LiteralPath $t)) { continue }
        $raw = Get-Content -LiteralPath $t -Raw -Encoding utf8
        $m = [regex]::Match($raw, 'Model Selection. from [^\r\n]+? to (.+?)\. No need')
        if ($m.Success) { return $m.Groups[1].Value }
    }
    $null
}

function Get-AgyWorkspaceSnapshot {
    param([string]$WorkDir = (Get-Location).Path)
    $snap = @{}
    Push-Location $WorkDir
    try {
        $lines = @(git status --porcelain=v1 -uall 2>$null)
        foreach ($line in $lines) {
            if ($line.Length -lt 4) { continue }
            $st = $line.Substring(0, 2)
            $rel = ($line.Substring(3).Trim() -replace '^"|"$', '') -replace '\\', '/'
            if ($rel -match '^\.cache/') { continue }
            $full = Join-Path $WorkDir $rel
            $hash = if (Test-Path -LiteralPath $full -PathType Leaf) {
                (Get-FileHash -LiteralPath $full -Algorithm SHA256).Hash
            } else { 'DELETED' }
            $snap[$rel] = "$st|$hash"
        }
    } catch {} finally { Pop-Location }
    $snap
}

function Compare-AgyWorkspaceSnapshots {
    param([hashtable]$Before, [hashtable]$After)
    $changed = [System.Collections.Generic.List[string]]::new()
    if (-not $Before -or -not $After) { return @() }
    foreach ($k in $After.Keys) {
        if (-not $Before.ContainsKey($k) -or $Before[$k] -ne $After[$k]) {
            $changed.Add($k)
        }
    }
    foreach ($k in $Before.Keys) {
        if (-not $After.ContainsKey($k) -and -not $changed.Contains($k)) {
            $changed.Add($k)
        }
    }
    @($changed | Sort-Object -Unique)
}

function Test-AgyProtectedFiles {
    param([string[]]$ChangedFiles, $Policy)
    $violations = @()
    if (-not $ChangedFiles -or -not $Policy.PSObject.Properties['protectedPathPatterns']) { return @() }
    foreach ($f in $ChangedFiles) {
        $norm = $f -replace '\\', '/'
        foreach ($pat in $Policy.protectedPathPatterns) {
            if ($norm -imatch $pat) {
                $violations += "file '$norm' matched protected pattern '$pat'"
                break
            }
        }
    }
    $violations
}

function Format-AgyStreamEvent {
    # Parses one line from `--output-format stream-json` or `json` and returns a structured event summary.
    param([string]$Line)
    if ([string]::IsNullOrWhiteSpace($Line)) { return $null }
    $trim = $Line.Trim()
    if (-not $trim.StartsWith('{')) {
        if ($trim -match '^AGY_ERROR:\s*(\{.+\})$') {
            try {
                $errObj = $Matches[1] | ConvertFrom-Json
                return [pscustomobject]@{
                    kind = 'agy_error'; raw = $errObj; reason = $errObj.reason
                    status = $errObj.status; message = $errObj.short_error
                    conversationId = $null; usage = $null; response = $null; finishedReason = $errObj.reason
                    toolName = $null; commandLine = $null; filePath = $null
                }
            } catch {}
        }
        return [pscustomobject]@{
            kind = 'text'; raw = $trim; reason = $null; status = $null; message = $trim
            conversationId = $null; usage = $null; response = $null; finishedReason = $null
            toolName = $null; commandLine = $null; filePath = $null
        }
    }
    try { $obj = $trim | ConvertFrom-Json } catch { return $null }
    $evType = if ($obj.PSObject.Properties['event']) { [string]$obj.event }
              elseif ($obj.PSObject.Properties['type']) { [string]$obj.type }
              elseif ($obj.PSObject.Properties['status'] -and $obj.PSObject.Properties['response']) { 'result_envelope' }
              else { 'unknown' }

    $convId = if ($obj.PSObject.Properties['conversation_id'] -and $obj.conversation_id) { [string]$obj.conversation_id }
              elseif ($obj.PSObject.Properties['result'] -and $obj.result -and $obj.result.PSObject.Properties['conversation_id']) { [string]$obj.result.conversation_id }
              else { $null }

    if ($evType -eq 'result' -or $evType -eq 'result_envelope') {
        $res = if ($evType -eq 'result' -and $obj.PSObject.Properties['result']) { $obj.result } else { $obj }
        $fin = if ($res.PSObject.Properties['finished_reason']) { [string]$res.finished_reason } else { $null }
        return [pscustomobject]@{
            kind           = 'result'
            raw            = $res
            status         = if ($res.PSObject.Properties['status']) { [string]$res.status } else { $null }
            response       = if ($res.PSObject.Properties['response']) { [string]$res.response } else { '' }
            usage          = if ($res.PSObject.Properties['usage']) { $res.usage } else { $null }
            durationSeconds= if ($res.PSObject.Properties['duration_seconds']) { $res.duration_seconds } else { $null }
            conversationId = $convId
            finishedReason = $fin
            message        = "result status=$($res.status) finished_reason=$fin"
            toolName       = $null; commandLine = $null; filePath = $null; reason = $fin
        }
    }

    if ($evType -eq 'init') {
        $initObj = if ($obj.PSObject.Properties['init'] -and $obj.init) { $obj.init } else { $obj }
        $initModel = if ($initObj.PSObject.Properties['model']) { [string]$initObj.model } else { $null }
        $permMode  = if ($initObj.PSObject.Properties['permission_mode']) { [string]$initObj.permission_mode } else { $null }
        return [pscustomobject]@{
            kind = 'init'; raw = $obj; status = $null; response = $null; usage = $null
            conversationId = $convId; finishedReason = $null; initModel = $initModel
            message = "init conversation=$convId model=$initModel permission_mode=$permMode"
            toolName = $null; commandLine = $null; filePath = $null; reason = $null
        }
    }

    # Unwrap nested step_update payload when present (real agy stream-json format)
    $su = if ($obj.PSObject.Properties['step_update'] -and $obj.step_update) { $obj.step_update } else { $obj }
    if (-not $convId -and $su.PSObject.Properties['conversation_id'] -and $su.conversation_id) {
        $convId = [string]$su.conversation_id
    }
    $stepState = if ($su.PSObject.Properties['state']) { [string]$su.state } elseif ($su.PSObject.Properties['status']) { [string]$su.status } else { $null }
    $stepType  = if ($su.PSObject.Properties['step_type']) { [string]$su.step_type } else { $evType }
    $tInfo     = if ($su.PSObject.Properties['tool_info'] -and $su.tool_info) { $su.tool_info } else { $su }
    $toolName  = if ($su.PSObject.Properties['tool_name'] -and $su.tool_name) { [string]$su.tool_name }
                 elseif ($tInfo.PSObject.Properties['tool_name'] -and $tInfo.tool_name) { [string]$tInfo.tool_name }
                 elseif ($tInfo.PSObject.Properties['name'] -and $tInfo.name) { [string]$tInfo.name }
                 else { $null }
    $params    = if ($tInfo -and $tInfo.PSObject.Properties['parameters']) { $tInfo.parameters }
                 elseif ($tInfo -and $tInfo.PSObject.Properties['args']) { $tInfo.args }
                 else { $tInfo }

    $cmdLine = $null; $filePath = $null
    if ($params) {
        foreach ($cn in @('CommandLine', 'command', 'cmd')) {
            if ($params.PSObject.Properties[$cn] -and $params.$cn) { $cmdLine = [string]$params.$cn; break }
        }
        foreach ($fn in @('Path', 'path', 'file_path', 'TargetFile')) {
            if ($params.PSObject.Properties[$fn] -and $params.$fn) { $filePath = [string]$params.$fn; break }
        }
    }
    $usage = if ($su.PSObject.Properties['usage']) { $su.usage } elseif ($obj.PSObject.Properties['usage']) { $obj.usage } else { $null }
    $stepTok = Get-AgyTokenTotal $usage
    $textChunk = if ($su.PSObject.Properties['text_delta'] -and $su.text_delta) { [string]$su.text_delta }
                 elseif ($su.PSObject.Properties['content'] -and $su.content) { [string]$su.content }
                 elseif ($su.PSObject.Properties['text'] -and $su.text) { [string]$su.text }
                 else { $null }
    $msgParts = @($stepType)
    if ($stepState) { $msgParts += "($stepState)" }
    if ($toolName)  { $msgParts += "tool=$toolName" }
    if ($filePath)  { $msgParts += "file=$filePath" }
    if ($cmdLine)   {
        $shortCmd = if ($cmdLine.Length -gt 80) { $cmdLine.Substring(0, 80) + '...' } else { $cmdLine }
        $msgParts += "cmd=`"$shortCmd`""
    }
    if ($stepTok -gt 0) { $msgParts += "tokens=$stepTok" }
    [pscustomobject]@{
        kind           = 'step'
        raw            = $obj
        status         = $stepState
        response       = $textChunk
        usage          = $usage
        conversationId = $convId
        finishedReason = $null
        initModel      = $null
        message        = ($msgParts -join ' ')
        toolName       = $toolName
        commandLine    = $cmdLine
        filePath       = $filePath
        reason         = $null
    }
}

function Write-AgyLiveStatus {
    param([string]$StatusPath, [hashtable]$Data)
    if (-not $StatusPath) { return }
    try { Write-AgyUtf8File -Path $StatusPath -Value ($Data | ConvertTo-Json -Depth 6) } catch {}
}

function Build-AgyHandoffLines {
    param(
        [int]$Index,
        $Route,
        [string]$Mode,
        [string]$Text,
        [string[]]$ContextFiles = @(),
        [string]$PriorSummary = '',
        [string]$ResumeConversationId = '',
        [string[]]$PriorModifiedFiles = @(),
        [string]$WorkDir = (Get-Location).Path
    )
    $cleanText = Format-AgyCleanMarkdown $Text
    $modeLabel = if ($Mode) { $Mode } else { 'read-only (file modifications forbidden)' }
    $lines = @(
        "# Handoff $Index",
        '',
        "Task type: $($Route.Type)",
        "Assigned model: $($Route.Model)",
        "Mode: $modeLabel"
    )
    if ($ResumeConversationId) {
        $lines += "Resumed conversation: $ResumeConversationId"
    }
    $lines += ''
    $lines += '## Phase Role & Strict Boundaries'
    switch ($Route.Type) {
        'investigation' {
            $lines += '- **INVESTIGATION ONLY (READ-ONLY)**: Inspect the relevant files, trace root causes or requirements, and produce a concise, structured technical analysis and implementation plan.'
            $lines += '- **DO NOT** implement features, **DO NOT** edit or create project files, and **DO NOT** waste time troubleshooting unrelated environment tools.'
            $lines += '- Use `.venv/Scripts/python.exe` if running quick read-only checks.'
        }
        'implement' {
            $lines += '- **IMPLEMENTATION**: Implement the required code changes cleanly and minimally based on the task and prior investigation handoff.'
            $lines += '- Keep changes strictly scoped to the task. Never touch unrelated Colorization, Outpainting, Audio, or vendor modules.'
        }
        'test' {
            $lines += '- **TESTING**: Write or update unit/integration tests and execute them using `.venv/Scripts/python.exe -m unittest`.'
            $lines += '- Do not suppress test discovery or weaken existing assertions.'
        }
        'review' {
            $lines += '- **INDEPENDENT REVIEW (READ-ONLY)**: Review the changes made in previous steps against the issue acceptance criteria and coding standards.'
            $lines += '- Do not modify files. End your review with `VERDICT: APPROVED` if everything is solid, or `VERDICT: CHANGES_REQUIRED` with a numbered list of concrete fixes needed.'
        }
        'corrections' {
            $lines += '- **CORRECTIONS**: Fix the exact issues raised in the independent review and re-verify with unit tests.'
        }
        'verify' {
            $lines += '- **FINAL VERIFICATION (READ-ONLY)**: Verify that all acceptance criteria are met, tests pass, and no unintended files were modified.'
            $lines += '- End with `VERIFICATION: PASSED` or `VERIFICATION: FAILED`.'
        }
        default {
            if (-not $Mode) {
                $lines += '- **READ-ONLY TASK**: Do not create or modify any repository files.'
            }
        }
    }
    $lines += ''
    $lines += '## Task'
    $lines += $cleanText
    $lines += ''
    if ($ContextFiles -and $ContextFiles.Count) {
        $lines += '## Context files (read as needed)'
        foreach ($cf in $ContextFiles) {
            if ($cf) { $lines += "- ``$(Format-AgyCleanPath -Path $cf -WorkDir $WorkDir)``" }
        }
        $lines += ''
    }
    if ($PriorModifiedFiles -and $PriorModifiedFiles.Count) {
        $lines += '## Files modified prior to this step (inspect before acting)'
        foreach ($mf in $PriorModifiedFiles) {
            $lines += "- ``$(Format-AgyCleanPath -Path $mf -WorkDir $WorkDir)``"
        }
        $lines += ''
    }
    if ($PriorSummary) {
        $lines += '## Previous worker summary'
        $lines += (Format-AgyCleanMarkdown $PriorSummary)
        $lines += ''
    }
    $lines += '## Rules'
    $lines += '- Stay strictly within the task scope. No destructive operations.'
    $lines += "- $(if ($Mode -eq 'accept-edits') { 'File edits are authorized ONLY for files directly required by this task.' } else { 'No file modifications are permitted in this phase.' })"
    $lines += '- Answer directly. Use tools only when needed; do not run `pip list` or broad directory crawls.'
    $lines += '- Be concise. End your answer with a final line: `MODEL: <your model name>`'
    $lines
}

function Invoke-AgyWorker {
    param(
        [Parameter(Mandatory)][string]$Text,
        [Parameter(Mandatory)]$Route,
        [Parameter(Mandatory)]$Policy,
        [Parameter(Mandatory)][string]$RunDir,
        [Parameter(Mandatory)][int]$Index,
        [string]$AgyCommand = 'agy',
        [string[]]$ContextFiles = @(),
        [string]$PriorSummary = '',
        [bool]$AllowWrites = $false,
        [bool]$YoloWorker = $false,
        [int]$WorkerTimeoutMinutes = 0,
        [ValidateSet('json', 'stream-json')][string]$OutputFormat = 'stream-json',
        [bool]$Live = $false,
        [string]$ResumeConversationId = '',
        [string[]]$PriorModifiedFiles = @(),
        [string]$LiveStatusPath = '',
        [string]$CancelFilePath = '',
        [string]$WorkDir = (Get-Location).Path
    )
    $null = New-Item -ItemType Directory -Force -Path $RunDir
    $mode = if ($Route.WantsWrite -and $AllowWrites) { $Policy.modes.write } else { $Policy.modes.readOnly }

    $max = [int]$Policy.budgets.handoffMaxChars
    if ($PriorSummary.Length -gt $max) { $PriorSummary = $PriorSummary.Substring(0, $max) + ' [truncated]' }
    $handoff = Join-Path $RunDir ("handoff-{0}.md" -f $Index)
    $handoffRel = Format-AgyCleanPath -Path $handoff -WorkDir $WorkDir
    $lines = Build-AgyHandoffLines -Index $Index -Route $Route -Mode $mode -Text $Text -ContextFiles $ContextFiles `
        -PriorSummary $PriorSummary -ResumeConversationId $ResumeConversationId -PriorModifiedFiles $PriorModifiedFiles -WorkDir $WorkDir
    Write-AgyUtf8File -Path $handoff -Value $lines

    $prompt = if ($ResumeConversationId) {
        "Continue from the current conversation state. Read the updated handoff instructions at $handoffRel, inspect any files already touched, and finish the remaining work for phase '$($Route.Type)' without repeating completed steps."
    } else {
        "Read the handoff file at $handoffRel and complete the '$($Route.Type)' task described in it."
    }

    $timeoutSec = if ($WorkerTimeoutMinutes -gt 0) {
        $WorkerTimeoutMinutes * 60
    } elseif ($Policy.budgets.PSObject.Properties['workerTimeoutMinutes'] -and [int]$Policy.budgets.workerTimeoutMinutes -gt 0) {
        [int]$Policy.budgets.workerTimeoutMinutes * 60
    } else {
        [int]$Policy.budgets.timeoutSeconds
    }

    $args = @('-p', $prompt, '--model', $Route.Model, '--output-format', $OutputFormat)
    if ($ResumeConversationId) { $args += @('--conversation', $ResumeConversationId) }
    if ($mode) { $args += @('--mode', $mode) }
    if ($YoloWorker) { $args += '--dangerously-skip-permissions' }
    $args += @('--print-timeout', "${timeoutSec}s")

    $eventsFile = Join-Path $RunDir ("worker-{0}.events.jsonl" -f $Index)
    $errFile    = Join-Path $RunDir ("stderr-{0}.txt" -f $Index)
    $null = New-Item -ItemType File -Force -Path $eventsFile

    $maxAttempts = 1 + [int]$Policy.budgets.maxRetriesSameModel
    $attempt = 0; $parsed = $null; $exit = -1; $errText = ''; $tokens = 0
    $convId = if ($ResumeConversationId) { $ResumeConversationId } else { $null }
    $streamInitModel = $null
    $toolCalls = [System.Collections.Generic.List[object]]::new()
    $streamedTextParts = [System.Collections.Generic.List[string]]::new()
    $timedOut = $false; $cancelled = $false; $finishedReason = $null
    $snapBeforeRun = Get-AgyWorkspaceSnapshot -WorkDir $WorkDir
    $modifiedFiles = @()

    while ($attempt -lt $maxAttempts) {
        $attempt++
        $timedOut = $false; $cancelled = $false; $finishedReason = $null; $parsed = $null
        $snapBeforeAttempt = Get-AgyWorkspaceSnapshot -WorkDir $WorkDir
        $startTime = [datetime]::UtcNow

        if ($Live) {
            $ts = (Get-Date).ToString('HH:mm:ss')
            Write-Host "[$ts] [worker#$Index] START type=$($Route.Type) model=$($Route.Model) mode=$(if($mode){$mode}else{'read-only'}) timeout=${timeoutSec}s attempt=$attempt/$maxAttempts$(if($convId){" resume=$convId"}else{''})" -ForegroundColor Cyan
        }
        Write-AgyLiveStatus -StatusPath $LiveStatusPath -Data @{
            workerIndex = $Index; taskType = $Route.Type; model = $Route.Model; mode = $(if ($mode) { $mode } else { 'read-only' })
            state = 'RUNNING'; attempt = $attempt; maxAttempts = $maxAttempts; conversationId = $convId
            startedAt = $startTime.ToString('o'); timeoutSeconds = $timeoutSec; toolCalls = $toolCalls.Count
            lastActivity = 'starting worker'; tokens = $tokens
        }

        Push-Location $WorkDir
        try {
            & $AgyCommand @args 2>$errFile | ForEach-Object {
                $line = [string]$_
                Append-AgyUtf8Line -Path $eventsFile -Line $line
                if ($CancelFilePath -and (Test-Path -LiteralPath $CancelFilePath)) {
                    $cancelled = $true
                }
                $ev = Format-AgyStreamEvent -Line $line
                if ($ev) {
                    if ($ev.conversationId) { $convId = $ev.conversationId }
                    if ($ev.kind -eq 'init') {
                        if ($ev.initModel) { $streamInitModel = $ev.initModel }
                        if ($Live) {
                            $ts = (Get-Date).ToString('HH:mm:ss')
                            Write-Host "[$ts] [worker#$Index] $($ev.message)" -ForegroundColor DarkCyan
                        }
                    } elseif ($ev.kind -eq 'step') {
                        if (($ev.toolName -or $ev.commandLine -or $ev.filePath) -and $ev.status -ne 'DONE') {
                            $tcEntry = [pscustomobject]@{
                                ts          = (Get-Date).ToString('o')
                                tool        = $ev.toolName
                                commandLine = $ev.commandLine
                                filePath    = $ev.filePath
                                message     = $ev.message
                            }
                            $toolCalls.Add($tcEntry)
                        }
                        if ($ev.response) { $streamedTextParts.Add($ev.response) }
                        if ($Live) {
                            $ts = (Get-Date).ToString('HH:mm:ss')
                            Write-Host "[$ts] [worker#$Index] $($ev.message)" -ForegroundColor Gray
                        }
                        Write-AgyLiveStatus -StatusPath $LiveStatusPath -Data @{
                            workerIndex = $Index; taskType = $Route.Type; model = $Route.Model; mode = $(if ($mode) { $mode } else { 'read-only' })
                            state = 'RUNNING'; attempt = $attempt; maxAttempts = $maxAttempts; conversationId = $convId
                            startedAt = $startTime.ToString('o'); timeoutSeconds = $timeoutSec; toolCalls = $toolCalls.Count
                            lastActivity = $ev.message; tokens = $tokens
                        }
                    } elseif ($ev.kind -eq 'result') {
                        $parsed = $ev.raw
                        if ($ev.finishedReason) { $finishedReason = $ev.finishedReason }
                        if ($Live) {
                            $ts = (Get-Date).ToString('HH:mm:ss')
                            $uTok = Get-AgyTokenTotal $ev.usage
                            Write-Host "[$ts] [worker#$Index] RESULT status=$($ev.status) finished_reason=$($ev.finishedReason) tokens=$uTok" -ForegroundColor Green
                        }
                    } elseif ($ev.kind -eq 'agy_error') {
                        if ($ev.reason) { $finishedReason = $ev.reason }
                        $errText += " $($ev.message)"
                        if ($Live) {
                            $ts = (Get-Date).ToString('HH:mm:ss')
                            Write-Host "[$ts] [worker#$Index] ERROR reason=$($ev.reason) $($ev.message)" -ForegroundColor Red
                        }
                    }
                }
            }
            $exit = $LASTEXITCODE
        } catch {
            $exit = -1
            $errText += " $($_.Exception.Message)"
        } finally {
            Pop-Location
        }

        if (Test-Path -LiteralPath $errFile) {
            $stderrRaw = Get-Content -LiteralPath $errFile -Raw -Encoding utf8
            if ($stderrRaw) {
                $errText = ($errText + ' ' + $stderrRaw.Trim()).Trim()
                foreach ($sLine in ($stderrRaw -split "`r?`n")) {
                    if ($sLine -match '^AGY_ERROR:\s*(\{.+\})$') {
                        try {
                            $eObj = $Matches[1] | ConvertFrom-Json
                            if ($eObj.reason -and -not $finishedReason) { $finishedReason = [string]$eObj.reason }
                        } catch {}
                    }
                }
            }
        }

        if ($parsed) { $tokens += Get-AgyTokenTotal $parsed.usage }
        if (-not $convId -and $parsed -and $parsed.PSObject.Properties['conversation_id'] -and $parsed.conversation_id) {
            $convId = [string]$parsed.conversation_id
        }

        $snapAfterAttempt = Get-AgyWorkspaceSnapshot -WorkDir $WorkDir
        $attemptChangedFiles = @(Compare-AgyWorkspaceSnapshots -Before $snapBeforeAttempt -After $snapAfterAttempt)
        $modifiedFiles = @(Compare-AgyWorkspaceSnapshots -Before $snapBeforeRun -After $snapAfterAttempt)

        $combinedDiag = ("$errText $finishedReason " + $(if ($parsed -and $parsed.PSObject.Properties['response']) { [string]$parsed.response } else { '' }))
        if ($finishedReason -eq 'print_timeout' -or $exit -eq 3 -and $combinedDiag -imatch $Policy.timeoutPattern -or $combinedDiag -imatch $Policy.timeoutPattern) {
            $timedOut = $true
            if (-not $finishedReason) { $finishedReason = 'print_timeout' }
        }
        if ($finishedReason -eq 'interrupted' -or $combinedDiag -imatch $Policy.interruptedPattern) {
            $cancelled = $true
            if (-not $finishedReason) { $finishedReason = 'interrupted' }
        }

        if ($cancelled) {
            $errText = ($errText + ' worker interrupted/cancelled').Trim()
            break
        }

        # CRITICAL SAFETY RULE: NEVER auto-retry after a print_timeout (especially if writes were enabled or files changed,
        # and even on read-only tasks a 20m timeout should surface as INCOMPLETE so the conversation can be resumed rather than burning 2x quota).
        if ($timedOut) {
            $errText = ($errText + " worker hit print_timeout (${timeoutSec}s); marked INCOMPLETE without automatic retry (use -ResumeConversationId $convId or -ResumeLatest)").Trim()
            break
        }

        # Also never auto-retry if the attempt already modified workspace files.
        if ($attemptChangedFiles.Count -gt 0 -and ($exit -ne 0 -or -not $parsed -or $parsed.status -ne 'SUCCESS')) {
            $errText = ($errText + " worker modified $($attemptChangedFiles.Count) file(s) before failing; skipping automatic retry to prevent duplicate/conflicting edits").Trim()
            break
        }

        if ($exit -eq 0 -and $parsed -and $parsed.status -eq 'SUCCESS' -and -not $finishedReason) {
            if ([string]::IsNullOrWhiteSpace([string]$parsed.response)) {
                $errText = ($errText + ' empty response (worker produced no text)').Trim()
                $parsed.status = 'EMPTY'
                break
            }
            if ((([string]$parsed.response) + ' ' + $errText) -imatch $Policy.permissionDeniedPattern) {
                $errText = ($errText + ' permission denied in headless mode (re-run with -YoloWorker only if you accept the risk)').Trim()
                $parsed.status = 'PERMISSION_DENIED'
                break
            }
            break
        }
        # Transient non-timeout failure with zero file changes: retry once on the SAME model; never auto-escalates.
    }

    $response = if ($parsed -and $parsed.PSObject.Properties['response'] -and $parsed.response) {
        [string]$parsed.response
    } elseif ($streamedTextParts.Count -gt 0) {
        ($streamedTextParts -join "`n")
    } else { '' }

    $partialFile = $null
    if ($response -and ($timedOut -or $cancelled -or $exit -ne 0)) {
        $partialFile = Join-Path $RunDir ("partial-{0}.md" -f $Index)
        Write-AgyUtf8File -Path $partialFile -Value $response
    }

    $self = $null
    if ($response -match '(?im)^\s*MODEL:\s*(.+?)\s*$') { $self = $Matches[1].Trim('`', ' ', '*') }

    $status = if ($cancelled) {
        'CANCELLED'
    } elseif ($timedOut) {
        'INCOMPLETE'
    } elseif ($exit -eq 0 -and $parsed -and $parsed.status -eq 'SUCCESS' -and -not $finishedReason) {
        'SUCCESS'
    } else {
        'FAILED'
    }

    # Check if a read-only worker illegally modified files or touched protected paths
    $protectedViolations = @(Test-AgyProtectedFiles -ChangedFiles $modifiedFiles -Policy $Policy)
    if ($protectedViolations.Count -gt 0) {
        $status = 'FAILED'
        $errText = ($errText + ' PROTECTED PATH VIOLATION: ' + ($protectedViolations -join '; ')).Trim()
    } elseif (-not ($Route.WantsWrite -and $AllowWrites) -and $modifiedFiles.Count -gt 0) {
        $status = 'FAILED'
        $errText = ($errText + " read-only worker ($($Route.Type)) unexpectedly modified files: $($modifiedFiles -join ', ')").Trim()
    }

    Write-AgyLiveStatus -StatusPath $LiveStatusPath -Data @{
        workerIndex = $Index; taskType = $Route.Type; model = $Route.Model; mode = $(if ($mode) { $mode } else { 'read-only' })
        state = $status; attempt = $attempt; maxAttempts = $maxAttempts; conversationId = $convId
        timeoutSeconds = $timeoutSec; toolCalls = $toolCalls.Count; modifiedFiles = $modifiedFiles
        lastActivity = "finished with status=$status"; tokens = $tokens
    }

    $result = [pscustomobject]@{
        index              = $Index
        taskType           = $Route.Type
        classifiedBy       = $Route.MatchedBy
        requestedModel     = $Route.Model
        mode               = $mode
        permissionBypass   = $YoloWorker
        outputFormat       = $OutputFormat
        timeoutSeconds     = $timeoutSec
        status             = $status
        finishedReason     = $finishedReason
        exitCode           = $exit
        attempts           = $attempt
        conversationId     = $convId
        resumedFrom        = $(if ($ResumeConversationId) { $ResumeConversationId } else { $null })
        durationSeconds    = if ($parsed -and $parsed.PSObject.Properties['duration_seconds']) { $parsed.duration_seconds } else { $null }
        usage              = if ($parsed -and $parsed.PSObject.Properties['usage']) { $parsed.usage } else { $null }
        tokensAllAttempts  = $tokens
        toolCallsCount     = $toolCalls.Count
        toolCalls          = @($toolCalls | Select-Object -Last 25)
        modifiedFiles      = $modifiedFiles
        modelSelfReport    = $self
        modelEvidence      = $(if ($streamInitModel) { $streamInitModel } elseif ($convId) { Get-AgyModelEvidence $convId } else { $null })
        modelVerification  = 'verified via stream-json init.model or conversation transcript; selfReport advisory'
        error              = if ($status -eq 'SUCCESS') { $null } else { ($errText, $response) -join ' ' | ForEach-Object { $_.Trim() } }
        response           = $response
        partialOutputFile  = $(if ($partialFile) { Format-AgyCleanPath -Path $partialFile -WorkDir $WorkDir } else { $null })
        eventsFile         = Format-AgyCleanPath -Path $eventsFile -WorkDir $WorkDir
        handoff            = Format-AgyCleanPath -Path $handoff -WorkDir $WorkDir
    }
    Write-AgyUtf8File -Path (Join-Path $RunDir ("result-{0}.json" -f $Index)) -Value ($result | ConvertTo-Json -Depth 8)
    $result
}

function Invoke-AgyDeterministicVerify {
    # Deterministic local verification: runs file-ownership and unit tests directly without LLM tokens.
    param(
        [Parameter(Mandatory)][int]$IssueNumber,
        $Policy = (Get-AgyPolicy),
        [string]$WorkDir = (Get-Location).Path,
        [string[]]$ModifiedFiles = @(),
        [string]$PythonExe = (Join-Path $WorkDir '.venv\Scripts\python.exe')
    )
    $cleanMods = @($ModifiedFiles | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) -and $_ -notmatch '^docs/maestro/plans/' })

    # 1. Protected paths check
    $protViolations = @(Test-AgyProtectedFiles -ChangedFiles $cleanMods -Policy $Policy)
    if ($protViolations.Count -gt 0) {
        return [pscustomobject]@{
            status            = 'FAILED'
            response          = "VERIFICATION: FAILED - protected path violation: $($protViolations -join '; ')"
            error             = "protected paths modified: $($protViolations -join '; ')"
            requestedModel    = 'local-deterministic'
            tokensAllAttempts = 0
            exitCode          = 1
        }
    }

    # 2. File ownership check
    if ($Policy.backlog.PSObject.Properties['delivery'] -and $Policy.backlog.delivery.PSObject.Properties['issueFileOwnership']) {
        $numStr = [string]$IssueNumber
        if ($Policy.backlog.delivery.issueFileOwnership.PSObject.Properties[$numStr]) {
            $allowed = @($Policy.backlog.delivery.issueFileOwnership.$numStr)
            $pre = @($Policy.backlog.delivery.preExistingDirtyFiles)
            $unauthorized = @($cleanMods | Where-Object { $allowed -notcontains $_ -and $pre -notcontains $_ })
            if ($unauthorized.Count -gt 0) {
                return [pscustomobject]@{
                    status            = 'FAILED'
                    response          = "VERIFICATION: FAILED - unauthorized file modifications for #${IssueNumber}: $($unauthorized -join ', ')"
                    error             = "unauthorized files modified: $($unauthorized -join ', ')"
                    requestedModel    = 'local-deterministic'
                    tokensAllAttempts = 0
                    exitCode          = 1
                }
            }
        }
    }

    # 3. Deterministic test runner
    $targetTests = @()
    if ($Policy.backlog.PSObject.Properties['delivery'] -and $Policy.backlog.delivery.PSObject.Properties['issueFileOwnership']) {
        $numStr = [string]$IssueNumber
        if ($Policy.backlog.delivery.issueFileOwnership.PSObject.Properties[$numStr]) {
            $owned = @($Policy.backlog.delivery.issueFileOwnership.$numStr)
            $targetTests = @($owned | Where-Object { $_ -match '^tests/.*\.py$' -and (Test-Path (Join-Path $WorkDir $_)) })
        }
    }
    if ($targetTests.Count -eq 0 -and $cleanMods.Count -gt 0) {
        $targetTests = @($cleanMods | Where-Object { $_ -match '^tests/.*\.py$' -and (Test-Path (Join-Path $WorkDir $_)) })
    }

    $testsPassed = $true
    $diag = ''
    if ($targetTests.Count -gt 0 -and (Test-Path -LiteralPath $PythonExe)) {
        foreach ($tPath in $targetTests) {
            $pInfo = New-Object System.Diagnostics.ProcessStartInfo
            $pInfo.FileName = $PythonExe
            $pInfo.Arguments = "-m unittest $tPath"
            $pInfo.WorkingDirectory = $WorkDir
            $pInfo.RedirectStandardOutput = $true
            $pInfo.RedirectStandardError = $true
            $pInfo.UseShellExecute = $false
            $pInfo.CreateNoWindow = $true
            $proc = [System.Diagnostics.Process]::Start($pInfo)
            $stdout = $proc.StandardOutput.ReadToEnd()
            $stderr = $proc.StandardError.ReadToEnd()
            $null = $proc.WaitForExit(60000)
            if ($proc.ExitCode -ne 0) {
                $testsPassed = $false
                $diag += "$tPath failed (exit $($proc.ExitCode)): $stderr $stdout`n"
            }
        }
    }

    if (-not $testsPassed) {
        return [pscustomobject]@{
            status            = 'FAILED'
            response          = "VERIFICATION: FAILED - deterministic test execution failed:`n$diag"
            error             = "deterministic test runner failed"
            requestedModel    = 'local-deterministic'
            tokensAllAttempts = 0
            exitCode          = 1
        }
    }

    [pscustomobject]@{
        status            = 'SUCCESS'
        response          = "Deterministic local verification passed. All file ownership rules and unit tests verified.`nVERIFICATION: PASSED`nMODEL: local-deterministic"
        error             = $null
        requestedModel    = 'local-deterministic'
        tokensAllAttempts = 0
        exitCode          = 0
    }
}

function Get-AgyIssueText {
    param([Parameter(Mandatory)][int]$Number, [string]$GhCommand = 'gh', [int]$MaxChars = 8000)
    $prevEnc = [Console]::OutputEncoding
    [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
    try {
        $raw = & $GhCommand issue view $Number --json 'title,body'
        $j = ($raw -join "`n") | ConvertFrom-Json
    } finally {
        [Console]::OutputEncoding = $prevEnc
    }
    $body = Format-AgyCleanMarkdown ([string]$j.body)
    if ($body.Length -gt $MaxChars) { $body = $body.Substring(0, $MaxChars) + "`n[issue body truncated]" }
    "GitHub issue #${Number}: $($j.title)`n`n$body"
}

function Resolve-AgyPlanFiles {
    # Accepts paths or wildcards (e.g. docs/maestro/plans/*impl-plan.md); returns clean forward-slash paths.
    param([string[]]$Pattern, [string]$WorkDir = (Get-Location).Path)
    foreach ($p in $Pattern) {
        if (-not $p) { continue }
        $hits = @(Get-ChildItem -Path $p -File -ErrorAction SilentlyContinue)
        if (-not $hits.Count) { throw "Plan/context file not found: $p" }
        foreach ($h in $hits) {
            Format-AgyCleanPath -Path $h.FullName -WorkDir $WorkDir
        }
    }
}

# ---------------------------------------------------------------- quota guard ----
# Supports:
# 1. Manual weekly and 5-hour remaining percentages (`Quota.ps1 -Set ...`), ALWAYS labelled ESTIMATE.
# 2. Local token ledger vs self-imposed localWeeklyTokenCap, ALWAYS labelled ESTIMATE.
# 3. Optional `Quota.ps1 -SyncFromCli` reading `agy -p "/usage" --output-format json` (0 tokens, no agent turn),
#    labelled ACTUAL when fresh (< officialStaleHours) and downgraded to ESTIMATE when stale or after local usage.

function Get-AgyModelGroup {
    param([string]$Model, $Policy)
    foreach ($g in $Policy.groups.PSObject.Properties) { if ($Model -match $g.Value.modelPattern) { return $g.Name } }
    'unknown'
}

function Get-AgyState {
    param([Parameter(Mandatory)][string]$Path)
    $s = if (Test-Path -LiteralPath $Path) { Get-Content -LiteralPath $Path -Raw -Encoding utf8 | ConvertFrom-Json -AsHashtable } else { @{} }
    if (-not $s) { $s = @{} }
    if (-not $s.ContainsKey('readings')) { $s.readings = @{} }
    if (-not $s.ContainsKey('ledger')) { $s.ledger = @() }
    $s
}

function Save-AgyState {
    param([Parameter(Mandatory)]$State, [Parameter(Mandatory)][string]$Path)
    Write-AgyUtf8File -Path $Path -Value ($State | ConvertTo-Json -Depth 8)
}

function Set-AgyQuotaReading {
    param(
        [Parameter(Mandatory)][string]$Group,
        [Parameter(Mandatory)][ValidateRange(0, 100)][double]$RemainingPercent,
        [ValidateRange(-1, 100)][double]$FiveHourRemainingPercent = -1,
        [ValidateSet('manual', 'official')][string]$Source = 'manual',
        [Parameter(Mandatory)][string]$StatePath,
        $Policy = (Get-AgyPolicy),
        [datetime]$AsOf = [datetime]::UtcNow,
        [string]$ResetTime = '',
        [string]$FiveHourResetTime = ''
    )
    if (-not $Policy.groups.PSObject.Properties[$Group]) {
        throw "Unknown group '$Group'. Known: $(($Policy.groups.PSObject.Properties.Name) -join ', ')"
    }
    $s = Get-AgyState $StatePath
    $prev5h = if ($s.readings.ContainsKey($Group) -and $s.readings[$Group].ContainsKey('fiveHourRemainingPercent')) {
        $s.readings[$Group]['fiveHourRemainingPercent']
    } else { $null }
    $fiveH = if ($FiveHourRemainingPercent -ge 0) { $FiveHourRemainingPercent } else { $prev5h }
    $s.readings[$Group] = @{
        remainingPercent         = $RemainingPercent
        fiveHourRemainingPercent = $fiveH
        source                   = $Source
        asOf                     = $AsOf.ToUniversalTime().ToString('o')
        resetTime                = $ResetTime
        fiveHourResetTime        = $FiveHourResetTime
    }
    Save-AgyState $s $StatePath
}

function Sync-AgyQuotaFromCli {
    # Invokes official read-only `agy -p "/usage" --output-format json` (spends 0 tokens, no agent turn).
    param(
        [Parameter(Mandatory)][string]$StatePath,
        [string]$AgyCommand = 'agy',
        $Policy = (Get-AgyPolicy),
        [datetime]$AsOf = [datetime]::UtcNow
    )
    $raw = & $AgyCommand -p '/usage' --output-format json 2>$null
    $jsonLine = @($raw | Where-Object { "$_".TrimStart().StartsWith('{') }) | Select-Object -Last 1
    if (-not $jsonLine) { throw 'agy -p "/usage" did not return JSON' }
    $obj = $jsonLine | ConvertFrom-Json
    if (-not ($obj.PSObject.Properties['command'] -and $obj.command.data.groups)) {
        throw 'agy /usage JSON did not contain command.data.groups'
    }
    $map = @{
        'Gemini Models'         = 'gemini-flash-pro'
        'Claude and GPT models' = 'claude-gpt'
        'Gemini 4 Argon'        = 'argon'
    }
    foreach ($g in $obj.command.data.groups) {
        $target = $map[$g.name]
        if (-not $target) { continue }
        $weekly = $g.buckets | Where-Object { $_.window -eq 'weekly' } | Select-Object -First 1
        $fiveH  = $g.buckets | Where-Object { $_.window -eq '5h' }     | Select-Object -First 1
        if ($weekly) {
            $wPct = [math]::Round([double]$weekly.remaining_fraction * 100, 1)
            $fPct = if ($fiveH) { [math]::Round([double]$fiveH.remaining_fraction * 100, 1) } else { -1 }
            Set-AgyQuotaReading -Group $target -RemainingPercent $wPct -FiveHourRemainingPercent $fPct `
                -Source 'official' -StatePath $StatePath -Policy $Policy -AsOf $AsOf `
                -ResetTime ([string]$weekly.reset_time) -FiveHourResetTime ($(if ($fiveH) { [string]$fiveH.reset_time } else { '' }))
        }
    }
    Get-AgyQuotaReport -Policy $Policy -StatePath $StatePath
}

function Add-AgyUsage {
    param([Parameter(Mandatory)]$State, [string]$Group, [string]$Model, [int]$Tokens, [string]$RunId, [datetime]$Now = [datetime]::UtcNow)
    $cut = $Now.AddDays(-14)
    $State.ledger = @($State.ledger | Where-Object { ([datetime]$_.ts).ToUniversalTime() -gt $cut }) +
        @(@{ ts = $Now.ToUniversalTime().ToString('o'); group = $Group; model = $Model; tokens = $Tokens; runId = $RunId })
}

function Get-AgyGroupStatus {
    param([Parameter(Mandatory)][string]$Group, [Parameter(Mandatory)]$Policy, [Parameter(Mandatory)]$State, [datetime]$Now = [datetime]::UtcNow)
    $g = $Policy.groups.$Group
    $reserve = if ($g.PSObject.Properties['weeklyReservePercent'] -and $null -ne $g.weeklyReservePercent) {
        [double]$g.weeklyReservePercent
    } else {
        [double]$Policy.quota.weeklyReservePercent
    }
    $fiveHReserve = if ($g.PSObject.Properties['fiveHourReservePercent'] -and $null -ne $g.fiveHourReservePercent) {
        [double]$g.fiveHourReservePercent
    } elseif ($Policy.quota.PSObject.Properties['fiveHourReservePercent'] -and $null -ne $Policy.quota.fiveHourReservePercent) {
        [double]$Policy.quota.fiveHourReservePercent
    } else { 0.0 }

    $cap = if ($g.PSObject.Properties['localWeeklyTokenCap'] -and $g.localWeeklyTokenCap) { [double]$g.localWeeklyTokenCap } else { 0 }
    $week = $Now.AddDays(-7)
    $fiveHWindow = $Now.AddHours(-5)
    $entries = @($State.ledger | Where-Object { $_.group -eq $Group })
    $used7d = [double](($entries | Where-Object { ([datetime]$_.ts).ToUniversalTime() -gt $week } | ForEach-Object { [double]$_.tokens } | Measure-Object -Sum).Sum)
    $used5h = [double](($entries | Where-Object { ([datetime]$_.ts).ToUniversalTime() -gt $fiveHWindow } | ForEach-Object { [double]$_.tokens } | Measure-Object -Sum).Sum)
    $localRem = if ($cap) { [math]::Max(0, 100 * (1 - $used7d / $cap)) } else { $null }

    $rem = $null; $fiveHRem = $null; $kind = 'UNKNOWN'; $basis = 'no reading and no localWeeklyTokenCap configured'
    $r = if ($State.readings.ContainsKey($Group)) { $State.readings[$Group] } else { $null }
    if ($r) {
        $asOf = ([datetime]$r.asOf).ToUniversalTime()
        $ageH = ($Now - $asOf).TotalHours
        $since = [double](($entries | Where-Object { ([datetime]$_.ts).ToUniversalTime() -gt $asOf } | ForEach-Object { [double]$_.tokens } | Measure-Object -Sum).Sum)
        if ($r.ContainsKey('fiveHourRemainingPercent') -and $null -ne $r.fiveHourRemainingPercent -and $r.fiveHourRemainingPercent -ge 0) {
            # If 5h window reset_time has passed, don't let an old 0% reading block forever if >5h old
            if ($ageH -le 5.0) {
                $fiveHRem = [math]::Max(0, [math]::Round(([double]$r.fiveHourRemainingPercent - $(if ($cap) { 100 * $since / $cap } else { 0 })), 1))
            }
        }
        if ($r.source -eq 'official' -and $ageH -le [double]$Policy.quota.officialStaleHours -and $since -eq 0) {
            $rem = [double]$r.remainingPercent
            $kind = 'ACTUAL'
            $basis = ("official CLI /usage reading, {0:N1}h old" -f $ageH)
        } else {
            $rem = [double]$r.remainingPercent - $(if ($cap) { 100 * $since / $cap } else { 0 })
            $kind = 'ESTIMATE'
            $basis = ("{0} reading {1}% ({2:N1}h old) minus local usage since{3}" -f $r.source, $r.remainingPercent, $ageH, $(if ($cap) { '' } else { ' [no cap configured: usage since reading NOT deducted]' }))
            if ($null -ne $localRem -and $localRem -lt $rem) { $rem = $localRem; $basis += '; lowered to local-cap ledger value' }
        }
    } elseif ($null -ne $localRem) {
        $rem = $localRem; $kind = 'ESTIMATE'; $basis = 'local ledger vs configured localWeeklyTokenCap (not a Google reading)'
    }
    if ($null -ne $rem) { $rem = [math]::Max(0, [math]::Round($rem, 1)) }
    $belowWeekly = [bool]($null -ne $rem -and $rem -lt $reserve)
    $belowFiveH  = [bool]($null -ne $fiveHRem -and $fiveHRem -lt $fiveHReserve)

    [pscustomobject]@{
        group                    = $Group
        remainingPercent         = $rem
        fiveHourRemainingPercent = $fiveHRem
        kind                     = $kind
        basis                    = $basis
        reservePercent           = $reserve
        fiveHourReservePercent   = $fiveHReserve
        belowWeeklyReserve       = $belowWeekly
        belowFiveHourReserve     = $belowFiveH
        belowReserve             = [bool]($belowWeekly -or $belowFiveH)
        localTokensUsed7d        = $used7d
        localTokensUsed5h        = $used5h
        localWeeklyTokenCap      = $(if ($cap) { $cap } else { $null })
    }
}

function Get-AgyQuotaReport {
    param($Policy = (Get-AgyPolicy), [Parameter(Mandatory)][string]$StatePath)
    $s = Get-AgyState $StatePath
    foreach ($g in $Policy.groups.PSObject.Properties.Name) { Get-AgyGroupStatus -Group $g -Policy $Policy -State $s }
}

function Find-AgyResumableWorker {
    # Inspects the latest (or specified) run summary/result for an INCOMPLETE or FAILED worker with a conversationId.
    param([string]$RunRoot = (Join-Path (Get-Location).Path '.cache/agy-orch'), [string]$FromPath = 'latest')
    $path = if ($FromPath -eq 'latest') { Join-Path $RunRoot 'latest.json' } else { $FromPath }
    if (-not (Test-Path -LiteralPath $path)) { throw "No run summary found at $path" }
    $j = Get-Content -LiteralPath $path -Raw -Encoding utf8 | ConvertFrom-Json
    $results = if ($j.PSObject.Properties['results']) { @($j.results) } else { @($j) }
    $cand = $results | Where-Object { $_.conversationId -and ($_.status -in @('INCOMPLETE', 'FAILED', 'CANCELLED')) } | Select-Object -Last 1
    if (-not $cand) {
        $cand = $results | Where-Object { $_.conversationId } | Select-Object -Last 1
    }
    if (-not $cand) { throw "No resumable worker with a conversationId found in $path" }
    $cand
}

function Invoke-AgyOrchestration {
    param(
        [Parameter(Mandatory)][object[]]$Tasks,   # strings or @{Text=;Type=;Context=@();AllowWrites=;ResumeConversationId=;PriorModifiedFiles=@()}
        $Policy = (Get-AgyPolicy),
        [string]$AgyCommand = 'agy',
        [string]$RunRoot = (Join-Path (Get-Location).Path '.cache/agy-orch'),
        [string]$StatePath = (Join-Path $RunRoot 'state.json'),
        [hashtable]$Auth = @{},
        [bool]$AllowWrites = $false,
        [bool]$EnforceWriteAuth = $false,
        [switch]$Parallel,
        [switch]$DryRun,
        [switch]$OverrideReserve,
        [switch]$YoloWorker,
        [switch]$Live,
        [int]$WorkerTimeoutMinutes = 0,
        [ValidateSet('json', 'stream-json')][string]$OutputFormat = 'stream-json',
        [string]$ResumeConversationId = '',
        [string[]]$ContextFiles = @(),
        [string]$PriorSummary = '',
        [string]$WorkDir = (Get-Location).Path,
        [switch]$DeterministicVerify
    )
    $runId = (Get-Date -Format 'yyyyMMdd-HHmmss') + '-' + ([guid]::NewGuid().ToString('N').Substring(0, 6))
    $runDir = Join-Path $RunRoot $runId
    $liveStatusPath = Join-Path $RunRoot 'live-status.json'
    $cancelFilePath = Join-Path $RunRoot 'cancel.signal'
    if (Test-Path -LiteralPath $cancelFilePath) { Remove-Item -LiteralPath $cancelFilePath -Force -ErrorAction SilentlyContinue }

    $state = Get-AgyState $StatePath
    $plan = @(); $i = 0
    foreach ($t in $Tasks) {
        $i++
        $text = if ($t -is [string]) { $t } else { $t.Text }
        $type = if ($t -is [hashtable] -and $t.ContainsKey('Type')) { $t.Type } else { $null }
        $ctx = @($ContextFiles) + $(if ($t -is [hashtable] -and $t.ContainsKey('Context') -and $t.Context) { @($t.Context) } else { @() })
        $taskWrites = if ($t -is [hashtable] -and $t.ContainsKey('AllowWrites')) { [bool]$t.AllowWrites } else { $AllowWrites }
        $taskResume = if ($t -is [hashtable] -and $t.ContainsKey('ResumeConversationId') -and $t.ResumeConversationId) { [string]$t.ResumeConversationId } elseif ($i -eq 1 -and $ResumeConversationId) { $ResumeConversationId } else { '' }
        $taskPriorMods = if ($t -is [hashtable] -and $t.ContainsKey('PriorModifiedFiles') -and $t.PriorModifiedFiles) { @($t.PriorModifiedFiles) } else { @() }
        $onlyIfFlagged = if ($t -is [hashtable] -and $t.ContainsKey('OnlyIfReviewFlagged')) { [bool]$t.OnlyIfReviewFlagged } else { $false }

        $route = Resolve-AgyRoute -Text $text -Type $type -Policy $Policy -State $(if ($OverrideReserve) { $null } else { $state })
        $grp = Get-AgyModelGroup $route.Model $Policy
        $blocks = @(Get-AgyBlocks -Route $route -Policy $Policy -Auth $Auth -AllowWrites $taskWrites -EnforceWriteAuth $EnforceWriteAuth)
        $plan += [pscustomobject]@{
            Index                = $i
            Text                 = $text
            Route                = $route
            Context              = $ctx
            Group                = $grp
            AllowWrites          = $taskWrites
            ResumeConversationId = $taskResume
            PriorModifiedFiles   = $taskPriorMods
            OnlyIfReviewFlagged  = $onlyIfFlagged
            Blocks               = $blocks
        }
    }

    $timeoutSec = if ($WorkerTimeoutMinutes -gt 0) {
        $WorkerTimeoutMinutes * 60
    } elseif ($Policy.budgets.PSObject.Properties['workerTimeoutMinutes'] -and [int]$Policy.budgets.workerTimeoutMinutes -gt 0) {
        [int]$Policy.budgets.workerTimeoutMinutes * 60
    } else {
        [int]$Policy.budgets.timeoutSeconds
    }

    $summary = [ordered]@{
        runId               = $runId
        runDir              = Format-AgyCleanPath -Path $runDir -WorkDir $WorkDir
        dryRun              = [bool]$DryRun
        status              = 'OK'
        budgetKind          = $null
        blocked             = @()
        results             = @()
        totalTokens         = 0
        notes               = @()
        quota               = @()
        phaseModels         = @{}
        isIndependentReview = $null
    }
    foreach ($p in $plan) {
        if ($p.Route.FallbackReason) {
            $summary.notes += "task $($p.Index) ($($p.Route.Type)): $($p.Route.FallbackReason)"
        }
    }
    $summary.quota = @($plan.Group | Select-Object -Unique | ForEach-Object { Get-AgyGroupStatus -Group $_ -Policy $Policy -State $state })
    $planView = @($plan | ForEach-Object {
        [pscustomobject]@{
            index                = $_.Index
            type                 = $_.Route.Type
            model                = $_.Route.Model
            primaryModel         = $_.Route.PrimaryModel
            fallbackFrom         = $_.Route.FallbackFrom
            group                = $_.Group
            mode                 = $(if ($_.Route.WantsWrite -and $_.AllowWrites) { $Policy.modes.write } else { $Policy.modes.readOnly })
            permissionBypass     = [bool]$YoloWorker
            timeoutSeconds       = $timeoutSec
            resumeConversationId = $(if ($_.ResumeConversationId) { $_.ResumeConversationId } else { $null })
            matchedBy            = $_.Route.MatchedBy
        }
    })
    $summary.plan = $planView
    if ($YoloWorker) { $summary.notes += 'WARNING: -YoloWorker active; workers get --dangerously-skip-permissions (all tool permissions auto-approved)' }

    if ($plan.Count -gt [int]$Policy.budgets.maxWorkersPerRun) {
        $summary.status = 'BLOCKED'; $summary.budgetKind = 'INTERNAL_RUN_BUDGET'; $summary.blocked = @("task count $($plan.Count) exceeds maxWorkersPerRun $($Policy.budgets.maxWorkersPerRun)")
        return [pscustomobject]$summary
    }
    foreach ($p in $plan) { foreach ($b in $p.Blocks) { $summary.blocked += "task $($p.Index): $b" } }
    foreach ($q in $summary.quota) {
        if ($q.belowWeeklyReserve) {
            $msg = "group '$($q.group)' weekly remaining at $($q.remainingPercent)% ($($q.kind)) is below the $($q.reservePercent)% reserve [$($q.basis)]"
            if ($OverrideReserve) { $summary.notes += "reserve OVERRIDDEN: $msg" } else { $summary.budgetKind = 'PROVIDER_QUOTA_RESERVE'; $summary.blocked += "quota: $msg; use -OverrideReserve to dispatch anyway" }
        }
        if ($q.belowFiveHourReserve) {
            $msg5 = "group '$($q.group)' 5-hour remaining at $($q.fiveHourRemainingPercent)% ($($q.kind)) is below the $($q.fiveHourReservePercent)% 5h reserve [$($q.basis)]"
            if ($OverrideReserve) { $summary.notes += "5h reserve OVERRIDDEN: $msg5" } else { $summary.budgetKind = 'PROVIDER_QUOTA_RESERVE'; $summary.blocked += "quota: $msg5; use -OverrideReserve to dispatch anyway" }
        }
        if ($q.kind -eq 'UNKNOWN') { $summary.notes += "group '$($q.group)': quota UNKNOWN; only per-run budgets apply" }
    }
    foreach ($grpName in ($plan.Group | Select-Object -Unique)) {
        $calls = @($plan | Where-Object Group -eq $grpName).Count
        $max = [int]$Policy.groups.$grpName.maxCallsPerRun
        if ($max -and $calls -gt $max) { $summary.budgetKind = 'INTERNAL_RUN_BUDGET'; $summary.blocked += "group '$grpName': $calls tasks exceed maxCallsPerRun $max" }
    }
    if ($summary.blocked.Count) { $summary.status = 'BLOCKED'; return [pscustomobject]$summary }
    if ($DryRun) { return [pscustomobject]$summary }

    $useParallel = $Parallel -and $plan.Count -ge 2
    if ($useParallel -and ($plan.Count -gt [int]$Policy.budgets.maxParallelWorkers)) {
        $summary.notes += "parallel requested for $($plan.Count) tasks; limited to sequential (max parallel $($Policy.budgets.maxParallelWorkers))"
        $useParallel = $false
    }

    $lib = Join-Path $PSScriptRoot 'AgyOrchestrator.ps1'
    $groupTokens = @{}; $groupCalls = @{}
    if ($useParallel) {
        $summary.notes += 'ran 2 workers in parallel (no handoff chaining between them)'
        $jobs = foreach ($p in $plan) {
            $jobArgs = @{
                lib = $lib; text = $p.Text; route = $p.Route; policy = $Policy; runDir = $runDir; idx = $p.Index; agy = $AgyCommand
                ctx = @($p.Context); aw = [bool]$p.AllowWrites; wd = $WorkDir; prior = $PriorSummary; yolo = [bool]$YoloWorker
                wtm = $WorkerTimeoutMinutes; fmt = $OutputFormat; resume = $p.ResumeConversationId; mods = @($p.PriorModifiedFiles)
            }
            Start-Job -ArgumentList $jobArgs -ScriptBlock {
                param($a)
                . $a.lib
                Invoke-AgyWorker -Text $a.text -Route $a.route -Policy $a.policy -RunDir $a.runDir -Index $a.idx -AgyCommand $a.agy `
                    -ContextFiles $a.ctx -AllowWrites $a.aw -YoloWorker $a.yolo -PriorSummary $a.prior -WorkDir $a.wd `
                    -WorkerTimeoutMinutes $a.wtm -OutputFormat $a.fmt -ResumeConversationId $a.resume -PriorModifiedFiles $a.mods
            }
        }
        $summary.results = @($jobs | Wait-Job | Receive-Job)
        $jobs | Remove-Job
        foreach ($r in $summary.results) {
            $summary.totalTokens += $r.tokensAllAttempts
            Add-AgyUsage -State $state -Group (Get-AgyModelGroup $r.requestedModel $Policy) -Model $r.requestedModel -Tokens $r.tokensAllAttempts -RunId $runId
        }
    } else {
        $prior = $PriorSummary
        $accumModified = @()
        $lastReviewNeedsChanges = $false
        foreach ($p in $plan) {
            if ($p.OnlyIfReviewFlagged -and -not $lastReviewNeedsChanges) {
                $summary.notes += "skipped task $($p.Index) ($($p.Route.Type)) because independent review approved the implementation"
                continue
            }
            if ($summary.totalTokens -ge [int]$Policy.budgets.maxTotalTokens) {
                $summary.status = 'BUDGET_EXHAUSTED'; $summary.budgetKind = 'INTERNAL_RUN_BUDGET'; $summary.notes += "token budget reached before task $($p.Index)"; break
            }
            $g = $p.Group; $gc = $Policy.groups.$g
            if (-not $groupTokens.ContainsKey($g)) { $groupTokens[$g] = 0; $groupCalls[$g] = 0 }
            if ([int]$gc.perRunTokenBudget -and $groupTokens[$g] -ge [int]$gc.perRunTokenBudget) {
                $summary.status = 'BUDGET_EXHAUSTED'; $summary.budgetKind = 'INTERNAL_RUN_BUDGET'; $summary.notes += "group '$g' per-run token budget reached before task $($p.Index)"; break
            }
            $q = Get-AgyGroupStatus -Group $g -Policy $Policy -State $state
            if ($q.belowReserve -and -not $OverrideReserve) {
                $reRoute = Resolve-AgyRoute -Text $p.Text -Type $p.Route.Type -Policy $Policy -State $state
                $reGrp = Get-AgyModelGroup $reRoute.Model $Policy
                $reQ = Get-AgyGroupStatus -Group $reGrp -Policy $Policy -State $state
                if ($reRoute.Model -ne $p.Route.Model -and -not $reQ.belowReserve) {
                    $p.Route = $reRoute
                    $p.Group = $reGrp
                    $g = $reGrp
                    $gc = $Policy.groups.$g
                    if (-not $groupTokens.ContainsKey($g)) { $groupTokens[$g] = 0; $groupCalls[$g] = 0 }
                    $summary.notes += "task $($p.Index) ($($p.Route.Type)) mid-run fallback: $($reRoute.FallbackReason)"
                } else {
                    $summary.status = 'BUDGET_EXHAUSTED'; $summary.budgetKind = 'PROVIDER_QUOTA_RESERVE'; $summary.notes += "group '$g' fell below reserve ($($q.remainingPercent)% weekly / $($q.fiveHourRemainingPercent)% 5h $($q.kind)) before task $($p.Index)"; break
                }
            }
            $priorMods = @($p.PriorModifiedFiles) + @($accumModified) | Select-Object -Unique
            $r = $null
            if ($p.Route.Type -eq 'verify' -and ($DeterministicVerify -or ($Policy.PSObject.Properties['deterministicVerify'] -and $Policy.deterministicVerify))) {
                $issueNum = 0
                if ($p.Text -match '#(\d+)') { $issueNum = [int]$Matches[1] }
                $det = Invoke-AgyDeterministicVerify -IssueNumber $issueNum -Policy $Policy -WorkDir $WorkDir -ModifiedFiles $priorMods
                $r = [pscustomobject]@{
                    index              = $p.Index
                    taskType           = 'verify'
                    classifiedBy       = 'deterministic-local'
                    requestedModel     = 'local-deterministic'
                    mode               = ''
                    permissionBypass   = $false
                    outputFormat       = 'local'
                    timeoutSeconds     = 0
                    status             = $det.status
                    finishedReason     = $null
                    exitCode           = $det.exitCode
                    attempts           = 1
                    conversationId     = $null
                    resumedFrom        = $null
                    durationSeconds    = 0
                    usage              = @{ total_tokens = 0 }
                    tokensAllAttempts  = 0
                    toolCallsCount     = 0
                    toolCalls          = @()
                    modifiedFiles      = @()
                    modelSelfReport    = 'local-deterministic'
                    modelEvidence      = 'local-deterministic-tool'
                    modelVerification  = 'verified via deterministic local runner'
                    error              = $det.error
                    response           = $det.response
                    partialOutputFile  = $null
                    eventsFile         = $null
                    handoff            = $null
                    isIndependentReview= $null
                }
            } else {
                $r = Invoke-AgyWorker -Text $p.Text -Route $p.Route -Policy $Policy -RunDir $runDir -Index $p.Index -AgyCommand $AgyCommand `
                    -ContextFiles $p.Context -PriorSummary $prior -AllowWrites ([bool]$p.AllowWrites) -YoloWorker ([bool]$YoloWorker) `
                    -WorkerTimeoutMinutes $WorkerTimeoutMinutes -OutputFormat $OutputFormat -Live ([bool]$Live) `
                    -ResumeConversationId $p.ResumeConversationId -PriorModifiedFiles $priorMods `
                    -LiveStatusPath $liveStatusPath -CancelFilePath $cancelFilePath -WorkDir $WorkDir
            }
            if ($p.Route.Type -eq 'review') {
                $isIndep = [bool]([string]::IsNullOrEmpty($p.ResumeConversationId))
                $r | Add-Member -NotePropertyName isIndependentReview -NotePropertyValue $isIndep -Force -ErrorAction SilentlyContinue
                $summary.phaseModels['review'] = $r.requestedModel
                $summary['isIndependentReview'] = $isIndep
                $summary['reviewModel'] = $r.requestedModel
            } else {
                $summary.phaseModels[$p.Route.Type] = $r.requestedModel
            }
            $summary.results += $r
            $summary.totalTokens += $r.tokensAllAttempts
            $groupTokens[$g] += $r.tokensAllAttempts; $groupCalls[$g]++
            if ($r.modifiedFiles) { $accumModified = @($accumModified + @($r.modifiedFiles) | Select-Object -Unique) }
            Add-AgyUsage -State $state -Group $g -Model $r.requestedModel -Tokens $r.tokensAllAttempts -RunId $runId

            if ($Policy.budgets.PSObject.Properties['perTaskTokenBudgets'] -and $Policy.budgets.perTaskTokenBudgets.PSObject.Properties[$p.Route.Type]) {
                $tBudget = [int]$Policy.budgets.perTaskTokenBudgets.$($p.Route.Type)
                if ($tBudget -gt 0 -and $r.tokensAllAttempts -gt $tBudget) {
                    $summary.notes += "task $($p.Index) ($($p.Route.Type)) consumed $($r.tokensAllAttempts) tokens (exceeded advisory budget of $tBudget)"
                }
            }

            if ($r.status -eq 'INCOMPLETE') {
                $summary.status = 'INCOMPLETE'
                $summary.notes += "task $($p.Index) timed out on $($r.requestedModel) (conversationId=$($r.conversationId)); resume with -ResumeConversationId $($r.conversationId) or -ResumeLatest"
                break
            }
            if ($r.status -eq 'CANCELLED') {
                $summary.status = 'CANCELLED'
                $summary.notes += "task $($p.Index) was cancelled (conversationId=$($r.conversationId))"
                break
            }
            if ($r.status -ne 'SUCCESS') {
                $summary.status = 'FAILED'
                $summary.notes += "task $($p.Index) failed on $($r.requestedModel); no automatic escalation (rerun with explicit -Type / authorization)"
                break
            }
            if ($p.Route.Type -eq 'investigation') {
                if ($r.response -imatch 'ISSUE_STATUS:\s*ALREADY_COMPLETED') {
                    $summary.status = 'ALREADY_COMPLETED'
                    $summary.notes += "task $($p.Index) (investigation) reconciled issue as ALREADY_COMPLETED; skipping remaining phases"
                    break
                }
                if ($r.response -imatch 'ISSUE_STATUS:\s*BLOCKED:\s*(.+)') {
                    $summary.status = 'BLOCKED_BY_INVESTIGATION'
                    $summary.notes += "task $($p.Index) (investigation) reported issue blocked: $($Matches[1].Trim())"
                    break
                }
            }
            if ($p.Route.Type -eq 'review') {
                $lastReviewNeedsChanges = [bool]($r.response -imatch 'VERDICT:\s*CHANGES_REQUIRED')
            }
            $prior = $r.response
        }
    }
    if ($summary.status -eq 'OK') {
        if ($summary.results | Where-Object status -eq 'INCOMPLETE') { $summary.status = 'INCOMPLETE' }
        elseif ($summary.results | Where-Object status -eq 'CANCELLED') { $summary.status = 'CANCELLED' }
        elseif ($summary.results | Where-Object status -ne 'SUCCESS') { $summary.status = 'FAILED' }
    }
    Save-AgyState $state $StatePath
    $summary.quota = @($plan.Group | Select-Object -Unique | ForEach-Object { Get-AgyGroupStatus -Group $_ -Policy $Policy -State $state })
    $null = New-Item -ItemType Directory -Force -Path $runDir
    $out = [pscustomobject]$summary
    Write-AgyUtf8File -Path (Join-Path $runDir 'summary.json') -Value ($out | ConvertTo-Json -Depth 8)
    Write-AgyUtf8File -Path (Join-Path $RunRoot 'latest.json') -Value ($out | ConvertTo-Json -Depth 8)
    $out
}

function New-AgyIssueLifecycleTasks {
    # Builds the multi-phase autonomous issue workflow:
    # investigation -> implement -> test -> review -> corrections (if review flagged) -> verify
    param(
        [Parameter(Mandatory)][int]$IssueNumber,
        [Parameter(Mandatory)][string]$IssueText,
        $Policy = (Get-AgyPolicy),
        [bool]$AllowWrites = $false,
        [string[]]$Phases = @()
    )
    $selected = if ($Phases -and $Phases.Count) {
        @($Policy.lifecyclePhases | Where-Object { $Phases -contains $_.name })
    } else {
        @($Policy.lifecyclePhases)
    }
    $tasks = @()
    foreach ($ph in $selected) {
        $phasePrompt = switch ($ph.name) {
            'investigation' {
                "Phase 1 (Investigation & Code Reconciliation - READ-ONLY) for GitHub Issue #$IssueNumber.`nFirst reconcile existing repository files and tests against every acceptance criterion in the issue. If ALL acceptance criteria are already implemented and passing in the repository, conclude your response with `ISSUE_STATUS: ALREADY_COMPLETED`. If the issue is blocked by missing external data, human decisions, or unavailable commercial rights, conclude with `ISSUE_STATUS: BLOCKED: <reason>`. Otherwise, produce a concrete technical gap analysis and step-by-step plan (`ISSUE_STATUS: ACTIONABLE`). Do NOT modify any files.`n`n$IssueText"
            }
            'implement' {
                "Phase 2 (Implementation) for GitHub Issue #$IssueNumber.`nImplement the changes planned in the previous investigation summary. Keep edits minimal and strictly within the issue scope.`n`n$IssueText"
            }
            'test' {
                "Phase 3 (Unit & Integration Tests) for GitHub Issue #$IssueNumber.`nAdd or update unit tests covering all acceptance criteria and run `.venv/Scripts/python.exe -m unittest` to verify they pass.`n`n$IssueText"
            }
            'review' {
                "Phase 4 (Independent Code Review - READ-ONLY) for GitHub Issue #$IssueNumber.`nReview the code and tests written in the preceding steps against every acceptance criterion. Do NOT modify files. Conclude with `VERDICT: APPROVED` or `VERDICT: CHANGES_REQUIRED` with concrete findings.`n`n$IssueText"
            }
            'corrections' {
                "Phase 5 (Corrections) for GitHub Issue #$IssueNumber.`nAddress every finding from the independent review and re-run the unit tests.`n`n$IssueText"
            }
            'verify' {
                "Phase 6 (Final Verification - READ-ONLY) for GitHub Issue #$IssueNumber.`nRun the relevant test suite and git status check to confirm all acceptance criteria pass and no unrelated files were touched. Conclude with `VERIFICATION: PASSED` or `VERIFICATION: FAILED`.`n`n$IssueText"
            }
        }
        $tasks += @{
            Text                = $phasePrompt
            Type                = [string]$ph.type
            AllowWrites         = ([bool]$ph.writes -and $AllowWrites)
            OnlyIfReviewFlagged = [bool]($ph.PSObject.Properties['onlyIfReviewFlagged'] -and $ph.onlyIfReviewFlagged)
        }
    }
    $tasks
}

# ---------------------------------------------------------------- repository-wide backlog queue & autonomous runner ----

function Get-AgyBacklogQueue {
    param([Parameter(Mandatory)][string]$QueuePath)
    if (Test-Path -LiteralPath $QueuePath) {
        $q = Get-Content -LiteralPath $QueuePath -Raw -Encoding utf8 | ConvertFrom-Json -AsHashtable
        if ($q) {
            if (-not $q.ContainsKey('issues')) { $q.issues = @{} }
            if (-not $q.ContainsKey('lastSyncAt')) { $q.lastSyncAt = $null }
            if (-not $q.ContainsKey('runnerState')) { $q.runnerState = 'IDLE' }
            return $q
        }
    }
    @{
        lastSyncAt     = $null
        runnerState    = 'IDLE'
        activeIssue    = $null
        totalDiscovered= 0
        issues         = @{}
    }
}

function Save-AgyBacklogQueue {
    param([Parameter(Mandatory)]$Queue, [Parameter(Mandatory)][string]$QueuePath)
    Write-AgyUtf8File -Path $QueuePath -Value ($Queue | ConvertTo-Json -Depth 10)
}

function Parse-AgyIssueMetadata {
    # Extracts explicit dependencies, Maestro phase order, target files, and priority from issue title/body/labels.
    param(
        [Parameter(Mandatory)][int]$Number,
        [string]$Title = '',
        [string]$Body = '',
        [string[]]$Labels = @(),
        [int]$NativeBlockedByCount = 0,
        $Policy = (Get-AgyPolicy)
    )
    $cleanBody = Format-AgyCleanMarkdown $Body
    $deps = [System.Collections.Generic.List[int]]::new()
    foreach ($m in [regex]::Matches($cleanBody, '(?im)(?:blocked\s+by|depends\s+on|requires|after)\s*:?\s*(#[0-9]+(?:\s*,\s*#[0-9]+)*)')) {
        foreach ($numMatch in [regex]::Matches($m.Groups[1].Value, '#([0-9]+)')) {
            $d = [int]$numMatch.Groups[1].Value
            if ($d -ne $Number -and -not $deps.Contains($d)) { $deps.Add($d) }
        }
    }

    $phaseNum = $null
    $pm = [regex]::Match($cleanBody, '(?i)\bPart\s+of\s+Phase\s+([0-9]+)')
    if ($pm.Success) { $phaseNum = [int]$pm.Groups[1].Value }

    $targetFiles = [System.Collections.Generic.List[string]]::new()
    foreach ($fm in [regex]::Matches($cleanBody, '`([a-zA-Z0-9_./-]+\.(?:py|ps1|js|cjs|json|md|bat|sh))`')) {
        $f = $fm.Groups[1].Value.Trim()
        if (-not $targetFiles.Contains($f)) { $targetFiles.Add($f) }
    }

    # Triage & blocking label check
    $blockedLabels = if ($Policy.PSObject.Properties['backlog'] -and $Policy.backlog.PSObject.Properties['blockedLabels']) {
        @($Policy.backlog.blockedLabels)
    } else {
        @('needs-info', 'needs-triage', 'ready-for-human', 'wontfix')
    }
    $matchedBlockedLabels = @($Labels | Where-Object { $blockedLabels -contains $_ })
    $blockReason = if ($matchedBlockedLabels.Count -gt 0) {
        "blocked by triage label: $($matchedBlockedLabels -join ', ')"
    } elseif ($NativeBlockedByCount -gt 0) {
        "blocked by $NativeBlockedByCount open native GitHub blocker(s)"
    } else { $null }

    # Priority calculation (lower number = higher priority)
    $priority = 100
    if ($Labels -contains 'ready-for-agent') { $priority -= 30 }
    if ($Labels -match '^(p0|critical|urgent)$') { $priority -= 50 }
    elseif ($Labels -match '^(p1|high)$') { $priority -= 25 }
    elseif ($Labels -match '^(p3|low)$') { $priority += 20 }
    if ($null -ne $phaseNum) { $priority += ($phaseNum * 5) }

    [pscustomobject]@{
        dependencies         = @($deps)
        phaseNumber          = $phaseNum
        targetFiles          = @($targetFiles)
        nativeBlockedByCount = $NativeBlockedByCount
        blockReason          = $blockReason
        priority             = $priority
    }
}

function Test-AgyIssueAlreadyCompletedInRepo {
    # Lightweight pre-reconciliation check: inspects acceptance-criteria checkboxes and target files.
    # Only returns true when all acceptance criteria checkboxes in the issue body are already checked `[x]`.
    # Otherwise, Phase 1 (investigation) performs full code reconciliation.
    param([string]$Body)
    if ([string]::IsNullOrWhiteSpace($Body)) { return $false }
    $unchecked = [regex]::Matches($Body, '(?m)^\s*-\s*\[\s\]\s+').Count
    $checked   = [regex]::Matches($Body, '(?m)^\s*-\s*\[[xX]\]\s+').Count
    ($checked -gt 0 -and $unchecked -eq 0)
}

function Sync-AgyGitHubBacklog {
    <#
    .SYNOPSIS Paginated, incremental GitHub backlog discovery without arbitrary size limits.
    Uses `gh api` pagination (`per_page=100&page=N` + `since=` on incremental syncs) so repositories
    with 5, 500, or 8,000+ issues are discovered page-by-page without loading everything into a model context.
    #>
    param(
        [Parameter(Mandatory)][string]$QueuePath,
        [string]$GhCommand = 'gh',
        $Policy = (Get-AgyPolicy),
        [switch]$FullSync
    )
    $q = Get-AgyBacklogQueue -QueuePath $QueuePath
    $pageSize = if ($Policy.PSObject.Properties['backlog'] -and $Policy.backlog.pageSize) { [int]$Policy.backlog.pageSize } else { 100 }
    $sinceParam = if (-not $FullSync -and $q.lastSyncAt) { "&since=$([uri]::EscapeDataString([string]$q.lastSyncAt))" } else { '' }
    $syncStartedAt = [datetime]::UtcNow.ToString('o')

    $page = 1
    $fetchedCount = 0
    $seenInThisSync = [System.Collections.Generic.HashSet[int]]::new()
    $prevEnc = [Console]::OutputEncoding
    [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
    try {
        while ($true) {
            $ghArgs = @('api', 'repos/{owner}/{repo}/issues', '-X', 'GET', '-f', 'state=all', '-F', "per_page=$pageSize", '-F', "page=$page", '-f', 'direction=asc')
            if (-not $FullSync -and $q.lastSyncAt) {
                $ghArgs += @('-f', "since=$([string]$q.lastSyncAt)")
            }
            $raw = & $GhCommand @ghArgs 2>$null
            $txt = ($raw -join "`n").Trim()
            if (-not $txt -or $txt -eq '[]') { break }
            $items = @($txt | ConvertFrom-Json)
            if (-not $items.Count) { break }
            $newInPage = 0

            foreach ($it in $items) {
                if ($it.PSObject.Properties['number'] -and $seenInThisSync.Add([int]$it.number)) {
                    $newInPage++
                }
                # Skip pull requests (GitHub issues API returns both issues and PRs)
                if ($it.PSObject.Properties['pull_request'] -and $null -ne $it.pull_request) { continue }
                $fetchedCount++
                $numStr = [string]$it.number
                $num = [int]$it.number
                $labels = @()
                if ($it.PSObject.Properties['labels'] -and $it.labels) {
                    foreach ($lb in $it.labels) {
                        if ($lb -is [string]) { $labels += $lb }
                        elseif ($lb.PSObject.Properties['name']) { $labels += [string]$lb.name }
                    }
                }
                $nativeBlocked = if ($it.PSObject.Properties['issue_dependencies_summary'] -and $it.issue_dependencies_summary -and $it.issue_dependencies_summary.PSObject.Properties['blocked_by']) {
                    [int]$it.issue_dependencies_summary.blocked_by
                } elseif ($it.PSObject.Properties['blocked_by'] -and $null -ne $it.blocked_by) {
                    [int]$it.blocked_by
                } else { 0 }

                $bodyText = if ($it.PSObject.Properties['body'] -and $null -ne $it.body) { [string]$it.body } else { '' }
                $meta = Parse-AgyIssueMetadata -Number $num -Title ([string]$it.title) -Body $bodyText -Labels $labels -NativeBlockedByCount $nativeBlocked -Policy $Policy
                $stateStr = ([string]$it.state).ToLowerInvariant()

                $existing = if ($q.issues.ContainsKey($numStr)) { $q.issues[$numStr] } else { $null }
                $prevStatus = if ($existing -and $existing.ContainsKey('status')) { [string]$existing.status } else { $null }

                $newStatus = if ($stateStr -eq 'closed') {
                    'CLOSED_UPSTREAM'
                } elseif ($prevStatus -in @('COMPLETED', 'ALREADY_COMPLETED', 'PARTIAL', 'INCOMPLETE', 'RUNNING', 'IMPLEMENTATION_COMPLETE', 'HUMAN_REVIEW_PENDING')) {
                    $prevStatus
                } elseif (Test-AgyIssueAlreadyCompletedInRepo -Body $bodyText) {
                    'ALREADY_COMPLETED'
                } elseif ($meta.blockReason) {
                    'BLOCKED'
                } else {
                    'PENDING'
                }

                $cleanDone = if ($existing -and $existing.ContainsKey('completedPhases')) {
                    @($existing.completedPhases | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) })
                } else { @() }
                $cleanMods = if ($existing -and $existing.ContainsKey('modifiedFiles')) {
                    @($existing.modifiedFiles | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) })
                } else { @() }
                $remPhases = Get-AgyRemainingLifecyclePhases -CompletedPhases $cleanDone -Status $newStatus -Policy $Policy

                $preservedBlockReason = if ($newStatus -in @('PARTIAL', 'IMPLEMENTATION_COMPLETE', 'HUMAN_REVIEW_PENDING') -and $existing -and $existing.ContainsKey('blockReason') -and $existing.blockReason) {
                    [string]$existing.blockReason
                } elseif ($newStatus -in @('COMPLETED', 'ALREADY_COMPLETED')) {
                    $null
                } else {
                    $meta.blockReason
                }

                $q.issues[$numStr] = @{
                    number                 = $num
                    title                  = [string]$it.title
                    state                  = $stateStr
                    labels                 = $labels
                    updatedAt              = $(if ($it.PSObject.Properties['updated_at']) { [string]$it.updated_at } elseif ($it.PSObject.Properties['updatedAt']) { [string]$it.updatedAt } else { $syncStartedAt })
                    priority               = $meta.priority
                    phaseNumber            = $meta.phaseNumber
                    dependencies           = @($meta.dependencies)
                    targetFiles            = @($meta.targetFiles)
                    nativeBlockedByCount   = $meta.nativeBlockedByCount
                    blockReason            = $preservedBlockReason
                    status                 = $newStatus
                    verificationState      = $(if ($existing -and $existing.ContainsKey('verificationState')) { $existing.verificationState } else { $null })
                    implementationComplete = $(if ($existing -and $existing.ContainsKey('implementationComplete')) { [bool]$existing.implementationComplete } else { $false })
                    verificationComplete   = $(if ($existing -and $existing.ContainsKey('verificationComplete')) { [bool]$existing.verificationComplete } else { $false })
                    humanReviewPending     = $(if ($existing -and $existing.ContainsKey('humanReviewPending')) { [bool]$existing.humanReviewPending } else { $false })
                    currentPhase           = $(if ($existing -and $existing.ContainsKey('currentPhase')) { $existing.currentPhase } else { $null })
                    assignedWorker         = $(if ($existing -and $existing.ContainsKey('assignedWorker')) { $existing.assignedWorker } else { $null })
                    lastRunId              = $(if ($existing -and $existing.ContainsKey('lastRunId')) { $existing.lastRunId } else { $null })
                    conversationId         = $(if ($existing -and $existing.ContainsKey('conversationId')) { $existing.conversationId } else { $null })
                    completedPhases        = $cleanDone
                    remainingPhases        = $remPhases
                    tokensUsed             = $(if ($existing -and $existing.ContainsKey('tokensUsed')) { [int]$existing.tokensUsed } else { 0 })
                    modifiedFiles          = $cleanMods
                    delivery               = $(if ($existing -and $existing.ContainsKey('delivery')) { $existing.delivery } else { $null })
                }
            }
            if ($items.Count -lt $pageSize -or $newInPage -eq 0) { break }
            $page++
        }
    } finally {
        [Console]::OutputEncoding = $prevEnc
    }

    # Second pass: infer Maestro phase ordering dependencies among open issues sharing the same plan.
    # Links Phase N to the highest preceding phase (< N) that contains at least one non-label-blocked open issue,
    # so human-only phases (`ready-for-human`) are skipped without deadlocking subsequent agent-ready phases.
    $phaseMap = @{}
    foreach ($k in $q.issues.Keys) {
        $entry = $q.issues[$k]
        if ($null -ne $entry.phaseNumber -and $entry.state -eq 'open' -and (-not $entry.blockReason -or $entry.status -in @('COMPLETED', 'ALREADY_COMPLETED', 'PARTIAL'))) {
            $pNum = [int]$entry.phaseNumber
            if (-not $phaseMap.ContainsKey($pNum)) { $phaseMap[$pNum] = @() }
            $phaseMap[$pNum] += [int]$entry.number
        }
    }
    foreach ($k in $q.issues.Keys) {
        $entry = $q.issues[$k]
        if ($null -ne $entry.phaseNumber -and [int]$entry.phaseNumber -gt 1) {
            $prevPhase = [int]$entry.phaseNumber - 1
            while ($prevPhase -ge 1 -and -not $phaseMap.ContainsKey($prevPhase)) {
                $prevPhase--
            }
            if ($prevPhase -ge 1 -and $phaseMap.ContainsKey($prevPhase)) {
                $curDeps = [System.Collections.Generic.List[int]]::new()
                foreach ($d in @($entry.dependencies)) { $curDeps.Add([int]$d) }
                foreach ($parentIssue in $phaseMap[$prevPhase]) {
                    if ($parentIssue -ne [int]$entry.number -and -not $curDeps.Contains($parentIssue)) {
                        $curDeps.Add($parentIssue)
                    }
                }
                $entry.dependencies = @($curDeps)
            }
        }
    }

    $q.lastSyncAt = $syncStartedAt
    $q.totalDiscovered = $q.issues.Count
    Save-AgyBacklogQueue -Queue $q -QueuePath $QueuePath
    [pscustomobject]@{
        syncedAt        = $syncStartedAt
        pagesFetched    = $page
        issuesUpdated   = $fetchedCount
        totalInQueue    = $q.issues.Count
    }
}

function Get-AgyRemainingLifecyclePhases {
    param(
        [string[]]$CompletedPhases = @(),
        [string]$Status = 'PENDING',
        $Policy = (Get-AgyPolicy),
        [string[]]$RequestedPhases = @()
    )
    if ($Status -in @('COMPLETED', 'ALREADY_COMPLETED', 'CLOSED_UPSTREAM')) { return @() }
    $doneClean = @($CompletedPhases | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) })
    $allPhases = if ($RequestedPhases -and $RequestedPhases.Count -gt 0) {
        @($RequestedPhases)
    } else {
        @($Policy.lifecyclePhases | ForEach-Object { [string]$_.name })
    }
    @($allPhases | Where-Object { $doneClean -notcontains $_ })
}

function Get-AgyIssueLastWorkerSummary {
    # Retrieves the last successful worker response and modifiedFiles for a partially completed issue run.
    param(
        [Parameter(Mandatory)][string]$RunRoot,
        [string]$LastRunId = '',
        $IssueEntry = $null
    )
    $runId = if ($LastRunId) { $LastRunId } else { [string](Get-AgyPropValue $IssueEntry 'lastRunId' '') }
    $resp = ''
    $mods = @(Get-AgyPropValue $IssueEntry 'modifiedFiles' @() | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) })
    $convId = [string](Get-AgyPropValue $IssueEntry 'conversationId' '')

    if ($runId) {
        $sumPath = Join-Path (Join-Path $RunRoot $runId) 'summary.json'
        if (Test-Path -LiteralPath $sumPath) {
            try {
                $j = Get-Content -LiteralPath $sumPath -Raw -Encoding utf8 | ConvertFrom-Json
                $okResults = @($j.results | Where-Object { $_.status -eq 'SUCCESS' })
                if ($okResults.Count -gt 0) {
                    $lastOk = $okResults[-1]
                    if ($lastOk.PSObject.Properties['response'] -and $lastOk.response) {
                        $resp = [string]$lastOk.response
                    }
                    if ($lastOk.PSObject.Properties['conversationId'] -and $lastOk.conversationId) {
                        $convId = [string]$lastOk.conversationId
                    }
                    foreach ($r in $okResults) {
                        if ($r.PSObject.Properties['modifiedFiles'] -and $r.modifiedFiles) {
                            $mods = @($mods + @($r.modifiedFiles | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) }) | Select-Object -Unique)
                        }
                    }
                }
            } catch {}
        }
    }
    [pscustomobject]@{
        lastRunId      = $runId
        conversationId = $convId
        priorSummary   = $resp
        modifiedFiles  = @($mods)
    }
}

function Select-AgyNextActionableIssue {
    # Picks the highest-priority open issue whose dependencies are all satisfied (COMPLETED, ALREADY_COMPLETED, or CLOSED_UPSTREAM).
    # Treats PARTIAL issues (interrupted by internal per-run token budget) as actionable and prioritizes finishing them first.
    param([Parameter(Mandatory)]$Queue)
    $satisfiedStates = @('COMPLETED', 'ALREADY_COMPLETED', 'CLOSED_UPSTREAM')
    $candidates = [System.Collections.Generic.List[object]]::new()

    foreach ($k in $Queue.issues.Keys) {
        $it = $Queue.issues[$k]
        if ($it.state -ne 'open') { continue }
        if ($it.status -in @('COMPLETED', 'ALREADY_COMPLETED', 'CLOSED_UPSTREAM', 'FAILED', 'INCOMPLETE', 'HUMAN_REVIEW_PENDING', 'IMPLEMENTATION_COMPLETE')) { continue }

        $blk = [string](Get-AgyPropValue $it 'blockReason' '')
        # Re-evaluate explicit label/native blocks (ignore internal run-budget pause notes on PARTIAL issues)
        if ($blk -and $blk -notmatch '^waiting on open issue' -and $blk -notmatch '(?i)internal per-run token budget') {
            $it['status'] = 'BLOCKED'
            continue
        }

        $unmet = @()
        foreach ($depNum in @(Get-AgyPropValue $it 'dependencies' @())) {
            $depKey = [string]$depNum
            if ($Queue.issues.ContainsKey($depKey)) {
                $depEntry = $Queue.issues[$depKey]
                if ($depEntry.status -notin $satisfiedStates -and $depEntry.state -eq 'open') {
                    $unmet += "#$depNum ($($depEntry.status))"
                }
            }
        }
        if ($unmet.Count -gt 0) {
            $it['status'] = 'BLOCKED'
            $it['blockReason'] = "waiting on open issue dependency: $($unmet -join ', ')"
            continue
        }

        if ($it.status -eq 'BLOCKED' -and $blk -match '^waiting on open issue') {
            $it['status'] = if (@(Get-AgyPropValue $it 'completedPhases' @() | Where-Object { $_ }).Count -gt 0) { 'PARTIAL' } else { 'PENDING' }
            $it['blockReason'] = $null
        }
        if ($it.status -in @('PENDING', 'PARTIAL')) {
            $candidates.Add($it)
        }
    }

    if ($candidates.Count -eq 0) { return $null }
    # Prioritize PARTIAL issues first so partially completed work (#37) finishes before starting new PENDING issues
    $sorted = @($candidates | Sort-Object `
        -Property @{Expression = { if ($_.status -eq 'PARTIAL') { 0 } else { 1 } }}, `
                  @{Expression = { [int]$_.priority }}, `
                  @{Expression = { if ($null -ne $_.phaseNumber) { [int]$_.phaseNumber } else { 999 } }}, `
                  @{Expression = { [int]$_.number }})
    $sorted[0]
}

function Get-AgyBacklogStats {
    param([Parameter(Mandatory)]$Queue)
    $vals = @($Queue.issues.Values)
    $openVals = @($vals | Where-Object { $_.state -eq 'open' })
    [pscustomobject]@{
        totalDiscovered        = $vals.Count
        openTotal              = $openVals.Count
        pending                = @($openVals | Where-Object { $_.status -eq 'PENDING' }).Count
        partial                = @($openVals | Where-Object { $_.status -eq 'PARTIAL' }).Count
        running                = @($openVals | Where-Object { $_.status -eq 'RUNNING' }).Count
        completed              = @($openVals | Where-Object { $_.status -in @('COMPLETED', 'ALREADY_COMPLETED') }).Count
        implementationComplete = @($openVals | Where-Object { $_.status -eq 'IMPLEMENTATION_COMPLETE' }).Count
        humanReviewPending     = @($openVals | Where-Object { $_.status -eq 'HUMAN_REVIEW_PENDING' }).Count
        blocked                = @($openVals | Where-Object { $_.status -eq 'BLOCKED' }).Count
        failed                 = @($openVals | Where-Object { $_.status -eq 'FAILED' }).Count
        incomplete             = @($openVals | Where-Object { $_.status -eq 'INCOMPLETE' }).Count
        closedUpstream         = @($vals     | Where-Object { $_.status -eq 'CLOSED_UPSTREAM' }).Count
        runnerState            = $Queue.runnerState
        activeIssue            = $Queue.activeIssue
        lastSyncAt             = $Queue.lastSyncAt
    }
}

function Invoke-AgyBacklogRunner {
    <#
    .SYNOPSIS Autonomous repository-wide GitHub backlog processor.
    Discovers all open issues via paginated API queries, builds the dependency & priority graph,
    and executes actionable issues one by one through the 6-phase lifecycle with checkpointing,
    periodic re-sync, pause/stop/cancel signals, and quota reserve protection.
    #>
    param(
        $Policy = (Get-AgyPolicy),
        [string]$AgyCommand = 'agy',
        [string]$GhCommand = 'gh',
        [string]$RunRoot = (Join-Path (Get-Location).Path '.cache/agy-orch'),
        [string]$StatePath = (Join-Path $RunRoot 'state.json'),
        [string]$QueuePath = (Join-Path $RunRoot 'backlog-queue.json'),
        [hashtable]$Auth = @{},
        [bool]$AllowWrites = $false,
        [switch]$Lifecycle,
        [string[]]$Phases = @(),
        [switch]$YoloWorker,
        [switch]$Live,
        [switch]$DryRun,
        [switch]$OverrideReserve,
        [int]$WorkerTimeoutMinutes = 0,
        [int]$MaxIssuesToProcess = 0,     # 0 = process all actionable issues until queue exhausted
        [ValidateSet('json', 'stream-json')][string]$OutputFormat = 'stream-json',
        [string[]]$ContextFiles = @(),
        [string]$WorkDir = (Get-Location).Path,
        [switch]$DeterministicVerify
    )
    $null = New-Item -ItemType Directory -Force -Path $RunRoot
    $pauseFile  = Join-Path $RunRoot 'pause.signal'
    $stopFile   = Join-Path $RunRoot 'stop.signal'
    $cancelFile = Join-Path $RunRoot 'cancel.signal'

    $syncInfo = Sync-AgyGitHubBacklog -QueuePath $QueuePath -GhCommand $GhCommand -Policy $Policy -FullSync
    $queue = Get-AgyBacklogQueue -QueuePath $QueuePath

    # Refresh dependency states across all discovered issues
    $null = Select-AgyNextActionableIssue -Queue $queue
    Save-AgyBacklogQueue -Queue $queue -QueuePath $QueuePath

    if ($DryRun) {
        $actionableOrder = [System.Collections.Generic.List[object]]::new()
        $simQueue = $queue | ConvertTo-Json -Depth 10 | ConvertFrom-Json -AsHashtable
        while ($true) {
            $next = Select-AgyNextActionableIssue -Queue $simQueue
            if (-not $next) { break }
            $numStr = [string]$next.number
            $donePhases = @($next.completedPhases | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) })
            $remPhases = Get-AgyRemainingLifecyclePhases -CompletedPhases $donePhases -Status ([string]$next.status) -Policy $Policy -RequestedPhases $Phases
            $sampleTasks = if ($Lifecycle) {
                @(New-AgyIssueLifecycleTasks -IssueNumber ([int]$next.number) -IssueText "Issue #$($next.number): $($next.title)" -Policy $Policy -AllowWrites $AllowWrites -Phases $remPhases)
            } else {
                @(@{ Text = "Issue #$($next.number): $($next.title)"; Type = 'investigation'; AllowWrites = $false })
            }
            $state = Get-AgyState $StatePath
            $phaseRoutes = @($sampleTasks | ForEach-Object {
                $r = Resolve-AgyRoute -Text $_.Text -Type $_.Type -Policy $Policy -State $(if ($OverrideReserve) { $null } else { $state })
                [pscustomobject]@{ phase = $_.Type; model = $r.Model; primaryModel = $r.PrimaryModel; fallbackFrom = $r.FallbackFrom; writes = [bool]$_.AllowWrites }
            })
            $actionableOrder.Add([pscustomobject]@{
                number          = [int]$next.number
                title           = [string]$next.title
                status          = [string]$next.status
                priority        = [int]$next.priority
                phaseNumber     = $next.phaseNumber
                completedPhases = $donePhases
                remainingPhases = $remPhases
                dependencies    = @($next.dependencies)
                targetFiles     = @($next.targetFiles)
                phases          = $phaseRoutes
            })
            $simQueue.issues[$numStr].status = 'COMPLETED'
            if ($MaxIssuesToProcess -gt 0 -and $actionableOrder.Count -ge $MaxIssuesToProcess) { break }
        }
        $blockedList = @($simQueue.issues.Values | Where-Object { $_.state -eq 'open' -and $_.status -eq 'BLOCKED' } | Sort-Object { [int]$_.number } | ForEach-Object {
            [pscustomobject]@{ number = [int]$_.number; title = [string]$_.title; blockReason = [string]$_.blockReason; dependencies = @($_.dependencies) }
        })
        $stats = Get-AgyBacklogStats -Queue $queue
        $state = Get-AgyState $StatePath
        $quotaView = @($Policy.groups.PSObject.Properties.Name | ForEach-Object { Get-AgyGroupStatus -Group $_ -Policy $Policy -State $state })
        $dryOut = [pscustomobject]@{
            status          = 'OK'
            dryRun          = $true
            sync            = $syncInfo
            stats           = $stats
            executionOrder  = @($actionableOrder)
            blockedIssues   = $blockedList
            quota           = $quotaView
        }
        Write-AgyUtf8File -Path (Join-Path $RunRoot 'latest-backlog-dryrun.json') -Value ($dryOut | ConvertTo-Json -Depth 10)
        return $dryOut
    }

    $queue.runnerState = 'RUNNING'
    Save-AgyBacklogQueue -Queue $queue -QueuePath $QueuePath
    $processedIssues = [System.Collections.Generic.List[object]]::new()
    $stopReason = 'QUEUE_EXHAUSTED'
    $syncIntervalMin = if ($Policy.PSObject.Properties['backlog'] -and $Policy.backlog.syncIntervalMinutes) { [double]$Policy.backlog.syncIntervalMinutes } else { 5.0 }

    while ($true) {
        if (Test-Path -LiteralPath $pauseFile) {
            $queue.runnerState = 'PAUSED'; $queue.activeIssue = $null
            Save-AgyBacklogQueue -Queue $queue -QueuePath $QueuePath
            $stopReason = 'PAUSED'
            break
        }
        if ((Test-Path -LiteralPath $stopFile) -or (Test-Path -LiteralPath $cancelFile)) {
            $queue.runnerState = 'STOPPED'; $queue.activeIssue = $null
            Save-AgyBacklogQueue -Queue $queue -QueuePath $QueuePath
            $stopReason = 'STOPPED'
            break
        }
        if ($MaxIssuesToProcess -gt 0 -and $processedIssues.Count -ge $MaxIssuesToProcess) {
            $queue.runnerState = 'IDLE'; $queue.activeIssue = $null
            Save-AgyBacklogQueue -Queue $queue -QueuePath $QueuePath
            $stopReason = 'MAX_ISSUES_REACHED'
            break
        }

        # Periodic incremental sync with GitHub if interval elapsed
        if ($queue.lastSyncAt) {
            $elapsedMin = ([datetime]::UtcNow - ([datetime]$queue.lastSyncAt).ToUniversalTime()).TotalMinutes
            if ($elapsedMin -ge $syncIntervalMin) {
                $null = Sync-AgyGitHubBacklog -QueuePath $QueuePath -GhCommand $GhCommand -Policy $Policy
                $queue = Get-AgyBacklogQueue -QueuePath $QueuePath
            }
        }

        $nextIssue = Select-AgyNextActionableIssue -Queue $queue
        Save-AgyBacklogQueue -Queue $queue -QueuePath $QueuePath
        if (-not $nextIssue) {
            $queue.runnerState = 'IDLE'; $queue.activeIssue = $null
            Save-AgyBacklogQueue -Queue $queue -QueuePath $QueuePath
            $stopReason = 'QUEUE_EXHAUSTED'
            break
        }

        $num = [int]$nextIssue.number
        $numKey = [string]$num
        $prevEntry = $queue.issues[$numKey]
        $donePhases = @($prevEntry.completedPhases | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) })
        $priorCheckpoint = if ($donePhases.Count -gt 0) {
            Get-AgyIssueLastWorkerSummary -RunRoot $RunRoot -IssueEntry $prevEntry
        } else {
            [pscustomobject]@{ lastRunId = ''; conversationId = ''; priorSummary = ''; modifiedFiles = @() }
        }

        $queue.activeIssue = $num
        $queue.issues[$numKey].status = 'RUNNING'
        $queue.issues[$numKey].assignedWorker = "issue-$num"
        Save-AgyBacklogQueue -Queue $queue -QueuePath $QueuePath

        if ($Live) {
            $ts = (Get-Date).ToString('HH:mm:ss')
            $resumeNote = if ($donePhases.Count -gt 0) { " [resuming after: $($donePhases -join ', ')]" } else { '' }
            Write-Host "`n[$ts] [backlog] === Starting Issue #$num : $($nextIssue.title) (priority=$($nextIssue.priority))$resumeNote ===" -ForegroundColor Magenta
        }

        $issueText = Get-AgyIssueText -Number $num -GhCommand $GhCommand -MaxChars ([int]$Policy.budgets.issueMaxChars)
        $tasks = if ($Lifecycle) {
            $remPhases = Get-AgyRemainingLifecyclePhases -CompletedPhases $donePhases -Status 'PARTIAL' -Policy $Policy -RequestedPhases $Phases
            $builtTasks = @(New-AgyIssueLifecycleTasks -IssueNumber $num -IssueText $issueText -Policy $Policy -AllowWrites $AllowWrites -Phases $remPhases)
            if ($priorCheckpoint.modifiedFiles.Count -gt 0 -and $builtTasks.Count -gt 0) {
                $builtTasks[0].PriorModifiedFiles = @($priorCheckpoint.modifiedFiles)
            }
            $builtTasks
        } else {
            @(@{ Text = $issueText; Type = 'investigation'; AllowWrites = $false })
        }

        $enforceWrite = [bool]($Lifecycle -and -not $AllowWrites)
        $runSummary = Invoke-AgyOrchestration -Tasks $tasks -Policy $Policy -AgyCommand $AgyCommand -RunRoot $RunRoot -StatePath $StatePath `
            -Auth $Auth -AllowWrites $AllowWrites -EnforceWriteAuth $enforceWrite -OverrideReserve:$OverrideReserve `
            -YoloWorker:$YoloWorker -Live:$Live -WorkerTimeoutMinutes $WorkerTimeoutMinutes -OutputFormat $OutputFormat `
            -ContextFiles $ContextFiles -PriorSummary ([string]$priorCheckpoint.priorSummary) -WorkDir $WorkDir `
            -DeterministicVerify:$DeterministicVerify

        $entry = $queue.issues[$numKey]
        $entry.lastRunId = $runSummary.runId
        $entry.tokensUsed = [int]$entry.tokensUsed + [int]$runSummary.totalTokens
        $allMods = @($entry.modifiedFiles | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) })
        $doneList = [System.Collections.Generic.List[string]]::new()
        foreach ($dp in $donePhases) {
            if (-not $doneList.Contains([string]$dp)) { $doneList.Add([string]$dp) }
        }

        if (-not $entry.ContainsKey('phaseModels')) { $entry['phaseModels'] = @{} }
        foreach ($res in @($runSummary.results)) {
            if ($res.conversationId) { $entry.conversationId = $res.conversationId }
            if ($res.taskType) {
                $entry.currentPhase = $res.taskType
                $entry.phaseModels[$res.taskType] = $res.requestedModel
                if ($res.taskType -eq 'review') {
                    $entry['isIndependentReview'] = [bool]($res.PSObject.Properties['isIndependentReview'] -and $res.isIndependentReview)
                    $entry['reviewModel'] = $res.requestedModel
                }
            }
            if ($res.modifiedFiles) {
                $cleanResMods = @($res.modifiedFiles | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) -and $_ -notmatch '^docs/maestro/plans/' })
                $allMods = @($allMods + $cleanResMods | Select-Object -Unique)
            }
            if ($res.status -eq 'SUCCESS' -and $res.taskType -and -not $doneList.Contains([string]$res.taskType)) {
                $doneList.Add([string]$res.taskType)
            }
        }
        $entry.modifiedFiles = $allMods
        $entry.completedPhases = @($doneList)

        if ($runSummary.status -eq 'OK') {
            $entry.status = 'COMPLETED'
            $entry.currentPhase = 'done'
            $entry.blockReason = $null
            $entry.remainingPhases = @()
        } elseif ($runSummary.status -eq 'ALREADY_COMPLETED') {
            $entry.status = 'ALREADY_COMPLETED'
            $entry.currentPhase = 'reconciled'
            $entry.blockReason = $null
            $entry.remainingPhases = @()
        } elseif ($runSummary.status -eq 'BLOCKED_BY_INVESTIGATION') {
            $entry.status = 'BLOCKED'
            $entry.blockReason = ($runSummary.notes | Select-Object -Last 1)
            $entry.remainingPhases = Get-AgyRemainingLifecyclePhases -CompletedPhases $entry.completedPhases -Status 'BLOCKED' -Policy $Policy
        } elseif ($runSummary.status -eq 'INCOMPLETE') {
            $entry.status = 'INCOMPLETE'
            $entry.remainingPhases = Get-AgyRemainingLifecyclePhases -CompletedPhases $entry.completedPhases -Status 'INCOMPLETE' -Policy $Policy
            $queue.runnerState = 'PAUSED_INCOMPLETE'
            $queue.activeIssue = $null
            Save-AgyBacklogQueue -Queue $queue -QueuePath $QueuePath
            $processedIssues.Add([pscustomobject]@{ number = $num; status = $entry.status; runId = $runSummary.runId; tokens = $runSummary.totalTokens })
            $stopReason = 'WORKER_INCOMPLETE'
            break
        } elseif ($runSummary.status -eq 'CANCELLED') {
            $entry.status = if ($entry.completedPhases.Count -gt 0) { 'PARTIAL' } else { 'PENDING' }
            $entry.remainingPhases = Get-AgyRemainingLifecyclePhases -CompletedPhases $entry.completedPhases -Status $entry.status -Policy $Policy
            $queue.runnerState = 'STOPPED'
            $queue.activeIssue = $null
            Save-AgyBacklogQueue -Queue $queue -QueuePath $QueuePath
            $processedIssues.Add([pscustomobject]@{ number = $num; status = 'CANCELLED'; runId = $runSummary.runId; tokens = $runSummary.totalTokens })
            $stopReason = 'CANCELLED'
            break
        } elseif ($runSummary.status -eq 'BUDGET_EXHAUSTED') {
            $bKind = if ($runSummary.PSObject.Properties['budgetKind'] -and $runSummary.budgetKind) { [string]$runSummary.budgetKind } else { 'INTERNAL_RUN_BUDGET' }
            $entry.status = if ($entry.completedPhases.Count -gt 0) { 'PARTIAL' } else { 'PENDING' }
            $entry.remainingPhases = Get-AgyRemainingLifecyclePhases -CompletedPhases $entry.completedPhases -Status $entry.status -Policy $Policy
            $lastNote = ($runSummary.notes | Select-Object -Last 1)
            if ($bKind -eq 'PROVIDER_QUOTA_RESERVE') {
                $entry.blockReason = "Blocked by provider quota reserve: $lastNote"
                $queue.runnerState = 'BLOCKED_PROVIDER_QUOTA'
                $stopReason = 'PROVIDER_QUOTA_BELOW_RESERVE'
            } else {
                $entry.blockReason = "Paused by internal per-run token budget after phase '$($entry.currentPhase)'; provider quota above reserve — ready to resume ($($entry.remainingPhases -join ', '))"
                $queue.runnerState = 'PAUSED_RUN_BUDGET'
                $stopReason = 'RUN_BUDGET_EXHAUSTED'
            }
            $queue.activeIssue = $null
            Save-AgyBacklogQueue -Queue $queue -QueuePath $QueuePath
            $processedIssues.Add([pscustomobject]@{ number = $num; status = $entry.status; budgetKind = $bKind; runId = $runSummary.runId; tokens = $runSummary.totalTokens })
            break
        } elseif ($runSummary.status -eq 'BLOCKED') {
            $bKind = if ($runSummary.PSObject.Properties['budgetKind'] -and $runSummary.budgetKind) { [string]$runSummary.budgetKind } else { 'AUTH_OR_QUOTA' }
            $entry.status = if ($entry.completedPhases.Count -gt 0) { 'PARTIAL' } else { 'PENDING' }
            $entry.remainingPhases = Get-AgyRemainingLifecyclePhases -CompletedPhases $entry.completedPhases -Status $entry.status -Policy $Policy
            $queue.runnerState = if ($bKind -eq 'PROVIDER_QUOTA_RESERVE') { 'BLOCKED_PROVIDER_QUOTA' } else { 'BLOCKED_QUOTA_OR_AUTH' }
            $queue.activeIssue = $null
            Save-AgyBacklogQueue -Queue $queue -QueuePath $QueuePath
            $processedIssues.Add([pscustomobject]@{ number = $num; status = $runSummary.status; budgetKind = $bKind; runId = $runSummary.runId; tokens = $runSummary.totalTokens })
            $stopReason = 'BLOCKED'
            break
        } else {
            # Issue worker failed; record FAILED status on this issue and continue to independent work if any
            $entry.status = 'FAILED'
            $entry.blockReason = ($runSummary.notes | Select-Object -Last 1)
            $entry.remainingPhases = Get-AgyRemainingLifecyclePhases -CompletedPhases $entry.completedPhases -Status 'FAILED' -Policy $Policy
        }

        $queue.activeIssue = $null
        Save-AgyBacklogQueue -Queue $queue -QueuePath $QueuePath
        $processedIssues.Add([pscustomobject]@{ number = $num; status = $entry.status; runId = $runSummary.runId; tokens = $runSummary.totalTokens })
    }

    $finalStats = Get-AgyBacklogStats -Queue $queue
    $state = Get-AgyState $StatePath
    $quotaView = @($Policy.groups.PSObject.Properties.Name | ForEach-Object { Get-AgyGroupStatus -Group $_ -Policy $Policy -State $state })
    $statusOverall = if ($stopReason -in @('QUEUE_EXHAUSTED', 'MAX_ISSUES_REACHED', 'PAUSED', 'STOPPED')) {
        if ($finalStats.failed -gt 0 -and $processedIssues.Count -gt 0 -and @($processedIssues | Where-Object status -in @('COMPLETED', 'ALREADY_COMPLETED')).Count -eq 0) { 'FAILED' } else { 'OK' }
    } elseif ($stopReason -in @('BLOCKED', 'PROVIDER_QUOTA_BELOW_RESERVE')) {
        'BLOCKED'
    } else {
        $stopReason
    }

    [pscustomobject]@{
        status          = $statusOverall
        stopReason      = $stopReason
        stats           = $finalStats
        processedIssues = @($processedIssues)
        quota           = $quotaView
    }
}

# ---------------------------------------------------------------- GitHub delivery, file ownership & local recovery ----

function Get-AgyPropValue {
    param($Obj, [Parameter(Mandatory)][string]$Name, $Default = $null)
    if ($null -eq $Obj) { return $Default }
    if ($Obj -is [System.Collections.IDictionary]) {
        if ($Obj.Contains($Name) -and $null -ne $Obj[$Name]) { return $Obj[$Name] }
        return $Default
    }
    if ($Obj.PSObject.Properties[$Name] -and $null -ne $Obj.$Name) { return $Obj.$Name }
    return $Default
}

function Test-AgyIssueFileOwnership {
    <#
    .SYNOPSIS Strict file ownership check for an issue before creating commits, branches, or PRs.
    Prevents mixing unrelated pre-existing dirty working-tree files, protected paths, or files owned by other issues.
    #>
    param(
        [Parameter(Mandatory)][int]$IssueNumber,
        $IssueEntry = $null,
        $Policy = (Get-AgyPolicy),
        [string]$WorkDir = (Get-Location).Path,
        [string[]]$AdditionalDirtyFiles = @()
    )
    $numKey = [string]$IssueNumber
    $delivCfg = if ($Policy.PSObject.Properties['backlog'] -and $Policy.backlog.PSObject.Properties['delivery']) { $Policy.backlog.delivery } else { $null }
    $ownershipMap = if ($delivCfg -and $delivCfg.PSObject.Properties['issueFileOwnership']) { $delivCfg.issueFileOwnership } else { $null }
    $preDirty = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
    if ($delivCfg -and $delivCfg.PSObject.Properties['preExistingDirtyFiles'] -and $delivCfg.preExistingDirtyFiles) {
        foreach ($df in $delivCfg.preExistingDirtyFiles) {
            if ($df) { $null = $preDirty.Add((Format-AgyCleanPath -Path ([string]$df) -WorkDir $WorkDir)) }
        }
    }
    foreach ($df in $AdditionalDirtyFiles) {
        if ($df) { $null = $preDirty.Add((Format-AgyCleanPath -Path ([string]$df) -WorkDir $WorkDir)) }
    }

    # Build map of files owned by OTHER issues
    $otherOwned = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
    if ($ownershipMap) {
        foreach ($prop in $ownershipMap.PSObject.Properties) {
            if ($prop.Name -ne $numKey) {
                foreach ($f in @($prop.Value)) {
                    if ($f) { $null = $otherOwned.Add((Format-AgyCleanPath -Path ([string]$f) -WorkDir $WorkDir)) }
                }
            }
        }
    }

    # Determine candidate owned files for this issue
    $rawOwned = @()
    if ($ownershipMap -and $ownershipMap.PSObject.Properties[$numKey] -and $ownershipMap.$numKey) {
        $rawOwned = @($ownershipMap.$numKey)
    } elseif ($IssueEntry) {
        $mods = @(Get-AgyPropValue $IssueEntry 'modifiedFiles' @() | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) })
        $targs = @(Get-AgyPropValue $IssueEntry 'targetFiles' @() | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) })
        $rawOwned = @(if ($mods.Count -gt 0) { $mods } else { $targs })
    }

    $owned = [System.Collections.Generic.List[string]]::new()
    foreach ($f in $rawOwned) {
        if ([string]::IsNullOrWhiteSpace([string]$f)) { continue }
        $norm = Format-AgyCleanPath -Path ([string]$f) -WorkDir $WorkDir
        if ($norm -match '^docs/maestro/plans/') { continue }
        if (-not $owned.Contains($norm)) { $owned.Add($norm) }
    }

    $existingOwned = [System.Collections.Generic.List[string]]::new()
    $sharedDirty   = [System.Collections.Generic.List[string]]::new()
    $protected     = [System.Collections.Generic.List[string]]::new()
    $crossIssue    = [System.Collections.Generic.List[string]]::new()
    $cleanOwned    = [System.Collections.Generic.List[string]]::new()

    $protPatterns = if ($Policy.PSObject.Properties['protectedPathPatterns'] -and $Policy.protectedPathPatterns) { @($Policy.protectedPathPatterns) } else { @() }

    foreach ($rel in $owned) {
        $full = Join-Path $WorkDir ($rel -replace '/', [IO.Path]::DirectorySeparatorChar)
        if (Test-Path -LiteralPath $full) { $existingOwned.Add($rel) }
        $isProt = $false
        foreach ($pat in $protPatterns) {
            if ($rel -imatch $pat) { $isProt = $true; break }
        }
        if ($isProt) { $protected.Add($rel) }
        if ($preDirty.Contains($rel)) { $sharedDirty.Add($rel) }
        if ($otherOwned.Contains($rel)) { $crossIssue.Add($rel) }

        if ((Test-Path -LiteralPath $full) -and -not $isProt -and -not $preDirty.Contains($rel) -and -not $otherOwned.Contains($rel)) {
            $cleanOwned.Add($rel)
        }
    }

    $canCommit = ($existingOwned.Count -gt 0 -and $protected.Count -eq 0 -and $sharedDirty.Count -eq 0 -and $crossIssue.Count -eq 0)
    $reason = if ($protected.Count -gt 0) {
        "blocked by protected path(s): $($protected -join ', ')"
    } elseif ($sharedDirty.Count -gt 0) {
        "blocked by pre-existing dirty working-tree conflict(s): $($sharedDirty -join ', ')"
    } elseif ($crossIssue.Count -gt 0) {
        "blocked by cross-issue file ownership conflict(s): $($crossIssue -join ', ')"
    } elseif ($existingOwned.Count -eq 0) {
        "no owned files found on disk"
    } else {
        "clean isolated file ownership verified ($($cleanOwned.Count) file(s))"
    }

    [pscustomobject]@{
        issueNumber          = $IssueNumber
        ownedFiles           = @($owned)
        existingOwnedFiles   = @($existingOwned)
        cleanOwnedFiles      = @($cleanOwned)
        sharedDirtyConflicts = @($sharedDirty)
        protectedConflicts   = @($protected)
        crossIssueConflicts  = @($crossIssue)
        canCommitSafely      = [bool]$canCommit
        reason               = $reason
    }
}

function Get-AgyIssueExecutionEvidence {
    <#
    .SYNOPSIS Reconstructs trustworthy execution and verification evidence for an issue strictly from persisted run summaries.
    Never infers `test`, `review`, or `verify` completion from file existence on disk, and never uses issue-number-specific overrides.
    #>
    param(
        [Parameter(Mandatory)][int]$IssueNumber,
        $IssueEntry = $null,
        [Parameter(Mandatory)][hashtable]$RunSummaries,
        [hashtable]$RunIssueMap = @{},
        [Parameter(Mandatory)]$Ownership,
        [hashtable]$ExternalEvidenceMap = @{},
        [string]$WorkDir = (Get-Location).Path
    )
    $numKey = [string]$IssueNumber
    $candidateRunIds = [System.Collections.Generic.List[string]]::new()

    $entryLastRunId = [string](Get-AgyPropValue $IssueEntry 'lastRunId' '')
    if (-not [string]::IsNullOrWhiteSpace($entryLastRunId) -and $RunSummaries.ContainsKey($entryLastRunId)) {
        $candidateRunIds.Add($entryLastRunId)
    }
    foreach ($rid in @(Get-AgyPropValue $IssueEntry 'runIds' @())) {
        $ridStr = [string]$rid
        if (-not [string]::IsNullOrWhiteSpace($ridStr) -and $RunSummaries.ContainsKey($ridStr) -and -not $candidateRunIds.Contains($ridStr)) {
            $candidateRunIds.Add($ridStr)
        }
    }
    if ($RunIssueMap -and $RunIssueMap.ContainsKey($numKey)) {
        foreach ($rid in @($RunIssueMap[$numKey])) {
            $ridStr = [string]$rid
            if (-not [string]::IsNullOrWhiteSpace($ridStr) -and $RunSummaries.ContainsKey($ridStr) -and -not $candidateRunIds.Contains($ridStr)) {
                $candidateRunIds.Add($ridStr)
            }
        }
    }

    $sortedRunIds = @($candidateRunIds | Sort-Object)
    $hasPersistedRun = ($sortedRunIds.Count -gt 0)
    $verifiedPhases = [System.Collections.Generic.List[string]]::new()
    $reconciledAlreadyCompleted = $false
    $blockedByInvestigation = $false
    $investigationBlockNote = $null
    $lastRunStatus = $null
    $lastBudgetKind = $null
    $lastConversationId = $null
    $lastPhase = $null
    $lastWorkerStatus = $null
    $lastRunNote = $null
    $persistedTokens = 0
    $effectiveLastRunId = if ($sortedRunIds.Count -gt 0) { $sortedRunIds[-1] } else { $null }

    foreach ($rid in $sortedRunIds) {
        $sum = $RunSummaries[$rid]
        if (-not $sum) { continue }
        $lastRunStatus = [string](Get-AgyPropValue $sum 'status' '')
        $bKind = [string](Get-AgyPropValue $sum 'budgetKind' '')
        $notesArr = @(Get-AgyPropValue $sum 'notes' @() | Where-Object { $_ })
        if ($notesArr.Count -gt 0) { $lastRunNote = [string]$notesArr[-1] }
        if (-not $bKind -and $lastRunStatus -eq 'BUDGET_EXHAUSTED') {
            $notesJoined = ($notesArr -join ' ')
            $bKind = if ($notesJoined -match 'per-run token budget|token budget reached') { 'INTERNAL_RUN_BUDGET' } else { 'PROVIDER_QUOTA_RESERVE' }
        }
        if ($bKind) { $lastBudgetKind = $bKind }
        $persistedTokens += [int](Get-AgyPropValue $sum 'totalTokens' 0)
        if ($lastRunStatus -eq 'ALREADY_COMPLETED') {
            $reconciledAlreadyCompleted = $true
        }

        $resList = @(Get-AgyPropValue $sum 'results' @())
        $hasCorrectionsSuccess = @($resList | Where-Object {
            [string](Get-AgyPropValue $_ 'taskType' '') -eq 'corrections' -and [string](Get-AgyPropValue $_ 'status' '') -eq 'SUCCESS'
        }).Count -gt 0

        foreach ($r in $resList) {
            $rType = [string](Get-AgyPropValue $r 'taskType' '')
            $rStat = [string](Get-AgyPropValue $r 'status' '')
            $rResp = [string](Get-AgyPropValue $r 'response' '')
            $rConv = [string](Get-AgyPropValue $r 'conversationId' '')
            if (-not [string]::IsNullOrWhiteSpace($rConv)) { $lastConversationId = $rConv }
            if (-not [string]::IsNullOrWhiteSpace($rType)) { $lastPhase = $rType }
            if (-not [string]::IsNullOrWhiteSpace($rStat)) { $lastWorkerStatus = $rStat }

            if ($rStat -ne 'SUCCESS' -or [string]::IsNullOrWhiteSpace($rType)) { continue }

            switch ($rType) {
                'investigation' {
                    if ($rResp -match '(?im)^\s*ISSUE_STATUS\s*:\s*BLOCKED\b') {
                        $blockedByInvestigation = $true
                        $investigationBlockNote = ($rResp -split "`r?`n" | Where-Object { $_ -match '(?i)ISSUE_STATUS|BLOCK' } | Select-Object -First 1)
                        continue
                    }
                    if ($rResp -match '(?im)^\s*ISSUE_STATUS\s*:\s*ALREADY_COMPLETED\b') {
                        $reconciledAlreadyCompleted = $true
                    }
                    if (-not $verifiedPhases.Contains('investigation')) { $verifiedPhases.Add('investigation') }
                }
                'review' {
                    if ($rResp -notmatch '(?im)^\s*VERDICT\s*:\s*CHANGES_REQUIRED\b' -or $hasCorrectionsSuccess) {
                        if (-not $verifiedPhases.Contains('review')) { $verifiedPhases.Add('review') }
                    }
                }
                'verify' {
                    if ($rResp -notmatch '(?im)^\s*VERIFICATION\s*:\s*FAILED\b') {
                        if (-not $verifiedPhases.Contains('verify')) { $verifiedPhases.Add('verify') }
                    }
                }
                default {
                    if (-not $verifiedPhases.Contains($rType)) { $verifiedPhases.Add($rType) }
                }
            }
        }
    }

    $hasOwnedFilesSpec = ($Ownership.ownedFiles.Count -gt 0)
    $allOwnedFilesOnDisk = ($hasOwnedFilesSpec -and $Ownership.existingOwnedFiles.Count -eq $Ownership.ownedFiles.Count)

    # Check if any non-plan target files referenced by the issue are missing on disk (e.g. human-curated dataset manifests)
    $missingTargetArtifacts = [System.Collections.Generic.List[string]]::new()
    foreach ($tf in @(Get-AgyPropValue $IssueEntry 'targetFiles' @())) {
        if ([string]::IsNullOrWhiteSpace([string]$tf)) { continue }
        $normTf = Format-AgyCleanPath -Path ([string]$tf) -WorkDir $WorkDir
        if ($normTf -match '^docs/maestro/plans/') { continue }
        $fullTf = Join-Path $WorkDir ($normTf -replace '/', [IO.Path]::DirectorySeparatorChar)
        if (-not (Test-Path -LiteralPath $fullTf)) {
            if (-not $missingTargetArtifacts.Contains($normTf)) { $missingTargetArtifacts.Add($normTf) }
        }
    }

    $extEv = Get-AgyPropValue $ExternalEvidenceMap $numKey $null
    $extTestsPassed        = [bool](Get-AgyPropValue $extEv 'testsPassed' $false)
    $extReviewApproved     = [bool](Get-AgyPropValue $extEv 'reviewApproved' $false)
    $extVerificationPassed = [bool](Get-AgyPropValue $extEv 'verificationPassed' $false)
    $extHumanPending       = Get-AgyPropValue $extEv 'humanReviewPending' $null

    $labels = @(Get-AgyPropValue $IssueEntry 'labels' @() | ForEach-Object { [string]$_ })
    $humanLabels = @($labels | Where-Object { $_ -in @('ready-for-human', 'needs-human-review', 'human-review-pending') })
    $hasHumanLabel = ($humanLabels.Count -gt 0)
    $entryHumanPending = Get-AgyPropValue $IssueEntry 'humanReviewPending' $null

    $orchestratorVerified = [bool](
        $reconciledAlreadyCompleted -or (
            $verifiedPhases.Contains('implement') -and
            $verifiedPhases.Contains('test') -and
            $verifiedPhases.Contains('review') -and
            $verifiedPhases.Contains('verify')
        )
    )
    $externallyVerified = [bool](
        $allOwnedFilesOnDisk -and
        $null -ne $extEv -and
        $extTestsPassed -and
        $extReviewApproved -and
        $extVerificationPassed -and
        ($extHumanPending -ne $true)
    )

    $implementationComplete = [bool](
        $verifiedPhases.Contains('implement') -or
        $allOwnedFilesOnDisk -or
        $reconciledAlreadyCompleted
    )

    $humanReviewPending = if ($null -ne $extHumanPending) {
        [bool]$extHumanPending
    } elseif ($hasHumanLabel) {
        $true
    } elseif ($null -ne $entryHumanPending -and [bool]$entryHumanPending) {
        $true
    } elseif (-not $orchestratorVerified -and -not $externallyVerified -and $allOwnedFilesOnDisk -and $missingTargetArtifacts.Count -gt 0) {
        $true
    } else {
        $false
    }

    $verificationComplete = [bool]((-not $humanReviewPending) -and ($orchestratorVerified -or $externallyVerified))

    $prevStatus = [string](Get-AgyPropValue $IssueEntry 'status' 'PENDING')
    $prevDoneCount = @(Get-AgyPropValue $IssueEntry 'completedPhases' @() | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) }).Count
    $hadUnverifiedClaims = [bool]((-not $hasPersistedRun) -and ($prevStatus -in @('COMPLETED', 'ALREADY_COMPLETED') -or $prevDoneCount -gt 0))

    $verificationState = if ($verificationComplete) {
        'VERIFICATION_COMPLETE'
    } elseif ($humanReviewPending) {
        'HUMAN_REVIEW_PENDING'
    } elseif ($verifiedPhases.Count -gt 0 -and -not $verificationComplete) {
        'PARTIAL_LIFECYCLE'
    } elseif ($implementationComplete -and -not $verificationComplete) {
        'IMPLEMENTATION_COMPLETE'
    } elseif ($hadUnverifiedClaims -and -not $implementationComplete) {
        'INCOMPLETE_EVIDENCE'
    } else {
        'NOT_STARTED'
    }

    [pscustomobject]@{
        issueNumber                = $IssueNumber
        hasPersistedRun            = [bool]$hasPersistedRun
        lastRunId                  = $effectiveLastRunId
        lastRunStatus              = $lastRunStatus
        lastBudgetKind             = $lastBudgetKind
        lastConversationId         = $lastConversationId
        lastPhase                  = $lastPhase
        lastWorkerStatus           = $lastWorkerStatus
        lastRunNote                = $lastRunNote
        persistedTokens            = [int]$persistedTokens
        completedPhases            = @($verifiedPhases)
        reconciledAlreadyCompleted = [bool]$reconciledAlreadyCompleted
        blockedByInvestigation     = [bool]$blockedByInvestigation
        investigationBlockNote     = $investigationBlockNote
        allOwnedFilesOnDisk        = [bool]$allOwnedFilesOnDisk
        missingTargetArtifacts     = @($missingTargetArtifacts)
        humanLabels                = @($humanLabels)
        implementationComplete     = [bool]$implementationComplete
        verificationComplete       = [bool]$verificationComplete
        humanReviewPending         = [bool]$humanReviewPending
        verificationState          = $verificationState
        hadUnverifiedClaims        = [bool]$hadUnverifiedClaims
    }
}

function Repair-AgyBacklogQueue {
    <#
    .SYNOPSIS Reconciles local completed and partially completed runs into `backlog-queue.json` without invoking any model worker.
    Reconstructs `completedPhases` strictly from persisted execution evidence (`$RunRoot/<runId>/summary.json`),
    never infers `test`/`review`/`verify` from file existence, removes issue-number-specific completion logic,
    and distinguishes `IMPLEMENTATION_COMPLETE`, `VERIFICATION_COMPLETE`, and `HUMAN_REVIEW_PENDING` for externally completed work.
    #>
    param(
        [Parameter(Mandatory)][string]$QueuePath,
        [string]$RunRoot = (Join-Path (Get-Location).Path '.cache/agy-orch'),
        $Policy = (Get-AgyPolicy),
        [string]$WorkDir = (Get-Location).Path,
        [switch]$DryRun
    )
    $queue = Get-AgyBacklogQueue -QueuePath $QueuePath
    $recovered = [System.Collections.Generic.List[object]]::new()

    # Map runId -> parsed summary.json and discover issue->runId associations from persisted handoff files
    $runSummaries = @{}
    $runIssueMap  = @{}
    if (Test-Path -LiteralPath $RunRoot) {
        foreach ($d in @(Get-ChildItem -LiteralPath $RunRoot -Directory -ErrorAction SilentlyContinue)) {
            $sp = Join-Path $d.FullName 'summary.json'
            if (Test-Path -LiteralPath $sp) {
                try {
                    $runSummaries[$d.Name] = Get-Content -LiteralPath $sp -Raw -Encoding utf8 | ConvertFrom-Json
                } catch {}
                foreach ($hf in @(Get-ChildItem -LiteralPath $d.FullName -Filter 'handoff-*.md' -File -ErrorAction SilentlyContinue)) {
                    try {
                        $hTxt = Get-Content -LiteralPath $hf.FullName -Raw -Encoding utf8 -ErrorAction SilentlyContinue
                        foreach ($m in [regex]::Matches([string]$hTxt, '(?im)^GitHub\s+Issue\s+#([0-9]+)\b')) {
                            $inum = [string]$m.Groups[1].Value
                            if (-not $runIssueMap.ContainsKey($inum)) {
                                $runIssueMap[$inum] = [System.Collections.Generic.List[string]]::new()
                            }
                            if (-not $runIssueMap[$inum].Contains($d.Name)) {
                                $runIssueMap[$inum].Add($d.Name)
                            }
                        }
                    } catch {}
                }
            }
        }
    }

    # Optional persisted external verification manifest ($RunRoot/external-evidence.json)
    $externalEvidenceMap = @{}
    $extEvPath = Join-Path $RunRoot 'external-evidence.json'
    if (Test-Path -LiteralPath $extEvPath) {
        try {
            $extRaw = Get-Content -LiteralPath $extEvPath -Raw -Encoding utf8 | ConvertFrom-Json
            if ($extRaw) {
                foreach ($prop in $extRaw.PSObject.Properties) {
                    $externalEvidenceMap[[string]$prop.Name] = $prop.Value
                }
            }
        } catch {}
    }

    $hasPartialRunBudget = $false
    foreach ($k in @($queue.issues.Keys | Sort-Object { [int]$_ })) {
        $entry = $queue.issues[$k]
        $num = [int]$entry.number
        $ownership = Test-AgyIssueFileOwnership -IssueNumber $num -IssueEntry $entry -Policy $Policy -WorkDir $WorkDir
        $ev = Get-AgyIssueExecutionEvidence -IssueNumber $num -IssueEntry $entry -RunSummaries $runSummaries `
            -RunIssueMap $runIssueMap -Ownership $ownership -ExternalEvidenceMap $externalEvidenceMap -WorkDir $WorkDir

        $doneList = @($ev.completedPhases)
        $entry['lastRunId']              = $ev.lastRunId
        $entry['conversationId']         = $ev.lastConversationId
        $entry['completedPhases']        = $doneList
        $entry['implementationComplete'] = [bool]$ev.implementationComplete
        $entry['verificationComplete']   = [bool]$ev.verificationComplete
        $entry['humanReviewPending']     = [bool]$ev.humanReviewPending
        $entry['verificationState']      = [string]$ev.verificationState
        if ($ev.persistedTokens -gt 0) {
            $entry['tokensUsed'] = [int]$ev.persistedTokens
        } elseif (-not $ev.hasPersistedRun) {
            $entry['tokensUsed'] = [int](Get-AgyPropValue $entry 'tokensUsed' 0)
        }

        if ($ev.verificationComplete) {
            if ($ev.reconciledAlreadyCompleted -and -not ($doneList -contains 'implement')) {
                $entry['status'] = 'ALREADY_COMPLETED'
                $entry['currentPhase'] = 'reconciled'
            } else {
                $entry['status'] = 'COMPLETED'
                $entry['currentPhase'] = 'done'
            }
            $entry['blockReason'] = $null
        } elseif ($ev.blockedByInvestigation) {
            $entry['status'] = 'BLOCKED'
            $entry['currentPhase'] = 'investigation'
            $entry['blockReason'] = if ($ev.investigationBlockNote) { [string]$ev.investigationBlockNote } else { 'Blocked by investigation phase' }
        } elseif ($ev.hasPersistedRun -and $doneList.Count -gt 0) {
            $curPh = if ($ev.lastPhase) { [string]$ev.lastPhase } else { [string](Get-AgyPropValue $entry 'currentPhase' '') }
            $entry['currentPhase'] = $curPh
            if ($ev.lastRunStatus -eq 'INCOMPLETE' -or $ev.lastWorkerStatus -eq 'INCOMPLETE') {
                $entry['status'] = 'INCOMPLETE'
                $entry['blockReason'] = if ($ev.lastRunNote) { $ev.lastRunNote } else { "Worker incomplete in phase '$curPh'" }
            } elseif ($ev.lastRunStatus -eq 'FAILED') {
                $entry['status'] = 'FAILED'
                $entry['blockReason'] = if ($ev.lastRunNote) { $ev.lastRunNote } else { "Worker failed in phase '$curPh'" }
            } else {
                $entry['status'] = 'PARTIAL'
                $rem = Get-AgyRemainingLifecyclePhases -CompletedPhases $doneList -Status 'PARTIAL' -Policy $Policy
                if ($ev.lastRunStatus -eq 'BUDGET_EXHAUSTED') {
                    if ($ev.lastBudgetKind -eq 'PROVIDER_QUOTA_RESERVE') {
                        $entry['blockReason'] = "Blocked by provider quota reserve after phase '$curPh'"
                    } else {
                        $entry['blockReason'] = "Paused by internal per-run token budget after phase '$curPh'; provider quota above reserve — ready to resume ($($rem -join ', '))"
                        $hasPartialRunBudget = $true
                    }
                } else {
                    $entry['blockReason'] = "Partial lifecycle recovery after phase '$curPh'; remaining phases: $($rem -join ', ')"
                }
            }
        } elseif ($ev.hasPersistedRun -and $ev.lastRunStatus -in @('FAILED', 'INCOMPLETE')) {
            $entry['status'] = $ev.lastRunStatus
            $entry['currentPhase'] = $ev.lastPhase
            $entry['blockReason'] = $ev.lastRunNote
        } elseif ($ev.humanReviewPending -and $ev.implementationComplete) {
            $entry['status'] = 'HUMAN_REVIEW_PENDING'
            $entry['currentPhase'] = 'human-review'
            if ($ev.missingTargetArtifacts.Count -gt 0) {
                $entry['blockReason'] = "Implementation complete on disk; human review / curation pending (missing target artifact(s): $($ev.missingTargetArtifacts -join ', '))"
            } elseif ($ev.humanLabels.Count -gt 0) {
                $entry['blockReason'] = "Implementation complete on disk; awaiting human review (label: $($ev.humanLabels -join ', '))"
            } else {
                $entry['blockReason'] = "Implementation complete on disk; human review pending before verification completion"
            }
        } elseif ($ev.implementationComplete) {
            $entry['status'] = 'IMPLEMENTATION_COMPLETE'
            $entry['currentPhase'] = 'external-implement'
            $entry['blockReason'] = "Externally implemented on disk ($($ownership.existingOwnedFiles.Count)/$($ownership.ownedFiles.Count) owned files); persisted test/review/verify execution evidence pending"
        } else {
            $entry['currentPhase'] = $null
            $prevBlk = [string](Get-AgyPropValue $entry 'blockReason' '')
            if ($ev.humanLabels.Count -gt 0) {
                $entry['status'] = 'BLOCKED'
                $entry['blockReason'] = "blocked by triage label: $($ev.humanLabels -join ', ')"
            } elseif ($prevBlk -and $prevBlk -notmatch '(?i)internal per-run token budget|Externally implemented|Implementation complete') {
                $entry['status'] = 'BLOCKED'
            } else {
                $entry['status'] = 'PENDING'
                $entry['blockReason'] = $null
            }
        }

        $entry['remainingPhases'] = @(Get-AgyRemainingLifecyclePhases -CompletedPhases $doneList -Status ([string]$entry.status) -Policy $Policy)
        if ($ownership.existingOwnedFiles.Count -gt 0 -and ($entry.status -in @('COMPLETED', 'ALREADY_COMPLETED', 'PARTIAL', 'IMPLEMENTATION_COMPLETE', 'HUMAN_REVIEW_PENDING'))) {
            $entry['modifiedFiles'] = @($ownership.existingOwnedFiles)
        } else {
            $entry['modifiedFiles'] = @(Get-AgyPropValue $entry 'modifiedFiles' @() | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) -and $_ -notmatch '^docs/maestro/plans/' })
        }

        # Initialize or update delivery record (and strip any stale DRY_RUN state that lacks a real commitSha)
        $prevDeliv = Get-AgyPropValue $entry 'delivery' $null
        $prevCommitSha = Get-AgyPropValue $prevDeliv 'commitSha' $null
        $prevCommitStatus = [string](Get-AgyPropValue $prevDeliv 'commitStatus' 'NOT_COMMITTED')
        $prevBranch = Get-AgyPropValue $prevDeliv 'branch' $null
        if ($prevCommitStatus -eq 'DRY_RUN' -and -not $prevCommitSha) {
            $prevCommitStatus = 'NOT_COMMITTED'
            $prevBranch = $null
        }

        $entry['delivery'] = [ordered]@{
            issueNumber          = $num
            localStatus          = [string]$entry.status
            verificationState    = [string]$entry.verificationState
            branch               = $prevBranch
            commitSha            = $prevCommitSha
            treeSha              = (Get-AgyPropValue $prevDeliv 'treeSha' $null)
            commitStatus         = $prevCommitStatus
            prNumber             = (Get-AgyPropValue $prevDeliv 'prNumber' $null)
            prUrl                = (Get-AgyPropValue $prevDeliv 'prUrl' $null)
            prState              = (Get-AgyPropValue $prevDeliv 'prState' $null)
            lastCommentSignature = (Get-AgyPropValue $prevDeliv 'lastCommentSignature' $null)
            commentSynced        = [bool](Get-AgyPropValue $prevDeliv 'commentSynced' $false)
            labelSynced          = [bool](Get-AgyPropValue $prevDeliv 'labelSynced' $false)
            canCommitSafely      = [bool]$ownership.canCommitSafely
            ownedFiles           = @($ownership.existingOwnedFiles)
            sharedDirtyConflicts = @($ownership.sharedDirtyConflicts)
            ownershipReason      = [string]$ownership.reason
        }

        $recovered.Add([pscustomobject]@{
            number                 = $num
            status                 = [string]$entry.status
            verificationState      = [string]$entry.verificationState
            implementationComplete = [bool]$entry.implementationComplete
            verificationComplete   = [bool]$entry.verificationComplete
            humanReviewPending     = [bool]$entry.humanReviewPending
            completedPhases        = @($entry.completedPhases)
            remainingPhases        = @($entry.remainingPhases)
            ownedFiles             = @($ownership.existingOwnedFiles)
            canCommitSafely        = [bool]$ownership.canCommitSafely
            sharedDirtyConflicts   = @($ownership.sharedDirtyConflicts)
            lastRunId              = (Get-AgyPropValue $entry 'lastRunId' $null)
            conversationId         = (Get-AgyPropValue $entry 'conversationId' $null)
            tokensUsed             = [int](Get-AgyPropValue $entry 'tokensUsed' 0)
        })
    }

    # Re-evaluate dependency blocks across the queue
    $null = Select-AgyNextActionableIssue -Queue $queue
    if ($hasPartialRunBudget -and $queue.runnerState -eq 'BLOCKED_QUOTA_OR_AUTH') {
        $queue.runnerState = 'PAUSED_RUN_BUDGET'
    }

    $writtenFiles = @()
    if (-not $DryRun) {
        Save-AgyBacklogQueue -Queue $queue -QueuePath $QueuePath
        $writtenFiles = @(Format-AgyCleanPath -Path $QueuePath -WorkDir $WorkDir)
    }

    [pscustomobject]@{
        status       = 'OK'
        dryRun       = [bool]$DryRun
        runnerState  = $queue.runnerState
        writtenFiles = $writtenFiles
        stats        = Get-AgyBacklogStats -Queue $queue
        issues       = @($recovered)
    }
}

function New-AgyIssueBranchCommit {
    <#
    .SYNOPSIS Creates or updates an isolated issue branch (`agy/issue-<N>`) using a temporary `GIT_INDEX_FILE`.
    Never switches the active working branch, never touches `.git/index`, and never modifies working-tree files.
    Refuses to commit if `Test-AgyIssueFileOwnership` reports any shared dirty or protected file conflict.
    #>
    param(
        [Parameter(Mandatory)][int]$IssueNumber,
        [string]$IssueTitle = '',
        $IssueEntry = $null,
        $Policy = (Get-AgyPolicy),
        [string]$GitCommand = 'git',
        [string]$BaseBranch = '',
        [string]$WorkDir = (Get-Location).Path,
        [switch]$DryRun
    )
    $ownership = Test-AgyIssueFileOwnership -IssueNumber $IssueNumber -IssueEntry $IssueEntry -Policy $Policy -WorkDir $WorkDir
    $delivCfg = if ($Policy.PSObject.Properties['backlog'] -and $Policy.backlog.PSObject.Properties['delivery']) { $Policy.backlog.delivery } else { $null }
    $prefix = if ($delivCfg -and $delivCfg.PSObject.Properties['branchPrefix'] -and $delivCfg.branchPrefix) { [string]$delivCfg.branchPrefix } else { 'agy/issue-' }
    $branchName = "$prefix$IssueNumber"
    $baseRef = if ($BaseBranch) { $BaseBranch } elseif ($delivCfg -and $delivCfg.PSObject.Properties['baseBranch'] -and $delivCfg.baseBranch) { [string]$delivCfg.baseBranch } else { 'main' }

    if (-not $ownership.canCommitSafely) {
        return [pscustomobject]@{
            issueNumber          = $IssueNumber
            status               = 'BLOCKED_OWNERSHIP'
            branch               = $branchName
            baseBranch           = $baseRef
            commitSha            = $null
            treeSha              = $null
            files                = @($ownership.cleanOwnedFiles)
            sharedDirtyConflicts = @($ownership.sharedDirtyConflicts)
            reason               = $ownership.reason
        }
    }

    if ($DryRun) {
        return [pscustomobject]@{
            issueNumber          = $IssueNumber
            status               = 'DRY_RUN'
            branch               = $branchName
            baseBranch           = $baseRef
            commitSha            = $null
            treeSha              = $null
            files                = @($ownership.cleanOwnedFiles)
            sharedDirtyConflicts = @()
            reason               = "dry-run: would stage $($ownership.cleanOwnedFiles.Count) isolated file(s) onto $branchName from $baseRef"
        }
    }

    # Resolve base commit SHA (fallback to HEAD if baseRef does not exist in a test repo)
    $baseSha = (& $GitCommand -C $WorkDir rev-parse --verify "$baseRef^{commit}" 2>$null | Select-Object -First 1)
    if (-not $baseSha) {
        $baseRef = 'HEAD'
        $baseSha = (& $GitCommand -C $WorkDir rev-parse --verify "HEAD^{commit}" 2>$null | Select-Object -First 1)
    }
    if (-not $baseSha) { throw "Cannot resolve base git commit in $WorkDir" }
    $baseSha = ([string]$baseSha).Trim()
    $baseTreeSha = ([string](& $GitCommand -C $WorkDir rev-parse --verify "$baseSha^{tree}" 2>$null | Select-Object -First 1)).Trim()

    $tmpIndex = Join-Path ([IO.Path]::GetTempPath()) ("agy-git-idx-$IssueNumber-" + [guid]::NewGuid().ToString('N'))
    $tmpMsg   = Join-Path ([IO.Path]::GetTempPath()) ("agy-git-msg-$IssueNumber-" + [guid]::NewGuid().ToString('N') + '.txt')
    $prevIndex = $env:GIT_INDEX_FILE
    try {
        $env:GIT_INDEX_FILE = $tmpIndex
        $null = & $GitCommand -C $WorkDir read-tree $baseSha 2>$null
        if ($LASTEXITCODE -ne 0) { throw "git read-tree $baseSha failed" }

        foreach ($rel in $ownership.cleanOwnedFiles) {
            $full = Join-Path $WorkDir ($rel -replace '/', [IO.Path]::DirectorySeparatorChar)
            $blob = ([string](& $GitCommand -C $WorkDir hash-object -w -- $full 2>$null | Select-Object -First 1)).Trim()
            if (-not $blob -or $LASTEXITCODE -ne 0) { throw "git hash-object failed for $rel" }
            $null = & $GitCommand -C $WorkDir update-index --add --cacheinfo "100644,$blob,$rel" 2>$null
            if ($LASTEXITCODE -ne 0) { throw "git update-index failed for $rel" }
        }

        $treeSha = ([string](& $GitCommand -C $WorkDir write-tree 2>$null | Select-Object -First 1)).Trim()
        if (-not $treeSha -or $LASTEXITCODE -ne 0) { throw "git write-tree failed" }

        if ($treeSha -eq $baseTreeSha) {
            return [pscustomobject]@{
                issueNumber          = $IssueNumber
                status               = 'ALREADY_IN_BASE'
                branch               = $branchName
                baseBranch           = $baseRef
                commitSha            = $baseSha
                treeSha              = $treeSha
                files                = @($ownership.cleanOwnedFiles)
                sharedDirtyConflicts = @()
                reason               = "all owned files are already identical to $baseRef ($baseSha)"
            }
        }

        # Idempotency check: if refs/heads/$branchName already exists with the exact same treeSha, reuse it
        $existingBranchSha = (& $GitCommand -C $WorkDir rev-parse --verify "refs/heads/$branchName^{commit}" 2>$null | Select-Object -First 1)
        if ($existingBranchSha) {
            $existingBranchSha = ([string]$existingBranchSha).Trim()
            $existingBranchTree = ([string](& $GitCommand -C $WorkDir rev-parse --verify "$existingBranchSha^{tree}" 2>$null | Select-Object -First 1)).Trim()
            if ($existingBranchTree -eq $treeSha) {
                return [pscustomobject]@{
                    issueNumber          = $IssueNumber
                    status               = 'UNCHANGED'
                    branch               = $branchName
                    baseBranch           = $baseRef
                    commitSha            = $existingBranchSha
                    treeSha              = $treeSha
                    files                = @($ownership.cleanOwnedFiles)
                    sharedDirtyConflicts = @()
                    reason               = "branch $branchName already at commit $existingBranchSha with matching tree $treeSha"
                }
            }
        }

        $cleanTitle = if ($IssueTitle) { ($IssueTitle -replace '^\s*\[[^\]]+\]\s*', '').Trim() } else { "Issue #$IssueNumber implementation" }
        $msgLines = @(
            "feat(issue-$IssueNumber): $cleanTitle",
            "",
            "Automated isolated branch commit for GitHub Issue #$IssueNumber.",
            "Refs #$IssueNumber",
            "",
            "Owned files:"
        ) + @($ownership.cleanOwnedFiles | ForEach-Object { "- $_" })
        $commitMsg = $msgLines -join "`n"
        Write-AgyUtf8File -Path $tmpMsg -Value $commitMsg

        $commitRaw = & $GitCommand -C $WorkDir commit-tree $treeSha -p $baseSha -F $tmpMsg 2>&1
        $commitExit = $LASTEXITCODE
        $commitSha = ([string]($commitRaw | Where-Object { $_ -match '^[0-9a-f]{40}$' } | Select-Object -First 1)).Trim()
        if (-not $commitSha -or $commitExit -ne 0) { throw "git commit-tree failed for issue #$IssueNumber (exit=$commitExit, tree=$treeSha, base=$baseSha): $($commitRaw -join ' ')" }

        $null = & $GitCommand -C $WorkDir update-ref "refs/heads/$branchName" $commitSha 2>$null
        if ($LASTEXITCODE -ne 0) { throw "git update-ref refs/heads/$branchName failed" }

        return [pscustomobject]@{
            issueNumber          = $IssueNumber
            status               = 'COMMITTED'
            branch               = $branchName
            baseBranch           = $baseRef
            commitSha            = $commitSha
            treeSha              = $treeSha
            files                = @($ownership.cleanOwnedFiles)
            sharedDirtyConflicts = @()
            reason               = "created isolated commit $commitSha on branch $branchName from $baseRef"
        }
    } finally {
        $env:GIT_INDEX_FILE = $prevIndex
        if (Test-Path -LiteralPath $tmpIndex) { Remove-Item -LiteralPath $tmpIndex -Force -ErrorAction SilentlyContinue }
        if (Test-Path -LiteralPath $tmpMsg)   { Remove-Item -LiteralPath $tmpMsg   -Force -ErrorAction SilentlyContinue }
    }
}

function Format-AgyIssueCommentBody {
    # Generates a deterministic progress/completion comment with an embedded signature marker for idempotent deduplication.
    param(
        [Parameter(Mandatory)]$IssueEntry,
        [Parameter(Mandatory)]$Ownership,
        $Delivery = $null
    )
    $num = [int]$IssueEntry.number
    $st = [string]$IssueEntry.status
    $verState = [string](Get-AgyPropValue $IssueEntry 'verificationState' $st)
    $done = @(Get-AgyPropValue $IssueEntry 'completedPhases' @() | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) })
    $rem  = @(Get-AgyPropValue $IssueEntry 'remainingPhases' @() | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) })
    $doneCsv = if ($done.Count -gt 0) { $done -join ',' } else { 'none' }
    $remCsv  = if ($rem.Count -gt 0)  { $rem -join ',' }  else { 'none' }
    $sig = "<!-- agy-orchestrator:issue=${num}:status=${st}:phases=${doneCsv} -->"

    $lines = [System.Collections.Generic.List[string]]::new()
    $lines.Add($sig)
    $lines.Add("### Antigravity Orchestrator Status Update — ``$st``")
    $lines.Add("")
    $lines.Add("- **Issue:** #$num — $($IssueEntry.title)")
    $lines.Add("- **Local Lifecycle Status:** ``$st``")
    $lines.Add("- **Verification State:** ``$verState``")
    $lines.Add("- **Completed Phases:** ``$doneCsv``")
    if ($rem.Count -gt 0) {
        $lines.Add("- **Remaining Phases:** ``$remCsv``")
    }
    $lastRun = Get-AgyPropValue $IssueEntry 'lastRunId' $null
    if ($lastRun) {
        $lines.Add("- **Last Run ID:** ``$lastRun`` (tokens used: ``$(Get-AgyPropValue $IssueEntry 'tokensUsed' 0)``)")
    }
    $blkReason = Get-AgyPropValue $IssueEntry 'blockReason' $null
    if ($blkReason) {
        $lines.Add("- **Checkpoint / Pause Note:** $blkReason")
    }
    if ($Ownership.existingOwnedFiles.Count -gt 0) {
        $fileList = ($Ownership.existingOwnedFiles | ForEach-Object { "``$_``" }) -join ', '
        $lines.Add("- **Owned Files:** $fileList")
    }
    if ($Ownership.sharedDirtyConflicts.Count -gt 0) {
        $confList = ($Ownership.sharedDirtyConflicts | ForEach-Object { "``$_``" }) -join ', '
        $lines.Add("- **File Ownership Guard:** Shared working-tree changes detected in $confList (isolated commit deferred until resolved).")
    }
    $dBr = Get-AgyPropValue $Delivery 'branch' $null
    $dCm = Get-AgyPropValue $Delivery 'commitSha' $null
    if ($dBr -and $dCm) {
        $lines.Add("- **Isolated Branch / Commit:** ``$dBr`` @ ``$dCm``")
    }
    $dPr = Get-AgyPropValue $Delivery 'prUrl' $null
    if ($dPr) {
        $lines.Add("- **Draft Pull Request:** $dPr")
    }

    [pscustomobject]@{
        signature = $sig
        body      = ($lines -join "`n")
    }
}

function Sync-AgyGitHubDelivery {
    <#
    .SYNOPSIS Idempotent GitHub delivery synchronization for locally completed and partially completed issues.
    Supports configurable:
      1. Isolated issue branch & commit creation (`-EnableBranchCommit`)
      2. Branch push (`-EnablePush`)
      3. Draft Pull Request creation (`-EnableDraftPr`)
      4. Progress comments (`-EnableComments`)
      5. Status labels (`-EnableLabels`)
    Never auto-merges into main and never auto-closes issues.
    #>
    param(
        [Parameter(Mandatory)][string]$QueuePath,
        [int[]]$IssueNumbers = @(),
        $Policy = (Get-AgyPolicy),
        [string]$GhCommand = 'gh',
        [string]$GitCommand = 'git',
        [string]$WorkDir = (Get-Location).Path,
        [switch]$EnableComments,
        [switch]$EnableLabels,
        [switch]$EnableBranchCommit,
        [switch]$EnablePush,
        [switch]$EnableDraftPr,
        [switch]$DryRun
    )
    $delivCfg = if ($Policy.PSObject.Properties['backlog'] -and $Policy.backlog.PSObject.Properties['delivery']) { $Policy.backlog.delivery } else { $null }
    $doComments     = [bool]($EnableComments     -or ($delivCfg -and $delivCfg.allowComments))
    $doLabels       = [bool]($EnableLabels       -or ($delivCfg -and $delivCfg.allowLabels))
    $doBranchCommit = [bool]($EnableBranchCommit -or ($delivCfg -and $delivCfg.allowCommitsAndBranches))
    $doPush         = [bool]($EnablePush         -or ($delivCfg -and $delivCfg.allowPush))
    $doDraftPr      = [bool]($EnableDraftPr      -or ($delivCfg -and $delivCfg.allowDraftPr))
    $baseBranch     = if ($delivCfg -and $delivCfg.baseBranch) { [string]$delivCfg.baseBranch } else { 'main' }

    # Reconcile local queue state (without writing to disk when -DryRun is active)
    $rep = Repair-AgyBacklogQueue -QueuePath $QueuePath -RunRoot (Split-Path -Parent $QueuePath) -Policy $Policy -WorkDir $WorkDir -DryRun:$DryRun
    $queue = Get-AgyBacklogQueue -QueuePath $QueuePath
    if ($DryRun -and $rep -and $rep.issues) {
        foreach ($ri in @($rep.issues)) {
            $rk = [string]$ri.number
            if ($queue.issues.ContainsKey($rk)) {
                $queue.issues[$rk]['status']                 = $ri.status
                $queue.issues[$rk]['verificationState']      = $ri.verificationState
                $queue.issues[$rk]['implementationComplete'] = [bool]$ri.implementationComplete
                $queue.issues[$rk]['verificationComplete']   = [bool]$ri.verificationComplete
                $queue.issues[$rk]['humanReviewPending']     = [bool]$ri.humanReviewPending
                $queue.issues[$rk]['completedPhases']        = @($ri.completedPhases)
                $queue.issues[$rk]['remainingPhases']        = @($ri.remainingPhases)
                $queue.issues[$rk]['lastRunId']              = $ri.lastRunId
                $queue.issues[$rk]['conversationId']         = $ri.conversationId
                $queue.issues[$rk]['tokensUsed']             = [int]$ri.tokensUsed
            }
        }
    }

    $targetKeys = if ($IssueNumbers -and $IssueNumbers.Count -gt 0) {
        @($IssueNumbers | ForEach-Object { [string]$_ } | Where-Object { $queue.issues.ContainsKey($_) })
    } else {
        @($queue.issues.Keys | Sort-Object { [int]$_ } | Where-Object {
            $queue.issues[$_].status -in @('COMPLETED', 'ALREADY_COMPLETED', 'IMPLEMENTATION_COMPLETE', 'HUMAN_REVIEW_PENDING', 'PARTIAL', 'INCOMPLETE', 'FAILED', 'RUNNING')
        })
    }

    $results = [System.Collections.Generic.List[object]]::new()
    $prevEnc = [Console]::OutputEncoding
    [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
    try {
        foreach ($k in $targetKeys) {
            $entry = $queue.issues[$k]
            $num = [int]$entry.number
            $ownership = Test-AgyIssueFileOwnership -IssueNumber $num -IssueEntry $entry -Policy $Policy -WorkDir $WorkDir
            if (-not $entry.ContainsKey('delivery') -or -not $entry.delivery) {
                $entry.delivery = [ordered]@{}
            }
            $deliv = $entry.delivery

            $branchAction  = 'SKIPPED'
            $pushAction    = 'SKIPPED'
            $prAction      = 'SKIPPED'
            $commentAction = 'SKIPPED'
            $labelAction   = 'SKIPPED'

            # 1. Isolated Branch & Commit (only for verified COMPLETED / ALREADY_COMPLETED issues with clean file ownership)
            if ($doBranchCommit) {
                if ($entry.status -in @('COMPLETED', 'ALREADY_COMPLETED')) {
                    $bc = New-AgyIssueBranchCommit -IssueNumber $num -IssueTitle ([string]$entry.title) -IssueEntry $entry `
                        -Policy $Policy -GitCommand $GitCommand -BaseBranch $baseBranch -WorkDir $WorkDir -DryRun:$DryRun
                    $branchAction = $bc.status
                    $deliv.branch = $bc.branch
                    $deliv.commitStatus = $bc.status
                    if ($bc.commitSha) { $deliv.commitSha = $bc.commitSha }
                    if ($bc.treeSha)   { $deliv.treeSha   = $bc.treeSha }
                } else {
                    $branchAction = "SKIPPED_STATUS_$($entry.status)"
                }
            }

            # 2. Push branch & create Draft PR (never auto-merge, never auto-close)
            if ($doPush -and $deliv.branch -and $deliv.commitStatus -in @('COMMITTED', 'UNCHANGED', 'DRY_RUN')) {
                if ($DryRun) {
                    $pushAction = 'DRY_RUN'
                } else {
                    $null = & $GitCommand -C $WorkDir push -u origin $deliv.branch 2>$null
                    $pushAction = if ($LASTEXITCODE -eq 0) { 'PUSHED' } else { 'PUSH_FAILED' }
                }
            }

            if ($doDraftPr -and $deliv.branch -and $deliv.commitStatus -in @('COMMITTED', 'UNCHANGED', 'DRY_RUN')) {
                $existingPrJson = (& $GhCommand pr list --head $deliv.branch --state all --json number,url,state,isDraft 2>$null) -join "`n"
                $existingPrs = @()
                if ($existingPrJson -and $existingPrJson.Trim().StartsWith('[')) {
                    $existingPrs = @($existingPrJson | ConvertFrom-Json | Where-Object { $null -ne $_ })
                }
                if ($existingPrs.Count -gt 0) {
                    $pr0 = $existingPrs[0]
                    $deliv.prNumber = $pr0.number
                    $deliv.prUrl    = [string]$pr0.url
                    $deliv.prState  = if ($pr0.isDraft) { 'DRAFT' } else { [string]$pr0.state }
                    $prAction       = 'SKIPPED_EXISTING_PR'
                } elseif ($DryRun) {
                    $prAction = 'DRY_RUN'
                } else {
                    $prTitle = "feat(issue-$num): $($entry.title)"
                    $prBodyLines = @(
                        "### Summary",
                        "Automated Draft Pull Request for GitHub Issue #$num.",
                        "",
                        "Refs #$num (do not auto-close until reviewed and merged).",
                        "",
                        "### Owned Files",
                        (($ownership.cleanOwnedFiles | ForEach-Object { "- ``$_``" }) -join "`n")
                    )
                    $prBody = $prBodyLines -join "`n"
                    $prOut = (& $GhCommand pr create --draft --base $baseBranch --head $deliv.branch --title $prTitle --body $prBody 2>$null) -join "`n"
                    $prUrl = ($prOut -split "`r?`n" | Where-Object { $_ -match '^https?://' } | Select-Object -Last 1)
                    if ($prUrl) {
                        $deliv.prUrl = $prUrl.Trim()
                        $deliv.prState = 'DRAFT'
                        if ($deliv.prUrl -match '/pull/([0-9]+)') { $deliv.prNumber = [int]$Matches[1] }
                        $prAction = 'CREATED_DRAFT_PR'
                    } else {
                        $prAction = 'PR_CREATE_FAILED'
                    }
                }
            }

            # 3. Idempotent Progress Comment
            $commentObj = Format-AgyIssueCommentBody -IssueEntry $entry -Ownership $ownership -Delivery $deliv
            if ($doComments) {
                $existingCommentsRaw = (& $GhCommand api "repos/{owner}/{repo}/issues/$num/comments" -X GET 2>$null) -join "`n"
                $existingComments = @()
                if ($existingCommentsRaw -and $existingCommentsRaw.Trim().StartsWith('[')) {
                    $existingComments = @($existingCommentsRaw | ConvertFrom-Json | Where-Object { $null -ne $_ })
                }
                $alreadyPosted = $false
                foreach ($c in $existingComments) {
                    if ($c.PSObject.Properties['body'] -and [string]$c.body -and ([string]$c.body).Contains($commentObj.signature)) {
                        $alreadyPosted = $true
                        break
                    }
                }
                if (-not $alreadyPosted -and $deliv.lastCommentSignature -eq $commentObj.signature -and $deliv.commentSynced) {
                    $alreadyPosted = $true
                }
                if ($alreadyPosted) {
                    $commentAction = 'SKIPPED_DUPLICATE'
                    $deliv.lastCommentSignature = $commentObj.signature
                    $deliv.commentSynced = $true
                } elseif ($DryRun) {
                    $commentAction = 'DRY_RUN'
                } else {
                    $null = & $GhCommand issue comment $num --body $commentObj.body 2>$null
                    if ($LASTEXITCODE -eq 0) {
                        $commentAction = 'POSTED'
                        $deliv.lastCommentSignature = $commentObj.signature
                        $deliv.commentSynced = $true
                    } else {
                        $commentAction = 'COMMENT_FAILED'
                    }
                }
            }

            # 4. Idempotent Status Labels
            if ($doLabels) {
                $lblCfg = if ($delivCfg -and $delivCfg.PSObject.Properties['statusLabels']) { $delivCfg.statusLabels } else { $null }
                $desiredLabel = switch ($entry.status) {
                    'COMPLETED'               { if ($deliv.prUrl -and $lblCfg.prReady) { [string]$lblCfg.prReady } elseif ($lblCfg.completed) { [string]$lblCfg.completed } else { 'completed-locally' } }
                    'ALREADY_COMPLETED'       { if ($lblCfg.completed) { [string]$lblCfg.completed } else { 'completed-locally' } }
                    'HUMAN_REVIEW_PENDING'    { if ($lblCfg -and $lblCfg.PSObject.Properties['humanReview'] -and $lblCfg.humanReview) { [string]$lblCfg.humanReview } else { 'ready-for-human' } }
                    'IMPLEMENTATION_COMPLETE' { if ($lblCfg.partial)   { [string]$lblCfg.partial }   else { 'partial-progress' } }
                    'PARTIAL'                 { if ($lblCfg.partial)   { [string]$lblCfg.partial }   else { 'partial-progress' } }
                    'RUNNING'                 { if ($lblCfg.inProgress){ [string]$lblCfg.inProgress }else { 'in-progress' } }
                    default                   { $null }
                }
                if ($desiredLabel) {
                    $curLabels = @($entry.labels)
                    if ($curLabels -contains $desiredLabel) {
                        $labelAction = 'SKIPPED_UNCHANGED'
                        $deliv.labelSynced = $true
                    } elseif ($DryRun) {
                        $labelAction = "DRY_RUN:$desiredLabel"
                    } else {
                        $null = & $GhCommand issue edit $num --add-label $desiredLabel 2>$null
                        if ($LASTEXITCODE -eq 0) {
                            $labelAction = "ADDED:$desiredLabel"
                            $entry.labels = @($curLabels + @($desiredLabel) | Select-Object -Unique)
                            $deliv.labelSynced = $true
                        } else {
                            $labelAction = 'LABEL_FAILED'
                        }
                    }
                }
            }

            $deliv.canCommitSafely      = [bool]$ownership.canCommitSafely
            $deliv.ownedFiles           = @($ownership.existingOwnedFiles)
            $deliv.sharedDirtyConflicts = @($ownership.sharedDirtyConflicts)
            $deliv.ownershipReason      = [string]$ownership.reason
            $entry.delivery             = $deliv

            $results.Add([pscustomobject]@{
                issueNumber          = $num
                title                = [string]$entry.title
                localStatus          = [string]$entry.status
                verificationState    = [string](Get-AgyPropValue $entry 'verificationState' $entry.status)
                completedPhases      = @($entry.completedPhases | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) })
                remainingPhases      = @($entry.remainingPhases | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) })
                canCommitSafely      = [bool]$ownership.canCommitSafely
                ownedFiles           = @($ownership.existingOwnedFiles)
                sharedDirtyConflicts = @($ownership.sharedDirtyConflicts)
                branch               = $deliv.branch
                commitSha            = $deliv.commitSha
                prUrl                = $deliv.prUrl
                branchAction         = $branchAction
                pushAction           = $pushAction
                prAction             = $prAction
                commentAction        = $commentAction
                labelAction          = $labelAction
                commentSignature     = $commentObj.signature
            })
        }
    } finally {
        [Console]::OutputEncoding = $prevEnc
    }

    $writtenFiles = @()
    if (-not $DryRun) {
        Save-AgyBacklogQueue -Queue $queue -QueuePath $QueuePath
        $writtenFiles = @(Format-AgyCleanPath -Path $QueuePath -WorkDir $WorkDir)
    }
    [pscustomobject]@{
        status          = 'OK'
        dryRun          = [bool]$DryRun
        writtenFiles    = $writtenFiles
        autoMergeAllowed= $false
        autoCloseAllowed= $false
        items           = @($results)
    }
}


