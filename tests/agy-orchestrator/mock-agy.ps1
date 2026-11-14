param()
# Mock agy: logs args, returns canned JSON or NDJSON stream-json.
# Behaviour via MOCK_AGY_MODE:
#   ok | stream | timeout | interrupted | denied | empty | badjson | fail | flaky | review_changes | usage_cmd
$a = $args
if ($env:MOCK_AGY_LOG) { ($a | ForEach-Object { "[$_]" }) -join ' ' | Add-Content -LiteralPath $env:MOCK_AGY_LOG -Encoding utf8 }

$promptIdx = [array]::IndexOf($a, '-p')
$prompt = if ($promptIdx -ge 0 -and ($promptIdx + 1) -lt $a.Count) { [string]$a[$promptIdx + 1] } else { '' }
$modelIdx = [array]::IndexOf($a, '--model')
$model = if ($modelIdx -ge 0 -and ($modelIdx + 1) -lt $a.Count) { [string]$a[$modelIdx + 1] } else { 'gemini-3.8-flash-medium' }
$convIdx = [array]::IndexOf($a, '--conversation')
$convId = if ($convIdx -ge 0 -and ($convIdx + 1) -lt $a.Count) { [string]$a[$convIdx + 1] } else { 'mock-conv' }

if ($prompt -eq '/usage' -or $env:MOCK_AGY_MODE -eq 'usage_cmd') {
    $u = @{
        conversation_id = ''; status = 'SUCCESS'; response = 'usage'; duration_seconds = 0; num_turns = 0
        usage = @{ input_tokens = 0; output_tokens = 0; thinking_tokens = 0; cache_read_tokens = 0; total_tokens = 0 }
        command = @{
            name = 'usage'
            data = @{
                groups = @(
                    @{ name = 'Gemini Models'; buckets = @(
                        @{ id = 'gemini-weekly'; window = 'weekly'; remaining_fraction = 0.96; reset_time = '2026-11-16T14:00:00Z' },
                        @{ id = 'gemini-5h';     window = '5h';     remaining_fraction = 0.92; reset_time = '2026-11-09T20:00:00Z' }
                    ) },
                    @{ name = 'Claude and GPT models'; buckets = @(
                        @{ id = '3p-weekly'; window = 'weekly'; remaining_fraction = 0.48; reset_time = '2026-11-16T15:00:00Z' },
                        @{ id = '3p-5h';     window = '5h';     remaining_fraction = 0.05; reset_time = '2026-11-09T20:00:00Z' }
                    ) },
                    @{ name = 'Gemini 4 Argon'; buckets = @(
                        @{ id = 'argon-weekly'; window = 'weekly'; remaining_fraction = 0.94; reset_time = '2026-11-16T13:00:00Z' }
                    ) }
                )
            }
        }
    }
    $u | ConvertTo-Json -Depth 10 -Compress
    exit 0
}

switch ($env:MOCK_AGY_MODE) {
    'fail'    { [Console]::Error.WriteLine('mock: quota exceeded'); exit 1 }
    'denied'  { '{"conversation_id":"d","status":"SUCCESS","response":"Encountered error: permission check failed for unsandboxed dir: user denied permission to run command","usage":{"total_tokens":28325}}'; exit 0 }
    'empty'   { '{"conversation_id":"x","status":"SUCCESS","response":"","usage":{"total_tokens":50}}'; exit 0 }
    'badjson' { 'not json at all'; exit 0 }
    'flaky'   {
        $c = "$env:MOCK_AGY_LOG.count"
        $n = if (Test-Path $c) { [int](Get-Content $c) } else { 0 }
        Set-Content $c ($n + 1)
        if ($n -eq 0) { [Console]::Error.WriteLine('mock: transient'); exit 1 }
    }
    'timeout' {
        '{"event":"init","conversation_id":"conv-timeout-42"}'
        '{"event":"step_update","step_type":"tool_call","tool_name":"read_file","tool_info":{"tool_name":"read_file","parameters":{"Path":"scripts/benchmark_defect_masks.py"}}}'
        '{"event":"step_update","step_type":"planner_response","content":"Partial investigation notes before timeout"}'
        '{"event":"result","result":{"conversation_id":"conv-timeout-42","status":"DEADLINE_EXCEEDED","finished_reason":"print_timeout","response":"Partial investigation notes before timeout","duration_seconds":1200,"usage":{"input_tokens":50000,"output_tokens":2000,"thinking_tokens":1000,"cache_read_tokens":0,"total_tokens":52000}}}'
        [Console]::Error.WriteLine('[agy] print timeout after 20m0s with turn in progress; returning partial output')
        [Console]::Error.WriteLine('AGY_ERROR: {"short_error":"print timeout after 20m0s with the turn still in progress; the response is partial","status":"DEADLINE_EXCEEDED","retryable":false,"reason":"print_timeout"}')
        exit 3
    }
    'interrupted' {
        '{"event":"init","conversation_id":"conv-int-7"}'
        '{"event":"step_update","step_type":"planner_response","content":"Partial output before cancel"}'
        '{"event":"result","result":{"conversation_id":"conv-int-7","status":"CANCELLED","finished_reason":"interrupted","response":"Partial output before cancel","usage":{"total_tokens":900}}}'
        exit 130
    }
    'stream' {
        '{"event":"init","conversation_id":"' + $convId + '"}'
        '{"event":"step_update","step_type":"tool_call","tool_name":"read_file","tool_info":{"tool_name":"read_file","parameters":{"Path":"docs/maestro/plans/x.md"}}}'
        '{"event":"step_update","step_type":"tool_call","tool_name":"run_command","tool_info":{"tool_name":"run_command","parameters":{"CommandLine":".venv/Scripts/python.exe -m unittest"}}}'
        '{"event":"step_update","step_type":"tool_call","tool_name":"edit_file","tool_info":{"tool_name":"edit_file","parameters":{"Path":"scripts/benchmark_defect_masks.py"}}}'
        $resp = "streamed-ok`nVERDICT: APPROVED`nVERIFICATION: PASSED`nMODEL: $model"
        @{ event = 'result'; result = @{
            conversation_id = $convId; status = 'SUCCESS'; response = $resp; duration_seconds = 0.5; num_turns = 2
            usage = @{ input_tokens = 200; output_tokens = 40; thinking_tokens = 10; cache_read_tokens = 50; total_tokens = 240 }
        } } | ConvertTo-Json -Depth 6 -Compress
        exit 0
    }
    'review_changes' {
        $verdict = if ($prompt -match "complete the 'review' task") { 'VERDICT: CHANGES_REQUIRED - fix edge case' } else { 'VERDICT: APPROVED' }
        @{ event = 'result'; result = @{
            conversation_id = $convId; status = 'SUCCESS'; response = "done`n$verdict`nVERIFICATION: PASSED`nMODEL: $model"; duration_seconds = 0.1
            num_turns = 1; usage = @{ input_tokens = 100; output_tokens = 20; thinking_tokens = 0; cache_read_tokens = 0; total_tokens = 120 }
        } } | ConvertTo-Json -Depth 6 -Compress
        exit 0
    }
}

$r = @{
    conversation_id = $convId; status = 'SUCCESS'; response = "done`nVERDICT: APPROVED`nVERIFICATION: PASSED`nMODEL: $model"; duration_seconds = 0.1
    num_turns = 1; usage = @{ input_tokens = 100; output_tokens = 20; thinking_tokens = 0; cache_read_tokens = 0; total_tokens = 120 }
}
$r | ConvertTo-Json -Compress
exit 0
