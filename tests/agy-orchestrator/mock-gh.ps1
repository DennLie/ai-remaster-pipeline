param()
# Mock gh CLI supporting:
# - `gh issue view <n> --json title,body`
# - `gh issue comment <n> --body <body>`
# - `gh issue edit <n> --add-label <label>`
# - `gh pr list --head <branch> ...`
# - `gh pr create --draft --base <base> --head <branch> --title <t> --body <b>`
# - `gh api repos/{owner}/{repo}/issues/<n>/comments`
# - paginated `gh api repos/{owner}/{repo}/issues?...`
$a = $args

function Get-MockGhState {
    if ($env:MOCK_GH_STATE_FILE -and (Test-Path -LiteralPath $env:MOCK_GH_STATE_FILE)) {
        return (Get-Content -LiteralPath $env:MOCK_GH_STATE_FILE -Raw -Encoding utf8 | ConvertFrom-Json -AsHashtable)
    }
    return @{ comments = @{}; labels = @{}; prs = @(); calls = @() }
}
function Save-MockGhState($st) {
    if ($env:MOCK_GH_STATE_FILE) {
        [System.IO.File]::WriteAllText($env:MOCK_GH_STATE_FILE, ($st | ConvertTo-Json -Depth 10), [System.Text.UTF8Encoding]::new($false))
    }
}

if ($a.Count -ge 2 -and $a[0] -eq 'issue' -and $a[1] -eq 'view') {
    $num = [int]$a[2]
    $obj = @{ title = "Mock issue #$num"; body = "Please implement feature #$num in ``scripts/feature_$num.py``." }
    $obj | ConvertTo-Json -Compress
    exit 0
}

if ($a.Count -ge 2 -and $a[0] -eq 'issue' -and $a[1] -eq 'comment') {
    $numKey = [string]$a[2]
    $bodyIdx = [Array]::IndexOf($a, '--body')
    $bodyTxt = if ($bodyIdx -ge 0 -and ($bodyIdx + 1) -lt $a.Count) { [string]$a[$bodyIdx + 1] } else { '' }
    $st = Get-MockGhState
    if (-not $st.comments.ContainsKey($numKey)) { $st.comments[$numKey] = @() }
    $st.comments[$numKey] = @($st.comments[$numKey]) + @(@{ id = ($st.comments[$numKey].Count + 1); body = $bodyTxt })
    $st.calls = @($st.calls) + @("issue comment $numKey")
    Save-MockGhState $st
    "https://github.com/DennLie/ai-remaster-pipeline/issues/$numKey#issuecomment-1"
    exit 0
}

if ($a.Count -ge 2 -and $a[0] -eq 'issue' -and $a[1] -eq 'edit') {
    $numKey = [string]$a[2]
    $lblIdx = [Array]::IndexOf($a, '--add-label')
    $lblTxt = if ($lblIdx -ge 0 -and ($lblIdx + 1) -lt $a.Count) { [string]$a[$lblIdx + 1] } else { '' }
    $st = Get-MockGhState
    if (-not $st.labels.ContainsKey($numKey)) { $st.labels[$numKey] = @() }
    $st.labels[$numKey] = @($st.labels[$numKey] + @($lblTxt) | Select-Object -Unique)
    $st.calls = @($st.calls) + @("issue edit $numKey --add-label $lblTxt")
    Save-MockGhState $st
    exit 0
}

if ($a.Count -ge 2 -and $a[0] -eq 'pr' -and $a[1] -eq 'list') {
    $headIdx = [Array]::IndexOf($a, '--head')
    $headBr = if ($headIdx -ge 0 -and ($headIdx + 1) -lt $a.Count) { [string]$a[$headIdx + 1] } else { '' }
    $st = Get-MockGhState
    $matched = @($st.prs | Where-Object { $_.head -eq $headBr })
    ConvertTo-Json -InputObject @($matched) -Depth 6 -Compress
    exit 0
}

if ($a.Count -ge 2 -and $a[0] -eq 'pr' -and $a[1] -eq 'create') {
    $isDraft = ($a -contains '--draft')
    $headIdx = [Array]::IndexOf($a, '--head')
    $baseIdx = [Array]::IndexOf($a, '--base')
    $titleIdx = [Array]::IndexOf($a, '--title')
    $bodyIdx = [Array]::IndexOf($a, '--body')
    $headBr = if ($headIdx -ge 0 -and ($headIdx + 1) -lt $a.Count) { [string]$a[$headIdx + 1] } else { '' }
    $baseBr = if ($baseIdx -ge 0 -and ($baseIdx + 1) -lt $a.Count) { [string]$a[$baseIdx + 1] } else { 'main' }
    $titleTxt = if ($titleIdx -ge 0 -and ($titleIdx + 1) -lt $a.Count) { [string]$a[$titleIdx + 1] } else { '' }
    $bodyTxt  = if ($bodyIdx -ge 0 -and ($bodyIdx + 1) -lt $a.Count) { [string]$a[$bodyIdx + 1] } else { '' }
    $st = Get-MockGhState
    $prNum = @($st.prs).Count + 201
    $prUrl = "https://github.com/DennLie/ai-remaster-pipeline/pull/$prNum"
    $st.prs = @($st.prs) + @(@{
        number  = $prNum
        url     = $prUrl
        state   = 'OPEN'
        isDraft = [bool]$isDraft
        head    = $headBr
        base    = $baseBr
        title   = $titleTxt
        body    = $bodyTxt
    })
    $st.calls = @($st.calls) + @("pr create --draft=$isDraft --head $headBr")
    Save-MockGhState $st
    $prUrl
    exit 0
}

if ($a.Count -ge 2 -and $a[0] -eq 'api') {
    $joined = $a -join ' '
    if ($joined -match 'issues/([0-9]+)/comments') {
        $numKey = [string]$Matches[1]
        $st = Get-MockGhState
        $cList = if ($st.comments.ContainsKey($numKey)) { @($st.comments[$numKey]) } else { @() }
        ConvertTo-Json -InputObject @($cList) -Depth 6 -Compress
        exit 0
    }
    $page = if ($joined -match '(?:[?&]|\b)page=([0-9]+)') { [int]$Matches[1] } else { 1 }
    $perPage = if ($joined -match '(?:[?&]|\b)per_page=([0-9]+)') { [int]$Matches[1] } else { 100 }

    if ($env:MOCK_GH_MODE -eq 'scale120') {
        $total = 120
        $start = (($page - 1) * $perPage) + 1
        if ($start -gt $total) { '[]'; exit 0 }
        $end = [Math]::Min($total, $start + $perPage - 1)
        $list = for ($i = $start; $i -le $end; $i++) {
            @{
                number = $i
                title  = "Scale issue #$i"
                state  = 'open'
                labels = @(@{ name = 'ready-for-agent' })
                body   = "Body for #$i"
                updated_at = '2026-11-09T15:00:00Z'
                issue_dependencies_summary = @{ blocked_by = 0 }
            }
        }
        ConvertTo-Json -InputObject @($list) -Depth 6 -Compress
        exit 0
    }

    if ($page -gt 1) { '[]'; exit 0 }
    # Default backlog dataset testing dependencies, triage blocks, already-completed checkboxes, PR filtering, and priorities
    $items = @(
        @{
            number = 101; title = 'Pull request (must be ignored)'; state = 'open'
            pull_request = @{ url = 'https://api.github.com/pulls/101' }
            labels = @(@{ name = 'ready-for-agent' }); body = 'PR'
        },
        @{
            number = 40; title = 'Dependent feature (after #33)'; state = 'open'
            labels = @(@{ name = 'ready-for-agent' })
            body = "Blocked by: #33`nImplement ``scripts/dep_40.py``."
            updated_at = '2026-11-09T15:05:00Z'; issue_dependencies_summary = @{ blocked_by = 0 }
        },
        @{
            number = 34; title = 'Human dataset curation'; state = 'open'
            labels = @(@{ name = 'ready-for-human' })
            body = 'Requires human annotation.'
            updated_at = '2026-11-09T15:04:30Z'; issue_dependencies_summary = @{ blocked_by = 0 }
        },
        @{
            number = 33; title = 'Foundation benchmark harness'; state = 'open'
            labels = @(@{ name = 'ready-for-agent' })
            body = "Part of Phase 1.`n- [ ] ``scripts/benchmark_defect_masks.py``"
            updated_at = '2026-11-09T15:04:20Z'; issue_dependencies_summary = @{ blocked_by = 0 }
        },
        @{
            number = 32; title = 'Already finished issue'; state = 'open'
            labels = @(@{ name = 'ready-for-agent' })
            body = "- [x] Done item 1`n- [x] Done item 2"
            updated_at = '2026-11-09T15:01:00Z'; issue_dependencies_summary = @{ blocked_by = 0 }
        }
    )
    ConvertTo-Json -InputObject @($items) -Depth 6 -Compress
    exit 0
}
echo '{"title":"Mock issue","body":"Please implement the thing."}'
exit 0
