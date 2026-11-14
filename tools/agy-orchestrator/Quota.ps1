#requires -Version 7
<#
.SYNOPSIS  Show or set weekly and 5-hour quota readings used by the orchestrator guards.
.EXAMPLE   ./Quota.ps1                                                    # report with ACTUAL/ESTIMATE/UNKNOWN labels
.EXAMPLE   ./Quota.ps1 -Set claude-gpt -Percent 47 -FiveHourPercent 0     # manual reading from /usage
.EXAMPLE   ./Quota.ps1 -SyncFromCli                                       # sync from `agy -p "/usage" --output-format json` (0 tokens)
#>
[CmdletBinding()]
param(
    [string]$Set,
    [double]$Percent = -1,
    [double]$FiveHourPercent = -1,
    [ValidateSet('manual', 'official')][string]$Source = 'manual',
    [switch]$SyncFromCli,
    [string]$AgyCommand = 'agy',
    [string]$PolicyPath = (Join-Path $PSScriptRoot 'policy.json'),
    [string]$StatePath = (Join-Path (Get-Location).Path '.cache/agy-orch/state.json')
)
. (Join-Path $PSScriptRoot 'AgyOrchestrator.ps1')
$policy = Get-AgyPolicy $PolicyPath
if ($SyncFromCli) {
    $null = Sync-AgyQuotaFromCli -StatePath $StatePath -AgyCommand $AgyCommand -Policy $policy
} elseif ($Set) {
    if ($Percent -lt 0) { throw 'Provide -Percent when using -Set.' }
    Set-AgyQuotaReading -Group $Set -RemainingPercent $Percent -FiveHourRemainingPercent $FiveHourPercent -Source $Source -StatePath $StatePath -Policy $policy
}
Get-AgyQuotaReport -Policy $policy -StatePath $StatePath | Format-Table group, remainingPercent, fiveHourRemainingPercent, kind, reservePercent, fiveHourReservePercent, belowReserve, localTokensUsed7d, basis -AutoSize -Wrap
