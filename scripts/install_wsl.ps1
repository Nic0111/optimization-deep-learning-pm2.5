[CmdletBinding()]
param(
    [switch]$ConfirmRestartRisk
)

$ErrorActionPreference = 'Stop'

if (-not $ConfirmRestartRisk) {
    throw 'This command may stage Windows features that require a restart. Re-run only after explicit user approval with -ConfirmRestartRisk.'
}

$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = [Security.Principal.WindowsPrincipal]::new($identity)
$isAdministrator = $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdministrator) {
    throw 'Open PowerShell as Administrator before running this script.'
}

$installed = @(wsl.exe --list --quiet 2>$null) -replace "`0", ''
if ($installed -contains 'Ubuntu-24.04') {
    Write-Output 'Ubuntu-24.04 is already installed; no installation change was made.'
    exit 0
}

Write-Output 'Installing Ubuntu-24.04 without launching or restarting Windows...'
wsl.exe --install --distribution Ubuntu-24.04 --no-launch
if ($LASTEXITCODE -ne 0) {
    throw "WSL installation returned exit code $LASTEXITCODE."
}

$pendingKeys = @(
    'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Component Based Servicing\RebootPending',
    'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\WindowsUpdate\Auto Update\RebootRequired'
)
$restartPending = ($pendingKeys | Where-Object { Test-Path -LiteralPath $_ }).Count -gt 0
if ($restartPending) {
    Write-Output 'A Windows restart is required. Do not restart until the user explicitly approves.'
} else {
    Write-Output 'No Windows restart marker was detected. Launch Ubuntu once to create its Linux user.'
}
