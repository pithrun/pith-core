# ============================================================
# Pith — Backup Schedule Setup (Windows Task Scheduler)
# ============================================================
# Creates a scheduled task that runs safe_backup.ps1 every 3h.
# Idempotent — safe to run multiple times.
#
# Usage: powershell -File setup_schedule.ps1 [--remove]
# ============================================================
param(
    [switch]$Remove = $false
)

$ErrorActionPreference = 'Stop'
$TaskName = "Pith-Backup-3h"
$PithHome = if ($env:PITH_HOME) { $env:PITH_HOME } else { "$env:USERPROFILE\.pith" }
$SafeBackupPath = "$PithHome\pith-server\scripts\backup\safe_backup.ps1"

if ($Remove) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
    Write-Host "Removed scheduled task: $TaskName"
    exit 0
}

# Verify safe_backup.ps1 exists
if (-not (Test-Path $SafeBackupPath)) {
    # Try distribution copy
    $ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
    $SafeBackupPath = "$ScriptDir\safe_backup.ps1"
    if (-not (Test-Path $SafeBackupPath)) {
        Write-Host "ERROR: safe_backup.ps1 not found" -ForegroundColor Red
        Write-Host "  Expected: $SafeBackupPath"
        exit 1
    }
}

# Create the scheduled task (every 3 hours)
$Action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$SafeBackupPath`" -Quiet"

$Trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).Date `
    -RepetitionInterval (New-TimeSpan -Hours 3)

$Principal = New-ScheduledTaskPrincipal -UserId (whoami) -RunLevel Limited

$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable

try {
    Register-ScheduledTask -TaskName $TaskName `
        -Action $Action `
        -Trigger $Trigger `
        -Principal $Principal `
        -Settings $Settings `
        -Force | Out-Null
    Write-Host "Scheduled task created: $TaskName" -ForegroundColor Green
    Write-Host "  Runs every 3 hours"
    Write-Host "  Script: $SafeBackupPath"
} catch {
    Write-Host "ERROR: Could not create scheduled task: $_" -ForegroundColor Red
    Write-Host "  Try running as Administrator"
    exit 1
}

# Remove legacy daily backup task if exists
$Legacy = Get-ScheduledTask -TaskName "Pith-Daily-Backup" -ErrorAction SilentlyContinue
if ($Legacy) {
    Unregister-ScheduledTask -TaskName "Pith-Daily-Backup" -Confirm:$false
    Write-Host "  Removed legacy daily backup task"
}
