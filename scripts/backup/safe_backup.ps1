# ============================================================
# Pith — Safe Database Backup (Windows)
# ============================================================
# Creates a consistent backup of the Pith database using Python's
# sqlite3.backup() API. This is the ONLY safe way to back up
# a WAL-mode database while the server is running.
#
# DO NOT use: Copy-Item pith.db backup.db (corrupts if WAL active)
# DO NOT use: Compress-Archive on live DB (same corruption risk)
#
# Usage: powershell -File safe_backup.ps1 [--output C:\path\backup.db]
# ============================================================
param(
    [string]$Output = "",
    [switch]$Quiet = $false
)

$ErrorActionPreference = 'Stop'

# Resolve paths
$PithHome = if ($env:PITH_HOME) { $env:PITH_HOME } else { "$env:USERPROFILE\.pith" }
# Resolve DB path: pith.db (post-Brand-001) first, brain.db fallback
$DbPath = if (Test-Path "$PithHome\data\pith.db") { "$PithHome\data\pith.db" } else { "$PithHome\data\brain.db" }
$ArchiveDir = "$PithHome\backups"
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$KeepBackups = if ($env:KEEP_BACKUPS) { [int]$env:KEEP_BACKUPS } else { 10 }

# Default output path
if (-not $Output) {
    New-Item -ItemType Directory -Path $ArchiveDir -Force | Out-Null
    $Output = "$ArchiveDir\pith_backup_$Timestamp.db"
}

function Log($msg) {
    if (-not $Quiet) { Write-Host $msg }
}

# Verify source exists
if (-not (Test-Path $DbPath)) {
    Write-Host "ERROR: Database not found at $DbPath" -ForegroundColor Red
    Write-Host "  Has Pith been started at least once?"
    exit 1
}

# Find Python in venv
$PythonExe = "$PithHome\venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    Write-Host "ERROR: Python venv not found at $PithHome\venv" -ForegroundColor Red
    Write-Host "  Re-run the installer: scripts\install.ps1"
    exit 1
}

$DbName = Split-Path $DbPath -Leaf
Log "Creating safe backup of $DbName..."
Log "  Source: $DbPath"
Log "  Output: $Output"

# Use Python sqlite3.backup() — WAL-safe
$BackupScript = @"
import sqlite3, sys
src = sqlite3.connect(r'$DbPath')
dst = sqlite3.connect(r'$Output')
src.backup(dst)
dst.close()
src.close()
# Verify
v = sqlite3.connect(r'$Output')
integrity = v.execute('PRAGMA integrity_check').fetchone()[0]
try:
    concepts = v.execute('SELECT COUNT(*) FROM concepts').fetchone()[0]
    latest = v.execute('SELECT MAX(created_at) FROM concepts').fetchone()[0] or 'N/A'
except:
    concepts = 0
    latest = 'N/A'
v.close()
print(f'{integrity}|{concepts}|{latest}')
"@

$Result = & $PythonExe -c $BackupScript 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Backup failed: $Result" -ForegroundColor Red
    Remove-Item -Path $Output -ErrorAction SilentlyContinue
    exit 1
}

$Parts = $Result -split '\|'
$Integrity = $Parts[0]
$Concepts = $Parts[1]
$Latest = $Parts[2]

if ($Integrity -eq "ok") {
    $Size = (Get-Item $Output).Length / 1KB
    Log ""
    Log "Backup successful!"
    Log "  Size: $([Math]::Round($Size, 1))KB"
    Log "  Concepts: $Concepts"
    Log "  Latest: $Latest"
    Log "  Integrity: OK"
} else {
    Write-Host ""
    Write-Host "BACKUP INTEGRITY CHECK FAILED!" -ForegroundColor Red
    Write-Host "  $Integrity"
    Remove-Item -Path $Output -ErrorAction SilentlyContinue
    exit 1
}

# Data presence warning
if ($Concepts -eq "0" -or -not $Concepts) {
    Write-Host "WARNING: Backup contains 0 concepts." -ForegroundColor Yellow
    Write-Host "  This might indicate the wrong database was backed up."
}

# Retention: keep only the last N backups
if (Test-Path $ArchiveDir) {
    $Backups = Get-ChildItem "$ArchiveDir\*_backup_*.db" -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -notmatch '-(shm|wal)$' } |
        Sort-Object LastWriteTime -Descending
    if ($Backups.Count -gt $KeepBackups) {
        $ToRemove = $Backups | Select-Object -Skip $KeepBackups
        $Pruned = 0
        foreach ($Old in $ToRemove) {
            Remove-Item $Old.FullName -Force
            Remove-Item "$($Old.FullName)-shm" -Force -ErrorAction SilentlyContinue
            Remove-Item "$($Old.FullName)-wal" -Force -ErrorAction SilentlyContinue
            $Pruned++
        }
        Log "  Retention: kept $KeepBackups, pruned $Pruned old backup(s)"
    } else {
        Log "  Retention: $($Backups.Count) backup(s), no pruning needed (keep=$KeepBackups)"
    }
}
