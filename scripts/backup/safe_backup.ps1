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
$PithServerPath = "$PithHome\pith-server"

function Resolve-PithDefaultDataDir {
    $Profile = if ($env:PITH_PROFILE) { $env:PITH_PROFILE } else { "default" }
    $DataRoot = Join-Path ([Environment]::GetFolderPath("UserProfile")) "pith-data"
    return (Join-Path $DataRoot $Profile)
}

function Resolve-PithDataDir {
    if ($env:PITH_DATA_DIR) { return $env:PITH_DATA_DIR }
    foreach ($EnvFile in @("$PithHome\.env", "$PithServerPath\.env")) {
        if (-not (Test-Path -LiteralPath $EnvFile -PathType Leaf)) {
            continue
        }
        $DataDirLine = Get-Content -LiteralPath $EnvFile |
            Where-Object { $_ -match '^PITH_DATA_DIR=' } |
            Select-Object -First 1
        if ($DataDirLine) {
            $ConfiguredDataDir = (($DataDirLine -split '=', 2)[1]).Trim()
            if ($ConfiguredDataDir) { return $ConfiguredDataDir }
        }
    }
    return (Resolve-PithDefaultDataDir)
}

# Resolve DB path: pith.db (post-Brand-001) first, real brain.db fallback.
function Resolve-DbPath {
    $DataDir = Resolve-PithDataDir
    $PithDb = Join-Path $DataDir "pith.db"
    $BrainDb = Join-Path $DataDir "brain.db"
    if (Test-Path -LiteralPath $PithDb -PathType Leaf) { return $PithDb }
    if (Test-Path -LiteralPath $BrainDb -PathType Leaf) { return $BrainDb }
    return $PithDb
}

$DbPath = Resolve-DbPath
$ArchiveDir = "$PithHome\backups"
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$KeepBackups = if ($env:KEEP_BACKUPS) { [int]$env:KEEP_BACKUPS } else { 3 }

# Default output path
if (-not $Output) {
    New-Item -ItemType Directory -Path $ArchiveDir -Force | Out-Null
    $Output = "$ArchiveDir\pith_backup_$Timestamp.db"
}

function Log($msg) {
    if (-not $Quiet) { Write-Host $msg }
}

# Verify source exists
if (-not (Test-Path -LiteralPath $DbPath -PathType Leaf)) {
    Write-Host "ERROR: Database not found at $DbPath" -ForegroundColor Red
    Write-Host "  Has Pith been started at least once?"
    exit 1
}

# Backup needs only sqlite3, so it does not depend on the application venv.
# Match the installer's deterministic managed venv for owned-Python opt-out installs.
$VenvRoot = if ($env:LOCALAPPDATA) {
    Join-Path $env:LOCALAPPDATA 'Pith\venvs'
} else {
    Join-Path (Split-Path -Parent $PithHome) 'AppData\Local\Pith\venvs'
}
$Hasher = [Security.Cryptography.SHA256]::Create()
try {
    $HomeHash = [BitConverter]::ToString($Hasher.ComputeHash([Text.Encoding]::UTF8.GetBytes($PithHome))).Replace('-', '').Substring(0, 12).ToLowerInvariant()
} finally {
    $Hasher.Dispose()
}
$ManagedPython = Join-Path (Join-Path $VenvRoot $HomeHash) 'Scripts\python.exe'
$PythonExe = @("$PithHome\runtime\python\python.exe", $ManagedPython, "$PithHome\venv\Scripts\python.exe") |
    Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } |
    Select-Object -First 1
if (-not $PythonExe) {
    Write-Host "ERROR: Pith backup Python runtime not found under $PithHome" -ForegroundColor Red
    Write-Host "  Re-run the installer: scripts\install.ps1"
    exit 1
}

$DbName = Split-Path $DbPath -Leaf
Log "Creating safe backup of $DbName..."
Log "  Source: $DbPath"
Log "  Output: $Output"

# Use Python sqlite3.backup() — WAL-safe
$BackupScript = @'
import sqlite3, sys
from pathlib import Path
src = sqlite3.connect(Path(sys.argv[1]).resolve().as_uri() + '?mode=ro', uri=True)
dst = sqlite3.connect(sys.argv[2])
src.backup(dst)
dst.close()
src.close()
# Verify
v = sqlite3.connect(sys.argv[2])
integrity = v.execute('PRAGMA integrity_check').fetchone()[0]
try:
    concepts = v.execute('SELECT COUNT(*) FROM concepts').fetchone()[0]
    latest = v.execute('SELECT MAX(created_at) FROM concepts').fetchone()[0] or 'N/A'
except:
    concepts = 0
    latest = 'N/A'
v.close()
print(f'{integrity}|{concepts}|{latest}')
'@

# Reserve a new destination so failures cannot delete a pre-existing user file.
$Reservation = [IO.File]::Open($Output, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
$Reservation.Dispose()
$Result = & $PythonExe -I -S -c $BackupScript $DbPath $Output 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Backup failed: $Result" -ForegroundColor Red
    Remove-Item -LiteralPath $Output -ErrorAction SilentlyContinue
    exit 1
}

$Parts = $Result -split '\|'
$Integrity = $Parts[0]
$Concepts = $Parts[1]
$Latest = $Parts[2]

if ($Integrity -eq "ok") {
    $Size = (Get-Item -LiteralPath $Output).Length / 1KB
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
    Remove-Item -LiteralPath $Output -ErrorAction SilentlyContinue
    exit 1
}

# Data presence warning
if ($Concepts -eq "0" -or -not $Concepts) {
    Write-Host "WARNING: Backup contains 0 concepts." -ForegroundColor Yellow
    Write-Host "  This might indicate the wrong database was backed up."
}

# Retention: keep only the last N backups
if (Test-Path -LiteralPath $ArchiveDir) {
    $Backups = Get-ChildItem -LiteralPath $ArchiveDir -Filter '*_backup_*.db' -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -notmatch '-(shm|wal)$' } |
        Sort-Object LastWriteTime -Descending
    if ($Backups.Count -gt $KeepBackups) {
        $ToRemove = $Backups | Select-Object -Skip $KeepBackups
        $Pruned = 0
        foreach ($Old in $ToRemove) {
            Remove-Item -LiteralPath $Old.FullName -Force
            Remove-Item -LiteralPath "$($Old.FullName)-shm" -Force -ErrorAction SilentlyContinue
            Remove-Item -LiteralPath "$($Old.FullName)-wal" -Force -ErrorAction SilentlyContinue
            $Pruned++
        }
        Log "  Retention: kept $KeepBackups, pruned $Pruned old backup(s)"
    } else {
        Log "  Retention: $($Backups.Count) backup(s), no pruning needed (keep=$KeepBackups)"
    }
}
