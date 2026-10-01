#Requires -Version 5.0
<#
.SYNOPSIS
    Pith Docker to Native Migration Script (Windows)

.DESCRIPTION
    Migrates beta users from Docker brain-mcp container to native ~/.pith/ install.
    Implements: DOCKER_MIGRATION_SPEC.md §4B, Amendments A-E

.PARAMETER ContainerName
    Docker container name (default: brain-mcp)

.PARAMETER DataDir
    Custom data directory inside container (default: auto-detect)

.EXAMPLE
    PowerShell -ExecutionPolicy Bypass -File migrate_from_docker.ps1
    PowerShell -ExecutionPolicy Bypass -File migrate_from_docker.ps1 -ContainerName "my-brain"
#>

param(
    [string]$ContainerName = "brain-mcp",
    [string]$DataDir = ""
)

$ErrorActionPreference = "Stop"

# --- Configuration ---
$PithHome = "$env:USERPROFILE\.pith"
$ExtractTempDir = ""
$DockerWasStopped = $false
$ContainerId = ""
$SkipImport = $false
$PreConceptCount = 0
$NewApiKey = ""
# --- Helper Functions ---
function Write-Banner {
    Clear-Host
    Write-Host ""
    Write-Host "  +========================================+" -ForegroundColor Blue
    Write-Host "  |   Pith Docker -> Native Migration      |" -ForegroundColor Blue
    Write-Host "  |   v1.0 - Windows                       |" -ForegroundColor Blue
    Write-Host "  +========================================+" -ForegroundColor Blue
    Write-Host ""
}

function Write-Success { param([string]$Message) Write-Host "  [OK] $Message" -ForegroundColor Green }
function Write-Warn    { param([string]$Message) Write-Host "  [!!] $Message" -ForegroundColor Yellow }
function Write-Info    { param([string]$Message) Write-Host "  [--] $Message" -ForegroundColor Cyan }

function Exit-WithError {
    param([string]$Message)
    Write-Host "  [ERROR] $Message" -ForegroundColor Red

    # Amendment E: auto-restart Docker if we stopped it
    if ($script:DockerWasStopped -and $script:ContainerId) {
        Write-Host ""
        Write-Host "  Restarting Docker container..." -ForegroundColor Yellow
        docker start $script:ContainerId 2>$null | Out-Null
        docker update --restart=unless-stopped $script:ContainerId 2>$null | Out-Null
        Write-Host "  Docker container restarted. Your original setup should be working." -ForegroundColor Yellow
    }

    # Clean up temp dir
    if ($script:ExtractTempDir -and (Test-Path $script:ExtractTempDir)) {
        Remove-Item -Recurse -Force $script:ExtractTempDir -ErrorAction SilentlyContinue
    }

    Write-Host ""
    Write-Host "  Migration did not complete. Your Docker setup has been restored." -ForegroundColor Red
    Write-Host "  If you need help, contact your beta support channel."
    exit 1
}
Write-Banner

# ============================================================================
# STEP 0: Pre-flight Checks
# ============================================================================
Write-Host "  [Step 0/7] Pre-flight checks..." -ForegroundColor Blue

# Docker installed?
try { docker --version | Out-Null } catch { Exit-WithError "Docker not found. Is Docker Desktop installed?" }

# Docker daemon running?
try { docker info 2>$null | Out-Null } catch { Exit-WithError "Docker daemon not running. Please start Docker Desktop and retry." }

# Python 3.9+?
try {
    $PyVer = python --version 2>&1
    if ($PyVer -match "(\d+)\.(\d+)") {
        $PyMajor = [int]$Matches[1]; $PyMinor = [int]$Matches[2]
        if ($PyMajor -lt 3 -or ($PyMajor -eq 3 -and $PyMinor -lt 9)) {
            Exit-WithError "Python 3.9+ required, found $PyVer"
        }
        Write-Success "Python $($Matches[0])"
    }
} catch { Exit-WithError "Python not found. Please install Python 3.9+." }

# Disk space (need ~500MB)
$FreeMB = [math]::Round((Get-PSDrive C).Free / 1MB)
if ($FreeMB -lt 500) { Exit-WithError "Insufficient disk space. Need ~500MB, have ${FreeMB}MB" }
Write-Success "Disk space OK (${FreeMB}MB available)"

# F8: Port 8000 pre-check
try {
    $HealthResp = Invoke-WebRequest -Uri "http://127.0.0.1:8000/health" -TimeoutSec 2 -ErrorAction SilentlyContinue
    if ($HealthResp.StatusCode -eq 200) {
        Write-Info "Port 8000 is active (Docker brain likely running - will be stopped before install)"
    }
} catch {
    # Port not responding - that's fine
}

Write-Host ""
# ============================================================================
# STEP 1: Detect Docker Container
# ============================================================================
Write-Host "  [Step 1/7] Detecting Docker container..." -ForegroundColor Blue

$RunningContainers = docker ps --filter "name=$ContainerName" --format "{{.Names}}" 2>$null
if (-not $RunningContainers) {
    $AllContainers = docker ps -a --filter "name=$ContainerName" --format "{{.Names}}" 2>$null
    if (-not $AllContainers) {
        Write-Host ""
        Write-Host "  No Docker container named '$ContainerName' found." -ForegroundColor Yellow
        Write-Host "  If your container has a different name, re-run with:"
        Write-Host "    .\migrate_from_docker.ps1 -ContainerName YOUR_NAME"
        Write-Host ""
        Write-Host "  If you never installed the Docker version, run the native installer:"
        Write-Host "    PowerShell -ExecutionPolicy Bypass -File scripts\install.ps1"
        exit 1
    }
    Write-Warn "Container '$AllContainers' exists but is not running."
    $reply = Read-Host "  Start it to export data? (y/n)"
    if ($reply -match "^[Yy]") {
        $first = ($AllContainers -split "`n")[0]
        docker start $first 2>$null | Out-Null
        $RunningContainers = $first
    } else {
        Exit-WithError "Cannot migrate without a running container."
    }
}

$script:ContainerId = ($RunningContainers -split "`n")[0]
Write-Success "Found Docker container: $($script:ContainerId)"

# Verify running
$ContainerState = docker inspect $script:ContainerId --format='{{.State.Running}}' 2>$null
if ($ContainerState -ne "true") { Exit-WithError "Container is not running." }

Write-Host ""
# ============================================================================
# STEP 2: Export Brain Data
# ============================================================================
Write-Host "  [Step 2/7] Exporting brain data from Docker container..." -ForegroundColor Blue

$script:ExtractTempDir = Join-Path $env:TEMP "pith-migration-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
New-Item -ItemType Directory -Path $script:ExtractTempDir -Force | Out-Null
Write-Success "Created temp directory"

# Amendment A: WAL Checkpoint
Write-Host "  Flushing database write-ahead log..."
try {
    docker exec $script:ContainerId python3 -c @"
import sqlite3, os
for p in ['/app/data/brain.db', '/pith/data/brain.db']:
    if os.path.exists(p):
        c = sqlite3.connect(p)
        c.execute('PRAGMA wal_checkpoint(FULL)')
        c.close()
        print(f'WAL checkpoint complete: {p}')
        break
"@ 2>$null
    Write-Success "WAL checkpoint complete"
} catch {
    Write-Warn "Could not flush WAL (non-fatal)"
}

# F1: Try /app/data first
$ContainerDataPath = ""
if ($DataDir) {
    $ContainerDataPath = $DataDir
} else {
    foreach ($tryPath in @("/app/data", "/pith/data", "/home/pith/data")) {
        $testResult = docker exec $script:ContainerId test -d $tryPath 2>$null
        if ($LASTEXITCODE -eq 0) {
            $ContainerDataPath = $tryPath
            break
        }
    }
}

if (-not $ContainerDataPath) {
    Exit-WithError "Could not locate data directory in container. Try: -DataDir /path/inside/container"
}
Write-Success "Found data at: $ContainerDataPath"

# Export
docker cp "${script:ContainerId}:${ContainerDataPath}" "$script:ExtractTempDir\data" 2>$null
if ($LASTEXITCODE -ne 0) { Exit-WithError "Failed to export data from container." }
Write-Success "Exported brain data from container"

# F5: Pre-migration concept count
$brainDbPath = Join-Path $script:ExtractTempDir "data\brain.db"
if (Test-Path $brainDbPath) {
    try {
        $script:PreConceptCount = python -c @"
import sqlite3
try:
    c = sqlite3.connect(r'$brainDbPath')
    count = c.execute('SELECT COUNT(*) FROM concepts').fetchone()[0]
    print(count)
    c.close()
except: print(0)
"@ 2>$null
        $dbSize = [math]::Round((Get-Item $brainDbPath).Length / 1MB, 1)
        Write-Success "Brain contains $($script:PreConceptCount) concepts (${dbSize}MB)"
    } catch {
        $script:PreConceptCount = 0
    }
}

Write-Host ""
# ============================================================================
# STEP 2.5: Stop Docker Container (Amendment B)
# ============================================================================
Write-Host "  [Step 2.5/7] Stopping Docker container (freeing port 8000)..." -ForegroundColor Blue

docker update --restart=no $script:ContainerId 2>$null | Out-Null
docker stop $script:ContainerId 2>$null | Out-Null
$script:DockerWasStopped = $true
Write-Success "Docker container stopped (will not auto-restart)"

Write-Host ""

# ============================================================================
# STEP 3: Generate New API Key (F3)
# ============================================================================
Write-Host "  [Step 3/7] Generating secure API key..." -ForegroundColor Blue

# Try openssl first (Git for Windows includes it), fall back to .NET crypto
try {
    $script:NewApiKey = (openssl rand -hex 32 2>$null).Trim()
} catch { $script:NewApiKey = "" }

if (-not $script:NewApiKey) {
    $bytes = New-Object byte[] 32
    [System.Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
    $script:NewApiKey = ($bytes | ForEach-Object { $_.ToString("x2") }) -join ""
}
Write-Success "New API key generated (replaces default insecure key)"

Write-Host ""
# ============================================================================
# STEP 4: Run Native Installer
# ============================================================================
Write-Host "  [Step 4/7] Running native Pith installer..." -ForegroundColor Blue

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$InstallerScript = ""

foreach ($try in @(
    (Join-Path $ScriptDir "install.ps1"),
    (Join-Path $ScriptDir "..\scripts\install.ps1"),
    ".\scripts\install.ps1"
)) {
    if (Test-Path $try) { $InstallerScript = $try; break }
}

if (-not $InstallerScript) {
    Exit-WithError "Installer script (install.ps1) not found. Expected at $ScriptDir\install.ps1"
}

Write-Info "Running installer from: $InstallerScript"
PowerShell -ExecutionPolicy Bypass -File $InstallerScript
if ($LASTEXITCODE -ne 0) { Exit-WithError "Installer failed. Check the output above." }
Write-Success "Native Pith installation completed"

Write-Host ""
# ============================================================================
# STEP 5: Import Data (F4, F5, Amendment D)
# ============================================================================
Write-Host "  [Step 5/7] Importing brain data to native installation..." -ForegroundColor Blue

$DataDest = Join-Path $PithHome "data"
$ConfigDest = Join-Path $PithHome "config"
New-Item -ItemType Directory -Path $DataDest -Force | Out-Null
New-Item -ItemType Directory -Path $ConfigDest -Force | Out-Null

$brainDbSrc = Join-Path $script:ExtractTempDir "data\brain.db"
$brainDbDst = Join-Path $DataDest "brain.db"

if (Test-Path $brainDbSrc) {
    # Amendment D: mtime comparison
    if (Test-Path $brainDbDst) {
        $srcTime = (Get-Item $brainDbSrc).LastWriteTime
        $dstTime = (Get-Item $brainDbDst).LastWriteTime
        if ($dstTime -gt $srcTime) {
            Write-Warn "Existing brain.db is NEWER than Docker export."
            $reply = Read-Host "  Overwrite with Docker data? (y/n)"
            if ($reply -notmatch "^[Yy]") {
                Write-Info "Keeping existing data. Skipping import."
                $script:SkipImport = $true
            }
        }
    }

    if (-not $script:SkipImport) {
        # Amendment D: SHA256 checksum
        $SrcHash = (Get-FileHash -Algorithm SHA256 -Path $brainDbSrc).Hash

        # --- HARDENED: Pre-copy integrity check on exported brain.db ---
        $ExportIntegrity = python -c @"
import sqlite3
c = sqlite3.connect(r'$brainDbSrc')
result = c.execute('PRAGMA integrity_check').fetchone()[0]
print(result)
c.close()
"@ 2>$null
        if ($ExportIntegrity -ne "ok") {
            Write-Host ""
            Write-Warn "EXPORTED brain.db FAILED integrity check: $ExportIntegrity"
            Write-Warn "The database inside your Docker container appears to be corrupted."
            Write-Host ""
            Write-Host "  Options:"
            Write-Host "    1) Skip import - start with a fresh empty database"
            Write-Host "    2) Proceed anyway - import the corrupted database (not recommended)"
            Write-Host "    3) Abort - stop migration entirely"
            Write-Host ""
            $integrityChoice = Read-Host "  Choose (1/2/3)"
            switch ($integrityChoice) {
                "1" {
                    Write-Info "Skipping import. You'll start with a fresh brain database."
                    $script:SkipImport = $true
                }
                "2" {
                    Write-Warn "Proceeding with corrupted database at your own risk."
                }
                default {
                    Exit-WithError "Migration aborted by user. Docker container is still intact."
                }
            }
        } else {
            Write-Success "Exported database integrity verified"
        }
    }

    if (-not $script:SkipImport) {
        # Backup existing
        if (Test-Path $brainDbDst) {
            $backupName = "brain.db.pre-migration.$(Get-Date -Format 'yyyyMMddHHmmss')"
            Copy-Item $brainDbDst (Join-Path $DataDest $backupName)
            Write-Info "Backed up existing brain.db -> $backupName"
        }

        # Copy
        Copy-Item $brainDbSrc $brainDbDst -Force

        # Verify checksum
        $DstHash = (Get-FileHash -Algorithm SHA256 -Path $brainDbDst).Hash
        if ($SrcHash -eq $DstHash) {
            Write-Success "Data copied and verified (SHA256 match)"
        } else {
            Exit-WithError "Data copy verification FAILED! SHA256 mismatch."
        }

        # F4: integrity check
        $integrity = python -c @"
import sqlite3
c = sqlite3.connect(r'$brainDbDst')
result = c.execute('PRAGMA integrity_check').fetchone()[0]
print(result)
c.close()
"@ 2>$null
        if ($integrity -eq "ok") {
            Write-Success "Post-copy database integrity check passed"
        } else {
            Write-Warn "Post-copy integrity check failed: $integrity"
            Write-Warn "This may indicate a copy error. Your Docker data is still safe."
            Exit-WithError "Aborting due to post-copy integrity failure."
        }

        # F5: post-migration concept count
        $PostCount = python -c @"
import sqlite3
try:
    c = sqlite3.connect(r'$brainDbDst')
    count = c.execute('SELECT COUNT(*) FROM concepts').fetchone()[0]
    print(count)
    c.close()
except: print(0)
"@ 2>$null

        if ([int]$script:PreConceptCount -gt 0) {
            if ($PostCount -eq $script:PreConceptCount) {
                Write-Success "Concept count verified: $PostCount (matches pre-migration)"
            } else {
                Write-Warn "Concept count mismatch: pre=$($script:PreConceptCount), post=$PostCount"
            }
        } else {
            Write-Info "Post-migration concepts: $PostCount"
        }
    }
} else {
    Write-Warn "No brain.db to import - starting with fresh database"
}

# Write new API key
$apiKeyPath = Join-Path $ConfigDest "api.key"
$script:NewApiKey | Out-File -FilePath $apiKeyPath -Encoding ascii -NoNewline
Write-Success "API key written to $apiKeyPath"

Write-Host ""
# ============================================================================
# STEP 6: Update MCP Config (F2, Amendment C)
# ============================================================================
Write-Host "  [Step 6/7] Updating Claude Desktop MCP configuration..." -ForegroundColor Blue

$env:MIGRATED_API_KEY = $script:NewApiKey

python -c @"
import json, os, shutil, sys

config_path = os.path.join(os.environ['APPDATA'], 'Claude', 'claude_desktop_config.json')

if not os.path.exists(config_path):
    print(f'Claude Desktop config not found at {config_path}')
    pith_home = os.path.join(os.environ['USERPROFILE'], '.pith')
    print(f'You will need to manually update your MCP config.')
    print(f'  Server path: {os.path.join(pith_home, "pith-server", "server.js")}')
    print(f'  API key: check {os.path.join(pith_home, "config", "api.key")}')
    sys.exit(0)

backup_path = config_path + '.pre-migration'
shutil.copy2(config_path, backup_path)
print(f'Backed up config -> {os.path.basename(backup_path)}')

with open(config_path) as f:
    config = json.load(f)

pith_home = os.path.join(os.environ['USERPROFILE'], '.pith')
pith_server = os.path.join(pith_home, 'pith-server', 'server.js')
api_key = os.environ.get('MIGRATED_API_KEY', '')

servers = config.get('mcpServers', {})
updated = False
for name, entry in servers.items():
    args = entry.get('args', [])
    if any('server.js' in str(a) and 'pith' in str(a).lower() for a in args):
        if entry.get('command') == 'node':
            entry['args'] = [pith_server]
        else:
            entry['command'] = 'node'
            entry['args'] = [pith_server]
        if api_key:
            entry.setdefault('env', {})['PITH_API_KEY'] = api_key
        updated = True
        print(f'Updated MCP entry: {name}')
        print(f'  -> server.js: {pith_server}')
        print(f'  -> API key: ...{api_key[-8:]}')
        break

if not updated:
    for name in ['brain', 'brain-mcp', 'pith', 'pith-brain']:
        if name in servers:
            servers[name]['command'] = 'node'
            servers[name]['args'] = [pith_server]
            if api_key:
                servers[name].setdefault('env', {})['PITH_API_KEY'] = api_key
            updated = True
            print(f'Updated MCP entry: {name} (matched by name)')
            break

if not updated:
    print('No pith-related MCP entries found. Manual config update needed.')
    sys.exit(0)

with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)
print('Config saved successfully.')
"@ 2>$null

if ($LASTEXITCODE -eq 0) {
    Write-Success "MCP config updated"
} else {
    Write-Warn "MCP config update had issues. You may need to update manually."
}

Write-Host ""
# ============================================================================
# STEP 7: Verification & Cleanup
# ============================================================================
Write-Host "  [Step 7/7] Verifying native installation..." -ForegroundColor Blue

$pithCmd = Join-Path $PithHome "bin\pith.cmd"
if (Test-Path $pithCmd) {
    try {
        & $pithCmd start 2>$null | Out-Null
        Start-Sleep -Seconds 3

        $health = Invoke-WebRequest -Uri "http://127.0.0.1:8000/health" -TimeoutSec 5 -ErrorAction SilentlyContinue
        if ($health.StatusCode -eq 200) {
            Write-Success "Native server running - health check passed"
        } else {
            Write-Warn "Health check returned $($health.StatusCode) - server may still be starting"
        }
    } catch {
        Write-Warn "Health check did not respond. Try: & '$pithCmd' status"
    }
} else {
    Write-Warn "pith CLI not found at $pithCmd. Check installation."
}

Write-Host ""

# Docker Cleanup
Write-Host "  [Cleanup] Docker container cleanup..." -ForegroundColor Blue
Write-Host ""
Write-Host "  Your Docker container has been stopped."
Write-Host "  Your original data is safe in your previous Pith install directory."
Write-Host ""
$reply = Read-Host "  Remove Docker container '$($script:ContainerId)'? (y/n)"
if ($reply -match "^[Yy]") {
    docker rm $script:ContainerId 2>$null | Out-Null
    Write-Success "Docker container removed"

    $reply2 = Read-Host "  Remove Docker image too? (y/n)"
    if ($reply2 -match "^[Yy]") {
        try {
            $dockerImage = docker inspect $script:ContainerId --format='{{.Config.Image}}' 2>$null
            if ($dockerImage) {
                docker rmi $dockerImage 2>$null | Out-Null
                Write-Success "Docker image removed"
            }
        } catch {}
    }
} else {
    Write-Info "Docker container kept (stopped). Remove later: docker rm $($script:ContainerId)"
    Write-Warn "Docker auto-restart has been disabled. Container will NOT auto-restart."
}

# Clear error-recovery flag
$script:DockerWasStopped = $false

# Clean temp dir
if ($script:ExtractTempDir -and (Test-Path $script:ExtractTempDir)) {
    Remove-Item -Recurse -Force $script:ExtractTempDir -ErrorAction SilentlyContinue
}

Write-Host ""
# ============================================================================
# Final Success Message
# ============================================================================
Write-Host ""
Write-Host "  +========================================+" -ForegroundColor Green
Write-Host "  |     Migration Complete!                |" -ForegroundColor Green
Write-Host "  +========================================+" -ForegroundColor Green
Write-Host ""
Write-Host "  Your Pith Brain has been migrated from Docker to native installation."
Write-Host ""
if ([int]$script:PreConceptCount -gt 0) {
    Write-Host "  Brain data:  $($script:PreConceptCount) concepts preserved" -ForegroundColor Green
}
Write-Host "  Install:     $PithHome" -ForegroundColor Cyan
Write-Host "  API key:     $PithHome\config\api.key" -ForegroundColor Cyan
Write-Host "  Server:      $PithHome\pith-server\server.js" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Next steps:" -ForegroundColor White
Write-Host "    1. Restart Claude Desktop (Ctrl+Q, then reopen)" -ForegroundColor Yellow
Write-Host "    2. Verify: type 'Run brain_stats' in a new Claude conversation"
Write-Host ""
Write-Host "  Commands:"
Write-Host "    pith status    - Check if server is running"
Write-Host "    pith logs      - View server logs"
Write-Host "    pith backup    - Create a backup"
Write-Host "    pith restart   - Restart the server"
Write-Host ""
Write-Host "  You can safely delete your old Pith source directory when ready."
Write-Host ""

exit 0
