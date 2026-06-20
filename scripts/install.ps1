# Pith Brain Installer v1.1 (Windows PowerShell)
# Windows equivalent installer

#Requires -Version 5.0

param(
    [switch]$Force = $false,
    [string]$PithVersion = "1.0.2"
)

# Strict error handling
$ErrorActionPreference = 'Stop'
$VerbosePreference = 'SilentlyContinue'

# Configuration
$DownloadUrl = if ($env:DOWNLOAD_URL) { $env:DOWNLOAD_URL } else { "https://github.com/pithrun/pith-core/releases/latest/download" }
$ChecksumUrl = if ($env:CHECKSUM_URL) { $env:CHECKSUM_URL } else { "https://github.com/pithrun/pith-core/releases/latest/download" }
$PithHome = if ($env:PITH_HOME) { $env:PITH_HOME } else { "$env:USERPROFILE\.pith" }
$StepCount = 8
$CurrentStep = 0

# File names
$PithServerFilename = "pith-server-latest.zip"
$PithChecksumFilename = "pith-server-latest.sha256"

# Color functions
function Write-Banner {
    Clear-Host
    Write-Host ""
    Write-Host "+========================================+" -ForegroundColor Cyan
    Write-Host "|   Pith Brain Installer v$PithVersion        |" -ForegroundColor Cyan
    Write-Host "|      Windows Edition                   |" -ForegroundColor Cyan
    Write-Host "+========================================+" -ForegroundColor Cyan
    Write-Host ""
}

function Write-Step {
    param([int]$StepNum, [string]$StepName)
    $global:CurrentStep = $StepNum
    Write-Host "[Step $StepNum/$StepCount] $StepName" -ForegroundColor Cyan
}

function Write-Success {
    param([string]$Message)
    Write-Host "[OK] $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[!] $Message" -ForegroundColor Yellow
}

function Write-Error-Custom {
    param([string]$Message)
    Write-Host "[X] ERROR: $Message" -ForegroundColor Red
    exit 1
}

# Trap for cleanup
trap {
    Write-Error-Custom "Installation interrupted at step $CurrentStep`n$_"
}

Write-Banner

# ============================================================================
# STEP 1: System Check
# ============================================================================
Write-Step 1 "System check (OS, Python, disk space, venv)"

# Verify Windows
$OSName = [System.Environment]::OSVersion.Platform
if ($OSName -ne "Win32NT") {
    Write-Error-Custom "This script requires Windows. Detected: $OSName"
}
Write-Success "OS: Windows"

# Check Python
$PythonPath = $null
$PythonVersion = $null

# Try to find Python in PATH
try {
    $PythonPath = (Get-Command python.exe -ErrorAction SilentlyContinue).Source
    if ($PythonPath) {
        $VersionOutput = & python.exe --version 2>&1
        $PythonVersion = ($VersionOutput -split ' ')[-1]
    }
}
catch {
    $PythonPath = $null
}

# If not found, check Microsoft Store Python and offer to install
if (-not $PythonPath) {
    Write-Warning "Python 3 not found in PATH"
    
    # Check if winget is available
    try {
        $WingetAvailable = $null -ne (Get-Command winget -ErrorAction SilentlyContinue)
        if ($WingetAvailable) {
            Write-Host "Attempting to install Python 3.11 via winget..."
            & winget install -e --id Python.Python.3.11 -y
            $PythonPath = (Get-Command python.exe -ErrorAction SilentlyContinue).Source
            if ($PythonPath) {
                Write-Success "Python installed successfully"
            }
            else {
                Write-Error-Custom "Python installation failed. Please install Python 3.9+ manually from python.org"
            }
        }
        else {
            Write-Error-Custom "Python not found and winget is unavailable. Please install Python 3.9+ from python.org"
        }
    }
    catch {
        Write-Error-Custom "Failed to install Python. Please install Python 3.9+ manually from python.org"
    }
}

# Verify Python version
$VersionParts = $PythonVersion -split '\.'
$MajorVersion = [int]$VersionParts[0]
$MinorVersion = [int]$VersionParts[1]

if ($MajorVersion -lt 3 -or ($MajorVersion -eq 3 -and $MinorVersion -lt 9)) {
    Write-Error-Custom "Python 3.9+ required. Found: $PythonVersion"
}
Write-Success "Python: $PythonVersion"

# Check disk space (3GB required)
$DriveLetter = $env:USERPROFILE.Substring(0, 1)
$Drive = Get-PSDrive $DriveLetter
$DiskAvailable = $Drive.Free / 1GB
$DiskRequired = 3

if ($DiskAvailable -lt $DiskRequired) {
    Write-Error-Custom "Insufficient disk space. Required: 3GB, Available: $([Math]::Round($DiskAvailable, 2))GB"
}
Write-Success "Disk space: $([Math]::Round($DiskAvailable, 2))GB available"

# Verify venv module
try {
    & python.exe -m venv --help *>$null
    Write-Success "Python venv module available"
}
catch {
    Write-Error-Custom "Python venv module not available. Re-install Python and ensure venv is included."
}

Write-Host ""

# ============================================================================
# STEP 2: Create Directory Structure
# ============================================================================
Write-Step 2 "Create directory structure (%USERPROFILE%\.pith\)"

$Dirs = @(
    "$PithHome",
    "$PithHome\bin",
    "$PithHome\data",
    "$PithHome\config",
    "$PithHome\logs",
    "$PithHome\cache",
    "$PithHome\backups"
)

foreach ($Dir in $Dirs) {
    if (-not (Test-Path $Dir)) {
        New-Item -ItemType Directory -Path $Dir -Force | Out-Null
    }
}
Write-Success "Created $PithHome with subdirectories"

Write-Host ""

# ============================================================================
# STEP 3: Install Pith Server Files
# ============================================================================
Write-Step 3 "Install Pith server files"

$PithServerPath = "$PithHome\pith-server"
$DownloadSuccess = $false

# Strategy 1: Detect running from distribution directory (most common for beta)
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$DistDir = Split-Path -Parent $ScriptDir

if ((Test-Path "$DistDir\app\api\server.py") -and (Test-Path "$DistDir\pith_mcp.py")) {
    Write-Host "  Detected distribution directory: $DistDir"
    if (-not (Test-Path $PithServerPath)) {
        New-Item -ItemType Directory -Path $PithServerPath -Force | Out-Null
    }
    # Copy app files from distribution to install location
    Copy-Item -Path "$DistDir\app" -Destination "$PithServerPath\app" -Recurse -Force
    Copy-Item -Path "$DistDir\pith_mcp.py" -Destination "$PithServerPath\pith_mcp.py" -Force
    Copy-Item -Path "$DistDir\skill_deployer.py" -Destination "$PithServerPath\skill_deployer.py" -Force
    Copy-Item -Path "$DistDir\requirements.txt" -Destination "$PithServerPath\requirements.txt" -Force
    if (Test-Path "$DistDir\scripts") {
        Copy-Item -Path "$DistDir\scripts" -Destination "$PithServerPath\scripts" -Recurse -Force
    }
    if (Test-Path "$DistDir\migrations") {
        Copy-Item -Path "$DistDir\migrations" -Destination "$PithServerPath\migrations" -Recurse -Force
    }
    Write-Success "Copied server files from distribution"
    $DownloadSuccess = $true
}

# Strategy 2: Local tarball/zip (created by build-release.sh)
if (-not $DownloadSuccess) {
    $LocalPackage = Join-Path $DistDir "pith-server-latest.zip"
    if (Test-Path $LocalPackage) {
        Write-Host "  Found local package: $LocalPackage"
        if (-not (Test-Path $PithServerPath)) {
            New-Item -ItemType Directory -Path $PithServerPath -Force | Out-Null
        }
        Expand-Archive -Path $LocalPackage -DestinationPath $PithServerPath -Force
        Write-Success "Extracted local server package"
        $DownloadSuccess = $true
    }
}

# Strategy 3: Download from hosted URL (future)
if (-not $DownloadSuccess) {
    Write-Host "Attempting download from: $DownloadUrl"
    
    $TempDir = [System.IO.Path]::GetTempPath() + [System.Guid]::NewGuid().ToString()
    New-Item -ItemType Directory -Path $TempDir -Force | Out-Null
    
    try {
        # Download server package
        $ServerUrl = "$DownloadUrl/$PithServerFilename"
        $ServerPath = Join-Path $TempDir $PithServerFilename
        Invoke-WebRequest -Uri $ServerUrl -OutFile $ServerPath -TimeoutSec 30 -ErrorAction SilentlyContinue
        
        # Download checksum
        $ChecksumPath = Join-Path $TempDir $PithChecksumFilename
        $ChecksumUrl = "$ChecksumUrl/$PithChecksumFilename"
        Invoke-WebRequest -Uri $ChecksumUrl -OutFile $ChecksumPath -TimeoutSec 30 -ErrorAction SilentlyContinue
        
        # Verify checksum
        if ((Test-Path $ServerPath) -and (Test-Path $ChecksumPath)) {
            $FileHash = (Get-FileHash -Path $ServerPath -Algorithm SHA256).Hash
            $ChecksumContent = (Get-Content $ChecksumPath | Select-Object -First 1) -split ' '
            $ExpectedHash = $ChecksumContent[0]
            
            if ($FileHash -eq $ExpectedHash) {
                Write-Success "Download successful and checksum verified"
                
                # Extract server
                if (-not (Test-Path $PithServerPath)) {
                    New-Item -ItemType Directory -Path $PithServerPath -Force | Out-Null
                }
                Expand-Archive -Path $ServerPath -DestinationPath $PithServerPath -Force
                $DownloadSuccess = $true
            }
            else {
                Write-Warning "Checksum verification failed, attempting fallback"
            }
        }
        else {
            Write-Warning "Download failed, attempting fallback"
        }
    }
    catch {
        Write-Warning "Download exception: $_"
    }
    finally {
        Remove-Item -Path $TempDir -Recurse -Force -ErrorAction SilentlyContinue
    }
}

if (-not $DownloadSuccess) {
    Write-Error-Custom "Could not locate Pith server files. Run this script from the distribution directory or provide DOWNLOAD_URL."
}

Write-Host ""

# ============================================================================
# STEP 4: Python venv Setup with Health Check
# ============================================================================
Write-Step 4 "Python venv setup with health check [FIX R1, R2, R3]"

$VenvPath = "$PithHome\venv"

# FIX R1: Detect broken existing venv, recreate if needed
if (Test-Path $VenvPath) {
    try {
        $PythonExe = "$VenvPath\Scripts\python.exe"
        & $PythonExe -c "import sys" 2>$null
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "Existing venv is broken, recreating"
            Remove-Item -Path $VenvPath -Recurse -Force
        }
    }
    catch {
        Write-Warning "Existing venv is broken, recreating"
        Remove-Item -Path $VenvPath -Recurse -Force
    }
}

# Create venv
if (-not (Test-Path $VenvPath)) {
    & python.exe -m venv $VenvPath
    Write-Success "Created Python virtual environment"
}
else {
    Write-Success "Using existing virtual environment"
}

# Prepare pip activation script
$PipExe = "$VenvPath\Scripts\pip.exe"
$PythonExe = "$VenvPath\Scripts\python.exe"

# Upgrade pip (use python -m pip to avoid self-replace lock issues)
$ErrorActionPreference = 'Continue'
& $PythonExe -m pip install --quiet --upgrade pip setuptools wheel 2>$null
$ErrorActionPreference = 'Stop'
Write-Success "Updated pip, setuptools, wheel"

# Install core dependencies from requirements.txt
Write-Host "Installing dependencies (this may take a moment)..."
$ReqFile = "$PithHome\pith-server\requirements.txt"
if (-not (Test-Path $ReqFile)) {
    $ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
    $ReqFile = Join-Path (Split-Path -Parent $ScriptDir) "requirements.txt"
}
$ErrorActionPreference = 'Continue'
& $PipExe install --quiet -r $ReqFile 2>$null
$ErrorActionPreference = 'Stop'
Write-Success "Installed core dependencies"

# Platform-aware embedding installation (F16)
$EmbedLog = "$PithHome\logs\embedding_install.log"
New-Item -ItemType Directory -Path "$PithHome\logs" -Force | Out-Null

function Install-Embeddings {
    # Windows - CPU-only torch
    $ErrorActionPreference = 'Continue'
    & $PipExe install --quiet torch --index-url https://download.pytorch.org/whl/cpu 2>"$EmbedLog"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  [!] PyTorch install failed. Using TF-IDF search." -ForegroundColor Yellow
        Write-Host "  Details: $EmbedLog"
        "embeddings=false`nreason=pytorch_install_failed" | Out-File "$PithHome\.install_capabilities"
        $ErrorActionPreference = 'Stop'
        return $false
    }

    & $PipExe install --quiet "sentence-transformers>=3.0.0,<4.0.0" 2>>"$EmbedLog"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  [!] sentence-transformers install failed. Using TF-IDF search." -ForegroundColor Yellow
        "embeddings=false`nreason=st_install_failed" | Out-File "$PithHome\.install_capabilities"
        $ErrorActionPreference = 'Stop'
        return $false
    }

    # Pre-download model
    & $PythonExe -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" 2>$null
    $TorchVer = & $PythonExe -c "import torch; print(torch.__version__)" 2>$null
    Write-Success "Semantic embeddings enabled (all-MiniLM-L6-v2)"
    "embeddings=true`npytorch=$TorchVer`narch=x86_64" | Out-File "$PithHome\.install_capabilities"
    $ErrorActionPreference = 'Stop'
    return $true
}

Write-Host "Installing embeddings (CPU-only PyTorch)..."
$EmbedResult = Install-Embeddings
if (-not $EmbedResult) {
    Write-Host "  Pith will run with TF-IDF search (fully functional, reduced semantic quality)." -ForegroundColor Yellow
}

Write-Host ""

# ============================================================================
# STEP 5: Generate API Key with Secure Permissions
# ============================================================================
Write-Step 5 "Generate API key with secure file permissions [FIX S2]"

$ApiKeyFile = "$PithHome\config\api.key"
if (-not (Test-Path $ApiKeyFile)) {
    # Generate 32-byte random key as hex (64 chars)
    $ApiKey = -join ((1..32) | ForEach-Object { "{0:x2}" -f (Get-Random -Minimum 0 -Maximum 256) })
    
    # Write with restricted permissions
    Set-Content -Path $ApiKeyFile -Value $ApiKey -NoNewline
    
    # FIX S2: Set restrictive ACL (owner read-only)
    $Acl = Get-Acl -Path $ApiKeyFile
    $Acl.SetAccessRuleProtection($true, $false)
    $Acl.Access | ForEach-Object { $Acl.RemoveAccessRule($_) } | Out-Null
    $Rule = New-Object System.Security.AccessControl.FileSystemAccessRule(
        [System.Security.Principal.WindowsIdentity]::GetCurrent().User,
        [System.Security.AccessControl.FileSystemRights]::Read,
        [System.Security.AccessControl.AccessControlType]::Allow
    )
    $Acl.AddAccessRule($Rule)
    Set-Acl -Path $ApiKeyFile -AclObject $Acl
    
    Write-Success "Generated API key: $($ApiKey.Substring(0, 16))... (saved to $ApiKeyFile)"
}
else {
    $ApiKey = (Get-Content -Path $ApiKeyFile -Raw).Trim()
    Write-Success "API key already exists"
}

# Create .env file (F21)
$EnvFile = "$PithServerPath\.env"
if (-not (Test-Path $EnvFile)) {
    @(
        "PITH_API_KEY=$ApiKey",
        "HOST=127.0.0.1",
        "PORT=8000"
    ) | Set-Content -Path $EnvFile
    Write-Success "Created .env file"
}

Write-Host ""

# ============================================================================
# STEP 6: Configure MCP Clients
# ============================================================================
Write-Step 6 "Configure MCP clients using configure_clients.py"

# Try real configure_clients.py first (supports 6 MCP clients)
$ApiKeyContent = (Get-Content -Path $ApiKeyFile -Raw).Trim()
$ConfigureScript = "$PithHome\pith-server\scripts\configure_clients.py"

$ConfigSuccess = $false
if (Test-Path $ConfigureScript) {
    & $PythonExe $ConfigureScript `
        --server-path "$PithHome\pith-server\pith_mcp.py" `
        --python-cmd "$VenvPath\Scripts\python.exe" `
        --api-key $ApiKeyContent `
        --project-dir "$PithHome\pith-server" `
        --platform "windows" `
        --json 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Success "MCP clients configured (6 clients)"
        $ConfigSuccess = $true
    }
}

if (-not $ConfigSuccess) {
    # Fallback: configure Claude Desktop only via standalone script (DEBT-084)
    Write-Host "  Configuring Claude Desktop only..." -ForegroundColor Yellow
    $ClaudeConfig = "$env:APPDATA\Claude\claude_desktop_config.json"
    $ConfigureDesktopScript = "$PithServerPath\scripts\configure_mcp_claude_desktop.py"
    if (-not (Test-Path $ConfigureDesktopScript)) {
        # Try distribution directory
        $ConfigureDesktopScript = Join-Path (Split-Path -Parent $PSScriptRoot) "scripts\configure_mcp_claude_desktop.py"
    }
    & $PythonExe $ConfigureDesktopScript `
        --config-path $ClaudeConfig `
        --python-cmd "$VenvPath\Scripts\python.exe" `
        --mcp-script "$PithHome\pith-server\pith_mcp.py" `
        --api-key $ApiKeyContent 2>$null
    Write-Success "Claude Desktop MCP configured"
}

Write-Host ""

# ============================================================================
# STEP 7: Auto-start Setup
# ============================================================================
Write-Step 7 "Auto-start setup (Task Scheduler) and backup scheduler"

# Create pith CLI scripts first
$PithBatPath = "$PithHome\bin\pith.cmd"
Set-Content -Path $PithBatPath -Value "@echo off`r`nsetlocal`r`nset PITH_HOME=%USERPROFILE%\.pith`r`nset VENV_PATH=%PITH_HOME%\venv`r`ncall `"%VENV_PATH%\Scripts\activate.bat`"`r`npowershell -NoProfile -ExecutionPolicy Bypass -File `"%PITH_HOME%\bin\pith.ps1`" %*"

# Create PowerShell CLI wrapper
# Load CLI wrapper from template file
$CliTemplatePath = Join-Path $PSScriptRoot "templates\pith_cli.ps1"
if (-not (Test-Path $CliTemplatePath)) {
    Write-Error-Custom "CLI template not found at $CliTemplatePath"
}
$PithPsContent = Get-Content -Path $CliTemplatePath -Raw
$PithPsContent = $PithPsContent -replace '__PITH_HOME__', $PithHome
$PithPsContent = $PithPsContent -replace '__PITH_VERSION__', $PithVersion


$PithPsPath = "$PithHome\bin\pith.ps1"
Set-Content -Path $PithPsPath -Value $PithPsContent

Write-Success "Created pith CLI wrapper"

# Create Task Scheduler entry for auto-start
Write-Host "Setting up Task Scheduler for auto-start..."
$TaskName = "Pith-Brain-Server"
$TaskPath = "\Pith\"

# Create task action
$Action = New-ScheduledTaskAction -Execute "$PithHome\bin\pith.cmd" -Argument "start"

# Create task trigger (at system startup)
$Trigger = New-ScheduledTaskTrigger -AtStartup

# Create task principal (run as user)
$Principal = New-ScheduledTaskPrincipal -UserId (whoami) -RunLevel Limited

# Register task
try {
    Register-ScheduledTask -TaskName $TaskName `
        -Action $Action `
        -Trigger $Trigger `
        -Principal $Principal `
        -Force -ErrorAction SilentlyContinue | Out-Null
    Write-Success "Registered Pith auto-start task"
}
catch {
    Write-Warning "Could not register Task Scheduler task"
}

# Copy safe_backup.ps1 if present in distribution
$SafeBackupSrc = "$PithServerPath\scripts\backup\safe_backup.ps1"
$SetupScheduleSrc = "$PithServerPath\scripts\backup\setup_schedule.ps1"
if (Test-Path $SafeBackupSrc) {
    Write-Success "WAL-safe backup script available"
} else {
    Write-Host "  [!] safe_backup.ps1 not found -- backup command will be unavailable" -ForegroundColor Yellow
}

# Schedule backup task (every 3 hours)
if (Test-Path $SafeBackupSrc) {
    $BackupAction = New-ScheduledTaskAction -Execute "powershell.exe" `
        -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$SafeBackupSrc`""
    $BackupTrigger = New-ScheduledTaskTrigger -Once -At (Get-Date).Date `
        -RepetitionInterval (New-TimeSpan -Hours 3)
    $BackupPrincipal = New-ScheduledTaskPrincipal -UserId (whoami)

    try {
        Register-ScheduledTask -TaskName "Pith-Backup-3h" `
            -Action $BackupAction `
            -Trigger $BackupTrigger `
            -Principal $BackupPrincipal `
            -Force -ErrorAction SilentlyContinue | Out-Null
        Write-Success "Scheduled backups every 3 hours (WAL-safe)"
    }
    catch {
        Write-Warning "Could not register backup task"
    }
}

# Remove legacy daily backup task if exists
Unregister-ScheduledTask -TaskName "Pith-Daily-Backup" -Confirm:$false -ErrorAction SilentlyContinue

Write-Host ""

# ============================================================================
# STEP 8: Health Check
# ============================================================================
Write-Step 8 "Health check (30s timeout)"

Write-Host "Performing health check..."

# Pre-check: if something is already running on port 8000 and healthy, skip
# Handles: (1) dev env where Docker Pith is on 8000, (2) re-running installer
$ExistingHealthy = $false
try {
    $existResp = Invoke-WebRequest -Uri "http://127.0.0.1:8000/health" -TimeoutSec 2 -ErrorAction SilentlyContinue
    if ($existResp.StatusCode -eq 200) { $ExistingHealthy = $true }
} catch {}

if ($ExistingHealthy) {
    Write-Success "Pith server already running on port 8000 - health check passed"
} else {

$HealthCheckTimeout = $false

try {
    $proc = Start-Process -FilePath $PythonExe `
        -ArgumentList "-m uvicorn app.api.server:app --host 127.0.0.1 --port 8000" `
        -WorkingDirectory $PithServerPath `
        -WindowStyle Hidden `
        -PassThru
    
    $Stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
    while ($Stopwatch.Elapsed.TotalSeconds -lt 30) {
        try {
            $Response = Invoke-WebRequest -Uri "http://127.0.0.1:8000/health" `
                -TimeoutSec 2 -ErrorAction SilentlyContinue
            if ($Response.StatusCode -eq 200) {
                Write-Success "Health check passed"
                Stop-Process -Id $proc.Id -ErrorAction SilentlyContinue
                break
            }
        }
        catch {
            Start-Sleep -Milliseconds 500
        }
    }
    
    Stop-Process -Id $proc.Id -ErrorAction SilentlyContinue
}
catch {
    Write-Warning "Health check did not complete (may complete on first run)"
}

}  # end of port pre-check else block

Write-Host ""

# ============================================================================
# STEP 8b: Auto-configure PATH [FIX A1 - Windows equivalent]
# ============================================================================
$PathAdded = $false
$PithBinDir = "$PithHome\bin"

# Add to current session
if ($env:PATH -notlike "*$PithBinDir*") {
    $env:PATH = "$PithBinDir;$env:PATH"
}

# Add to PowerShell profile (persistent)
$ProfilePath = $PROFILE.CurrentUserAllHosts
if ($ProfilePath) {
    # Create profile if it doesn't exist
    if (-not (Test-Path $ProfilePath)) {
        New-Item -Path $ProfilePath -ItemType File -Force | Out-Null
    }
    $ProfileContent = Get-Content $ProfilePath -Raw -ErrorAction SilentlyContinue
    if ($ProfileContent -notlike "*$PithBinDir*") {
        Add-Content -Path $ProfilePath -Value "`n# Pith Brain CLI"
        Add-Content -Path $ProfilePath -Value "`$env:PATH = `"$PithBinDir;`$env:PATH`""
        Write-Host "  [OK] Added Pith to PATH in $ProfilePath" -ForegroundColor Green
        $PathAdded = $true
    } else {
        Write-Host "  [OK] PATH already configured in $ProfilePath" -ForegroundColor Green
        $PathAdded = $true
    }
}

# Also add to User environment variable (persists across all terminals)
try {
    $UserPath = [Environment]::GetEnvironmentVariable("PATH", "User")
    if ($UserPath -notlike "*$PithBinDir*") {
        [Environment]::SetEnvironmentVariable("PATH", "$PithBinDir;$UserPath", "User")
        Write-Host "  [OK] Added Pith to User PATH environment variable" -ForegroundColor Green
        $PathAdded = $true
    }
} catch {
    Write-Warning "Could not update User PATH environment variable"
}

# ============================================================================
# Final Success Message
# ============================================================================
Write-Banner

Write-Host "[OK] Installation Complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Pith Brain is installed at: " -NoNewline
Write-Host "$PithHome" -ForegroundColor Cyan
Write-Host ""

if ($PathAdded) {
    Write-Host "Quick Start:" -ForegroundColor Cyan
    Write-Host "  PATH has been auto-configured. Open a new terminal to use 'pith' command."
    Write-Host ""
} else {
    Write-Host "Quick Start:" -ForegroundColor Cyan
    Write-Host "  1. Add to PATH: Add '$PithHome\bin' to your system PATH"
    Write-Host "     (System Settings > Environment Variables)"
    Write-Host ""
}

Write-Host "Available Commands:" -ForegroundColor Cyan
Write-Host "  pith start       Start the Pith Brain server"
Write-Host "  pith stop        Stop the server"
Write-Host "  pith restart     Restart the server"
Write-Host "  pith status      Check server status"
Write-Host "  pith logs        Tail server logs"
Write-Host "  pith backup      Create WAL-safe backup"
Write-Host "  pith restore     Restore from backup"
Write-Host "  pith update      Update deps + embeddings"
Write-Host "  pith version     Show version + capabilities"
Write-Host "  pith maintenance run   Run maintenance cycle"
Write-Host "  pith maintenance status Show maintenance task status"
Write-Host "  pith uninstall   Remove Pith completely"
Write-Host ""

Write-Host "Next Steps:" -ForegroundColor Cyan
if ($PathAdded) {
    Write-Host "  1. Open a new terminal (PATH is already configured)"
    Write-Host "  2. Start server: pith start"
    Write-Host "  3. Check status: pith version"
} else {
    Write-Host "  1. Add to PATH:  `$env:PATH += ';$PithHome\bin'"
    Write-Host "  2. Start server: & '$PithHome\bin\pith.cmd' start"
    Write-Host "  3. Check status: pith version"
}
Write-Host ""

exit 0
