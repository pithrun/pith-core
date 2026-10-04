# Pith Installer v1.0.9 (Windows PowerShell)
# Windows equivalent installer

#Requires -Version 5.0

param(
    [switch]$Force = $false,
    [string]$PithVersion = "1.0.9"
)

# Strict error handling
$ErrorActionPreference = 'Stop'
$VerbosePreference = 'SilentlyContinue'

# Configuration
$DownloadUrl = if ($env:DOWNLOAD_URL) { $env:DOWNLOAD_URL } else { "https://github.com/pithrun/pith-core/releases/latest/download" }
$ChecksumUrl = if ($env:CHECKSUM_URL) { $env:CHECKSUM_URL } else { "https://github.com/pithrun/pith-core/releases/latest/download" }
$DefaultPithHome = "$env:USERPROFILE\.pith"
$PithHomeOverridden = [bool]$env:PITH_HOME
$PithHome = if ($PithHomeOverridden) { $env:PITH_HOME } else { $DefaultPithHome }
$PithDataDirOverridden = [bool]$env:PITH_DATA_DIR
$PithProfile = if ($env:PITH_PROFILE) { $env:PITH_PROFILE } else { "default" }
$DefaultPithDataDir = Join-Path (Join-Path $env:USERPROFILE "pith-data") $PithProfile
$PithDataDir = if ($PithDataDirOverridden) { $env:PITH_DATA_DIR } else { $DefaultPithDataDir }
$PithPort = if ($env:PITH_PORT) { $env:PITH_PORT } else { "8000" }
$PithRuntimeId = if ($env:PITH_RUNTIME_ID) { $env:PITH_RUNTIME_ID } else { "cpython-3.12.13+20260504-x86_64-pc-windows-msvc-install_only_stripped" }
$PithRuntimeVersion = if ($env:PITH_RUNTIME_VERSION) { $env:PITH_RUNTIME_VERSION } else { "3.12.13" }
$PithRuntimePlatform = "windows"
$PithRuntimeArch = if ($env:PITH_RUNTIME_ARCH) { $env:PITH_RUNTIME_ARCH } else { "x86_64" }
$PithRuntimeSource = if ($env:PITH_RUNTIME_SOURCE) { $env:PITH_RUNTIME_SOURCE } else { "astral-sh/python-build-standalone" }
$PithRuntimeLicense = if ($env:PITH_RUNTIME_LICENSE) { $env:PITH_RUNTIME_LICENSE } else { "CPython distribution from astral-sh/python-build-standalone; preserve upstream runtime notices" }
$PithRuntimeUrl = if ($env:PITH_RUNTIME_URL) { $env:PITH_RUNTIME_URL } else { "https://github.com/astral-sh/python-build-standalone/releases/download/20260504/cpython-3.12.13%2B20260504-x86_64-pc-windows-msvc-install_only_stripped.tar.gz" }
$PithRuntimeSha256 = if ($env:PITH_RUNTIME_SHA256) { $env:PITH_RUNTIME_SHA256 } else { "35804c0ca7fb01987d4754d66e7621d19b0a8fb39b6812e112fb41ec243113e7" }
$PithRuntimeSizeBytes = if ($env:PITH_RUNTIME_SIZE_BYTES) { [int64]$env:PITH_RUNTIME_SIZE_BYTES } else { [int64]21926840 }
$PithRuntimeRoot = "$PithHome\runtime\python"
$PithRuntimeMetaPath = "$PithHome\config\python-runtime.json"
$StepCount = 9
$CurrentStep = 0

# File names
$PithServerFilename = "pith-server-latest.zip"
$PithChecksumFilename = "pith-server-latest.zip.sha256"

# Color functions
function Write-Banner {
    Clear-Host
    Write-Host ""
    Write-Host "+========================================+" -ForegroundColor Cyan
    Write-Host "|   Pith Installer v$PithVersion              |" -ForegroundColor Cyan
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

function Get-PithSha256 {
    param([Parameter(Mandatory = $true)][string]$Path)

    $Stream = [System.IO.File]::OpenRead($Path)
    try {
        $Sha256 = [System.Security.Cryptography.SHA256]::Create()
        try {
            $HashBytes = $Sha256.ComputeHash($Stream)
        }
        finally {
            $Sha256.Dispose()
        }
    }
    finally {
        $Stream.Dispose()
    }

    return ([System.BitConverter]::ToString($HashBytes)).Replace("-", "").ToLowerInvariant()
}

function Get-PithUninstallMarkerPaths {
    return @(
        "$PithHome\config\uninstalling",
        "$PithHome.uninstalling"
    )
}

function Test-PithPendingUninstall {
    foreach ($MarkerPath in (Get-PithUninstallMarkerPaths)) {
        if (Test-Path -LiteralPath $MarkerPath) {
            return $true
        }
    }
    return $false
}

function Stop-PithMcpBridgeProcessesForCleanup {
    $EscapedPithHome = [regex]::Escape($PithHome)
    try {
        $Processes = Get-CimInstance Win32_Process -ErrorAction Stop |
            Where-Object {
                ($_.CommandLine -match "pith_mcp\.py") -and
                ($_.CommandLine -match $EscapedPithHome)
            }
    }
    catch {
        return
    }

    foreach ($Process in $Processes) {
        try {
            Stop-Process -Id $Process.ProcessId -Force -ErrorAction Stop
        }
        catch {
            Write-Warning "Could not stop Pith MCP bridge process PID $($Process.ProcessId): $($_.Exception.Message)"
        }
    }
}

function Get-PithStablePathHash {
    param([string]$Value)

    $Sha = [System.Security.Cryptography.SHA256]::Create()
    try {
        $Bytes = [System.Text.Encoding]::UTF8.GetBytes($Value)
        $HashBytes = $Sha.ComputeHash($Bytes)
        return ([System.BitConverter]::ToString($HashBytes)).Replace("-", "").Substring(0, 12).ToLowerInvariant()
    }
    finally {
        $Sha.Dispose()
    }
}

function Get-PithShortVenvPath {
    param([string]$PithHome)

    $BaseDir = if ($env:LOCALAPPDATA) {
        Join-Path $env:LOCALAPPDATA "Pith\venvs"
    }
    else {
        Join-Path (Split-Path -Parent $PithHome) "AppData\Local\Pith\venvs"
    }
    $PathHash = Get-PithStablePathHash -Value $PithHome
    return (Join-Path $BaseDir $PathHash)
}

function Get-PithOwnedRuntimePrefixes {
    $CandidateRoots = @(
        $PithHome,
        (Get-PithShortVenvPath -PithHome $PithHome)
    )
    return @($CandidateRoots |
        Where-Object { $_ } |
        ForEach-Object { ([System.IO.Path]::GetFullPath($_)).Replace('/', '\').TrimEnd('\') + '\' } |
        Select-Object -Unique)
}

function Get-PithInstallProcessesForCleanup {
    $NormalizedOwnedPrefixes = @(Get-PithOwnedRuntimePrefixes)
    try {
        return @(Get-CimInstance Win32_Process -ErrorAction Stop |
            Where-Object {
                $NormalizedCommandLine = if ($_.CommandLine) { ([string]$_.CommandLine).Replace('/', '\') } else { "" }
                $NormalizedExecutablePath = if ($_.ExecutablePath) { ([string]$_.ExecutablePath).Replace('/', '\') } else { "" }
                $OwnedProcess = $false
                foreach ($OwnedPrefix in $NormalizedOwnedPrefixes) {
                    if (
                        $NormalizedCommandLine.IndexOf($OwnedPrefix, [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -or
                        $NormalizedExecutablePath.StartsWith($OwnedPrefix, [System.StringComparison]::OrdinalIgnoreCase)
                    ) {
                        $OwnedProcess = $true
                        break
                    }
                }
                $_.ProcessId -ne $PID -and $OwnedProcess
            })
    }
    catch {
        Write-Error-Custom "Could not enumerate Pith runtime processes before install: $($_.Exception.Message)"
    }
}

function Stop-PithInstallProcessesForCleanup {
    $Processes = @(Get-PithInstallProcessesForCleanup)
    foreach ($Process in $Processes) {
        try {
            Stop-Process -Id $Process.ProcessId -Force -ErrorAction Stop
        }
        catch {
            Write-Warning "Could not stop partial-install process PID $($Process.ProcessId): $($_.Exception.Message)"
        }
    }

    $Deadline = (Get-Date).AddSeconds(15)
    do {
        $Survivors = @(Get-PithInstallProcessesForCleanup)
        if ($Survivors.Count -eq 0) {
            return
        }
        Start-Sleep -Milliseconds 250
    } while ((Get-Date) -lt $Deadline)

    $SurvivorIds = @($Survivors | ForEach-Object { $_.ProcessId }) -join ", "
    Write-Error-Custom "Pith runtime processes did not stop before install. Surviving PIDs: $SurvivorIds"
}

function Get-PithServerTreeInventory {
    param([Parameter(Mandatory=$true)][string]$PithHome, [Parameter(Mandatory=$true)][string]$ServerPath)
    if (-not [IO.Path]::IsPathRooted($PithHome) -or -not [IO.Path]::IsPathRooted($ServerPath)) { throw 'Pith install paths must be absolute' }
    $HomeFull = [IO.Path]::GetFullPath($PithHome).TrimEnd('\')
    $ServerFull = [IO.Path]::GetFullPath($ServerPath).TrimEnd('\')
    if ($PithHome -match '^[A-Za-z]:(?![\\/])' -or [IO.Path]::GetPathRoot($PithHome).Length -le 1 -or $HomeFull -eq [IO.Path]::GetPathRoot($HomeFull).TrimEnd('\')) { throw 'Pith home must not be relative to a drive or be a volume root' }
    $Expected = [IO.Path]::GetFullPath((Join-Path $HomeFull 'pith-server')).TrimEnd('\')
    if ($ServerFull -ine $Expected) { throw 'Pith server path is not the managed child' }
    $Ancestor = $ServerFull
    while ($Ancestor) {
        if (Test-Path -LiteralPath $Ancestor) {
            $Item = Get-Item -LiteralPath $Ancestor -Force -ErrorAction Stop
            if (-not $Item.PSIsContainer -or ($Item.Attributes -band [IO.FileAttributes]::ReparsePoint)) { throw 'Unsafe Pith server ancestor' }
        }
        $Ancestor = [IO.Path]::GetDirectoryName($Ancestor)
    }
    $Inventory = New-Object System.Collections.Generic.List[object]
    if (-not (Test-Path -LiteralPath $ServerFull)) { return $Inventory.ToArray() }
    $Pending = New-Object System.Collections.Generic.Stack[string]
    $Pending.Push($ServerFull)
    $MaxItems = 100000
    while ($Pending.Count -gt 0) {
        $Directory = $Pending.Pop()
        Get-ChildItem -LiteralPath $Directory -Force -ErrorAction Stop | ForEach-Object {
            $Item = $_
            if ($Item.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Reparse point in Pith server tree' }
            $Inventory.Add($Item)
            if ($Inventory.Count -gt $MaxItems) { throw 'Pith server tree exceeds safety inventory limit' }
            if ($Item.PSIsContainer) { $Pending.Push($Item.FullName) }
        }
    }
    return $Inventory.ToArray()
}

function Test-PithOwnedApplicationBytecode {
    param([Parameter(Mandatory=$true)][string]$RelativePath)
    $Parts = $RelativePath.Split('\')
    foreach ($Part in $Parts) {
        if (-not $Part -or $Part.IndexOfAny([IO.Path]::GetInvalidFileNameChars()) -ge 0) { return $false }
    }
    if ([IO.Path]::IsPathRooted($RelativePath) -or $Parts -contains '..' -or $Parts -contains '.' -or $RelativePath.Contains('/')) { return $false }
    if ([IO.Path]::GetExtension($RelativePath) -ine '.pyc') { return $false }
    if ($Parts.Count -gt 1 -and $Parts[0] -in @('app', 'pith_client', 'scripts', 'migrations', 'integrations')) { return $true }
    if ($Parts.Count -eq 1 -and $Parts[0] -in @('pith_mcp.pyc', 'skill_deployer.pyc')) { return $true }
    return ($Parts.Count -eq 2 -and $Parts[0] -ieq '__pycache__' -and $Parts[1] -match '^(pith_mcp|skill_deployer)\.[^.]+(?:\.opt-[0-9]+)?\.pyc$')
}

function Clear-PithApplicationBytecode {
    param([Parameter(Mandatory=$true)][string]$PithHome, [Parameter(Mandatory=$true)][string]$ServerPath)
    $Inventory = @(Get-PithServerTreeInventory -PithHome $PithHome -ServerPath $ServerPath)
    $ServerFull = [IO.Path]::GetFullPath($ServerPath).TrimEnd('\')
    $Removed = 0
    foreach ($Item in $Inventory) {
        if ($Item.PSIsContainer -or $Item.Extension -ine '.pyc') { continue }
        $Relative = $Item.FullName.Substring($ServerFull.Length + 1)
        if (-not (Test-PithOwnedApplicationBytecode -RelativePath $Relative)) { continue }
        # Recheck all ancestors immediately before file-only deletion.
        $Ancestor = $Item.FullName
        while ($Ancestor -and $Ancestor.Length -ge $ServerFull.Length) {
            $Current = Get-Item -LiteralPath $Ancestor -Force -ErrorAction Stop
            if ($Current.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Reparse point before Pith bytecode deletion' }
            $Ancestor = [IO.Path]::GetDirectoryName($Ancestor)
        }
        Remove-Item -LiteralPath $Item.FullName -Force -ErrorAction Stop
        if (Test-Path -LiteralPath $Item.FullName) { throw 'Pith application bytecode remains after deletion' }
        $Removed++
    }
    foreach ($Item in @(Get-PithServerTreeInventory -PithHome $PithHome -ServerPath $ServerPath)) {
        if (-not $Item.PSIsContainer -and (Test-PithOwnedApplicationBytecode -RelativePath $Item.FullName.Substring($ServerFull.Length + 1))) { throw 'Residual Pith application bytecode after cleanup' }
    }
    return $Removed
}

function Wait-PithPendingUninstallCleanup {
    if (-not (Test-PithPendingUninstall)) {
        return
    }

    Write-Warning "Previous Pith uninstall cleanup is still finishing; waiting before reinstall..."
    Stop-PithMcpBridgeProcessesForCleanup
    $Deadline = (Get-Date).AddSeconds(300)
    while ((Get-Date) -lt $Deadline) {
        if ((-not (Test-PithPendingUninstall)) -and (-not (Test-Path -LiteralPath $PithHome))) {
            Start-Sleep -Milliseconds 1000
            if ((-not (Test-PithPendingUninstall)) -and (-not (Test-Path -LiteralPath $PithHome))) {
                return
            }
        }
        Start-Sleep -Milliseconds 500
    }

    # Never reuse or move this path while the deferred remover still owns it.
    # Doing so allows the old remover to delete a newly created installation.
    Write-Error-Custom "Previous uninstall cleanup is still holding $PithHome after 300 seconds. Wait for cleanup to finish, then rerun the installer."
}

Wait-PithPendingUninstallCleanup

function Repair-PithPartialInstallState {
    $InstalledWrapper = Join-Path $PithHome "bin\pith.cmd"
    if ((-not (Test-Path -LiteralPath $PithHome)) -or (Test-Path -LiteralPath $InstalledWrapper -PathType Leaf)) {
        return
    }

    Write-Warning "Incomplete Pith install detected at $PithHome; retiring it before clean reinstall"
    Stop-PithInstallProcessesForCleanup
    Start-Sleep -Seconds 2
    $RetiredPithHome = "$PithHome.partial-recovery-$((Get-Date).ToString('yyyyMMddHHmmss'))"
    try {
        Move-Item -LiteralPath $PithHome -Destination $RetiredPithHome -Force -ErrorAction Stop
        Write-Warning "Retired incomplete install directory to $RetiredPithHome"
    }
    catch {
        Write-Error-Custom "Incomplete install at $PithHome could not be retired: $($_.Exception.Message)"
    }
}

Repair-PithPartialInstallState

function Normalize-PithScheduledTaskPath {
    param([string]$Path)

    if (-not $Path) {
        return ""
    }
    return ([Environment]::ExpandEnvironmentVariables($Path.Trim().Trim('"'))).TrimEnd('\')
}

function Test-PithScheduledTaskUserMatchesCurrentUser {
    param(
        [string]$TaskUser,
        [string]$CurrentUser
    )

    if (-not $TaskUser) {
        return $false
    }

    $NormalizedTaskUser = $TaskUser.Trim().ToLowerInvariant()
    $Aliases = New-Object System.Collections.Generic.List[string]
    foreach ($Alias in @($CurrentUser, $env:USERNAME, "$env:USERDOMAIN\$env:USERNAME", "$env:COMPUTERNAME\$env:USERNAME")) {
        if (-not [string]::IsNullOrWhiteSpace([string]$Alias)) {
            $Aliases.Add(([string]$Alias).Trim().ToLowerInvariant()) | Out-Null
        }
    }

    foreach ($Alias in $Aliases) {
        if ($NormalizedTaskUser -eq $Alias) {
            return $true
        }
    }
    return $false
}

function Test-PithScheduledTaskIdentityMatchesInstall {
    param(
        [object]$Task,
        [string]$ExpectedCommand,
        [string]$ExpectedArguments,
        [string]$CurrentUser
    )

    if (-not $Task) {
        return $false
    }

    $TaskAction = @($Task.Actions) | Select-Object -First 1
    if (-not $TaskAction) {
        return $false
    }

    $ActualCommand = Normalize-PithScheduledTaskPath -Path ([string]$TaskAction.Execute)
    $ExpectedCommandNormalized = Normalize-PithScheduledTaskPath -Path $ExpectedCommand
    $ActualArguments = ([string]$TaskAction.Arguments).Trim()
    $ExpectedArgumentsNormalized = ([string]$ExpectedArguments).Trim()
    $ActualUser = [string]$Task.Principal.UserId

    return (
        ($ActualCommand -ieq $ExpectedCommandNormalized) -and
        ($ActualArguments -ieq $ExpectedArgumentsNormalized) -and
        (Test-PithScheduledTaskUserMatchesCurrentUser -TaskUser $ActualUser -CurrentUser $CurrentUser)
    )
}

function Test-PithScheduledTaskMatchesInstall {
    param(
        [object]$Task,
        [string]$ExpectedCommand,
        [string]$ExpectedArguments,
        [string]$CurrentUser
    )

    if (-not (Test-PithScheduledTaskIdentityMatchesInstall -Task $Task -ExpectedCommand $ExpectedCommand -ExpectedArguments $ExpectedArguments -CurrentUser $CurrentUser)) {
        return $false
    }

    $Triggers = @($Task.Triggers)
    $LogonTrigger = if ($Triggers.Count -eq 1) { $Triggers[0] } else { $null }
    $LogonTriggerMatches = (
        ($null -ne $LogonTrigger) -and
        ([string]$LogonTrigger.CimClass.CimClassName -eq "MSFT_TaskLogonTrigger") -and
        ($LogonTrigger.Enabled -eq $true) -and
        (Test-PithScheduledTaskUserMatchesCurrentUser -TaskUser ([string]$LogonTrigger.UserId) -CurrentUser $CurrentUser)
    )
    $ExecutionTimeLimit = [string]$Task.Settings.ExecutionTimeLimit
    $SettingsMatch = (
        (-not [bool]$Task.Settings.DisallowStartIfOnBatteries) -and
        (-not [bool]$Task.Settings.StopIfGoingOnBatteries) -and
        ($ExecutionTimeLimit -in @("PT0S", "00:00:00"))
    )

    return ($LogonTriggerMatches -and $SettingsMatch)
}

function Register-PithAutoStartTask {
    param(
        [string]$TaskName,
        [string]$ExpectedCommand,
        [string]$ExpectedArguments,
        [string]$CurrentUser
    )

    try {
        $ExistingTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
        if ($ExistingTask -and (-not (Test-PithScheduledTaskMatchesInstall -Task $ExistingTask -ExpectedCommand $ExpectedCommand -ExpectedArguments $ExpectedArguments -CurrentUser $CurrentUser))) {
            Write-Warning "Replacing stale Pith auto-start task '$TaskName' with current user/profile task"
            Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction Stop
        }

        $Action = New-ScheduledTaskAction -Execute $ExpectedCommand -Argument $ExpectedArguments
        $Trigger = New-ScheduledTaskTrigger -AtLogOn -User $CurrentUser
        $Principal = New-ScheduledTaskPrincipal -UserId $CurrentUser -LogonType Interactive -RunLevel Limited
        $Settings = New-ScheduledTaskSettingsSet `
            -AllowStartIfOnBatteries `
            -DontStopIfGoingOnBatteries `
            -ExecutionTimeLimit ([TimeSpan]::Zero)

        Register-ScheduledTask -TaskName $TaskName `
            -Action $Action `
            -Trigger $Trigger `
            -Principal $Principal `
            -Settings $Settings `
            -Force -ErrorAction Stop | Out-Null
        $RegisteredTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
        if (-not $RegisteredTask) {
            Write-Warning "Could not verify Pith auto-start task '$TaskName' after registration"
            return $false
        }
        if (-not (Test-PithScheduledTaskMatchesInstall -Task $RegisteredTask -ExpectedCommand $ExpectedCommand -ExpectedArguments $ExpectedArguments -CurrentUser $CurrentUser)) {
            Write-Warning "Pith auto-start task '$TaskName' registered but does not match current user/profile"
            return $false
        }
        Write-Success "Registered Pith auto-start task '$TaskName'"
        return $true
    }
    catch {
        $ExistingTaskAfterError = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
        if (
            $ExistingTaskAfterError -and
            (Test-PithScheduledTaskMatchesInstall -Task $ExistingTaskAfterError -ExpectedCommand $ExpectedCommand -ExpectedArguments $ExpectedArguments -CurrentUser $CurrentUser)
        ) {
            Write-Warning "Could not refresh Pith auto-start task '$TaskName', but its existing identity, logon trigger, and durability settings match this install: $($_.Exception.Message)"
            return $true
        }
        Write-Warning "Could not register Pith auto-start task '$TaskName': $($_.Exception.Message)"
        return $false
    }
}

function Get-PithEnvFileValue {
    param([string]$EnvFile, [string]$Name)

    if (-not (Test-Path $EnvFile)) {
        return ""
    }

    $Pattern = "^$([regex]::Escape($Name))="
    $Match = Get-Content -Path $EnvFile |
        Where-Object { $_ -match $Pattern } |
        Select-Object -First 1
    if (-not $Match) {
        return ""
    }
    return (($Match -split "=", 2)[1]).Trim()
}

function Get-PithVenvPath {
    param([string]$PithHome)

    $DefaultVenvPath = "$PithHome\venv"
    $ShortVenvPath = Get-PithShortVenvPath -PithHome $PithHome
    $ConfiguredVenvPathFile = "$PithHome\config\venv.path"
    if (Test-Path -LiteralPath $ConfiguredVenvPathFile -PathType Leaf) {
        try {
            $ConfiguredVenvPath = (Get-Content -LiteralPath $ConfiguredVenvPathFile -Raw -ErrorAction Stop).Trim()
            $NormalizedConfiguredVenvPath = [System.IO.Path]::GetFullPath($ConfiguredVenvPath).TrimEnd('\')
            $NormalizedShortVenvPath = [System.IO.Path]::GetFullPath($ShortVenvPath).TrimEnd('\')
            if ($NormalizedConfiguredVenvPath.Equals($NormalizedShortVenvPath, [System.StringComparison]::OrdinalIgnoreCase)) {
                return $ShortVenvPath
            }
        }
        catch {
            Write-Warning "Ignoring unreadable or invalid configured venv path; installer will repair it."
        }
    }

    $TorchPathStressSuffix = "Lib\site-packages\torch-2.13.0+cpu.dist-info\licenses\third_party\kineto\libkineto\third_party\dynolog\third_party\prometheus-cpp\3rdparty\civetweb\src\third_party\duktape-1.5.2"
    $ProjectedTorchPath = Join-Path $DefaultVenvPath $TorchPathStressSuffix
    if ($ProjectedTorchPath.Length -lt 245) {
        return $DefaultVenvPath
    }
    return $ShortVenvPath
}

function Test-PithPathEntry {
    param(
        [string]$PathValue,
        [string]$Entry
    )

    if (-not $PathValue) {
        return $false
    }

    $NormalizedEntry = $Entry.TrimEnd('\')
    foreach ($Part in ($PathValue -split ';')) {
        if ($Part.Trim().TrimEnd('\') -ieq $NormalizedEntry) {
            return $true
        }
    }
    return $false
}

function Normalize-PithPathEntry {
    param([string]$Entry)

    if (-not $Entry) {
        return ""
    }

    return ([Environment]::ExpandEnvironmentVariables($Entry.Trim().Trim('"'))).TrimEnd('\')
}

function Test-PithOwnedPathEntry {
    param(
        [string]$Entry,
        [string]$PithBinDir
    )

    $NormalizedEntry = Normalize-PithPathEntry -Entry $Entry
    if (-not $NormalizedEntry) {
        return $false
    }

    $NormalizedCurrent = Normalize-PithPathEntry -Entry $PithBinDir
    if ($NormalizedCurrent -and ($NormalizedEntry -ieq $NormalizedCurrent)) {
        return $true
    }

    if ($NormalizedEntry -match '(?i)\\PithProofs\\.*\\\.pith\\bin$') {
        return $true
    }

    if ($NormalizedEntry -match '(?i)\\\.pith\\bin$') {
        $CandidateCmd = Join-Path $NormalizedEntry "pith.cmd"
        if (Test-Path -LiteralPath $CandidateCmd) {
            return $true
        }
    }

    return $false
}

function Update-PithPathValue {
    param(
        [string]$PathValue,
        [string]$PithBinDir,
        [bool]$PrependPithBin = $false
    )

    $Remaining = New-Object System.Collections.Generic.List[string]
    $Seen = @{}

    if ($PathValue) {
        foreach ($Part in ($PathValue -split ';')) {
            $Trimmed = $Part.Trim()
            $Normalized = Normalize-PithPathEntry -Entry $Trimmed
            if (-not $Normalized) {
                continue
            }
            if (Test-PithOwnedPathEntry -Entry $Normalized -PithBinDir $PithBinDir) {
                continue
            }
            $SeenKey = $Normalized.ToLowerInvariant()
            if (-not $Seen.ContainsKey($SeenKey)) {
                $Seen[$SeenKey] = $true
                $Remaining.Add($Normalized)
            }
        }
    }

    $NormalizedCurrent = Normalize-PithPathEntry -Entry $PithBinDir
    if ($PrependPithBin -and $NormalizedCurrent) {
        $Remaining.Insert(0, $NormalizedCurrent)
    }

    return ($Remaining -join ';')
}

function Remove-PithProfilePathLines {
    param([string]$PithBinDir)

    $ChangedAny = $false
    $ProfilePaths = @(
        $PROFILE.CurrentUserAllHosts,
        $PROFILE.CurrentUserCurrentHost
    ) | Where-Object { $_ } | Select-Object -Unique

    foreach ($ProfilePath in $ProfilePaths) {
        if (-not (Test-Path $ProfilePath)) {
            continue
        }

        $Lines = @(Get-Content -Path $ProfilePath -ErrorAction SilentlyContinue)
        $Filtered = New-Object System.Collections.Generic.List[string]
        $Changed = $false
        foreach ($Line in $Lines) {
            if (
                $Line -match '^\s*#\s*Pith (Brain )?CLI\s*$' -or
                $Line -like "*$PithBinDir*" -or
                $Line -match '(?i)PithProofs.*\.pith\\bin'
            ) {
                $Changed = $true
                continue
            }
            $Filtered.Add($Line)
        }

        $Remaining = ($Filtered -join "`r`n").Trim()
        if ($Remaining) {
            if ($Changed) {
                Set-Content -Path $ProfilePath -Value $Remaining
                $ChangedAny = $true
            }
        }
        else {
            Remove-Item -LiteralPath $ProfilePath -Force -ErrorAction SilentlyContinue
            $ChangedAny = $true
        }
    }

    return $ChangedAny
}

function Normalize-SurfaceList {
    param([string]$Value)

    if ($null -eq $Value) {
        return ""
    }

    $Raw = $Value.Trim().ToLowerInvariant()
    if (-not $Raw) {
        return ""
    }
    if ($Raw -in @("all", "detected", "default")) {
        return "all"
    }
    if ($Raw -in @("none", "skip", "no", "false", "0")) {
        return "none"
    }

    $Aliases = @{
        "1" = "claude_desktop"
        "claude" = "claude_desktop"
        "claude-desktop" = "claude_desktop"
        "claude_desktop" = "claude_desktop"
        "2" = "claude_code"
        "claude-code" = "claude_code"
        "claude_code" = "claude_code"
        "3" = "codex"
        "codex" = "codex"
        "8" = "chatgpt"
        "chat" = "chatgpt"
        "chatgpt" = "chatgpt"
        "chatgpt-desktop" = "chatgpt"
        "chatgpt_desktop" = "chatgpt"
        "4" = "vscode"
        "vs-code" = "vscode"
        "vs_code" = "vscode"
        "vscode" = "vscode"
        "5" = "cursor"
        "cursor" = "cursor"
        "6" = "windsurf"
        "windsurf" = "windsurf"
        "7" = "cline"
        "cline" = "cline"
    }

    $Selected = New-Object System.Collections.Generic.List[string]
    foreach ($Token in ($Raw -split "[,;\s]+")) {
        if (-not $Token) {
            continue
        }
        if ($Token -in @("all", "detected", "default")) {
            return "all"
        }
        if ($Token -in @("none", "skip", "no", "false", "0")) {
            return "none"
        }
        if ($Aliases.ContainsKey($Token)) {
            $Surface = $Aliases[$Token]
            if (-not $Selected.Contains($Surface)) {
                $Selected.Add($Surface)
            }
        }
    }

    return ($Selected -join ",")
}

function Get-SurfaceLabel {
    param([string]$Surface)

    $Labels = @{
        "claude_desktop" = "Claude Desktop"
        "claude_code" = "Claude Code"
        "codex" = "Codex"
        "chatgpt" = "ChatGPT"
        "vscode" = "VS Code"
        "cursor" = "Cursor"
        "windsurf" = "Windsurf"
        "cline" = "Cline"
    }
    if ($Labels.ContainsKey($Surface)) {
        return $Labels[$Surface]
    }
    return $Surface
}

function Test-SurfaceSelected {
    param(
        [string]$SelectedSurfaces,
        [string]$Surface
    )

    if (-not $SelectedSurfaces -or $SelectedSurfaces -eq "all") {
        return $true
    }
    if ($SelectedSurfaces -eq "none") {
        return $false
    }
    $Items = @($SelectedSurfaces -split "," | ForEach-Object { $_.Trim() } | Where-Object { $_ })
    return $Items -contains $Surface
}

function Show-SelectedSurfaces {
    param([string]$SelectedSurfaces)

    Write-Host ""
    Write-Host "Selected AI app surfaces:" -ForegroundColor Cyan
    if (-not $SelectedSurfaces -or $SelectedSurfaces -eq "all") {
        Write-Host "  - All supported detected surfaces"
        return
    }
    if ($SelectedSurfaces -eq "none") {
        Write-Host "  - None (AI app configuration skipped)"
        return
    }
    foreach ($Surface in ($SelectedSurfaces -split ",")) {
        $Trimmed = $Surface.Trim()
        if ($Trimmed) {
            Write-Host "  - $(Get-SurfaceLabel $Trimmed)"
        }
    }
}

function Select-InstallSurfaces {
    $PublicDefault = "claude_desktop,claude_code,chatgpt,codex,vscode,cursor"
    $EnvSelectedExists = $null -ne [Environment]::GetEnvironmentVariable("PITH_SELECTED_SURFACES", "Process")
    if ($EnvSelectedExists) {
        $Selected = Normalize-SurfaceList $env:PITH_SELECTED_SURFACES
        if (-not $Selected) {
            $Selected = "none"
        }
        Write-Host "  AI app surfaces selected from PITH_SELECTED_SURFACES: $Selected"
        return $Selected
    }

    $SelectedFromClients = if ($env:PITH_CLIENTS) { Normalize-SurfaceList $env:PITH_CLIENTS } else { "all" }
    if (-not $SelectedFromClients -or $SelectedFromClients -eq "all") {
        $SelectedFromClients = $PublicDefault
    }

    $PrivateBeta = $env:PITH_PRIVATE_BETA -eq "1"
    $SkipPauses = $env:PITH_SKIP_PAUSES -eq "1"
    $Interactive = [Environment]::UserInteractive -and -not [Console]::IsInputRedirected -and -not $SkipPauses
    if (-not $PrivateBeta -or -not $Interactive) {
        Write-Host "  AI app surfaces selected: $SelectedFromClients"
        return $SelectedFromClients
    }

    Write-Host ""
    Write-Host "AI app setup:" -ForegroundColor Cyan
    Write-Host "  1. Claude Desktop"
    Write-Host "  2. Claude Code"
    Write-Host "  3. Codex"
    Write-Host "  4. VS Code"
    Write-Host "  5. Cursor"
    Write-Host "  6. Windsurf"
    Write-Host "  7. Cline"
    Write-Host "  8. ChatGPT"
    $Answer = Read-Host "Install Pith into which surfaces? [all detected]"
    if (-not $Answer) {
        Write-Host "  AI app surfaces selected: $SelectedFromClients"
        return $SelectedFromClients
    }

    $Selected = Normalize-SurfaceList $Answer
    if (-not $Selected) {
        Write-Warning "No recognized AI app surfaces selected; skipping AI app configuration"
        return "none"
    }
    Write-Host "  AI app surfaces selected: $Selected"
    return $Selected
}

$script:ClientReadinessEntries = @()
$script:ChatGptTunnelReady = $false

function Show-ClientReadiness {
    param([string]$JsonPath)

    if (-not (Test-Path $JsonPath)) {
        return $false
    }

    try {
        $Report = Get-Content -Path $JsonPath -Raw | ConvertFrom-Json
    }
    catch {
        Write-Warning "Could not parse AI client readiness summary"
        return $true
    }

    $Readiness = @($Report.readiness | Where-Object { $_ })
    $script:ClientReadinessEntries = $Readiness
    if ($Readiness.Count -eq 0) {
        return $false
    }

    $Incomplete = $false
    Write-Host ""
    Write-Host "AI client readiness:" -ForegroundColor Cyan
    foreach ($Entry in $Readiness) {
        $State = [string]$Entry.state
        $Label = if ($Entry.client) { [string]$Entry.client } else { "Client" }
        Write-Host "  - $($Label): $State"
        if ($Entry.reason) {
            Write-Host "    $($Entry.reason)"
        }
        if ($Entry.path) {
            Write-Host "    Path: $($Entry.path)"
        }
        if ($State -ne "ready") {
            $Incomplete = $true
        }
    }

    return $Incomplete
}

function Open-PithClaudeExtensionInstaller {
    param([string]$JsonPath)

    if ($env:PITH_SKIP_CLIENT_UI -eq "1" -or -not [Environment]::UserInteractive -or [Console]::IsInputRedirected) {
        return
    }
    if (-not (Test-Path -LiteralPath $JsonPath)) {
        return
    }
    try {
        $Report = Get-Content -LiteralPath $JsonPath -Raw | ConvertFrom-Json
        $Extension = @($Report.configured | Where-Object {
            $_.client_id -eq "claude_desktop" -and
            $_.scope -eq "desktop_extension" -and
            $_.action -eq "user_action_required"
        } | Select-Object -First 1)
        if ($Extension.Count -eq 0 -or -not $Extension[0].package_path) {
            return
        }
        $PackagePath = [string]$Extension[0].package_path
        if (-not (Test-Path -LiteralPath $PackagePath -PathType Leaf)) {
            Write-Warning "Claude extension package was prepared but is missing at $PackagePath"
            return
        }
        Start-Process -FilePath $PackagePath -ErrorAction Stop
        Write-Host "  Opened the Pith extension in Claude Desktop. Complete the installation dialog." -ForegroundColor Yellow
    }
    catch {
        Write-Warning "Could not open the Claude extension automatically. Install it from Claude Settings > Extensions > Advanced settings > Install Extension."
    }
}

function Get-ReadinessEntry {
    param(
        [string]$ClientId,
        [string]$Label
    )

    foreach ($Entry in @($script:ClientReadinessEntries)) {
        if ($Entry.client_id -and [string]$Entry.client_id -eq $ClientId) {
            return $Entry
        }
        if ($Entry.client -and [string]$Entry.client -eq $Label) {
            return $Entry
        }
    }
    return $null
}

function Show-RequiredClientSetup {
    param(
        [string]$SelectedSurfaces,
        [bool]$PathAdded,
        [string]$PithHome
    )

    Write-Host "Required setup:" -ForegroundColor Cyan
    $Step = 1
    if ($PathAdded) {
        Write-Host "  $Step. Open a new terminal (PATH is already configured)"
    }
    else {
        Write-Host "  $Step. Add to PATH:  `$env:PATH += ';$PithHome\bin'"
    }
    $Step += 1

    if ($SelectedSurfaces -eq "none") {
            Write-Host "  $Step. AI app configuration was skipped. To configure later, rerun the installer with PITH_SELECTED_SURFACES=claude_desktop,claude_code,chatgpt,codex"
        $Step += 1
    }
    else {
        if (@($script:ClientReadinessEntries).Count -gt 0) {
            Write-Host "  $Step. Restart each configured AI app completely before testing it"
            $Step += 1
        }
        else {
            Write-Host "  $Step. No selected AI apps were detected/configured. Review $PithHome\logs\configure-clients.json, then rerun after installing the target clients."
            $Step += 1
        }

        $ClaudeDesktop = Get-ReadinessEntry -ClientId "claude_desktop" -Label "Claude Desktop"
        if ((Test-SurfaceSelected -SelectedSurfaces $SelectedSurfaces -Surface "claude_desktop") -and $ClaudeDesktop) {
            if ($ClaudeDesktop.state -eq "manual_action_required" -and $ClaudeDesktop.scope -eq "desktop_extension" -and $ClaudeDesktop.path -and (Test-Path -LiteralPath ([string]$ClaudeDesktop.path) -PathType Leaf)) {
                Write-Host "  $Step. Claude Desktop: finish installing the prepared .mcpb in Settings > Extensions > Advanced settings > Install Extension"
                Write-Host "     Package: $($ClaudeDesktop.path)"
            }
            else {
                Write-Host "  $Step. Claude Desktop: repair client configuration, then rerun setup"
                Write-Host "     Diagnostics: $PithHome\logs\configure-clients.json"
            }
            $Step += 1
            Write-Host "  $Step. Claude Desktop / Claude Chat: fully quit and restart the app, then run pith_connection_proof in a fresh turn"
            Write-Host "     A same-turn pith_conversation_turn is also valid; bridge/status checks or MCP config presence alone are not connected proof"
            $Step += 1
        }

        $ClaudeCode = Get-ReadinessEntry -ClientId "claude_code" -Label "Claude Code"
        if ((Test-SurfaceSelected -SelectedSurfaces $SelectedSurfaces -Surface "claude_code") -and $ClaudeCode) {
            Write-Host "  $Step. Claude Code: restart Claude Code, run /mcp and /status, then confirm a fresh turn can call pith_conversation_turn"
            $Step += 1
        }

        $Codex = Get-ReadinessEntry -ClientId "codex" -Label "Codex"
        if ((Test-SurfaceSelected -SelectedSurfaces $SelectedSurfaces -Surface "codex") -and $Codex) {
            Write-Host "  $Step. Codex: fully quit and restart the app after plugin/config changes, then verify Pith from a fresh task"
            $Step += 1
        }

        $ChatGPT = Get-ReadinessEntry -ClientId "chatgpt" -Label "ChatGPT"
        if ((Test-SurfaceSelected -SelectedSurfaces $SelectedSurfaces -Surface "chatgpt") -and $ChatGPT) {
            if ($script:ChatGptTunnelReady) {
                Write-Host "  $Step. ChatGPT Chat: the enrolled Secure MCP Tunnel was repaired and started"
                $Step += 1
            }
            else {
                Write-Host "  $Step. ChatGPT Chat: localhost MCP is not supported; enable Developer mode and configure a remote MCP app or OpenAI Secure MCP Tunnel"
                $Step += 1
                Write-Host "     Connector settings: $($ChatGPT.connector_settings_url)"
                Write-Host "     Tunnel management:  $($ChatGPT.tunnel_management_url)"
            }
            Write-Host "  $Step. ChatGPT Chat: start a fresh chat with the remote Pith app enabled, then require same-turn pith_conversation_turn proof"
            $Step += 1
        }
    }

    Write-Host "  $Step. Verify core health: pith status (expect Health: OK (Pith))"
    $Step += 1
    Write-Host "  $Step. Verify client diagnostics: pith clients --json"
}

# Trap for cleanup
trap {
    $TrapMessage = "Installation interrupted at step $CurrentStep`n$($_.Exception.Message)"
    if ($_.InvocationInfo) {
        $TrapMessage += "`nLine: $($_.InvocationInfo.ScriptLineNumber)"
        if ($_.InvocationInfo.Line) {
            $TrapMessage += "`nCommand: $($_.InvocationInfo.Line.Trim())"
        }
    }
    Write-Error-Custom $TrapMessage
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

function Test-PythonCandidate {
    param([string]$CandidatePath)

    if (-not $CandidatePath) {
        return $null
    }

    try {
        $VersionOutput = & $CandidatePath --version 2>&1
        if ($LASTEXITCODE -ne 0) {
            return $null
        }

        $VersionText = ($VersionOutput -join " ").Trim()
        if ($VersionText -notmatch "Python\s+([0-9]+)\.([0-9]+)(?:\.([0-9]+))?") {
            return $null
        }

        return [PSCustomObject]@{
            Path = $CandidatePath
            Version = if ($Matches[3]) { "$($Matches[1]).$($Matches[2]).$($Matches[3])" } else { "$($Matches[1]).$($Matches[2]).0" }
            Major = [int]$Matches[1]
            Minor = [int]$Matches[2]
        }
    }
    catch {
        return $null
    }
}

function Test-PythonVenvModule {
    param(
        [Parameter(Mandatory = $true)][string]$PythonExe,
        [int]$Attempts = 10
    )

    for ($Attempt = 1; $Attempt -le $Attempts; $Attempt++) {
        try {
            & $PythonExe -c "import venv" *>$null
            if ($LASTEXITCODE -eq 0) {
                return $true
            }
        }
        catch {
            # A freshly replaced executable can be briefly unavailable to the user token.
        }
        if ($Attempt -lt $Attempts) {
            Start-Sleep -Seconds 1
        }
    }
    return $false
}

function Test-PithCompatiblePythonExe {
    param([string]$CandidatePath)

    if (-not $CandidatePath -or -not (Test-Path -LiteralPath $CandidatePath)) {
        return $false
    }
    try {
        & $CandidatePath -c "import sys; raise SystemExit(0 if ((sys.version_info >= (3, 10)) and (sys.version_info < (3, 13))) else 1)" 2>$null
        return ($LASTEXITCODE -eq 0)
    }
    catch {
        return $false
    }
}

function Get-PithPythonVersion {
    param([string]$PythonExe)

    try {
        return (& $PythonExe -c "import sys; print('.'.join(map(str, sys.version_info[:3])))" 2>$null).Trim()
    }
    catch {
        return ""
    }
}

function Find-PythonCandidate {
    $CandidatePaths = New-Object System.Collections.Generic.List[string]

    Get-Command python.exe -All -ErrorAction SilentlyContinue | ForEach-Object {
        if ($_.Source -and -not $CandidatePaths.Contains($_.Source)) {
            $CandidatePaths.Add($_.Source)
        }
    }

    $KnownRoots = @(
        "$env:LOCALAPPDATA\Programs\Python",
        "$env:ProgramFiles",
        "${env:ProgramFiles(x86)}"
    )

    foreach ($Root in $KnownRoots) {
        if (-not $Root -or -not (Test-Path $Root)) {
            continue
        }
        Get-ChildItem -Path $Root -Directory -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -like "Python*" } |
            Sort-Object Name -Descending |
            ForEach-Object {
                $Candidate = Join-Path $_.FullName "python.exe"
                if ((Test-Path $Candidate) -and -not $CandidatePaths.Contains($Candidate)) {
                    $CandidatePaths.Add($Candidate)
                }
            }
    }

    foreach ($CandidatePath in $CandidatePaths) {
        $PythonCandidate = Test-PythonCandidate -CandidatePath $CandidatePath
        if ($PythonCandidate) {
            return $PythonCandidate
        }
    }

    return $null
}

function Install-PythonDirect {
    $PythonInstallVersion = "3.11.9"
    $PythonInstallArch = "amd64"
    if ($env:PROCESSOR_ARCHITECTURE -eq "ARM64") {
        Write-Host "  Windows ARM64 detected; installing x64 Python for package wheel compatibility."
    }

    New-Item -ItemType Directory -Path "$PithHome\logs" -Force | Out-Null
    $PythonInstallerUrl = "https://www.python.org/ftp/python/$PythonInstallVersion/python-$PythonInstallVersion-$PythonInstallArch.exe"
    $PythonInstallerPath = Join-Path $env:TEMP "python-$PythonInstallVersion-$PythonInstallArch.exe"
    $PythonInstallLog = "$PithHome\logs\python_install.log"

    Write-Host "Downloading Python $PythonInstallVersion from python.org..."
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    $PreviousProgressPreference = $ProgressPreference
    $ProgressPreference = "SilentlyContinue"
    try {
        Invoke-WebRequest -Uri $PythonInstallerUrl -OutFile $PythonInstallerPath -UseBasicParsing -TimeoutSec 300 -ErrorAction Stop
    }
    finally {
        $ProgressPreference = $PreviousProgressPreference
    }

    Write-Host "Installing Python $PythonInstallVersion for current user..."
    # Start-Process flattens string arrays under Windows PowerShell 5. Build one
    # command line so the log path remains a single argument when PITH_HOME has spaces.
    $PythonInstallArgumentLine = "/quiet InstallAllUsers=0 PrependPath=1 Include_launcher=0 InstallLauncherAllUsers=0 Include_test=0 /log `"$PythonInstallLog`""
    $PythonInstallProcess = New-Object System.Diagnostics.Process
    $PythonInstallProcess.StartInfo.FileName = $PythonInstallerPath
    $PythonInstallProcess.StartInfo.Arguments = $PythonInstallArgumentLine
    $PythonInstallProcess.StartInfo.UseShellExecute = $false
    [void]$PythonInstallProcess.Start()
    $PythonInstallCompleted = $PythonInstallProcess.WaitForExit(10 * 60 * 1000)
    if (-not $PythonInstallCompleted) {
        & taskkill.exe /PID $PythonInstallProcess.Id /T /F 2>$null | Out-Null
        [void]$PythonInstallProcess.WaitForExit(5000)
        Write-Error-Custom "Python installation timed out after 10 minutes. Details: $PythonInstallLog"
    }
    $PythonInstallExit = $PythonInstallProcess.ExitCode
    $PythonInstallProcess.Dispose()
    if ($null -eq $PythonInstallExit -or $PythonInstallExit -ne 0) {
        Write-Error-Custom "Python installation failed. Details: $PythonInstallLog"
    }
}

function Write-PithPythonRuntimeMetadata {
    param(
        [string]$ManagedBy,
        [string]$PythonExe,
        [string]$BasePythonExe,
        [string]$PythonVersion,
        [string]$VenvPath
    )

    $RuntimeArch = if ($ManagedBy -eq "pith") { $PithRuntimeArch } elseif ($env:PROCESSOR_ARCHITECTURE) { $env:PROCESSOR_ARCHITECTURE } else { "unknown" }
    $RuntimeIdVersion = if ($PythonVersion) { $PythonVersion } else { "unknown" }
    $RuntimeId = if ($ManagedBy -eq "pith") { $PithRuntimeId } else { "external-python-$RuntimeIdVersion-windows-$RuntimeArch".ToLowerInvariant() }
    $RuntimeMeta = [ordered]@{
        managed_by = $ManagedBy
        runtime_id = $RuntimeId
        python_version = $PythonVersion
        platform = $PithRuntimePlatform
        arch = $RuntimeArch
        source = if ($ManagedBy -eq "pith") { $PithRuntimeSource } else { "external_python" }
        source_url = if ($ManagedBy -eq "pith") { $PithRuntimeUrl } else { $null }
        sha256 = if ($ManagedBy -eq "pith") { $PithRuntimeSha256 } else { $null }
        license = if ($ManagedBy -eq "pith") { $PithRuntimeLicense } else { "External Python installation selected during Windows install; not bundled or managed by Pith." }
        installed_at = (Get-Date).ToUniversalTime().ToString("o")
        python_executable = $PythonExe
        base_python_executable = $BasePythonExe
        venv_path = $VenvPath
    }
    New-Item -ItemType Directory -Path (Split-Path -Parent $PithRuntimeMetaPath) -Force | Out-Null
    $RuntimeMetaJson = $RuntimeMeta | ConvertTo-Json -Depth 8
    [System.IO.File]::WriteAllText(
        $PithRuntimeMetaPath,
        $RuntimeMetaJson + [Environment]::NewLine,
        (New-Object System.Text.UTF8Encoding($false))
    )
}

function Get-PithRuntimeMetadataManagedBy {
    if (-not (Test-Path -LiteralPath $PithRuntimeMetaPath)) {
        return ""
    }
    try {
        $Meta = Get-Content -LiteralPath $PithRuntimeMetaPath -Raw | ConvertFrom-Json
        if ($Meta.PSObject.Properties["managed_by"]) {
            return [string]$Meta.managed_by
        }
    }
    catch {
        return ""
    }
    return ""
}

function Remove-PithManagedPythonRuntime {
    if ((Get-PithRuntimeMetadataManagedBy) -eq "pith") {
        Remove-Item -LiteralPath $PithRuntimeRoot -Recurse -Force -ErrorAction SilentlyContinue
        Remove-Item -LiteralPath $PithRuntimeMetaPath -Force -ErrorAction SilentlyContinue
        Write-Warning "Removed corrupt Pith-managed Python runtime"
    }
}

function Test-PithManagedPythonRuntimeReady {
    $RuntimePythonExe = Join-Path $PithRuntimeRoot "python.exe"
    if ((Get-PithRuntimeMetadataManagedBy) -ne "pith") {
        return $false
    }
    if (-not (Test-PithCompatiblePythonExe -CandidatePath $RuntimePythonExe)) {
        Remove-PithManagedPythonRuntime
        return $false
    }
    return $true
}

function Test-PithTarAvailable {
    try {
        $TarCommand = Get-Command tar.exe -ErrorAction Stop | Select-Object -First 1
        return [bool]$TarCommand
    }
    catch {
        return $false
    }
}

function Get-PithInstallerTempRoot {
    $Candidates = @(
        $env:TEMP,
        $env:TMP,
        $(if ($env:USERPROFILE) { Join-Path $env:USERPROFILE "AppData\Local\Temp" } else { $null }),
        (Join-Path $PithHome "tmp")
    )
    foreach ($Candidate in $Candidates) {
        if ([string]::IsNullOrWhiteSpace([string]$Candidate)) {
            continue
        }
        try {
            New-Item -ItemType Directory -Path $Candidate -Force -ErrorAction Stop | Out-Null
            return (Resolve-Path -LiteralPath $Candidate -ErrorAction Stop).Path
        }
        catch {
            continue
        }
    }
    Write-Error-Custom "Could not create a temporary directory for Pith-managed Python runtime download."
}

function Save-PithRuntimeArchive {
    param(
        [string]$Uri,
        [string]$OutFile
    )

    if ([string]::IsNullOrWhiteSpace($Uri)) {
        Write-Error-Custom "Pith-managed Python runtime URL is empty."
    }
    if ([string]::IsNullOrWhiteSpace($OutFile)) {
        Write-Error-Custom "Pith-managed Python runtime archive path is empty."
    }

    $OutDir = Split-Path -Parent $OutFile
    if ([string]::IsNullOrWhiteSpace($OutDir)) {
        Write-Error-Custom "Pith-managed Python runtime archive parent path is empty."
    }
    New-Item -ItemType Directory -Path $OutDir -Force | Out-Null

    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    Invoke-WebRequest -Uri $Uri -OutFile $OutFile -UseBasicParsing -TimeoutSec 300 -ErrorAction Stop

    if (-not (Test-Path -LiteralPath $OutFile)) {
        Write-Error-Custom "Pith-managed Python runtime download did not create archive: $OutFile"
    }
}

function Install-PithManagedPythonRuntime {
    if (-not (Test-PithTarAvailable)) {
        Write-Error-Custom "Pith-managed Python runtime extraction requires tar.exe. Set PITH_NO_AUTO_PYTHON=1 to use an existing Python installation instead."
    }

    New-Item -ItemType Directory -Path "$PithHome\logs", (Split-Path -Parent $PithRuntimeRoot), "$PithHome\config" -Force | Out-Null
    $TempRoot = Get-PithInstallerTempRoot
    $TempDir = Join-Path -Path $TempRoot -ChildPath "pith-python-runtime-$([System.Guid]::NewGuid().ToString())"
    $ArchivePath = Join-Path -Path $TempDir -ChildPath "python-runtime.tar.gz"
    $ExtractPath = Join-Path -Path $TempDir -ChildPath "extract"
    $RuntimeTmp = "$PithRuntimeRoot.tmp"
    New-Item -ItemType Directory -Path $TempDir, $ExtractPath -Force | Out-Null

    try {
        Write-Host "Downloading Pith-managed Python $PithRuntimeVersion ($([Math]::Round($PithRuntimeSizeBytes / 1MB, 1)) MB)..."
        $PreviousProgressPreference = $ProgressPreference
        $ProgressPreference = "SilentlyContinue"
        try {
            Save-PithRuntimeArchive -Uri $PithRuntimeUrl -OutFile $ArchivePath
        }
        finally {
            $ProgressPreference = $PreviousProgressPreference
        }

        $ActualHash = Get-PithSha256 -Path $ArchivePath
        if ($ActualHash -ne $PithRuntimeSha256.ToLowerInvariant()) {
            Write-Error-Custom "Python runtime checksum mismatch. Actual=$ActualHash Expected=$PithRuntimeSha256"
        }

        $ArchiveMembers = @(& tar.exe -tzf $ArchivePath)
        if ($LASTEXITCODE -ne 0) {
            Write-Error-Custom "Could not inspect Pith-managed Python runtime archive."
        }
        $BadMember = $ArchiveMembers | Where-Object { $_ -notmatch '^python/' } | Select-Object -First 1
        if ($BadMember) {
            Write-Error-Custom "Python runtime archive contains unexpected path: $BadMember"
        }
        if ($ArchiveMembers -notcontains "python/python.exe") {
            Write-Error-Custom "Python runtime archive missing python/python.exe"
        }

        & tar.exe -xzf $ArchivePath -C $ExtractPath
        if ($LASTEXITCODE -ne 0) {
            Write-Error-Custom "Could not extract Pith-managed Python runtime archive."
        }
        $RuntimePythonExe = Join-Path (Join-Path $ExtractPath "python") "python.exe"
        if (-not (Test-PithCompatiblePythonExe -CandidatePath $RuntimePythonExe)) {
            Write-Error-Custom "Extracted Pith-managed Python runtime is not compatible."
        }

        Remove-Item -LiteralPath $RuntimeTmp -Recurse -Force -ErrorAction SilentlyContinue
        Move-Item -LiteralPath (Join-Path $ExtractPath "python") -Destination $RuntimeTmp -Force
        Remove-Item -LiteralPath $PithRuntimeRoot -Recurse -Force -ErrorAction SilentlyContinue
        Move-Item -LiteralPath $RuntimeTmp -Destination $PithRuntimeRoot -Force
        $InstalledPythonExe = Join-Path $PithRuntimeRoot "python.exe"
        $InstalledVersion = Get-PithPythonVersion -PythonExe $InstalledPythonExe
        Write-PithPythonRuntimeMetadata `
            -ManagedBy "pith" `
            -PythonExe $InstalledPythonExe `
            -BasePythonExe $InstalledPythonExe `
            -PythonVersion $InstalledVersion `
            -VenvPath ""
        Write-Success "Installed Pith-managed Python runtime"
        return $InstalledPythonExe
    }
    finally {
        Remove-Item -LiteralPath $TempDir -Recurse -Force -ErrorAction SilentlyContinue
        Remove-Item -LiteralPath $RuntimeTmp -Recurse -Force -ErrorAction SilentlyContinue
    }
}

function Ensure-PithPythonRuntime {
    if ($env:PITH_REPAIR_RUNTIME -eq "1") {
        Remove-PithManagedPythonRuntime
    }
    if ($env:PITH_NO_AUTO_PYTHON -eq "1") {
        if (-not (Test-SurfaceSelected -SelectedSurfaces $PithSelectedSurfaces -Surface "claude_desktop")) {
            return $null
        }
        Write-Warning "Claude Desktop requires the Pith-managed Python runtime; PITH_NO_AUTO_PYTHON=1 is ignored for this selected surface."
    }

    $RuntimePythonExe = Join-Path $PithRuntimeRoot "python.exe"
    if (Test-PithManagedPythonRuntimeReady) {
        return $RuntimePythonExe
    }
    return (Install-PithManagedPythonRuntime)
}

# Check Python. Windows Store execution aliases resolve as python.exe but fail
# --version; treat those aliases as missing so direct install can recover.
$PithSelectedSurfaces = Select-InstallSurfaces
Show-SelectedSurfaces -SelectedSurfaces $PithSelectedSurfaces
$PythonPath = Ensure-PithPythonRuntime
$PithSelectedRuntimeManagedBy = if ($PythonPath) { "pith" } else { "external" }
$PythonCandidate = $null
if (-not $PythonPath) {
    $PythonCandidate = Find-PythonCandidate
}

# If not found, install Python directly for the current user. Winget can route
# the official Python installer through an elevated launcher path in unattended
# runs; direct per-user install avoids UAC and keeps the setup flow noninteractive.
if ((-not $PythonPath) -and (-not $PythonCandidate)) {
    Write-Warning "Python 3 not found in PATH"
    try {
        Install-PythonDirect

        # Refresh PATH for this process because the installer updates persisted env vars.
        $MachinePath = [Environment]::GetEnvironmentVariable("PATH", "Machine")
        $UserPath = [Environment]::GetEnvironmentVariable("PATH", "User")
        $env:PATH = "$UserPath;$MachinePath;$env:PATH"

        $PythonCandidate = Find-PythonCandidate
        if ($PythonCandidate) {
            Write-Success "Python installed successfully"
        }
        else {
            Write-Error-Custom "Python installation failed. Please install Python 3.9+ manually from python.org"
        }
    }
    catch {
        Write-Error-Custom "Failed to install Python. Please install Python 3.9+ manually from python.org"
    }
}

# Verify Python version
if ($PythonPath) {
    $PythonVersion = Get-PithPythonVersion -PythonExe $PythonPath
    if (-not $PythonVersion) {
        Write-Error-Custom "Pith-managed Python runtime did not report a version."
    }
}
elseif (
    ($PythonCandidate.Major -lt 3) -or
    (($PythonCandidate.Major -eq 3) -and ($PythonCandidate.Minor -lt 10)) -or
    (($PythonCandidate.Major -eq 3) -and ($PythonCandidate.Minor -gt 12)) -or
    ($PythonCandidate.Major -gt 3)
) {
    $PythonVersion = $PythonCandidate.Version
    Write-Error-Custom "Python 3.10-3.12 required. Found: $PythonVersion"
}
if (-not $PythonPath) {
    $PythonPath = $PythonCandidate.Path
    $PythonVersion = $PythonCandidate.Version
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

# Verify venv module. Windows security scanners can briefly delay execution
# immediately after the managed runtime directory is atomically replaced.
if (Test-PythonVenvModule -PythonExe $PythonPath) {
    Write-Success "Python venv module available"
}
else {
    Write-Error-Custom "Python venv module did not become available after bounded retries. Rerun the installer to repair the managed runtime."
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

# Windows cannot replace or remove venv binaries while the existing API or MCP
# process has them loaded. Stop only processes owned by this installation before
# mutating server files or dependency state.
Stop-PithInstallProcessesForCleanup

# ============================================================================
# STEP 3: Install Pith Server Files
# ============================================================================
Write-Step 3 "Install Pith server files"

$PithServerPath = "$PithHome\pith-server"
Get-PithServerTreeInventory -PithHome $PithHome -ServerPath $PithServerPath | Out-Null
$DownloadSuccess = $false

# Strategy 1: Detect running from distribution directory (most common for beta)
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ParentDir = Split-Path -Parent $ScriptDir
$AdjacentPackage = Join-Path $ScriptDir "pith-server-latest.zip"
$DistDir = if (Test-Path -LiteralPath $AdjacentPackage) { $ScriptDir } else { $ParentDir }

if (
    -not (Test-Path -LiteralPath $AdjacentPackage) -and
    (Test-Path -LiteralPath "$DistDir\app\api\server.py") -and
    (Test-Path -LiteralPath "$DistDir\pith_client\cli.py") -and
    (Test-Path -LiteralPath "$DistDir\pith_mcp.py")
) {
    Write-Host "  Detected distribution directory: $DistDir"
    if (-not (Test-Path -LiteralPath $PithServerPath)) {
        New-Item -ItemType Directory -Path $PithServerPath -Force | Out-Null
    }
    # Copy directories to the server parent so upgrades merge without nesting.
    Copy-Item -LiteralPath "$DistDir\app" -Destination $PithServerPath -Recurse -Force -ErrorAction Stop
    Copy-Item -LiteralPath "$DistDir\pith_client" -Destination $PithServerPath -Recurse -Force -ErrorAction Stop
    Copy-Item -LiteralPath "$DistDir\pith_mcp.py" -Destination "$PithServerPath\pith_mcp.py" -Force -ErrorAction Stop
    Copy-Item -LiteralPath "$DistDir\skill_deployer.py" -Destination "$PithServerPath\skill_deployer.py" -Force -ErrorAction Stop
    Copy-Item -LiteralPath "$DistDir\requirements.txt" -Destination "$PithServerPath\requirements.txt" -Force -ErrorAction Stop
    if (Test-Path -LiteralPath "$DistDir\scripts") {
        Copy-Item -LiteralPath "$DistDir\scripts" -Destination $PithServerPath -Recurse -Force -ErrorAction Stop
    }
    if (Test-Path -LiteralPath "$DistDir\migrations") {
        Copy-Item -LiteralPath "$DistDir\migrations" -Destination $PithServerPath -Recurse -Force -ErrorAction Stop
    }
    if (Test-Path -LiteralPath "$DistDir\integrations") {
        Copy-Item -LiteralPath "$DistDir\integrations" -Destination $PithServerPath -Recurse -Force -ErrorAction Stop
    }
    Write-Success "Copied server files from distribution"
    Write-Host "PITH_PACKAGE_SOURCE=distribution_directory"
    $DownloadSuccess = $true
}

# Strategy 2: Local tarball/zip (created by build-release.sh)
if (-not $DownloadSuccess) {
    $LocalPackage = Join-Path $DistDir "pith-server-latest.zip"
    if (Test-Path -LiteralPath $LocalPackage) {
        Write-Host "  Found local package: $LocalPackage"
        $LocalChecksum = "$LocalPackage.sha256"
        if (-not (Test-Path -LiteralPath $LocalChecksum -PathType Leaf)) {
            throw "Local release package checksum is missing: $LocalChecksum"
        }
        $ChecksumLine = (Get-Content -LiteralPath $LocalChecksum -Raw).Trim()
        if ($ChecksumLine -notmatch '^([0-9A-Fa-f]{64})\s+\*?pith-server-latest\.zip$') {
            throw "Local release package checksum format is invalid: $LocalChecksum"
        }
        $ExpectedLocalHash = $Matches[1].ToLowerInvariant()
        $ActualLocalHash = Get-PithSha256 -Path $LocalPackage
        if ($ActualLocalHash -ne $ExpectedLocalHash) {
            throw "Local release package checksum verification failed"
        }
        if (-not (Test-Path -LiteralPath $PithServerPath)) {
            New-Item -ItemType Directory -Path $PithServerPath -Force | Out-Null
        }
        Expand-Archive -LiteralPath $LocalPackage -DestinationPath $PithServerPath -Force
        Write-Success "Extracted local server package"
        Write-Host "PITH_PACKAGE_SOURCE=adjacent_verified_zip"
        $DownloadSuccess = $true
    }
}

# Strategy 3: Download from hosted URL (future)
if (-not $DownloadSuccess) {
    if ($env:PITH_REQUIRE_LOCAL_PACKAGE -eq "1") {
        throw "Required local release package was not found beside the installer or in its parent directory."
    }
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
        if ((Test-Path -LiteralPath $ServerPath) -and (Test-Path -LiteralPath $ChecksumPath)) {
            $FileHash = Get-PithSha256 -Path $ServerPath
            $ChecksumContent = (Get-Content $ChecksumPath | Select-Object -First 1) -split ' '
            $ExpectedHash = $ChecksumContent[0]

            if ($FileHash -eq $ExpectedHash) {
                Write-Success "Download successful and checksum verified"

                # Extract server
                if (-not (Test-Path -LiteralPath $PithServerPath)) {
                    New-Item -ItemType Directory -Path $PithServerPath -Force | Out-Null
                }
                Expand-Archive -LiteralPath $ServerPath -DestinationPath $PithServerPath -Force
                Write-Host "PITH_PACKAGE_SOURCE=hosted_verified_zip"
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
if (-not (Test-Path -LiteralPath "$PithServerPath\pith_client\cli.py" -PathType Leaf)) {
    throw "Installed Pith server tree is incomplete: missing pith_client\cli.py"
}

$BytecodeRemoved = Clear-PithApplicationBytecode -PithHome $PithHome -ServerPath $PithServerPath
Write-Host "PITH_APPLICATION_BYTECODE_REMOVED=$BytecodeRemoved"
Write-Host ""

# ============================================================================
# STEP 4: Python venv Setup with Health Check
# ============================================================================
Write-Step 4 "Python venv setup with health check [FIX R1, R2, R3]"

$WindowsEmbeddingRuntime = "$PithServerPath\scripts\windows_embedding_runtime.ps1"
if (-not (Test-Path $WindowsEmbeddingRuntime)) {
    Write-Error-Custom "Embedding runtime helper missing: $WindowsEmbeddingRuntime"
}
. $WindowsEmbeddingRuntime

$DefaultVenvPath = "$PithHome\venv"
$VenvPath = Get-PithVenvPath -PithHome $PithHome

function Test-PithVenvHealth {
    param([string]$Path)

    $VenvPythonExe = "$Path\Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $VenvPythonExe -PathType Leaf)) {
        return $false
    }
    try {
        $VenvHealthExit = Invoke-PithEmbeddingCommand `
            -PithHome $PithHome `
            -EmbedLog "$PithHome\logs\venv_setup.log" `
            -FilePath $VenvPythonExe `
            -Arguments @("-c", "import sys") `
            -Name "venv_health" `
            -TimeoutSeconds 30
        return ($VenvHealthExit -eq 0)
    }
    catch {
        return $false
    }
}

# FIX R1: Detect broken existing venv, recreate if needed
if (Test-Path $VenvPath) {
    if (-not (Test-PithVenvHealth -Path $VenvPath)) {
        Write-Warning "Existing venv is broken, recreating"
        try {
            Remove-Item -Path $VenvPath -Recurse -Force -ErrorAction Stop
        }
        catch {
            if ($VenvPath -ne $DefaultVenvPath) {
                Write-Error-Custom "Broken managed venv could not be removed: $($_.Exception.Message)"
            }
            $LockedDefaultVenv = $VenvPath
            $VenvPath = Get-PithShortVenvPath -PithHome $PithHome
            Write-Warning "Default venv is locked and cannot be rebuilt in place; switching to managed recovery venv: $VenvPath"
            if (Test-Path $VenvPath) {
                if (Test-PithVenvHealth -Path $VenvPath) {
                    Write-Warning "Reusing healthy managed recovery venv."
                }
                else {
                    try {
                        Remove-Item -Path $VenvPath -Recurse -Force -ErrorAction Stop
                    }
                    catch {
                        Write-Error-Custom "Managed recovery venv could not be reset: $($_.Exception.Message)"
                    }
                }
            }
            Write-Warning "Locked legacy venv remains at $LockedDefaultVenv and is no longer used."
        }
    }
}

# Create venv
if (-not (Test-Path $VenvPath)) {
    New-Item -ItemType Directory -Path (Split-Path -Parent $VenvPath) -Force | Out-Null
    $VenvCreateExit = Invoke-PithEmbeddingCommand `
        -PithHome $PithHome `
        -EmbedLog "$PithHome\logs\venv_setup.log" `
        -FilePath $PythonPath `
        -Arguments @("-m", "venv", $VenvPath) `
        -Name "venv_create" `
        -TimeoutSeconds 300
    if ($VenvCreateExit -ne 0) {
        Write-Error-Custom "Python virtual environment creation failed or timed out. Details: $PithHome\logs\venv_setup.log"
    }
    Write-Success "Created Python virtual environment"
}
else {
    Write-Success "Using existing virtual environment"
}

if (-not (Test-PithVenvHealth -Path $VenvPath)) {
    Write-Error-Custom "Selected Python virtual environment is not runnable: $VenvPath"
}

# Prepare pip activation script
$PipExe = "$VenvPath\Scripts\pip.exe"
$PythonExe = "$VenvPath\Scripts\python.exe"

# Upgrade pip (use python -m pip to avoid self-replace lock issues)
$PipBootstrapExit = Invoke-PithEmbeddingCommand `
    -PithHome $PithHome `
    -EmbedLog "$PithHome\logs\dependency_install.log" `
    -FilePath $PythonExe `
    -Arguments @("-m", "pip", "install", "--quiet", "--upgrade", "pip", "setuptools", "wheel") `
    -Name "pip_bootstrap" `
    -TimeoutSeconds 900
if ($PipBootstrapExit -eq 0) {
    Write-Success "Updated pip, setuptools, wheel"
}
else {
    Write-Warning "pip bootstrap failed or timed out; continuing with the installed pip. Details: $PithHome\logs\dependency_install.log"
}

# Install core dependencies from requirements.txt
Write-Host "Installing dependencies (this may take a moment)..."
$ReqFile = "$PithHome\pith-server\requirements.txt"
if (-not (Test-Path $ReqFile)) {
    $ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
    $ReqFile = Join-Path (Split-Path -Parent $ScriptDir) "requirements.txt"
}
$CoreDepsLog = "$PithHome\logs\core_dependency_install.log"
$CoreReqFile = "$PithHome\logs\core_requirements_windows.txt"
New-Item -ItemType Directory -Path "$PithHome\logs" -Force | Out-Null
Get-Content -Path $ReqFile | Where-Object {
    $_ -notmatch '^\s*sentence-transformers\b' -and $_ -notmatch '^\s*torch\b'
} | Set-Content -Path $CoreReqFile
$CoreDepsExit = Invoke-PithEmbeddingCommand `
    -PithHome $PithHome `
    -EmbedLog $CoreDepsLog `
    -FilePath $PipExe `
    -Arguments @("install", "--quiet", "-r", $CoreReqFile) `
    -Name "core_dependency_install" `
    -TimeoutSeconds 1800
if ($CoreDepsExit -ne 0) {
    Write-Error-Custom "Core dependency installation failed. Details: $CoreDepsLog"
}
Write-Success "Installed core dependencies"

Write-Host "Installing embeddings (CPU-only PyTorch)..."
$EmbedResult = Install-PithEmbeddings `
    -PithHome $PithHome `
    -VenvPath $VenvPath `
    -PipExe $PipExe `
    -PythonExe $PythonExe
if (-not $EmbedResult) {
    Write-Host "  Pith will run with TF-IDF search (fully functional, reduced semantic quality)." -ForegroundColor Yellow
}

$VenvPathRecord = "$PithHome\config\venv.path"
$ExistingVenvPathRecord = Get-Item -LiteralPath $VenvPathRecord -Force -ErrorAction SilentlyContinue
if ($ExistingVenvPathRecord) {
    try {
        Remove-Item -LiteralPath $VenvPathRecord -Force -ErrorAction Stop
    }
    catch {
        Write-Error-Custom "Could not replace configured venv path record: $($_.Exception.Message)"
    }
}
if ($VenvPath -ne $DefaultVenvPath) {
    Write-Warning "Using managed short virtual environment path: $VenvPath"
    Set-Content -LiteralPath $VenvPathRecord -Value $VenvPath -Encoding UTF8
}
Write-PithPythonRuntimeMetadata `
    -ManagedBy $PithSelectedRuntimeManagedBy `
    -PythonExe $PythonPath `
    -BasePythonExe $PythonPath `
    -PythonVersion $PythonVersion `
    -VenvPath $VenvPath

Write-Host ""

# ============================================================================
# STEP 5: Generate API Key with Secure Permissions
# ============================================================================
Write-Step 5 "Generate API key with secure file permissions [FIX S2]"

$ApiKeyFile = "$PithHome\config\api.key"

function New-PithApiKey {
    return -join ((1..32) | ForEach-Object { "{0:x2}" -f (Get-Random -Minimum 0 -Maximum 256) })
}

function Get-ApiKeyAccessIdentities {
    $IdentityNames = New-Object System.Collections.Generic.List[string]
    $CurrentUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
    $IdentityNames.Add($CurrentUser)
    $IdentityNames.Add("NT AUTHORITY\SYSTEM")

    try {
        $ProfileDir = Split-Path -Parent $PithHome
        $ProfileUser = Split-Path -Leaf $ProfileDir
        if (-not [string]::IsNullOrWhiteSpace($ProfileUser)) {
            $IdentityNames.Add(("{0}\{1}" -f $env:COMPUTERNAME, $ProfileUser))
        }
    }
    catch {
        Write-Warning "Could not resolve profile user while setting API key ACL"
    }

    try {
        $PithHomeOwner = (Get-Acl -Path $PithHome).Owner
        if ((-not [string]::IsNullOrWhiteSpace($PithHomeOwner)) -and ($PithHomeOwner -ne "NT AUTHORITY\SYSTEM")) {
            $IdentityNames.Add($PithHomeOwner)
        }
    }
    catch {
        Write-Warning "Could not resolve Pith home owner while setting API key ACL"
    }

    return @($IdentityNames | Select-Object -Unique)
}

function Repair-ApiKeyPathAccess {
    param([string]$Path)

    $ConfigDir = Split-Path -Parent $Path
    $AccessIdentities = Get-ApiKeyAccessIdentities

    if ([System.IO.Directory]::Exists($ConfigDir)) {
        foreach ($IdentityName in $AccessIdentities) {
            $DirectoryGrant = "{0}:(OI)(CI)F" -f $IdentityName
            & icacls.exe $ConfigDir /inheritance:e /grant:r $DirectoryGrant | Out-Null
        }
    }
    if ([System.IO.File]::Exists($Path)) {
        foreach ($IdentityName in $AccessIdentities) {
            $FileGrant = "{0}:F" -f $IdentityName
            & icacls.exe $Path /inheritance:e /grant:r $FileGrant | Out-Null
        }
        try {
            Set-ItemProperty -LiteralPath $Path -Name IsReadOnly -Value $false -ErrorAction SilentlyContinue
        }
        catch {
            Write-Warning "Could not clear read-only flag on API key file"
        }
    }
}

function Test-ApiKeyFileExists {
    param([string]$Path)

    try {
        return Test-Path -LiteralPath $Path -ErrorAction Stop
    }
    catch {
        Write-Warning "Could not inspect API key path; repairing permissions and retrying"
        Repair-ApiKeyPathAccess -Path $Path
        return Test-Path -LiteralPath $Path -ErrorAction Stop
    }
}

function Protect-ApiKeyFile {
    param([string]$Path)

    $AccessIdentities = Get-ApiKeyAccessIdentities
    & icacls.exe $Path /inheritance:r | Out-Null
    foreach ($IdentityName in $AccessIdentities) {
        $OwnerGrant = "{0}:F" -f $IdentityName
        & icacls.exe $Path /grant:r $OwnerGrant | Out-Null
    }
}

function Get-PithAppSandboxAccessIdentities {
    $IdentityNames = New-Object System.Collections.Generic.List[string]
    $CandidateNames = @(
        "CodexSandboxUsers",
        ("{0}\CodexSandboxUsers" -f $env:COMPUTERNAME)
    )

    foreach ($CandidateName in $CandidateNames) {
        if ([string]::IsNullOrWhiteSpace($CandidateName)) {
            continue
        }
        try {
            $Account = New-Object System.Security.Principal.NTAccount($CandidateName)
            [void]$Account.Translate([System.Security.Principal.SecurityIdentifier])
            $IdentityNames.Add($Account.Value)
        }
        catch {
            continue
        }
    }

    return @($IdentityNames | Select-Object -Unique)
}

function Grant-PithAppSandboxReadExecute {
    param([string]$TargetPath)

    if ([string]::IsNullOrWhiteSpace($TargetPath) -or -not (Test-Path -LiteralPath $TargetPath)) {
        return
    }

    $AccessIdentities = Get-PithAppSandboxAccessIdentities
    if (-not $AccessIdentities -or $AccessIdentities.Count -eq 0) {
        return
    }

    foreach ($IdentityName in $AccessIdentities) {
        try {
            if ([System.IO.Directory]::Exists($TargetPath)) {
                $Grant = "{0}:(OI)(CI)RX" -f $IdentityName
                & icacls.exe $TargetPath /inheritance:e /grant:r $Grant /T /C | Out-Null
            }
            elseif ([System.IO.File]::Exists($TargetPath)) {
                $Grant = "{0}:RX" -f $IdentityName
                & icacls.exe $TargetPath /inheritance:e /grant:r $Grant /C | Out-Null
            }
        }
        catch {
            Write-Warning "Could not grant app sandbox read/execute access to $TargetPath for $IdentityName"
        }
    }
}

function Repair-PithAppSandboxRuntimeAccess {
    $Targets = New-Object System.Collections.Generic.List[string]

    foreach ($Candidate in @($PithServerPath, $VenvPath, $PithRuntimeRoot, $PythonExe, $PythonPath)) {
        if (-not [string]::IsNullOrWhiteSpace([string]$Candidate)) {
            $Targets.Add([string]$Candidate)
        }
    }
    foreach ($Candidate in @($PythonPath, $PythonExe)) {
        if (-not [string]::IsNullOrWhiteSpace([string]$Candidate)) {
            $Parent = Split-Path -Parent $Candidate -ErrorAction SilentlyContinue
            if (-not [string]::IsNullOrWhiteSpace($Parent)) {
                $Targets.Add($Parent)
            }
        }
    }

    foreach ($Target in @($Targets | Select-Object -Unique)) {
        Grant-PithAppSandboxReadExecute -TargetPath $Target
    }
}

function Read-PithApiKeyFile {
    param([string]$Path)

    try {
        return (Get-Content -LiteralPath $Path -Raw).Trim()
    }
    catch {
        Write-Warning "Existing API key is not readable; repairing permissions and retrying"
        Repair-ApiKeyPathAccess -Path $Path
        return (Get-Content -LiteralPath $Path -Raw).Trim()
    }
}

function Write-PithApiKeyFile {
    param(
        [string]$Path,
        [string]$Value
    )

    try {
        Set-Content -LiteralPath $Path -Value $Value -NoNewline
    }
    catch {
        Write-Warning "Could not write API key file; repairing permissions and retrying"
        Repair-ApiKeyPathAccess -Path $Path
        Set-Content -LiteralPath $Path -Value $Value -NoNewline
    }
}

$ApiKey = $null
if (Test-ApiKeyFileExists -Path $ApiKeyFile) {
    $ApiKey = Read-PithApiKeyFile -Path $ApiKeyFile
}

if ([string]::IsNullOrWhiteSpace($ApiKey)) {
    $ApiKey = New-PithApiKey
    Write-PithApiKeyFile -Path $ApiKeyFile -Value $ApiKey
    Protect-ApiKeyFile -Path $ApiKeyFile
    Write-Success "Generated API key: $($ApiKey.Substring(0, 16))... (saved to $ApiKeyFile)"
}
else {
    Protect-ApiKeyFile -Path $ApiKeyFile
    Write-Success "API key already exists"
}

# Create .env file (F21)
$EnvFile = "$PithServerPath\.env"
if (-not $PithDataDirOverridden) {
    $ExistingPithDataDir = Get-PithEnvFileValue -EnvFile $EnvFile -Name "PITH_DATA_DIR"
    if ($ExistingPithDataDir) {
        $PithDataDir = $ExistingPithDataDir
    }
}
$EnvLines = @(
    "PITH_API_KEY=$ApiKey",
    "HOST=127.0.0.1",
    "PORT=$PithPort",
    "PITH_PORT=$PithPort",
    "PITH_DATA_DIR=$PithDataDir"
)

if (-not (Test-Path $EnvFile)) {
    $EnvLines | Set-Content -Path $EnvFile
    Write-Success "Created .env file"
}
else {
    $ExistingEnvLines = @(Get-Content -Path $EnvFile)
    $UpdatedEnvLines = New-Object System.Collections.Generic.List[string]
    $UpdatedApiKey = $false
    $UpdatedHost = $false
    $UpdatedPort = $false
    $UpdatedPithPort = $false
    $UpdatedDataDir = $false
    foreach ($Line in $ExistingEnvLines) {
        if ($Line -match '^PITH_API_KEY=') {
            $UpdatedEnvLines.Add("PITH_API_KEY=$ApiKey")
            $UpdatedApiKey = $true
        }
        elseif ($Line -match '^HOST=') {
            $UpdatedEnvLines.Add("HOST=127.0.0.1")
            $UpdatedHost = $true
        }
        elseif ($Line -match '^PORT=') {
            $UpdatedEnvLines.Add("PORT=$PithPort")
            $UpdatedPort = $true
        }
        elseif ($Line -match '^PITH_PORT=') {
            $UpdatedEnvLines.Add("PITH_PORT=$PithPort")
            $UpdatedPithPort = $true
        }
        elseif ($Line -match '^PITH_DATA_DIR=') {
            $UpdatedEnvLines.Add("PITH_DATA_DIR=$PithDataDir")
            $UpdatedDataDir = $true
        }
        else {
            $UpdatedEnvLines.Add($Line)
        }
    }
    if (-not $UpdatedApiKey) {
        $UpdatedEnvLines.Add("PITH_API_KEY=$ApiKey")
    }
    if (-not $UpdatedHost) {
        $UpdatedEnvLines.Add("HOST=127.0.0.1")
    }
    if (-not $UpdatedPort) {
        $UpdatedEnvLines.Add("PORT=$PithPort")
    }
    if (-not $UpdatedPithPort) {
        $UpdatedEnvLines.Add("PITH_PORT=$PithPort")
    }
    if (-not $UpdatedDataDir) {
        $UpdatedEnvLines.Add("PITH_DATA_DIR=$PithDataDir")
    }
    $UpdatedEnvLines | Set-Content -Path $EnvFile
    Write-Success "Persisted API key, port, and data directory in .env"
}

# Claude Code can keep MCP bridge subprocesses alive across reinstall and API-key refresh.
# Stop Pith-owned bridges now so the next client tool call uses the current credentials.
Stop-PithMcpBridgeProcessesForCleanup

Write-Host ""

# ============================================================================
# STEP 6: Configure MCP Clients
# ============================================================================
Write-Step 6 "Configure MCP clients using configure_clients.py"

# Try real configure_clients.py first (supports standard MCP clients including Codex)
$ConfigureScript = "$PithHome\pith-server\scripts\configure_clients.py"
$ClientConfigResultJson = "$PithHome\logs\configure-clients.json"
$ClientConfigResultErr = "$PithHome\logs\configure-clients.err"
$ClientReadinessIncomplete = $false
Repair-PithAppSandboxRuntimeAccess

if ($PithSelectedSurfaces -eq "none") {
    Write-Warning "AI app configuration skipped by surface selection; local service installation will continue"
}
elseif (Test-Path $ConfigureScript) {
    $ConfigureArgs = @(
        $ConfigureScript,
        "--server-path", "$PithHome\pith-server\pith_mcp.py",
        "--python-cmd", "$VenvPath\Scripts\python.exe",
        "--source-key-from-file", "$PithServerPath\.env",
        "--api-url", "http://localhost:$PithPort",
        "--pith-version", "$PithVersion",
        "--project-dir", "$PithHome\pith-server",
        "--platform", "windows",
        "--clients", "$PithSelectedSurfaces",
        "--skip-project",
        "--json"
    )
    if ($PithHomeOverridden) {
        $ConfigureArgs += "--allow-noncanonical-server"
    }
    & $PythonExe @ConfigureArgs > $ClientConfigResultJson 2> $ClientConfigResultErr
    $ConfigureExitCode = $LASTEXITCODE
    $ClientReadinessIncomplete = Show-ClientReadiness -JsonPath $ClientConfigResultJson
    Open-PithClaudeExtensionInstaller -JsonPath $ClientConfigResultJson
    if ($ConfigureExitCode -eq 0) {
        Write-Success "MCP client configuration completed"
        Write-Success "Client configuration diagnostics saved to $ClientConfigResultJson"
    }
    else {
        Write-Warning "MCP client configuration did not complete; local service installation will continue"
        Write-Warning "Client configuration diagnostics saved to $ClientConfigResultErr"
    }
}
else {
    Write-Warning "configure_clients.py not found; local service installation will continue without MCP client config"
}

Write-Host ""

# ============================================================================
# STEP 7: Agent cognitive loop instructions
# ============================================================================
Write-Step 7 "Agent cognitive loop instructions"

$SystemPromptPath = "$PithHome\SYSTEM_PROMPT.md"
$CursorGlobalRulePath = "$PithHome\CURSOR_GLOBAL_RULE.txt"

$SystemPromptContent = @'
# Pith - Cognitive Runtime for AI Agents

Pith is running as a local MCP server giving you persistent memory across
conversations. All Pith tools use the pith_ prefix, such as
pith_conversation_turn.

Disclosure: The user has installed Pith and consented to persistent memory.
Tool names are prefixed with pith_ to avoid naming conflicts with other MCPs.

## Cognitive Loop - Call on EVERY user message

Call pith_conversation_turn BEFORE composing any substantive response. This
retrieves context about the user, their projects, and prior sessions.

### Parameters
- message: the user's current message
- previous_message: the user's previous message after the first exchange
- previous_response: your last response, required after the first exchange
- extracted_concepts_json: 1-5 high-quality insights from your previous response

### Concept Format
```json
[{
  "summary": "30-500 chars - the insight, not the transcript",
  "confidence": 0.6,
  "knowledge_area": "domain",
  "evidence": ["source >= 10 chars"],
  "concept_type": "decision"
}]
```
concept_type must be: observation, pattern, decision, principle, method,
heuristic, or cognitive_strategy. If the exchange was trivial, send '[]' and do
not invent filler.

## Session Lifecycle
- pith_conversation_turn - Primary loop: call every exchange
- pith_session_end - Call when conversation ends, include previous_response
- pith_checkpoint - Save/resume state for long-running work when appropriate
- pith_session_learn - Explicit high-quality extraction when needed
- pith_search - Semantic search across knowledge base
- pith_reflect - Run consolidation/reflection cycle
- pith_orient - Situational awareness summary

## Fallback if MCP Transport Fails
If direct pith_ MCP tools return transport errors or time out, use the local CLI
fallback for the same lifecycle operation:

```powershell
%USERPROFILE%\.pith\bin\pith.cmd api conversation_turn --stdin-json
%USERPROFILE%\.pith\bin\pith.cmd api checkpoint --stdin-json
%USERPROFILE%\.pith\bin\pith.cmd api session_end --stdin-json
```

Send the JSON payload on stdin. The command may print a profile banner before
the payload; parse the last non-empty output line as JSON. For lifecycle
evidence reports, use pith api lifecycle_status --stdin-json. For cross-surface
source coverage evidence, use pith api surface_activity --stdin-json; treat this
as coverage evidence, not a semantic summary.
'@

Set-Content -Path $SystemPromptPath -Value $SystemPromptContent -Encoding UTF8
Write-Success "System prompt saved to $SystemPromptPath"

$CursorGlobalRuleContent = @'
Pith is installed locally as a cognitive runtime for this AI agent.

Before every substantive response, retrieve Pith context first. Prefer the Pith
MCP tool pith_conversation_turn. Send the user's current message. After the
first exchange, also send the previous user message, your previous response, and
extracted_concepts_json.

Use Pith's canonical concept format for extracted concepts: summary, confidence,
knowledge_area, evidence, and concept_type. If the exchange is trivial, send an
empty concept list [] instead of inventing filler.

If direct Pith MCP tools are unavailable or transport-broken, use the local API
command with JSON on stdin:

%USERPROFILE%\.pith\bin\pith.cmd api conversation_turn --stdin-json

For long-running work, use:

%USERPROFILE%\.pith\bin\pith.cmd api checkpoint --stdin-json

When a conversation ends, use:

%USERPROFILE%\.pith\bin\pith.cmd api session_end --stdin-json
'@

Set-Content -Path $CursorGlobalRulePath -Value $CursorGlobalRuleContent -Encoding UTF8
Write-Success "Cursor global rule saved to $CursorGlobalRulePath"

Write-Host ""

# ============================================================================
# STEP 8: Auto-start Setup
# ============================================================================
Write-Step 8 "Auto-start setup (Task Scheduler) and backup scheduler"

# Create pith CLI scripts first
$PithBatPath = "$PithHome\bin\pith.cmd"
$PithCmdContent = @"
@echo off
setlocal
set "PITH_HOME=$PithHome"
set "PITH_DATA_DIR=$PithDataDir"
set "VENV_PATH=$VenvPath"
if /I "%~1"=="start" goto pith_uninstall_guard
if /I "%~1"=="restart" goto pith_uninstall_guard
if /I "%~1"=="serve" goto pith_uninstall_guard
if /I "%~1"=="restore" goto pith_uninstall_guard
if /I "%~1"=="update" goto pith_uninstall_guard
goto pith_after_uninstall_guard
:pith_uninstall_guard
if exist "%PITH_HOME%\config\uninstalling" (
  echo Pith uninstall is pending or complete for %PITH_HOME%.
  echo Reinstall Pith before running 'pith %~1'.
  exit /b 1
)
if exist "%PITH_HOME%.uninstalling" (
  echo Pith uninstall is pending or complete for %PITH_HOME%.
  echo Reinstall Pith before running 'pith %~1'.
  exit /b 1
)
:pith_after_uninstall_guard
set "POWERSHELL_EXE=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"
if not exist "%POWERSHELL_EXE%" set "POWERSHELL_EXE=powershell.exe"
call "%VENV_PATH%\Scripts\activate.bat"
"%POWERSHELL_EXE%" -NoProfile -ExecutionPolicy Bypass -File "%PITH_HOME%\bin\pith-cli.ps1" %*
if errorlevel 1 exit /b 1
exit /b 0
"@
Set-Content -Path $PithBatPath -Value $PithCmdContent

# Create PowerShell CLI wrapper
# Load CLI wrapper from template file. Hosted installs run from a top-level
# install.ps1 after extracting the server under $PithServerPath, while local
# distribution installs may run from scripts\install.ps1.
$CliTemplateCandidates = @(
    (Join-Path $PithServerPath "scripts\templates\pith_cli.ps1"),
    (Join-Path $PSScriptRoot "templates\pith_cli.ps1"),
    (Join-Path (Split-Path -Parent $PSScriptRoot) "scripts\templates\pith_cli.ps1"),
    (Join-Path (Split-Path -Parent $PSScriptRoot) "templates\pith_cli.ps1")
)
$CliTemplatePath = $CliTemplateCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $CliTemplatePath) {
    Write-Error-Custom "CLI template not found. Checked: $($CliTemplateCandidates -join '; ')"
}
$PithPsContent = Get-Content -Path $CliTemplatePath -Raw
$PithPsContent = $PithPsContent -replace '__PITH_HOME__', $PithHome
$PithPsContent = $PithPsContent.Replace('__VENV_PATH__', $VenvPath)
$PithPsContent = $PithPsContent -replace '__PITH_VERSION__', $PithVersion


$LegacyPithPsPath = "$PithHome\bin\pith.ps1"
Remove-Item -LiteralPath $LegacyPithPsPath -Force -ErrorAction SilentlyContinue

$PithPsPath = "$PithHome\bin\pith-cli.ps1"
Set-Content -Path $PithPsPath -Value $PithPsContent

Write-Success "Created pith CLI wrapper"

# Create Task Scheduler entry for auto-start
Write-Host "Setting up Task Scheduler for auto-start..."
$TaskName = "Pith-Server"

# Create task trigger. Per-user installs mirror the macOS LaunchAgent behavior:
# start when the user logs in, without requiring elevation.
$CurrentUser = (whoami)
$ExpectedTaskCommand = "$PithHome\bin\pith.cmd"
$ExpectedTaskArguments = "start"
$FallbackTaskName = "Pith-Server-$($env:USERNAME)"
$PithAutoStartTaskName = $null
$PrimaryTaskRegistered = Register-PithAutoStartTask -TaskName $TaskName -ExpectedCommand $ExpectedTaskCommand -ExpectedArguments $ExpectedTaskArguments -CurrentUser $CurrentUser
if ($PrimaryTaskRegistered) {
    $PithAutoStartTaskName = $TaskName
    $ExistingFallbackTask = Get-ScheduledTask -TaskName $FallbackTaskName -ErrorAction SilentlyContinue
    if (
        $ExistingFallbackTask -and
        (Test-PithScheduledTaskIdentityMatchesInstall -Task $ExistingFallbackTask -ExpectedCommand $ExpectedTaskCommand -ExpectedArguments $ExpectedTaskArguments -CurrentUser $CurrentUser)
    ) {
        Unregister-ScheduledTask -TaskName $FallbackTaskName -Confirm:$false -ErrorAction SilentlyContinue
        if (Get-ScheduledTask -TaskName $FallbackTaskName -ErrorAction SilentlyContinue) {
            Write-Warning "Could not remove redundant Pith auto-start fallback task '$FallbackTaskName'"
        }
        else {
            Write-Success "Removed redundant Pith auto-start fallback task '$FallbackTaskName'"
        }
    }
}
else {
    Write-Warning "Primary Pith-Server task is unavailable; installing current-user fallback task '$FallbackTaskName'"
    $FallbackTaskRegistered = Register-PithAutoStartTask -TaskName $FallbackTaskName -ExpectedCommand $ExpectedTaskCommand -ExpectedArguments $ExpectedTaskArguments -CurrentUser $CurrentUser
    if ($FallbackTaskRegistered) {
        $PithAutoStartTaskName = $FallbackTaskName
        Write-Warning "Leaving inaccessible stale Pith-Server task in place; fallback task '$FallbackTaskName' provides current-user auto-start. Remove stale task later from elevated PowerShell with: schtasks /Delete /TN Pith-Server /F"
    }
    else {
        Write-Warning "Pith auto-start task was not registered. Pith can still run now, but will not auto-start on next login until the stale Pith-Server task is removed."
    }
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
    $BackupSettings = New-ScheduledTaskSettingsSet `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -ExecutionTimeLimit (New-TimeSpan -Hours 2)

    try {
        Register-ScheduledTask -TaskName "Pith-Backup-3h" `
            -Action $BackupAction `
            -Trigger $BackupTrigger `
            -Principal $BackupPrincipal `
            -Settings $BackupSettings `
            -Force -ErrorAction Stop | Out-Null
        $RegisteredBackupTask = Get-ScheduledTask -TaskName "Pith-Backup-3h" -ErrorAction SilentlyContinue
        if (-not $RegisteredBackupTask) {
            Write-Warning "Could not verify backup task after registration"
        }
        else {
            Write-Success "Scheduled backups every 3 hours (WAL-safe)"
        }
    }
    catch {
        Write-Warning "Could not register backup task: $($_.Exception.Message)"
    }
}

# Remove legacy daily backup task if exists
Unregister-ScheduledTask -TaskName "Pith-Daily-Backup" -Confirm:$false -ErrorAction SilentlyContinue

Write-Host ""

# ============================================================================
# STEP 9: Health Check
# ============================================================================
Write-Step 9 "Health check (90s timeout)"

Write-Host "Performing health check..."

function Test-PithInstallerConversationReady {
    param([int]$Port)

    try {
        $Response = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/readyz" -UseBasicParsing `
            -TimeoutSec 2 -ErrorAction Stop
        if ($Response.StatusCode -ne 200) { return $false }
        $Ready = $Response.Content | ConvertFrom-Json
        return (
            ([string]$Ready.process_state -eq "running") -and
            ([string]$Ready.write_state -eq "accepting") -and
            ([string]$Ready.retrieval_state -ne "recovering")
        )
    }
    catch {
        return $false
    }
}

# Pre-check: if something is already running on the selected port and healthy, skip
# Handles: (1) dev env where Docker Pith is on the selected port, (2) re-running installer
$ExistingHealthy = $false
$ExistingHealthy = Test-PithInstallerConversationReady -Port $PithPort

if ($ExistingHealthy) {
    Write-Success "Pith server already running on port $PithPort - conversation readiness passed"
} else {

$HealthCheckTimeout = $false

try {
    $HealthCheckPassed = $false
    New-Item -ItemType Directory -Path "$PithHome\logs" -Force | Out-Null
    $ServerLog = "$PithHome\logs\server.log"
    $ServerErrLog = "$PithHome\logs\server.err.log"
    $proc = $null
    if ($PithAutoStartTaskName) {
        Start-ScheduledTask -TaskName $PithAutoStartTaskName -ErrorAction Stop
        Write-Host "  Starting Pith through limited current-user task '$PithAutoStartTaskName'"
    }
    else {
        $proc = Start-PithDetachedPythonProcess `
            -PythonExe $PythonExe `
            -WorkingDirectory $PithServerPath `
            -Arguments @('-m', 'app.api.serve') `
            -StdoutPath $ServerLog `
            -StderrPath $ServerErrLog
        Set-Content -Path "$PithHome\pith.pid" -Value $proc.Id -Encoding ASCII
    }

    $Stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
    while ($Stopwatch.Elapsed.TotalSeconds -lt 90) {
        try {
            if ($proc -and $proc.HasExited) {
                break
            }
            if (Test-PithInstallerConversationReady -Port $PithPort) {
                Start-Sleep -Seconds 1
                if (-not (Test-PithInstallerConversationReady -Port $PithPort)) {
                    continue
                }
                Write-Success "Conversation readiness check passed"
                $StartedPid = if (Test-Path -LiteralPath "$PithHome\pith.pid") {
                    (Get-Content -LiteralPath "$PithHome\pith.pid" -Raw -ErrorAction SilentlyContinue).Trim()
                } elseif ($proc) { [string]$proc.Id } else { "unknown" }
                Write-Success "Pith server started (PID: $StartedPid)"
                $HealthCheckPassed = $true
                break
            }
        }
        catch {
            Start-Sleep -Milliseconds 500
        }
    }

    if (-not $HealthCheckPassed) {
        Write-Warning "Conversation readiness did not pass within 90 seconds"
        if ($proc -and $proc.HasExited) {
            Write-Warning "Pith server exited during startup. See $ServerErrLog"
            if (Test-Path $ServerErrLog) {
                Get-Content $ServerErrLog -Tail 20
            }
        }
    }
}
catch {
    Write-Warning "Health check did not complete (may complete on first run)"
}

}  # end of port pre-check else block

Write-Host ""

# Reconcile an existing, explicitly enrolled ChatGPT Secure MCP Tunnel only
# after the managed runtime and local API are ready. Enrollment credentials and
# tunnel creation remain user/OpenAI-owned.
if (Test-SurfaceSelected -SelectedSurfaces $PithSelectedSurfaces -Surface "chatgpt") {
    $TunnelReconciliationScript = "$PithServerPath\scripts\windows_tunnel_reconciliation.ps1"
    $TunnelReconciliationResultPath = "$PithHome\logs\chatgpt-tunnel-reconciliation.json"
    if (Test-Path -LiteralPath $TunnelReconciliationScript -PathType Leaf) {
        try {
            . $TunnelReconciliationScript
            $TunnelReconciliation = Invoke-PithTunnelReconciliation `
                -PithHome $PithHome `
                -VenvPath $VenvPath `
                -PithServerPath $PithServerPath `
                -PithPort ([int]$PithPort) `
                -CurrentUser (whoami) `
                -AllowLegacyTaskElevation
            $TunnelReconciliation | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $TunnelReconciliationResultPath -Encoding UTF8
            switch ([string]$TunnelReconciliation.state) {
                "ready" {
                    $script:ChatGptTunnelReady = $true
                    Write-Success "Repaired and started the enrolled ChatGPT Secure MCP Tunnel"
                    $NonChatGptIncomplete = @($script:ClientReadinessEntries | Where-Object {
                        ([string]$_.state -ne "ready") -and ([string]$_.client_id -ne "chatgpt")
                    })
                    if ($NonChatGptIncomplete.Count -eq 0) {
                        $ClientReadinessIncomplete = $false
                    }
                }
                "not_enrolled" {
                    Write-Host "  ChatGPT Secure MCP Tunnel is not enrolled; remote setup remains a manual account action."
                }
                "incomplete_enrollment" {
                    Write-Warning "ChatGPT tunnel enrollment is incomplete; missing components: $($TunnelReconciliation.missing -join ', ')"
                    $ClientReadinessIncomplete = $true
                }
                "legacy_task_elevation_required" {
                    Write-Warning "ChatGPT tunnel migration requires administrator approval to replace a legacy SYSTEM-owned task. Rerun the installer and approve the Windows elevation prompt."
                    $ClientReadinessIncomplete = $true
                }
                "legacy_task_elevation_failed" {
                    Write-Warning "ChatGPT tunnel migration could not replace the legacy task. Review $TunnelReconciliationResultPath and rerun the installer with administrator approval."
                    $ClientReadinessIncomplete = $true
                }
                default {
                    Write-Warning "ChatGPT tunnel reconciliation did not become ready (state: $($TunnelReconciliation.state)). Review $TunnelReconciliationResultPath"
                    $ClientReadinessIncomplete = $true
                }
            }
        }
        catch {
            Write-Warning "ChatGPT tunnel reconciliation failed: $($_.Exception.Message)"
            $ClientReadinessIncomplete = $true
        }
    }
    elseif (Test-Path -LiteralPath "$PithHome\tunnel") {
        Write-Warning "ChatGPT tunnel enrollment exists, but the reconciliation helper is missing from this release"
        $ClientReadinessIncomplete = $true
    }
}

Write-Host ""

# ============================================================================
# STEP 9b: Auto-configure PATH [FIX A1 - Windows equivalent]
# ============================================================================
$PathAdded = $false
$PithBinDir = "$PithHome\bin"

# Scrub stale Pith-owned shims from the current process and put this install first.
$env:PATH = Update-PithPathValue -PathValue $env:PATH -PithBinDir $PithBinDir -PrependPithBin:$true

# Remove Pith-owned PowerShell profile PATH edits from earlier installers.
# User PATH environment variables work without profile script execution.
if (Remove-PithProfilePathLines -PithBinDir $PithBinDir) {
    Write-Host "  [OK] Removed legacy Pith PATH entry from PowerShell profile" -ForegroundColor Green
}

# Add to User environment variable (persists across all terminals without profile scripts)
try {
    $UserPath = [Environment]::GetEnvironmentVariable("Path", "User")
    $NewUserPath = Update-PithPathValue -PathValue $UserPath -PithBinDir $PithBinDir -PrependPithBin:$true
    if ($NewUserPath -ne $UserPath) {
        [Environment]::SetEnvironmentVariable("Path", $NewUserPath, "User")
        Write-Host "  [OK] Added Pith to User PATH environment variable and removed stale Pith shims" -ForegroundColor Green
        $PathAdded = $true
    }
    else {
        Write-Host "  [OK] PATH already configured in User environment variable with Pith first" -ForegroundColor Green
        $PathAdded = $true
    }
} catch {
    Write-Warning "Could not update User PATH environment variable"
}

# ============================================================================
# Final Success Message
# ============================================================================
if (-not (Test-Path -LiteralPath $PithBatPath)) {
    Write-Error-Custom "Install verification failed: pith command was not created at $PithBatPath"
}

Write-Banner

Write-Host "[OK] Installation Complete!" -ForegroundColor Green
Write-Host ""
Write-Host "[OK] Core Pith service installed" -ForegroundColor Green
Write-Host ""
Write-Host "Pith is installed at: " -NoNewline
Write-Host "$PithHome" -ForegroundColor Cyan
Write-Host "Pith API URL: " -NoNewline
Write-Host "http://localhost:$PithPort" -ForegroundColor Cyan
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
Write-Host "  pith start       Start the Pith server"
Write-Host "  pith stop        Stop the server"
Write-Host "  pith restart     Restart the server"
Write-Host "  pith status      Check server status"
Write-Host "  pith logs        Tail server logs"
Write-Host "  pith backup      Create WAL-safe backup"
Write-Host "  pith restore     Restore from backup"
Write-Host "  pith update      Update deps + embeddings"
Write-Host "  pith version     Show version + capabilities"
Write-Host "  pith protocol    Show AI client setup instructions"
Write-Host "  pith maintenance run   Run maintenance cycle"
Write-Host "  pith maintenance status Show maintenance task status"
Write-Host "  pith uninstall   Remove Pith completely"
Write-Host ""

if ($ClientReadinessIncomplete) {
    Write-Warning "Some AI app surfaces require manual action before they are fully ready. Review the AI client readiness summary above."
    Write-Host ""
}

Write-Host "Next Steps:" -ForegroundColor Cyan
Show-RequiredClientSetup -SelectedSurfaces $PithSelectedSurfaces -PathAdded $PathAdded -PithHome $PithHome
Write-Host ""

exit 0
