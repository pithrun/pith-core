#Requires -Version 5.0

function ConvertTo-PithTunnelPath {
    param([Parameter(Mandatory = $true)][string]$Path)

    return ([System.IO.Path]::GetFullPath($Path)).Replace('\', '/')
}

function Test-PithTunnelRegularFile {
    param([Parameter(Mandatory = $true)][string]$Path)

    $Item = Get-Item -LiteralPath $Path -Force -ErrorAction SilentlyContinue
    if (-not $Item -or $Item.PSIsContainer) {
        return $false
    }
    return (($Item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -eq 0)
}

function Test-PithTunnelPathWithinRoot {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Root
    )

    $FullPath = [System.IO.Path]::GetFullPath($Path).TrimEnd('\')
    $FullRoot = [System.IO.Path]::GetFullPath($Root).TrimEnd('\')
    return (
        $FullPath.Equals($FullRoot, [System.StringComparison]::OrdinalIgnoreCase) -or
        $FullPath.StartsWith($FullRoot + '\', [System.StringComparison]::OrdinalIgnoreCase)
    )
}

function Test-PithTunnelPathChainSafe {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Root
    )

    if (-not (Test-PithTunnelPathWithinRoot -Path $Path -Root $Root)) { return $false }
    $FullRoot = [System.IO.Path]::GetFullPath($Root).TrimEnd('\')
    $CursorPath = [System.IO.Path]::GetFullPath($Path)
    if (-not (Test-Path -LiteralPath $CursorPath)) {
        $CursorPath = Split-Path -Parent $CursorPath
    }
    while ($CursorPath -and (Test-PithTunnelPathWithinRoot -Path $CursorPath -Root $FullRoot)) {
        $Item = Get-Item -LiteralPath $CursorPath -Force -ErrorAction SilentlyContinue
        if ($Item -and (($Item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0)) {
            return $false
        }
        if ([System.IO.Path]::GetFullPath($CursorPath).TrimEnd('\').Equals(
                $FullRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
            break
        }
        $CursorPath = Split-Path -Parent $CursorPath
    }
    return $true
}

function Get-PithTunnelEnrollmentState {
    param([Parameter(Mandatory = $true)][string]$PithHome)

    $TunnelRoot = Join-Path $PithHome "tunnel"
    $Required = [ordered]@{
        tunnel_client = Join-Path $TunnelRoot "client\tunnel-client.exe"
        profile = Join-Path $TunnelRoot "profiles\pith-windows.yaml"
        control_plane_key = Join-Path $TunnelRoot "control-plane.key"
    }
    if (-not (Test-Path -LiteralPath $TunnelRoot)) {
        return [PSCustomObject]@{
            state = "not_enrolled"
            tunnel_root = $TunnelRoot
            missing = @($Required.Keys)
        }
    }

    if (-not (Test-PithTunnelPathChainSafe -Path $TunnelRoot -Root $PithHome)) {
        return [PSCustomObject]@{
            state = "incomplete_enrollment"
            tunnel_root = $TunnelRoot
            missing = @("unsafe_tunnel_root")
        }
    }

    $Missing = New-Object System.Collections.Generic.List[string]
    foreach ($Name in $Required.Keys) {
        if ((-not (Test-PithTunnelRegularFile -Path $Required[$Name])) -or
            (-not (Test-PithTunnelPathChainSafe -Path $Required[$Name] -Root $TunnelRoot))) {
            $Missing.Add($Name) | Out-Null
        }
    }

    return [PSCustomObject]@{
        state = if ($Missing.Count -eq 0) { "repairable" } else { "incomplete_enrollment" }
        tunnel_root = $TunnelRoot
        missing = @($Missing)
    }
}

function ConvertTo-PithTunnelYamlString {
    param([Parameter(Mandatory = $true)][string]$Value)

    return "'" + $Value.Replace("'", "''") + "'"
}

function Update-PithTunnelProfile {
    param(
        [Parameter(Mandatory = $true)][string]$ProfilePath,
        [Parameter(Mandatory = $true)][string]$PythonExe,
        [Parameter(Mandatory = $true)][string]$McpServerPath,
        [Parameter(Mandatory = $true)][string]$ControlPlaneKeyPath,
        [Parameter(Mandatory = $true)][string]$TunnelRoot
    )

    foreach ($TunnelPath in @($ProfilePath, $ControlPlaneKeyPath)) {
        if (-not (Test-PithTunnelPathChainSafe -Path $TunnelPath -Root $TunnelRoot)) {
            throw "Tunnel reconciliation rejected an unsafe or out-of-root path: $TunnelPath"
        }
    }
    foreach ($RequiredPath in @($ProfilePath, $PythonExe, $McpServerPath, $ControlPlaneKeyPath)) {
        if (-not (Test-PithTunnelRegularFile -Path $RequiredPath)) {
            throw "Tunnel reconciliation requires a regular file: $RequiredPath"
        }
    }

    $Lines = @(Get-Content -LiteralPath $ProfilePath -ErrorAction Stop)
    $CommandIndexes = New-Object System.Collections.Generic.List[int]
    $KeyIndexes = New-Object System.Collections.Generic.List[int]
    $TunnelIds = New-Object System.Collections.Generic.List[string]
    for ($Index = 0; $Index -lt $Lines.Count; $Index++) {
        $Line = [string]$Lines[$Index]
        if ($Line -match '^\s*command\s*:') {
            $CommandIndexes.Add($Index) | Out-Null
        }
        elseif ($Line -match '^\s*api_key\s*:') {
            $KeyIndexes.Add($Index) | Out-Null
        }
        elseif ($Line -match '^\s*tunnel_id\s*:\s*["'']?([^\s"''#]+)') {
            $TunnelIds.Add([string]$Matches[1]) | Out-Null
        }
    }

    if ($TunnelIds.Count -ne 1 -or [string]::IsNullOrWhiteSpace($TunnelIds[0])) {
        throw "Tunnel profile must contain exactly one nonempty tunnel_id field."
    }
    if ($CommandIndexes.Count -ne 1) {
        throw "Tunnel profile must contain exactly one MCP command field."
    }
    if ($KeyIndexes.Count -ne 1) {
        throw "Tunnel profile must contain exactly one control-plane api_key field."
    }

    $PythonPath = ConvertTo-PithTunnelPath -Path $PythonExe
    $ServerPath = ConvertTo-PithTunnelPath -Path $McpServerPath
    $KeyPath = ConvertTo-PithTunnelPath -Path $ControlPlaneKeyPath
    $CommandValue = ConvertTo-PithTunnelYamlString -Value ('"{0}" "{1}"' -f $PythonPath, $ServerPath)

    $CommandIndent = ([regex]::Match([string]$Lines[$CommandIndexes[0]], '^\s*')).Value
    $KeyIndent = ([regex]::Match([string]$Lines[$KeyIndexes[0]], '^\s*')).Value
    $Lines[$CommandIndexes[0]] = "$CommandIndent" + "command: $CommandValue"
    $Lines[$KeyIndexes[0]] = "$KeyIndent" + 'api_key: "file:' + $KeyPath + '"'

    $ProfileDir = Split-Path -Parent $ProfilePath
    $TempPath = Join-Path $ProfileDir ("pith-windows.yaml.tmp-{0}" -f [guid]::NewGuid().ToString("N"))
    $BackupPath = Join-Path $ProfileDir ("pith-windows.yaml.backup-{0}" -f (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssfffZ"))
    try {
        Set-Content -LiteralPath $TempPath -Value $Lines -Encoding UTF8 -ErrorAction Stop
        [System.IO.File]::Replace($TempPath, $ProfilePath, $BackupPath, $true)
    }
    finally {
        Remove-Item -LiteralPath $TempPath -Force -ErrorAction SilentlyContinue
    }

    return [PSCustomObject]@{
        tunnel_id = $TunnelIds[0]
        profile_path = $ProfilePath
        backup_path = $BackupPath
        command_python = $PythonExe
        command_server = $McpServerPath
    }
}

function Write-PithTunnelLauncher {
    param(
        [Parameter(Mandatory = $true)][string]$PithHome,
        [Parameter(Mandatory = $true)][int]$PithPort
    )

    $TunnelRoot = Join-Path $PithHome "tunnel"
    $LauncherPath = Join-Path $TunnelRoot "START-PITH-TUNNEL.ps1"
    if (-not (Test-PithTunnelPathChainSafe -Path $LauncherPath -Root $TunnelRoot)) {
        throw "Tunnel launcher path is unsafe or outside the enrolled tunnel root."
    }
    $EscapedPithHome = $PithHome.Replace("'", "''")
    $LauncherTemplate = @'
param(
    [ValidateSet("doctor", "run")]
    [string]$Mode = "run"
)

$ErrorActionPreference = "Stop"
$PithHome = '__PITH_HOME__'
$TunnelRoot = Join-Path $PithHome "tunnel"
$PithEnvPath = Join-Path $PithHome "pith-server\.env"
$TunnelClient = Join-Path $TunnelRoot "client\tunnel-client.exe"
$ProfileFile = Join-Path $TunnelRoot "profiles\pith-windows.yaml"

if (
    (Test-Path -LiteralPath (Join-Path $PithHome "config\uninstalling")) -or
    (Test-Path -LiteralPath "$PithHome.uninstalling")
) {
    Write-Error "Pith uninstall is pending or complete; refusing to start the ChatGPT tunnel."
    exit 1
}

if (-not (Test-Path -LiteralPath $PithEnvPath -PathType Leaf)) {
    throw "Pith environment file not found: $PithEnvPath"
}
foreach ($Line in Get-Content -LiteralPath $PithEnvPath) {
    $Trimmed = ([string]$Line).Trim()
    if (-not $Trimmed -or $Trimmed.StartsWith("#") -or -not $Trimmed.Contains("=")) { continue }
    $Parts = $Trimmed.Split("=", 2)
    $Name = $Parts[0].Trim()
    if ($Name -in @("PITH_API_KEY", "PITH_DATA_DIR")) {
        Set-Item -Path "Env:$Name" -Value $Parts[1].Trim().Trim('"').Trim("'")
    }
}
if (-not $env:PITH_API_KEY) { throw "PITH_API_KEY is missing from the installed Pith environment." }

$env:PITH_API_URL = "http://127.0.0.1:__PITH_PORT__"
$env:PITH_SURFACE_ID = "chatgpt_tunnel"
if ($Mode -eq "doctor") {
    & $TunnelClient doctor --profile-file $ProfileFile --explain --json
    exit $LASTEXITCODE
}

Remove-Item -LiteralPath (Join-Path $TunnelRoot "health.url") -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath (Join-Path $TunnelRoot "tunnel-client.pid") -Force -ErrorAction SilentlyContinue
& $TunnelClient run --profile-file $ProfileFile --pid.file (Join-Path $TunnelRoot "tunnel-client.pid") --health.url-file (Join-Path $TunnelRoot "health.url") --log.file (Join-Path $TunnelRoot "tunnel-client.log") --log.format json
exit $LASTEXITCODE
'@
    $LauncherContent = $LauncherTemplate.Replace('__PITH_HOME__', $EscapedPithHome).Replace('__PITH_PORT__', [string]$PithPort)
    Set-Content -LiteralPath $LauncherPath -Value $LauncherContent -Encoding UTF8 -ErrorAction Stop
    return $LauncherPath
}

function Protect-PithTunnelControlPlaneKey {
    param([Parameter(Mandatory = $true)][string]$Path)

    $CurrentUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
    & icacls.exe $Path /inheritance:r | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Could not disable inherited ACLs on the ChatGPT tunnel control-plane key."
    }
    & icacls.exe $Path /grant:r ("{0}:F" -f $CurrentUser) | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Could not restrict the ChatGPT tunnel control-plane key ACL."
    }
}

function Test-PithTunnelTask {
    param(
        [object]$Task,
        [Parameter(Mandatory = $true)][string]$PowerShellExe,
        [Parameter(Mandatory = $true)][string]$Arguments,
        [Parameter(Mandatory = $true)][string]$CurrentUser
    )

    if (-not $Task) { return $false }
    $Action = @($Task.Actions) | Select-Object -First 1
    $Triggers = @($Task.Triggers)
    if (-not $Action -or $Triggers.Count -ne 1) { return $false }
    $ActualExecute = ([Environment]::ExpandEnvironmentVariables([string]$Action.Execute)).Trim('"')
    $ExpectedExecute = ([Environment]::ExpandEnvironmentVariables($PowerShellExe)).Trim('"')
    $ActualUser = ([string]$Task.Principal.UserId).ToLowerInvariant()
    $ExpectedUsers = @(Get-PithTunnelExpectedTaskUsers -CurrentUser $CurrentUser)
    return (
        ($ActualExecute -ieq $ExpectedExecute) -and
        (([string]$Action.Arguments).Trim() -ieq $Arguments.Trim()) -and
        ($ExpectedUsers -contains $ActualUser) -and
        ([string]$Triggers[0].CimClass.CimClassName -eq "MSFT_TaskLogonTrigger") -and
        (-not [bool]$Task.Settings.DisallowStartIfOnBatteries) -and
        (-not [bool]$Task.Settings.StopIfGoingOnBatteries) -and
        ([int]$Task.Settings.RestartCount -ge 3)
    )
}

function Get-PithTunnelExpectedTaskUsers {
    param([Parameter(Mandatory = $true)][string]$CurrentUser)

    return @($CurrentUser, $env:USERNAME, "$env:USERDOMAIN\$env:USERNAME", "$env:COMPUTERNAME\$env:USERNAME") |
        Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) } |
        ForEach-Object { ([string]$_).ToLowerInvariant() } |
        Select-Object -Unique
}

function Get-PithTunnelTaskMigrationDisposition {
    param(
        [object]$Task,
        [Parameter(Mandatory = $true)][string]$CurrentUser
    )

    if (-not $Task) { return "absent" }
    $ActualUser = ([string]$Task.Principal.UserId).ToLowerInvariant()
    if (@(Get-PithTunnelExpectedTaskUsers -CurrentUser $CurrentUser) -contains $ActualUser) {
        return "current_user"
    }
    return "legacy_other_principal"
}

function Test-PithTunnelAccessDeniedError {
    param([Parameter(Mandatory = $true)][System.Management.Automation.ErrorRecord]$ErrorRecord)

    $Exception = $ErrorRecord.Exception
    while ($Exception) {
        if (($Exception -is [System.UnauthorizedAccessException]) -or
            ($Exception.HResult -eq -2147024891)) {
            return $true
        }
        $Exception = $Exception.InnerException
    }
    return (
        ([string]$ErrorRecord.FullyQualifiedErrorId -match 'AccessDenied|Unauthorized') -or
        ([string]$ErrorRecord.Exception.Message -match 'Access is denied|0x80070005')
    )
}

function Invoke-PithLegacyTunnelTaskElevatedDeletion {
    $Result = [ordered]@{
        state = "legacy_task_elevation_failed"
        principal = "hidden_or_other_principal"
        elevation_attempted = $true
        removed = $false
        error = $null
    }
    $SchtasksExe = Join-Path $env:SystemRoot "System32\schtasks.exe"
    try {
        $Elevated = Start-Process -FilePath $SchtasksExe -Verb RunAs `
            -ArgumentList @("/Delete", "/TN", "Pith-OpenAI-Tunnel", "/F") `
            -Wait -PassThru -ErrorAction Stop
        if ($Elevated.ExitCode -ne 0) {
            throw "Elevated task deletion exited with code $($Elevated.ExitCode)."
        }
        if (Get-ScheduledTask -TaskName "Pith-OpenAI-Tunnel" -ErrorAction SilentlyContinue) {
            throw "The legacy ChatGPT tunnel task is still registered after elevated deletion."
        }
        $Result.state = "legacy_task_removed"
        $Result.removed = $true
    }
    catch {
        $Result.error = $_.Exception.Message
    }
    return [PSCustomObject]$Result
}

function Remove-PithTunnelTaskForReconciliation {
    param(
        [Parameter(Mandatory = $true)][string]$CurrentUser,
        [switch]$AllowLegacyTaskElevation
    )

    $TaskName = "Pith-OpenAI-Tunnel"
    $Existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    $Disposition = Get-PithTunnelTaskMigrationDisposition -Task $Existing -CurrentUser $CurrentUser
    $Result = [ordered]@{
        state = $Disposition
        principal = if ($Existing) { [string]$Existing.Principal.UserId } else { $null }
        elevation_attempted = $false
        removed = $false
        error = $null
    }
    if (-not $Existing) { return [PSCustomObject]$Result }

    if ($Disposition -eq "current_user") {
        try {
            Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
            Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction Stop
            $Result.removed = $true
            return [PSCustomObject]$Result
        }
        catch {
            if (-not (Test-PithTunnelAccessDeniedError -ErrorRecord $_)) { throw }
            if (-not $AllowLegacyTaskElevation) {
                $Result.state = "legacy_task_elevation_required"
                $Result.error = $_.Exception.Message
                return [PSCustomObject]$Result
            }
            return Invoke-PithLegacyTunnelTaskElevatedDeletion
        }
    }

    if (-not $AllowLegacyTaskElevation) {
        $Result.state = "legacy_task_elevation_required"
        return [PSCustomObject]$Result
    }

    return Invoke-PithLegacyTunnelTaskElevatedDeletion
}

function Stop-PithTunnelProcessForReconciliation {
    param([Parameter(Mandatory = $true)][string]$TunnelRoot)

    $PidFile = Join-Path $TunnelRoot "tunnel-client.pid"
    if ((-not (Test-PithTunnelRegularFile -Path $PidFile)) -or
        (-not (Test-PithTunnelPathChainSafe -Path $PidFile -Root $TunnelRoot))) {
        return $false
    }

    $TunnelPid = 0
    $PidText = (Get-Content -LiteralPath $PidFile -Raw -ErrorAction SilentlyContinue).Trim()
    if ((-not [int]::TryParse($PidText, [ref]$TunnelPid)) -or $TunnelPid -le 0) {
        return $false
    }
    $TunnelProcess = Get-Process -Id $TunnelPid -ErrorAction SilentlyContinue
    if (-not $TunnelProcess) { return $false }
    if ([string]$TunnelProcess.ProcessName -notlike "tunnel-client*") {
        throw "Tunnel reconciliation refused to stop PID $TunnelPid because it is not a tunnel-client process."
    }
    Stop-Process -Id $TunnelPid -Force -ErrorAction Stop
    Remove-Item -LiteralPath $PidFile -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath (Join-Path $TunnelRoot "health.url") -Force -ErrorAction SilentlyContinue
    return $true
}

function Register-PithTunnelTask {
    param(
        [Parameter(Mandatory = $true)][string]$LauncherPath,
        [Parameter(Mandatory = $true)][string]$CurrentUser
    )

    $TaskName = "Pith-OpenAI-Tunnel"
    $PowerShellExe = "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe"
    if (-not (Test-Path -LiteralPath $PowerShellExe -PathType Leaf)) { $PowerShellExe = "powershell.exe" }
    $Arguments = "-NoProfile -ExecutionPolicy Bypass -File `"$LauncherPath`" -Mode run"
    $Action = New-ScheduledTaskAction -Execute $PowerShellExe -Argument $Arguments
    $Trigger = New-ScheduledTaskTrigger -AtLogOn -User $CurrentUser
    $Principal = New-ScheduledTaskPrincipal -UserId $CurrentUser -LogonType Interactive -RunLevel Limited
    $Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
        -ExecutionTimeLimit ([TimeSpan]::Zero) -StartWhenAvailable -RestartCount 5 `
        -RestartInterval (New-TimeSpan -Minutes 1)
    Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Principal $Principal `
        -Settings $Settings -Force -ErrorAction Stop | Out-Null
    $Registered = Get-ScheduledTask -TaskName $TaskName -ErrorAction Stop
    if (-not (Test-PithTunnelTask -Task $Registered -PowerShellExe $PowerShellExe -Arguments $Arguments -CurrentUser $CurrentUser)) {
        throw "Registered ChatGPT tunnel task does not match the current user or launcher."
    }
    Start-ScheduledTask -TaskName $TaskName -ErrorAction Stop
    return $Registered
}

function Wait-PithTunnelHealth {
    param(
        [Parameter(Mandatory = $true)][string]$TunnelRoot,
        [int]$TimeoutSeconds = 30
    )

    $HealthUrlFile = Join-Path $TunnelRoot "health.url"
    $PidFile = Join-Path $TunnelRoot "tunnel-client.pid"
    $Deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    do {
        $RuntimeFilesSafe = (
            (Test-PithTunnelRegularFile -Path $HealthUrlFile) -and
            (Test-PithTunnelRegularFile -Path $PidFile) -and
            (Test-PithTunnelPathChainSafe -Path $HealthUrlFile -Root $TunnelRoot) -and
            (Test-PithTunnelPathChainSafe -Path $PidFile -Root $TunnelRoot)
        )
        if ($RuntimeFilesSafe) {
            $TunnelPid = 0
            $PidText = (Get-Content -LiteralPath $PidFile -Raw -ErrorAction SilentlyContinue).Trim()
            $PidValid = [int]::TryParse($PidText, [ref]$TunnelPid)
            $TunnelProcess = if ($PidValid -and $TunnelPid -gt 0) {
                Get-Process -Id $TunnelPid -ErrorAction SilentlyContinue
            }
            else { $null }
            $BaseUrl = (Get-Content -LiteralPath $HealthUrlFile -Raw -ErrorAction SilentlyContinue).Trim().TrimEnd('/')
            if ($TunnelProcess -and ([string]$TunnelProcess.ProcessName -like "tunnel-client*") -and
                ($BaseUrl -match '^http://127\.0\.0\.1:\d+$')) {
                try {
                    $Response = Invoke-WebRequest -UseBasicParsing -Uri "$BaseUrl/healthz" -TimeoutSec 2 -ErrorAction Stop
                    if ($Response.StatusCode -eq 200) {
                        return [PSCustomObject]@{ url = $BaseUrl; process_id = $TunnelPid }
                    }
                }
                catch { }
            }
        }
        Start-Sleep -Milliseconds 500
    } while ((Get-Date) -lt $Deadline)
    return $null
}

function Invoke-PithTunnelReconciliation {
    param(
        [Parameter(Mandatory = $true)][string]$PithHome,
        [Parameter(Mandatory = $true)][string]$VenvPath,
        [Parameter(Mandatory = $true)][string]$PithServerPath,
        [Parameter(Mandatory = $true)][int]$PithPort,
        [Parameter(Mandatory = $true)][string]$CurrentUser,
        [int]$HealthTimeoutSeconds = 30,
        [switch]$SkipTaskRegistration,
        [switch]$AllowLegacyTaskElevation
    )

    $Enrollment = Get-PithTunnelEnrollmentState -PithHome $PithHome
    $Result = [ordered]@{
        schema_version = "pith_windows_tunnel_reconciliation.v2"
        state = [string]$Enrollment.state
        missing = @($Enrollment.missing)
        doctor_exit_code = $null
        task_registered = $false
        task_identity_valid = $false
        health_reachable = $false
        health_url = $null
        tunnel_process_id = $null
        task_migration_state = "not_checked"
        task_migration_principal = $null
        task_migration_elevation_attempted = $false
        error_code = $null
        error = $null
    }
    if ($Enrollment.state -ne "repairable") { return [PSCustomObject]$Result }

    $TunnelRoot = [string]$Enrollment.tunnel_root
    $TunnelClient = Join-Path $TunnelRoot "client\tunnel-client.exe"
    $ProfilePath = Join-Path $TunnelRoot "profiles\pith-windows.yaml"
    $ControlPlaneKey = Join-Path $TunnelRoot "control-plane.key"
    $PythonExe = Join-Path $VenvPath "Scripts\python.exe"
    $McpServerPath = Join-Path $PithServerPath "pith_mcp.py"
    try {
        [void](Update-PithTunnelProfile -ProfilePath $ProfilePath -PythonExe $PythonExe `
            -McpServerPath $McpServerPath -ControlPlaneKeyPath $ControlPlaneKey -TunnelRoot $TunnelRoot)
        Protect-PithTunnelControlPlaneKey -Path $ControlPlaneKey
        $LauncherPath = Write-PithTunnelLauncher -PithHome $PithHome -PithPort $PithPort
        if (-not $SkipTaskRegistration) {
            $Migration = Remove-PithTunnelTaskForReconciliation -CurrentUser $CurrentUser `
                -AllowLegacyTaskElevation:$AllowLegacyTaskElevation
            $Result.task_migration_state = [string]$Migration.state
            $Result.task_migration_principal = $Migration.principal
            $Result.task_migration_elevation_attempted = [bool]$Migration.elevation_attempted
            if ($Migration.state -eq "legacy_task_elevation_required") {
                $Result.state = "legacy_task_elevation_required"
                $Result.error_code = "LEGACY_TUNNEL_TASK_ELEVATION_REQUIRED"
                $Result.error = "The existing ChatGPT tunnel task belongs to another principal and must be removed with administrator approval."
                return [PSCustomObject]$Result
            }
            if ($Migration.state -eq "legacy_task_elevation_failed") {
                $Result.state = "legacy_task_elevation_failed"
                $Result.error_code = "LEGACY_TUNNEL_TASK_ELEVATION_FAILED"
                $Result.error = [string]$Migration.error
                return [PSCustomObject]$Result
            }
            [void](Stop-PithTunnelProcessForReconciliation -TunnelRoot $TunnelRoot)
        }
        $DoctorLog = Join-Path $TunnelRoot "doctor.json"
        $DoctorErrorLog = Join-Path $TunnelRoot "doctor.err.log"
        & $TunnelClient doctor --profile-file $ProfilePath --explain --json > $DoctorLog 2> $DoctorErrorLog
        $Result.doctor_exit_code = if ($null -eq $LASTEXITCODE) { 0 } else { $LASTEXITCODE }
        if ($Result.doctor_exit_code -ne 0) {
            $Result.state = "doctor_failed"
            return [PSCustomObject]$Result
        }
        if ($SkipTaskRegistration) {
            $Result.state = "profile_reconciled"
            return [PSCustomObject]$Result
        }

        try {
            $Task = Register-PithTunnelTask -LauncherPath $LauncherPath -CurrentUser $CurrentUser
        }
        catch {
            if (-not (Test-PithTunnelAccessDeniedError -ErrorRecord $_)) { throw }
            $Result.task_migration_state = "hidden_legacy_task_detected"
            $Result.task_migration_principal = "hidden_or_other_principal"
            if (-not $AllowLegacyTaskElevation) {
                $Result.state = "legacy_task_elevation_required"
                $Result.error_code = "LEGACY_TUNNEL_TASK_ELEVATION_REQUIRED"
                $Result.error = "A hidden or privileged ChatGPT tunnel task blocks current-user registration and must be removed with administrator approval."
                return [PSCustomObject]$Result
            }
            $HiddenMigration = Invoke-PithLegacyTunnelTaskElevatedDeletion
            $Result.task_migration_state = [string]$HiddenMigration.state
            $Result.task_migration_principal = $HiddenMigration.principal
            $Result.task_migration_elevation_attempted = [bool]$HiddenMigration.elevation_attempted
            if (-not $HiddenMigration.removed) {
                $Result.state = "legacy_task_elevation_failed"
                $Result.error_code = "LEGACY_TUNNEL_TASK_ELEVATION_FAILED"
                $Result.error = [string]$HiddenMigration.error
                return [PSCustomObject]$Result
            }
            $Task = Register-PithTunnelTask -LauncherPath $LauncherPath -CurrentUser $CurrentUser
        }
        $Result.task_registered = $true
        $PowerShellExe = "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe"
        if (-not (Test-Path -LiteralPath $PowerShellExe -PathType Leaf)) { $PowerShellExe = "powershell.exe" }
        $TaskArguments = "-NoProfile -ExecutionPolicy Bypass -File `"$LauncherPath`" -Mode run"
        $Result.task_identity_valid = Test-PithTunnelTask -Task $Task -PowerShellExe $PowerShellExe `
            -Arguments $TaskArguments -CurrentUser $CurrentUser
        $HealthProof = Wait-PithTunnelHealth -TunnelRoot $TunnelRoot -TimeoutSeconds $HealthTimeoutSeconds
        if ($HealthProof) {
            $Result.health_reachable = $true
            $Result.health_url = [string]$HealthProof.url
            $Result.tunnel_process_id = [int]$HealthProof.process_id
            $Result.state = "ready"
        }
        else {
            $Result.state = "health_unreachable"
        }
    }
    catch {
        $Result.state = "repair_failed"
        $Result.error = $_.Exception.Message
    }
    return [PSCustomObject]$Result
}
