# Shared Windows embedding runtime setup for fresh install and update paths.

function Set-PithEmbeddingCapability {
    param(
        [string]$PithHome,
        [string]$Content
    )

    $Content | Out-File "$PithHome\.install_capabilities"
}

function ConvertTo-PithWindowsCommandLineArgument {
    param([AllowEmptyString()][string]$Argument)

    if ($Argument.Length -gt 0 -and $Argument -notmatch '[\s"]') {
        return $Argument
    }

    $Builder = New-Object System.Text.StringBuilder
    $Backslash = [char]92
    $Quote = [char]34
    $BackslashCount = 0
    [void]$Builder.Append($Quote)
    foreach ($Character in $Argument.ToCharArray()) {
        if ($Character -eq $Backslash) {
            $BackslashCount += 1
            continue
        }
        if ($Character -eq $Quote) {
            [void]$Builder.Append($Backslash, ($BackslashCount * 2) + 1)
            [void]$Builder.Append($Quote)
        }
        else {
            if ($BackslashCount -gt 0) {
                [void]$Builder.Append($Backslash, $BackslashCount)
            }
            [void]$Builder.Append($Character)
        }
        $BackslashCount = 0
    }
    if ($BackslashCount -gt 0) {
        [void]$Builder.Append($Backslash, $BackslashCount * 2)
    }
    [void]$Builder.Append($Quote)
    return $Builder.ToString()
}

function Stop-PithEmbeddingProcessTree {
    param([System.Diagnostics.Process]$Process)

    $ErrorActionPreference = 'Stop'
    # A finished launcher cannot be used as taskkill /T's root. Discover its
    # surviving descendants while retaining the owned parent process handle.
    $InventoryError = $null
    $Processes = @()
    try {
        $Processes = @(Get-CimInstance Win32_Process -OperationTimeoutSec 2 -ErrorAction Stop)
    }
    catch {
        # Still attempt taskkill /T against a live parent when CIM is unavailable.
        $InventoryError = $_.Exception.Message
    }
    $TreeIds = New-Object 'System.Collections.Generic.HashSet[int]'
    [void]$TreeIds.Add($Process.Id)
    do {
        $Added = $false
        foreach ($Candidate in $Processes) {
            if ($TreeIds.Contains([int]$Candidate.ParentProcessId) -and
                $Candidate.CreationDate -ge $Process.StartTime -and
                $TreeIds.Add([int]$Candidate.ProcessId)) {
                $Added = $true
            }
        }
    } while ($Added)

    $KillArguments = @('/T', '/F')
    foreach ($TreeId in $TreeIds) {
        if (($TreeId -ne $Process.Id) -or -not $Process.HasExited) {
            $KillArguments += @('/PID', [string]$TreeId)
        }
    }
    if ($KillArguments.Count -eq 2) {
        if ($InventoryError) { throw "Descendant inventory failed: $InventoryError" }
        return
    }

    $Killer = New-Object System.Diagnostics.Process
    $Killer.StartInfo.FileName = "$env:SystemRoot\System32\taskkill.exe"
    $Killer.StartInfo.Arguments = $KillArguments -join ' '
    $Killer.StartInfo.UseShellExecute = $false
    $Killer.StartInfo.CreateNoWindow = $true
    try {
        [void]$Killer.Start()
        if (-not $Killer.WaitForExit(5000)) {
            $Killer.Kill()
            [void]$Killer.WaitForExit(1000)
            throw 'Process-tree cleanup timed out.'
        }
        $Survivors = @(Get-Process -Id ([int[]]@($TreeIds)) -ErrorAction SilentlyContinue)
        if ($null -eq $Killer.ExitCode -or $Survivors.Count -gt 0) {
            throw "Process-tree cleanup failed (taskkill exit $($Killer.ExitCode))."
        }
        if ($InventoryError) { throw "Descendant inventory failed: $InventoryError" }
    }
    finally {
        $Killer.Dispose()
    }
}

function Invoke-PithEmbeddingCommand {
    param(
        [string]$PithHome,
        [string]$EmbedLog,
        [string]$FilePath,
        [string[]]$Arguments,
        [string]$Name,
        [int]$TimeoutSeconds = 900
    )

    $ErrorActionPreference = 'Stop'
    $ExitCode = $null
    $OutLog = "$PithHome\logs\$Name.stdout.log"
    $ErrLog = "$PithHome\logs\$Name.stderr.log"
    Remove-Item -Path $OutLog, $ErrLog -Force -ErrorAction SilentlyContinue

    $ArgumentLine = (@($Arguments | ForEach-Object { ConvertTo-PithWindowsCommandLineArgument -Argument $_ }) -join " ")
    # Own the process handle: Windows PowerShell Start-Process can lose ExitCode
    # after a timed wait. Casting that null to int would silently report success.
    $Process = New-Object System.Diagnostics.Process
    $Process.StartInfo.FileName = $FilePath
    $Process.StartInfo.Arguments = $ArgumentLine
    $Process.StartInfo.UseShellExecute = $false
    $Process.StartInfo.CreateNoWindow = $true
    $Process.StartInfo.RedirectStandardOutput = $true
    $Process.StartInfo.RedirectStandardError = $true
    $OutStream = $null
    $ErrStream = $null
    $Started = $false
    $CleanupAttempted = $false
    try {
        # Copy bytes as they arrive, not only after EOF. A descendant can keep
        # inherited pipes open after the owned process has already exited.
        $OutStream = [System.IO.FileStream]::new($OutLog, [System.IO.FileMode]::Create, [System.IO.FileAccess]::Write, [System.IO.FileShare]::Read, 1)
        $ErrStream = [System.IO.FileStream]::new($ErrLog, [System.IO.FileMode]::Create, [System.IO.FileAccess]::Write, [System.IO.FileShare]::Read, 1)
        [void]$Process.Start()
        $Started = $true
        $StdoutRead = $Process.StandardOutput.BaseStream.CopyToAsync($OutStream)
        $StderrRead = $Process.StandardError.BaseStream.CopyToAsync($ErrStream)
        $Completed = $Process.WaitForExit($TimeoutSeconds * 1000)
        if (-not $Completed) {
            $ExitCode = 124
            Add-Content -Path $EmbedLog -Value "===== $Name timed out after $TimeoutSeconds seconds ====="
        }
        else {
            $ExitCode = $Process.ExitCode
            if ($null -eq $ExitCode) {
                throw "$Name completed without an exit code."
            }
        }
        $Drained = $false
        if ($Completed) {
            $Drained = [System.Threading.Tasks.Task]::WaitAll([System.Threading.Tasks.Task[]]@($StdoutRead, $StderrRead), 5000)
        }
        if (-not $Drained) {
            if ($ExitCode -eq 0) { $ExitCode = 124 }
            Add-Content -Path $EmbedLog -Value "===== $Name output/cleanup required; exit=$ExitCode; partial output retained ====="
            try {
                $CleanupAttempted = $true
                Stop-PithEmbeddingProcessTree -Process $Process
            }
            catch {
                Add-Content -Path $EmbedLog -Value "===== $Name cleanup error: $($_.Exception.Message) ====="
            }
            if (-not [System.Threading.Tasks.Task]::WaitAll([System.Threading.Tasks.Task[]]@($StdoutRead, $StderrRead), 5000)) {
                Add-Content -Path $EmbedLog -Value "===== $Name output streams still open after bounded cleanup ====="
            }
        }
    }
    catch {
        $CommandError = $_
        if ($Started -and -not $CleanupAttempted) {
            try {
                Stop-PithEmbeddingProcessTree -Process $Process
            }
            catch {
                Add-Content -Path $EmbedLog -Value "===== $Name cleanup error: $($_.Exception.Message) ====="
            }
        }
        throw $CommandError
    }
    finally {
        $Process.Dispose()
        if ($OutStream) { $OutStream.Dispose() }
        if ($ErrStream) { $ErrStream.Dispose() }
        Add-Content -Path $EmbedLog -Value "===== $Name stdout ====="
        if (Test-Path $OutLog) {
            Get-Content $OutLog | Add-Content -Path $EmbedLog
        }
        Add-Content -Path $EmbedLog -Value "===== $Name stderr ====="
        if (Test-Path $ErrLog) {
            Get-Content $ErrLog | Add-Content -Path $EmbedLog
        }
    }

    return $ExitCode
}

function Start-PithDetachedPythonProcess {
    param(
        [Parameter(Mandatory = $true)][string]$PythonExe,
        [Parameter(Mandatory = $true)][string]$WorkingDirectory,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string]$StdoutPath,
        [Parameter(Mandatory = $true)][string]$StderrPath
    )

    $LauncherId = [System.Guid]::NewGuid().ToString('N')
    $LauncherPath = Join-Path ([System.IO.Path]::GetTempPath()) "pith-detached-launch-$LauncherId.py"
    $ArgumentsPath = Join-Path ([System.IO.Path]::GetTempPath()) "pith-detached-launch-$LauncherId.json"
    $Launcher = @'
import json
import os
from pathlib import Path
import subprocess
import sys

python_exe, working_directory, arguments_path, stdout_path, stderr_path = sys.argv[1:]
arguments = json.loads(Path(arguments_path).read_text(encoding="utf-8-sig"))
stdout_file = open(stdout_path, "ab", buffering=0)
stderr_file = open(stderr_path, "ab", buffering=0)
try:
    process = subprocess.Popen(
        [python_exe, *arguments],
        cwd=working_directory,
        stdin=subprocess.DEVNULL,
        stdout=stdout_file,
        stderr=stderr_file,
        env=os.environ.copy(),
        close_fds=True,
        creationflags=subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP,
    )
    print(process.pid, flush=True)
finally:
    stdout_file.close()
    stderr_file.close()
'@
    try {
        Set-Content -LiteralPath $LauncherPath -Value $Launcher -Encoding UTF8
        ConvertTo-Json -InputObject @($Arguments) -Compress |
            Set-Content -LiteralPath $ArgumentsPath -Encoding UTF8
        $LaunchOutput = @(& $PythonExe -I -S $LauncherPath $PythonExe $WorkingDirectory $ArgumentsPath $StdoutPath $StderrPath 2>&1)
        $LaunchExit = if ($null -eq $LASTEXITCODE) { 0 } else { $LASTEXITCODE }
        if ($LaunchExit -ne 0) {
            throw "Detached Python launch failed ($LaunchExit): $($LaunchOutput -join [Environment]::NewLine)"
        }
        $PidText = @($LaunchOutput | ForEach-Object { [string]$_ } | Where-Object { $_ -match '^\d+$' } | Select-Object -Last 1)
        if ($PidText.Count -ne 1) {
            throw "Detached Python launch did not return one PID: $($LaunchOutput -join [Environment]::NewLine)"
        }
        $Process = Get-Process -Id ([int]$PidText[0]) -ErrorAction Stop
        return $Process
    }
    finally {
        Remove-Item -LiteralPath $LauncherPath, $ArgumentsPath -Force -ErrorAction SilentlyContinue
    }
}

function Invoke-PithPythonScriptCommand {
    param(
        [string]$PithHome,
        [string]$EmbedLog,
        [string]$PythonExe,
        [string]$Name,
        [string]$ScriptContent,
        [int]$TimeoutSeconds = 120
    )

    $ScriptPath = "$PithHome\logs\$Name.py"
    Set-Content -Path $ScriptPath -Value $ScriptContent -Encoding UTF8
    return Invoke-PithEmbeddingCommand `
        -PithHome $PithHome `
        -EmbedLog $EmbedLog `
        -FilePath $PythonExe `
        -Arguments @($ScriptPath) `
        -Name $Name `
        -TimeoutSeconds $TimeoutSeconds
}

function Install-PithTorchRuntimeDlls {
    param(
        [string]$PithHome,
        [string]$VenvPath,
        [string]$EmbedLog
    )

    $VcLibsUrl = "https://aka.ms/Microsoft.VCLibs.x64.14.00.Desktop.appx"
    $VcLibsPath = Join-Path $env:TEMP "Microsoft.VCLibs.x64.14.00.Desktop.appx"
    $VcLibsZipPath = Join-Path $env:TEMP "Microsoft.VCLibs.x64.14.00.Desktop.zip"
    $VcLibsExtractPath = Join-Path $env:TEMP "Microsoft.VCLibs.x64.14.00.Desktop"
    $VcLibsLog = "$PithHome\logs\torch_runtime_dlls_install.log"
    $TorchLibPath = "$VenvPath\Lib\site-packages\torch\lib"

    Write-Host "Installing Microsoft C++ runtime DLLs for PyTorch..."
    Set-Content -Path $VcLibsLog -Value "Pith PyTorch runtime DLL install log"
    if (-not (Test-Path $TorchLibPath)) {
        Add-Content -Path $VcLibsLog -Value "Torch library path missing: $TorchLibPath"
        Add-Content -Path $EmbedLog -Value "Torch library path missing: $TorchLibPath"
        return 1
    }

    $PreviousProgressPreference = $ProgressPreference
    $ProgressPreference = "SilentlyContinue"
    try {
        Invoke-WebRequest -Uri $VcLibsUrl -OutFile $VcLibsPath -UseBasicParsing -TimeoutSec 60 -ErrorAction Stop
        Add-Content -Path $VcLibsLog -Value "Downloaded $VcLibsUrl to $VcLibsPath"
    }
    catch {
        Add-Content -Path $VcLibsLog -Value "VCLibs package download failed: $($_.Exception.Message)"
        Add-Content -Path $EmbedLog -Value "VCLibs package download failed: $($_.Exception.Message)"
        return 1
    }
    finally {
        $ProgressPreference = $PreviousProgressPreference
    }

    try {
        Remove-Item -Path $VcLibsZipPath -Force -ErrorAction SilentlyContinue
        Remove-Item -Path $VcLibsExtractPath -Recurse -Force -ErrorAction SilentlyContinue
        Copy-Item -Path $VcLibsPath -Destination $VcLibsZipPath -Force
        New-Item -ItemType Directory -Path $VcLibsExtractPath -Force | Out-Null
        Expand-Archive -Path $VcLibsZipPath -DestinationPath $VcLibsExtractPath -Force

        $Dlls = Get-ChildItem -Path $VcLibsExtractPath -Recurse -Filter "*.dll"
        if (-not $Dlls) {
            Add-Content -Path $VcLibsLog -Value "VCLibs package contained no DLLs"
            Add-Content -Path $EmbedLog -Value "VCLibs package contained no DLLs"
            return 1
        }

        foreach ($Dll in $Dlls) {
            Copy-Item -Path $Dll.FullName -Destination (Join-Path $TorchLibPath $Dll.Name) -Force
            Add-Content -Path $VcLibsLog -Value "copied $($Dll.Name)"
        }
        Add-Content -Path $EmbedLog -Value "Copied $($Dlls.Count) Microsoft C++ runtime DLLs into torch lib"
        return 0
    }
    catch {
        Add-Content -Path $VcLibsLog -Value "VCLibs package extraction/copy failed: $($_.Exception.Message)"
        Add-Content -Path $EmbedLog -Value "VCLibs package extraction/copy failed: $($_.Exception.Message)"
        return 1
    }
}

function Install-PithEmbeddings {
    param(
        [string]$PithHome,
        [string]$VenvPath,
        [string]$PipExe,
        [string]$PythonExe
    )

    $PreviousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    $EmbedLog = "$PithHome\logs\embedding_install.log"
    New-Item -ItemType Directory -Path "$PithHome\logs" -Force | Out-Null
    Set-Content -Path $EmbedLog -Value "Pith embedding install log"

    $TorchInstallExit = Invoke-PithEmbeddingCommand `
        -PithHome $PithHome `
        -EmbedLog $EmbedLog `
        -FilePath $PipExe `
        -Arguments @("install", "--quiet", "torch", "--index-url", "https://download.pytorch.org/whl/cpu") `
        -Name "torch_install" `
        -TimeoutSeconds 1800
    if ($TorchInstallExit -ne 0) {
        Write-Host "  [!] PyTorch install failed. Using TF-IDF search." -ForegroundColor Yellow
        Write-Host "  Details: $EmbedLog"
        Set-PithEmbeddingCapability -PithHome $PithHome -Content "embeddings=false`nreason=pytorch_install_failed"
        $ErrorActionPreference = $PreviousErrorActionPreference
        return $false
    }

    $RuntimeDllExit = Install-PithTorchRuntimeDlls `
        -PithHome $PithHome `
        -VenvPath $VenvPath `
        -EmbedLog $EmbedLog
    if ($RuntimeDllExit -ne 0) {
        $TorchImportAfterDllFailureExit = Invoke-PithPythonScriptCommand `
            -PithHome $PithHome `
            -EmbedLog $EmbedLog `
            -PythonExe $PythonExe `
            -Name "torch_import_after_dll_install_failure" `
            -ScriptContent "import torch`nprint(torch.__version__)`n"
        if ($TorchImportAfterDllFailureExit -ne 0) {
            Write-Host "  [!] Microsoft C++ runtime DLL install failed. Using TF-IDF search." -ForegroundColor Yellow
            Write-Host "  Details: $PithHome\logs\torch_runtime_dlls_install.log"
            Set-PithEmbeddingCapability -PithHome $PithHome -Content "embeddings=false`nreason=torch_runtime_dlls_install_failed"
            $ErrorActionPreference = $PreviousErrorActionPreference
            return $false
        }
        Write-Host "  [!] Microsoft C++ runtime DLL install failed, but PyTorch imports successfully. Continuing." -ForegroundColor Yellow
        Add-Content -Path $EmbedLog -Value "VCLibs helper failed, but PyTorch import succeeded; continuing with semantic embedding install"
    }

    $SentenceTransformersInstallExit = Invoke-PithEmbeddingCommand `
        -PithHome $PithHome `
        -EmbedLog $EmbedLog `
        -FilePath $PipExe `
        -Arguments @("install", "--quiet", "sentence-transformers>=3.0.0,<4.0.0") `
        -Name "sentence_transformers_install" `
        -TimeoutSeconds 1800
    if ($SentenceTransformersInstallExit -ne 0) {
        Write-Host "  [!] sentence-transformers install failed. Using TF-IDF search." -ForegroundColor Yellow
        Set-PithEmbeddingCapability -PithHome $PithHome -Content "embeddings=false`nreason=st_install_failed"
        $ErrorActionPreference = $PreviousErrorActionPreference
        return $false
    }

    $ModelWarmupExit = Invoke-PithPythonScriptCommand `
        -PithHome $PithHome `
        -EmbedLog $EmbedLog `
        -PythonExe $PythonExe `
        -Name "sentence_transformers_runtime_check" `
        -ScriptContent "from sentence_transformers import SentenceTransformer`nSentenceTransformer('all-MiniLM-L6-v2')`n" `
        -TimeoutSeconds 600
    if ($ModelWarmupExit -ne 0) {
        Write-Host "  [!] sentence-transformers runtime check failed. Using TF-IDF search." -ForegroundColor Yellow
        Set-PithEmbeddingCapability -PithHome $PithHome -Content "embeddings=false`nreason=st_runtime_check_failed"
        $ErrorActionPreference = $PreviousErrorActionPreference
        return $false
    }

    $TorchVersionExit = Invoke-PithPythonScriptCommand `
        -PithHome $PithHome `
        -EmbedLog $EmbedLog `
        -PythonExe $PythonExe `
        -Name "torch_version" `
        -ScriptContent "import torch`nprint(torch.__version__)`n"
    $TorchVersionOut = "$PithHome\logs\torch_version.stdout.log"
    $TorchVer = if (Test-Path $TorchVersionOut) { (Get-Content $TorchVersionOut -Raw).Trim() } else { "" }
    if ($TorchVersionExit -ne 0 -or -not ($TorchVer -match "^[0-9]+\.[0-9]+")) {
        Write-Host "  [!] PyTorch runtime check failed. Using TF-IDF search." -ForegroundColor Yellow
        Set-PithEmbeddingCapability -PithHome $PithHome -Content "embeddings=false`nreason=pytorch_runtime_check_failed"
        $ErrorActionPreference = $PreviousErrorActionPreference
        return $false
    }

    Write-Host "[OK] Semantic embeddings enabled (all-MiniLM-L6-v2)" -ForegroundColor Green
    Set-PithEmbeddingCapability -PithHome $PithHome -Content "embeddings=true`npytorch=$TorchVer`narch=x86_64"
    $ErrorActionPreference = $PreviousErrorActionPreference
    return $true
}
