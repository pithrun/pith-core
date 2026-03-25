# Pith CLI wrapper for Windows
$PithHome = "__PITH_HOME__"
$VenvPath = "$PithHome\venv"
$PithServerPath = "$PithHome\pith-server"
$PythonExe = "$VenvPath\Scripts\python.exe"

# DEBT-082: Single DB resolution function (pith.db first, brain.db fallback)
function Resolve-DbPath {
    if (Test-Path "$PithHome\data\pith.db") { return "$PithHome\data\pith.db" }
    return "$PithHome\data\brain.db"
}

$action = 'status'
if ($args.Count -gt 0) { $action = $args[0] }

switch ($action) {
    "start" {
        Write-Host "Starting Pith Brain Server..."
        Push-Location $PithServerPath
        $proc = Start-Process -FilePath $PythonExe `
            -ArgumentList "-m uvicorn app:app --host 127.0.0.1 --port 8000" `
            -WindowStyle Hidden -PassThru
        $proc.Id | Out-File "$PithHome\pith.pid"
        Start-Sleep -Seconds 2
        Write-Host "Pith started successfully (PID: $($proc.Id))"
        Pop-Location
    }
    "stop" {
        Write-Host "Stopping Pith Brain Server..."
        if (Test-Path "$PithHome\pith.pid") {
            $pidVal = Get-Content "$PithHome\pith.pid"
            Stop-Process -Id $pidVal -ErrorAction SilentlyContinue
            Remove-Item "$PithHome\pith.pid" -ErrorAction SilentlyContinue
            Write-Host "Pith stopped"
        } else {
            Write-Host "Pith is not running"
        }
    }
    "restart" {
        & $MyInvocation.MyCommand.Path stop
        Start-Sleep -Seconds 1
        & $MyInvocation.MyCommand.Path start
    }
    "status" {
        if (Test-Path "$PithHome\pith.pid") {
            $pidVal = Get-Content "$PithHome\pith.pid"
            if (Get-Process -Id $pidVal -ErrorAction SilentlyContinue) {
                Write-Host "Pith is running (PID: $pidVal)"
            } else {
                Write-Host "Pith is not running (stale PID file)"
                Remove-Item "$PithHome\pith.pid" -ErrorAction SilentlyContinue
            }
        } else {
            Write-Host "Pith is not running"
        }
    }
    "logs" {
        if (Test-Path "$PithHome\logs\pith.log") {
            Get-Content "$PithHome\logs\pith.log" -Tail 50 -Wait
        } else {
            Write-Host "No logs found"
        }
    }
    "backup" {
        $SafeBackup = "$PithHome\pith-server\scripts\backup\safe_backup.ps1"
        if (Test-Path $SafeBackup) {
            & powershell -NoProfile -ExecutionPolicy Bypass -File $SafeBackup
        } else {
            Write-Host "Backup script not found: $SafeBackup" -ForegroundColor Red
        }
    }
    "restore" {        if ($args.Count -lt 2) {
            Write-Host "Usage: pith restore <backup_file.db>"
            Write-Host "Available backups:"
            Get-ChildItem "$PithHome\backups\*.db" | Sort-Object LastWriteTime -Descending | Select-Object -First 10 Name, LastWriteTime
            exit 1
        }
        $BackupFile = $args[1]
        if (-not (Test-Path $BackupFile)) {
            $BackupFile = "$PithHome\backups\$($args[1])"
        }
        if (-not (Test-Path $BackupFile)) {
            Write-Host "Backup file not found: $($args[1])" -ForegroundColor Red
            exit 1
        }
        Write-Host "Restoring from: $BackupFile"
        & $MyInvocation.MyCommand.Path stop
        Start-Sleep -Seconds 1
        $dbPath = Resolve-DbPath
        Copy-Item -Path $BackupFile -Destination $dbPath -Force
        # Verify integrity
        $IntCheck = & $PythonExe -c "import sqlite3; c=sqlite3.connect(r'$dbPath'); print(c.execute('PRAGMA integrity_check').fetchone()[0]); c.close()"
        if ($IntCheck -eq "ok") {
            Write-Host "Database integrity verified" -ForegroundColor Green
            & $MyInvocation.MyCommand.Path start
        } else {
            Write-Host "WARNING: Database integrity check failed: $IntCheck" -ForegroundColor Red
            Write-Host "Restore aborted. Original database may need manual recovery."
        }
    }
    "update" {
        Write-Host "Updating Pith..."
        & $MyInvocation.MyCommand.Path stop
        Start-Sleep -Seconds 1
        & $VenvPath\Scripts\pip.exe install --quiet --upgrade -r "$PithServerPath\requirements.txt" 2>$null
        # Re-run embedding installation
        & $VenvPath\Scripts\pip.exe install --quiet torch --index-url https://download.pytorch.org/whl/cpu 2>"$PithHome\logs\embedding_install.log"
        if ($LASTEXITCODE -eq 0) {
            & $VenvPath\Scripts\pip.exe install --quiet "sentence-transformers>=3.0.0,<4.0.0" 2>>"$PithHome\logs\embedding_install.log"
        }
        & $MyInvocation.MyCommand.Path start
        Write-Host "Update complete"
    }
    "uninstall" {
        Write-Host "This will remove Pith and all its data from $PithHome"
        $Confirm = Read-Host "Are you sure? (yes/no)"
        if ($Confirm -eq "yes") {
            & $MyInvocation.MyCommand.Path stop
            Start-Sleep -Seconds 1
            # Remove scheduled tasks
            Unregister-ScheduledTask -TaskName "Pith-Brain-Server" -Confirm:$false -ErrorAction SilentlyContinue
            Unregister-ScheduledTask -TaskName "Pith-Daily-Backup" -Confirm:$false -ErrorAction SilentlyContinue
            Unregister-ScheduledTask -TaskName "Pith-Backup-3h" -Confirm:$false -ErrorAction SilentlyContinue
            Remove-Item -Path $PithHome -Recurse -Force
            Write-Host "Pith has been uninstalled"
        } else {
            Write-Host "Uninstall cancelled"
        }
    }
    "maintenance" {
        Write-Host "Running maintenance..."
        $dbPath = Resolve-DbPath
        & $PythonExe -c "import sqlite3; c=sqlite3.connect(r'$dbPath'); c.execute('VACUUM'); c.execute('ANALYZE'); print('VACUUM + ANALYZE complete'); c.close()"
        Write-Host "Maintenance complete"
    }
    "version" {
        $CapFile = "$PithHome\.install_capabilities"
        Write-Host "Pith v__PITH_VERSION__"
        if (Test-Path $CapFile) {
            Get-Content $CapFile | ForEach-Object { Write-Host "  $_" }
        } else {
            Write-Host "  (no capabilities info - re-run installer)"
        }
    }
    "report" {
        Write-Host "Pith Brain Diagnostics Report"
        Write-Host "=============================="
        Write-Host "Generated: $(Get-Date -Format 'yyyy-MM-ddTHH:mm:ssZ')"
        Write-Host ""

        # System
        Write-Host "[System]"
        Write-Host "  OS:           $([System.Environment]::OSVersion.VersionString)"
        Write-Host "  PowerShell:   $($PSVersionTable.PSVersion)"
        $pyVer = & python.exe --version 2>&1
        Write-Host "  Python:       $pyVer ($PythonExe)"
        $nodeVer = & node --version 2>$null
        if ($nodeVer) { Write-Host "  Node:         $nodeVer" }
        $drive = Get-PSDrive ($PithHome.Substring(0,1))
        Write-Host "  Disk Free:    $([Math]::Round($drive.Free / 1GB, 1)) GB"
        Write-Host ""

        # Installation
        Write-Host "[Installation]"
        Write-Host "  Pith Home:    $PithHome"
        Write-Host "  Version:      __PITH_VERSION__"
        $pkgJson = "$PithServerPath\package.json"
        if (Test-Path $pkgJson) {
            $pkg = Get-Content $pkgJson -Raw | ConvertFrom-Json -ErrorAction SilentlyContinue
            if ($pkg.version) { Write-Host "  Server Ver:   $($pkg.version)" }
        }
        $capFile = "$PithHome\.install_capabilities"
        if (Test-Path $capFile) {
            Get-Content $capFile | ForEach-Object { Write-Host "  $_" }
        } else {
            Write-Host "  Embeddings:   unknown (no .install_capabilities)"
        }
        Write-Host ""

        # Server
        Write-Host "[Server]"
        if (Test-Path "$PithHome\pith.pid") {
            $pidVal = Get-Content "$PithHome\pith.pid"
            $proc = Get-Process -Id $pidVal -ErrorAction SilentlyContinue
            if ($proc) {
                Write-Host "  Status:       Running (PID $pidVal)"
                $uptime = (Get-Date) - $proc.StartTime
                Write-Host "  Uptime:       $([Math]::Floor($uptime.TotalHours))h $($uptime.Minutes)m"
            } else {
                Write-Host "  Status:       Not running (stale PID)"
            }
        } else {
            Write-Host "  Status:       Not running"
        }
        Write-Host "  Port:         8000"
        try {
            $health = Invoke-WebRequest -Uri "http://127.0.0.1:8000/health" -TimeoutSec 3 -ErrorAction SilentlyContinue
            Write-Host "  Health:       OK ($($health.StatusCode))"
        } catch {
            Write-Host "  Health:       Unreachable"
        }
        Write-Host ""

        # Database
        Write-Host "[Database]"
        $dbPath = Resolve-DbPath
        if (Test-Path $dbPath) {
            $dbSize = [Math]::Round((Get-Item $dbPath).Length / 1MB, 2)
            Write-Host "  Path:         $dbPath"
            Write-Host "  Size:         $dbSize MB"
            try {
                $concepts = & $PythonExe -c "import sqlite3; c=sqlite3.connect(r'$dbPath'); print(c.execute('SELECT COUNT(*) FROM concepts').fetchone()[0]); c.close()" 2>$null
                if ($concepts) { Write-Host "  Concepts:     $concepts" }
                $wal = & $PythonExe -c "import sqlite3; c=sqlite3.connect(r'$dbPath'); print(c.execute('PRAGMA journal_mode').fetchone()[0]); c.close()" 2>$null
                if ($wal) { Write-Host "  Journal:      $wal" }
            } catch {}
        } else {
            Write-Host "  Path:         $dbPath (not created yet)"
        }
        Write-Host ""

        # MCP Clients
        Write-Host "[MCP Clients]"
        $clients = @{
            "Claude Desktop" = "$env:APPDATA\Claude\claude_desktop_config.json"
            "Claude Code"    = "$env:APPDATA\Claude\claude_code_config.json"
            "Cursor"         = "$env:APPDATA\Cursor\User\globalStorage\cursor.mcp\config.json"
            "Windsurf"       = "$env:APPDATA\Windsurf\User\globalStorage\windsurf.mcp\config.json"
            "Cline"          = "$env:APPDATA\Code\User\globalStorage\saoudrizwan.claude-dev\settings\cline_mcp_settings.json"
            "Continue"       = "$env:USERPROFILE\.continue\config.json"
        }
        foreach ($name in $clients.Keys) {
            $path = $clients[$name]
            if (Test-Path $path) {
                $content = Get-Content $path -Raw -ErrorAction SilentlyContinue
                if ($content -match '"pith"') {
                    Write-Host "  ${name}:$((' ' * (16 - $name.Length)))configured"
                } elseif ($content -match "brain-mcp") {
                    Write-Host "  ${name}:$((' ' * (16 - $name.Length)))configured (legacy brain-mcp — run installer to update)"
                } else {
                    Write-Host "  ${name}:$((' ' * (16 - $name.Length)))present (pith not found)"
                }
            } else {
                Write-Host "  ${name}:$((' ' * (16 - $name.Length)))not found"
            }
        }
        Write-Host ""

        # Scheduled Tasks
        Write-Host "[Scheduled Tasks]"
        foreach ($taskName in @("Pith-Brain-Server", "Pith-Backup-3h")) {
            try {
                $task = Get-ScheduledTask -TaskName $taskName -ErrorAction Stop
                Write-Host "  ${taskName}:$((' ' * (22 - $taskName.Length)))$($task.State)"
            } catch {
                Write-Host "  ${taskName}:$((' ' * (22 - $taskName.Length)))not registered"
            }
        }
        Write-Host ""

        # Backups
        Write-Host "[Backups]"
        $backupDir = "$PithHome\backups"
        if (Test-Path $backupDir) {
            $backups = Get-ChildItem "$backupDir\*.db" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending
            Write-Host "  Count:        $($backups.Count)"
            if ($backups.Count -gt 0) {
                Write-Host "  Latest:       $($backups[0].LastWriteTime.ToString('yyyy-MM-ddTHH:mm:ss'))"
                Write-Host "  Oldest:       $($backups[-1].LastWriteTime.ToString('yyyy-MM-ddTHH:mm:ss'))"
            }
        } else {
            Write-Host "  No backup directory"
        }
        Write-Host ""

        # API Key (redacted)
        $keyFile = "$PithHome\config\api.key"
        if (Test-Path $keyFile) {
            $key = (Get-Content $keyFile -Raw -ErrorAction SilentlyContinue).Trim()
            if ($key.Length -gt 8) {
                Write-Host "[API Key]"
                Write-Host "  Status:       Present ($($key.Substring(0,8))...)"
            }
        }
    }
    default {
        Write-Host "Pith Brain - Personal Knowledge Server"
        Write-Host ""
        Write-Host "Usage: pith <command>"
        Write-Host ""
        Write-Host "Commands:"
        Write-Host "  start       Start the Pith server"
        Write-Host "  stop        Stop the Pith server"
        Write-Host "  restart     Restart the Pith server"
        Write-Host "  status      Check server status"
        Write-Host "  logs        Tail server logs"
        Write-Host "  backup      Create a WAL-safe backup"
        Write-Host "  restore     Restore from a backup file"
        Write-Host "  update      Update dependencies + embeddings"
        Write-Host "  uninstall   Remove Pith completely"
        Write-Host "  maintenance Run VACUUM + ANALYZE on database"
        Write-Host "  version     Show version and capabilities"
        Write-Host "  report      Generate diagnostics report"
    }
}
