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
            -ArgumentList "-m uvicorn app.api.server:app --host 127.0.0.1 --port 8000" `
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
        Push-Location $PithServerPath
        if ($args.Count -gt 1) {
            & $PythonExe -m app.ops.support_cli status @($args[1..($args.Count - 1)])
        } else {
            & $PythonExe -m app.ops.support_cli status
        }
        Pop-Location
    }
    "health" {
        Push-Location $PithServerPath
        if ($args.Count -gt 1) {
            & $PythonExe -m app.ops.health_cli @($args[1..($args.Count - 1)])
        } else {
            & $PythonExe -m app.ops.health_cli
        }
        Pop-Location
    }
    "logs" {
        if (($args.Count -gt 1) -and ($args[1] -eq "snapshot")) {
            Push-Location $PithServerPath
            & $PythonExe -m app.ops.read_cli logs @($args[1..($args.Count - 1)])
            Pop-Location
        } elseif (Test-Path "$PithHome\logs\pith.log") {
            Get-Content "$PithHome\logs\pith.log" -Tail 50 -Wait
        } else {
            Write-Host "No logs found"
        }
    }
    "import" {
        Push-Location $PithServerPath
        if ($args.Count -gt 1) {
            & $PythonExe -m app.ops.import_cli @($args[1..($args.Count - 1)])
        } else {
            & $PythonExe -m app.ops.import_cli
        }
        Pop-Location
    }
    { $_ -in @("search", "concept", "orient", "sessions", "metrics") } {
        Push-Location $PithServerPath
        if ($args.Count -gt 1) {
            & $PythonExe -m app.ops.read_cli $args[0] @($args[1..($args.Count - 1)])
        } else {
            & $PythonExe -m app.ops.read_cli $args[0]
        }
        Pop-Location
    }
    { $_ -in @("doctor", "clients", "support") } {
        Push-Location $PithServerPath
        if ($args.Count -gt 1) {
            & $PythonExe -m app.ops.support_cli $args[0] @($args[1..($args.Count - 1)])
        } else {
            & $PythonExe -m app.ops.support_cli $args[0]
        }
        Pop-Location
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
        $maintenanceAction = if ($args.Count -gt 1) { $args[1] } else { "run" }
        switch ($maintenanceAction) {
            "run" {
                Write-Host "Running maintenance..."
                if ($args.Count -gt 2) {
                    & $PythonExe -m app.ops.maintenance_cli run @($args[2..($args.Count - 1)])
                } else {
                    & $PythonExe -m app.ops.maintenance_cli run
                }
            }
            "status" {
                & $PythonExe -m app.ops.maintenance_cli status
            }
            default {
                Write-Host "Usage: pith maintenance {run|status}" -ForegroundColor Yellow
                exit 1
            }
        }
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
        Push-Location $PithServerPath
        if ($args.Count -gt 1) {
            & $PythonExe -m app.ops.support_cli report @($args[1..($args.Count - 1)])
        } else {
            & $PythonExe -m app.ops.support_cli report
        }
        Pop-Location
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
        Write-Host "  health      Check operational health/readiness"
        Write-Host "  logs        Tail server logs"
        Write-Host "  search      Search concepts"
        Write-Host "  concept     Read concept details"
        Write-Host "  orient      Show present-moment orientation"
        Write-Host "  sessions    List cognitive sessions"
        Write-Host "  metrics     Show metrics snapshots"
        Write-Host "  doctor      Run read-only install diagnostics"
        Write-Host "  clients     Show detected/configured client surfaces"
        Write-Host "  support     Create redacted support bundles"
        Write-Host "  import      Import conversation exports safely"
        Write-Host "  backup      Create a WAL-safe backup"
        Write-Host "  restore     Restore from a backup file"
        Write-Host "  update      Update dependencies + embeddings"
        Write-Host "  uninstall   Remove Pith completely"
        Write-Host "  maintenance run [--phases 1,2,3] [--dry-run]"
        Write-Host "                Run maintenance cycle"
        Write-Host "  maintenance status"
        Write-Host "                Show maintenance task status"
        Write-Host "  version     Show version and capabilities"
        Write-Host "  report      Generate diagnostics report"
    }
}
