# Exact-artifact native acceptance fixtures. Never execute the complete installer.
param(
    [Parameter(Mandatory=$true)][string]$InstallerPath,
    [Parameter(Mandatory=$true)][string]$CheckerPath,
    [Parameter(Mandatory=$true)][string]$RuntimeHelperPath,
    [Parameter(Mandatory=$true)][string]$PythonExePath,
    [Parameter(Mandatory=$true)][string]$OutputPath
)
$ErrorActionPreference = 'Stop'
$Rows = New-Object System.Collections.Generic.List[object]
$FixtureRoot = Join-Path $env:TEMP ('pith installer health ' + [guid]::NewGuid().ToString('N'))
$Summary = [ordered]@{ started_utc=[DateTime]::UtcNow.ToString('o'); status='running'; powershell=$PSVersionTable.PSVersion.ToString() }
function Assert-Contract {
    param([string]$Name, [bool]$Passed)
    $Rows.Add([ordered]@{ name=$Name; passed=$Passed })
    if (-not $Passed) { throw "Contract failed: $Name" }
}
function Write-FixtureFile {
    param([string]$Path, [string]$Text)
    [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($Path)) | Out-Null
    [IO.File]::WriteAllText($Path, $Text, (New-Object Text.UTF8Encoding($false)))
}
function Get-ProductionFunction {
    param([string]$Name)
    $Found = @($Ast.FindAll({param($Node) $Node -is [Management.Automation.Language.FunctionDefinitionAst] -and $Node.Name -eq $Name}, $true))
    Assert-Contract "single_production_function:$Name" ($Found.Count -eq 1)
    return $Found[0].Extent.Text
}
function Write-Error-Custom { param([string]$Message) throw "FIXTURE_FATAL: $Message" }
function Write-Success { param([string]$Message) $script:SuccessMessages.Add($Message) }
function Invoke-WebRequest {
    param($Uri, [switch]$UseBasicParsing, $TimeoutSec, $ErrorAction)
    if ($script:HttpThrows) { throw 'fixture HTTP error' }
    Assert-Contract 'readyz_loopback_only' ($Uri -eq "http://127.0.0.1:$PithPort/readyz")
    # A one-element @($false) plan is falsey in PowerShell but is still a plan.
    if ($null -ne $script:ProbePlan) {
        $Index = [Math]::Min($script:ProbeIndex, $script:ProbePlan.Count - 1)
        $script:ProbeIndex++
        $Content = if ($script:ProbePlan[$Index]) { $script:ReadyText } else { '{}' }
        return [pscustomobject]@{ StatusCode=200; Content=$Content }
    }
    return [pscustomobject]@{ StatusCode=$script:HttpStatus; Content=$script:HttpText }
}
function Start-ScheduledTask { param($TaskName, $ErrorAction) if ($script:StartThrows) { throw 'fixture start failure' } }
function Start-PithDetachedPythonProcess {
    param($PythonExe, $WorkingDirectory, $Arguments, $StdoutPath, $StderrPath)
    if ($script:StartThrows) { throw 'fixture start failure' }
    return [pscustomobject]@{ Id=12345; HasExited=$script:EarlyExit }
}
function Start-Sleep { param($Seconds, $Milliseconds) } # Fixture clock only: no user process.
try {
    foreach ($Path in @($InstallerPath, $CheckerPath, $RuntimeHelperPath, $PythonExePath)) {
        Assert-Contract 'explicit_input_exists' ([IO.File]::Exists([IO.Path]::GetFullPath($Path)))
    }
    Assert-Contract 'fresh_output_path' (-not [IO.File]::Exists([IO.Path]::GetFullPath($OutputPath)))
    $Summary.installer_sha256 = (Get-FileHash -LiteralPath $InstallerPath -Algorithm SHA256).Hash.ToLowerInvariant()
    $Summary.checker_sha256 = (Get-FileHash -LiteralPath $CheckerPath -Algorithm SHA256).Hash.ToLowerInvariant()
    $Summary.helper_sha256 = (Get-FileHash -LiteralPath $RuntimeHelperPath -Algorithm SHA256).Hash.ToLowerInvariant()
    $Tokens=$null; $ParseErrors=$null
    $Ast = [Management.Automation.Language.Parser]::ParseFile([IO.Path]::GetFullPath($InstallerPath), [ref]$Tokens, [ref]$ParseErrors)
    Assert-Contract 'installer_parses_natively' ($ParseErrors.Count -eq 0)
    $InstallerText = [IO.File]::ReadAllText([IO.Path]::GetFullPath($InstallerPath))
    $FatalSource = Get-ProductionFunction 'Write-Error-Custom'
    Assert-Contract 'production_fatal_exit_one' ($FatalSource -match '(?m)^\s*exit 1\s*$')
    $GateSource = Get-ProductionFunction 'Assert-PithCoreDependencyHealth'
    $ReadySource = Get-ProductionFunction 'Test-PithInstallerConversationReady'
    . ([scriptblock]::Create($GateSource + "`r`n" + $ReadySource))
    $script:SuccessMessages = New-Object System.Collections.Generic.List[string]
    $PithVersion='1.0.10'; $PithPort=8765
    $ReadyObject = [ordered]@{ service='pith'; version=$PithVersion; process_state='running'; write_state='accepting'; retrieval_state='ready'; status='degraded'; semantic_full_search='disabled' }
    $script:ReadyText = $ReadyObject | ConvertTo-Json -Compress
    $script:HttpStatus=200; $script:HttpThrows=$false; $script:HttpText=$script:ReadyText
    Assert-Contract 'tfidf_ready_maintenance_degraded_allowed' (Test-PithInstallerConversationReady $PithPort)
    foreach ($Field in @('service','version','process_state','write_state','retrieval_state')) {
        foreach ($BadValue in @($null, 42, @('ready'), 'unknown', 'READY')) {
            $Object = $script:ReadyText | ConvertFrom-Json
            $Object.$Field=$BadValue
            $script:HttpText=$Object | ConvertTo-Json -Depth 5 -Compress
            Assert-Contract "readyz_reject:${Field}:$([string]$BadValue)" (-not (Test-PithInstallerConversationReady $PithPort))
        }
        $Object = $script:ReadyText | ConvertFrom-Json
        $Object.PSObject.Properties.Remove($Field)
        $script:HttpText=$Object | ConvertTo-Json -Compress
        Assert-Contract "readyz_missing:$Field" (-not (Test-PithInstallerConversationReady $PithPort))
    }
    foreach ($BadText in @('null','[]',('['+$script:ReadyText+']'),'{','{}',($script:ReadyText+' trailing'))) {
        $script:HttpText=$BadText
        Assert-Contract 'readyz_invalid_root_or_json' (-not (Test-PithInstallerConversationReady $PithPort))
    }
    $script:HttpText=$script:ReadyText; $script:HttpStatus=503
    Assert-Contract 'readyz_non200' (-not (Test-PithInstallerConversationReady $PithPort))
    $script:HttpStatus=200; $script:HttpThrows=$true
    Assert-Contract 'readyz_http_throw' (-not (Test-PithInstallerConversationReady $PithPort))
    $script:HttpThrows=$false

    # Actual owned command + exact checker: setup-python may have no packaging.
    # Both missing requirements and missing packaging must fail, never false-pass.
    $PithHome=Join-Path $FixtureRoot 'home'; $PithServerPath=Join-Path $PithHome 'pith-server'
    [IO.Directory]::CreateDirectory((Join-Path $PithHome 'logs')) | Out-Null
    [IO.Directory]::CreateDirectory((Join-Path $PithServerPath 'scripts')) | Out-Null
    Copy-Item -LiteralPath $CheckerPath -Destination (Join-Path $PithServerPath 'scripts\windows_dependency_health.py')
    $ResultPath=Join-Path $PithHome 'logs\core_dependency_health.stdout.log'
    $GoodReceipt=[ordered]@{schema_version=1;scope='core_runtime_dependencies';status='pass';distribution_count=2;runtime_file_count=2;failure_count=0;failures=@();advisory_count=0;advisories=@();import_status='pass';imports_checked=13}
    $GoodText=$GoodReceipt | ConvertTo-Json -Depth 5 -Compress
    Write-FixtureFile $ResultPath $GoodText
    # Only the trusted helper's definitions are loaded; no installation invoked.
    . $RuntimeHelperPath
    $RealExit=Invoke-PithEmbeddingCommand -PithHome $PithHome -EmbedLog "$PithHome\logs\actual.log" -FilePath $PythonExePath -Arguments @('-I','-B',$CheckerPath,'--requirements',"$FixtureRoot\missing.txt") -Name 'core_dependency_health' -TimeoutSeconds 15
    Assert-Contract 'exact_checker_actual_nonzero' ($RealExit -eq 1)
    $ActualText=[IO.File]::ReadAllText($ResultPath)
    $ActualReceipt=$ActualText | ConvertFrom-Json
    Assert-Contract 'actual_command_replaced_stale_pass' ($ActualReceipt.status -ceq 'fail' -and $ActualText -cne $GoodText)
    $RealExit=Invoke-PithEmbeddingCommand -PithHome $PithHome -EmbedLog "$PithHome\logs\actual.log" -FilePath $PythonExePath -Arguments @('-I','-B','-c','print("current-owned-command")') -Name 'fixture_command' -TimeoutSeconds 15
    Assert-Contract 'actual_command_zero_and_current_stdout' ($RealExit -eq 0 -and [IO.File]::ReadAllText("$PithHome\logs\fixture_command.stdout.log").Trim() -ceq 'current-owned-command')
    # Override only fixture command output; production gate is unchanged.
    function Invoke-PithEmbeddingCommand {
        param($PithHome,$EmbedLog,$FilePath,$Arguments,$Name,$TimeoutSeconds)
        Assert-Contract 'dependency_isolated_argv' ($Arguments[0] -ceq '-I' -and $Arguments[1] -ceq '-B' -and $Arguments[3] -ceq '--requirements' -and $TimeoutSeconds -eq 180)
        if ($script:CommandThrows) { throw 'fixture timeout/launch failure' }
        # Recreate current stdout as the real helper does; never retain old pass.
        Write-FixtureFile "$PithHome\logs\$Name.stdout.log" $script:CommandText
        return $script:CommandExit
    }
    $script:CommandText=$GoodText; $script:CommandExit=0; $script:CommandThrows=$false
    Assert-PithCoreDependencyHealth $PythonExePath $PithHome $PithServerPath "$FixtureRoot\roots.txt"
    Assert-Contract 'valid_dependency_receipt_passes' ($script:SuccessMessages.Contains('Verified core runtime dependencies'))
    $BadReceipts=New-Object System.Collections.Generic.List[string]
    foreach ($Text in @('','{}','null','[]',('['+$GoodText+']'),($GoodText+' trailing'),(' '*1048577))) { $BadReceipts.Add($Text) }
    foreach ($Field in @('schema_version','distribution_count','runtime_file_count','failure_count','imports_checked')) {
        foreach ($Value in @($null,$true,1.5,'1',-1,0,400001)) {
            $Object=$GoodText | ConvertFrom-Json; $Object.$Field=$Value
            # Zero failure_count is valid; all other chosen mutations must fail.
            if ($Field -eq 'failure_count' -and $Value -is [int] -and $Value -eq 0) { continue }
            $BadReceipts.Add(($Object | ConvertTo-Json -Depth 5 -Compress))
        }
    }
    foreach ($Field in @('scope','status','import_status')) {
        foreach ($Value in @($null,@('pass'),'unknown')) {
            $Object=$GoodText | ConvertFrom-Json; $Object.$Field=$Value
            $BadReceipts.Add(($Object | ConvertTo-Json -Depth 5 -Compress))
        }
    }
    foreach ($Value in @($null,'',@('failure'))) {
        $Object=$GoodText | ConvertFrom-Json; $Object.failures=$Value
        $BadReceipts.Add(($Object | ConvertTo-Json -Depth 5 -Compress))
    }
    foreach ($Text in $BadReceipts) {
        $script:CommandText=$Text; $script:SuccessMessages.Clear(); $Rejected=$false
        try { Assert-PithCoreDependencyHealth $PythonExePath $PithHome $PithServerPath "$FixtureRoot\roots.txt" } catch { $Rejected=$_.Exception.Message.StartsWith('FIXTURE_FATAL:') }
        Assert-Contract 'invalid_current_dependency_receipt_fatal' ($Rejected -and $script:SuccessMessages.Count -eq 0)
    }
    foreach ($Code in @($null,1,124)) {
        $script:CommandText=$GoodText; $script:CommandExit=$Code; $Rejected=$false
        try { Assert-PithCoreDependencyHealth $PythonExePath $PithHome $PithServerPath "$FixtureRoot\roots.txt" } catch { $Rejected=$_.Exception.Message.StartsWith('FIXTURE_FATAL:') }
        Assert-Contract 'nonzero_null_timeout_even_valid_receipt_fatal' $Rejected
    }
    $script:CommandThrows=$true; $Rejected=$false
    try { Assert-PithCoreDependencyHealth $PythonExePath $PithHome $PithServerPath "$FixtureRoot\roots.txt" } catch { $Rejected=$_.Exception.Message.StartsWith('FIXTURE_FATAL:') }
    Assert-Contract 'dependency_launch_throw_fatal' $Rejected
    $Rejected=$false
    try { Assert-PithCoreDependencyHealth $PythonExePath $PithHome "$FixtureRoot\missing-server" "$FixtureRoot\roots.txt" } catch { $Rejected=$_.Exception.Message.StartsWith('FIXTURE_FATAL:') }
    Assert-Contract 'dependency_checker_missing_fatal' $Rejected

    # Extract COMPLETE production readiness/startup/final-fatal block using AST.
    $ReadyNode=@($Ast.EndBlock.Statements | Where-Object { $_ -is [Management.Automation.Language.FunctionDefinitionAst] -and $_.Name -eq 'Test-PithInstallerConversationReady' })
    $FatalNode=@($Ast.EndBlock.Statements | Where-Object { $_ -is [Management.Automation.Language.IfStatementAst] -and $_.Extent.Text.Contains('Installation incomplete: Pith conversation readiness failed.') })
    Assert-Contract 'single_readiness_and_final_fatal_ast' ($ReadyNode.Count -eq 1 -and $FatalNode.Count -eq 1)
    $Start=$ReadyNode[0].Extent.StartOffset; $End=$FatalNode[0].Extent.EndOffset
    $HealthBlock=$InstallerText.Substring($Start,$End-$Start)
    Assert-Contract 'fatal_outside_warning_catch' ($HealthBlock.IndexOf('}  # end of port pre-check else block') -lt $HealthBlock.IndexOf('Installation incomplete: Pith conversation readiness failed.'))
    Assert-Contract 'exact_production_budget_anchor' ([regex]::Matches($HealthBlock,'Elapsed.TotalSeconds -lt 90').Count -eq 1)
    $FixtureBlock=$HealthBlock.Replace('Elapsed.TotalSeconds -lt 90','Elapsed.TotalSeconds -lt 0.2')
    Assert-Contract 'fixture_only_clock_transformation' ($FixtureBlock.Replace('Elapsed.TotalSeconds -lt 0.2','Elapsed.TotalSeconds -lt 90') -ceq $HealthBlock)
    # Restore process mock overwritten by the pure runtime helper definitions.
    function Start-PithDetachedPythonProcess {
        param($PythonExe,$WorkingDirectory,$Arguments,$StdoutPath,$StderrPath)
        if ($script:StartThrows) { throw 'fixture start failure' }
        return [pscustomobject]@{Id=12345;HasExited=$script:EarlyExit}
    }
    $PythonExe=$PythonExePath
    $script:ProbePlan=@($false); $script:ProbeIndex=0
    Assert-Contract 'false_only_probe_plan_is_not_ready' (-not (Test-PithInstallerConversationReady $PithPort))
    $script:ProbePlan=$null
    $Cases=@(
        @{name='existing_ready';plan=@($true);pass=$true},
        @{name='new_double_probe';plan=@($false,$true,$true);pass=$true},
        @{name='scheduled_double_probe';plan=@($false,$true,$true);task='fixture';pass=$true},
        @{name='second_probe_fails';plan=@($false,$true,$false);pass=$false},
        @{name='early_exit';plan=@($false);early=$true;pass=$false},
        @{name='process_start_throws';plan=@($false);throws=$true;pass=$false},
        @{name='task_start_throws';plan=@($false);task='fixture';throws=$true;pass=$false},
        @{name='deadline';plan=@($false);pass=$false}
    )
    foreach ($Case in $Cases) {
        $script:ProbePlan=$Case.plan; $script:ProbeIndex=0; $script:StartThrows=[bool]$Case.throws; $script:EarlyExit=[bool]$Case.early
        $PithAutoStartTaskName=$Case.task; $SuccessReached=$false; $Rejected=$false
        try { . ([scriptblock]::Create($FixtureBlock)); $SuccessReached=$true } catch { $Rejected=$_.Exception.Message.StartsWith('FIXTURE_FATAL:') }
        Assert-Contract "startup_acceptance:$($Case.name)" ($SuccessReached -eq $Case.pass -and $Rejected -eq (-not $Case.pass))
        if ($Case.pass) { Assert-Contract "positive_latch:$($Case.name)" $HealthCheckPassed }
    }
    $Summary.status='completed'
}
catch {
    $Summary.status='error'; $Summary.error=$_.Exception.Message; $Summary.error_position=$_.InvocationInfo.PositionMessage
}
finally {
    $Summary.rows=$Rows.ToArray(); $Summary.finished_utc=[DateTime]::UtcNow.ToString('o')
    # Retain owned fixtures on ephemeral CI for diagnosis, no recursive cleanup.
    Write-FixtureFile ([IO.Path]::GetFullPath($OutputPath)) ($Summary | ConvertTo-Json -Depth 8)
}
if ($Summary.status -ne 'completed') { Write-Error $Summary.error -ErrorAction Continue; exit 1 }
Write-Host "Windows installer health contract passed: $($Rows.Count) rows. Proof: $OutputPath"
exit 0
