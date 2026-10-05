# Windows upgrade contract. Never executes the complete installer.
param(
    [Parameter(Mandatory=$true)][string]$InstallerPath,
    [Parameter(Mandatory=$true)][string]$PythonExePath,
    [Parameter(Mandatory=$true)][string]$OutputPath
)
$ErrorActionPreference = 'Stop'
$Rows = New-Object System.Collections.Generic.List[object]
$FixtureRoot = Join-Path $env:TEMP ('pith upgrade [literal] ' + [guid]::NewGuid().ToString('N'))
$Summary = [ordered]@{ started_utc=[DateTime]::UtcNow.ToString('o'); fixture_root=$FixtureRoot; powershell=$PSVersionTable.PSVersion.ToString(); status='running' }
$OwnedHttp = $null
function Assert-Contract {
    param([string]$Name, [bool]$Passed)
    $Rows.Add([ordered]@{ name=$Name; passed=$Passed })
    if (-not $Passed) { throw "Contract failed: $Name" }
}
function Assert-Throws {
    param([string]$Name, [scriptblock]$Action, [string]$Pattern='.')
    $Message = ''
    try { & $Action | Out-Null } catch { $Message=$_.Exception.Message }
    Assert-Contract $Name ($Message -match $Pattern)
}
function Write-FixtureFile {
    param([string]$Path, [string]$Text)
    [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($Path)) | Out-Null
    [IO.File]::WriteAllText($Path, $Text, (New-Object Text.UTF8Encoding($false)))
}
function Quote-NativeArgument {
    param([string]$Value)
    # Windows argv quoting, including backslashes immediately before a quote/end.
    return '"' + [regex]::Replace([regex]::Replace($Value, '(\\*)"', '$1$1\"'), '(\\+)$', '$1$1') + '"'
}
function Invoke-FixtureProcess {
    param([string]$Exe, [string[]]$Arguments, [string]$Stem, [int]$TimeoutSeconds=180)
    $Stdout = Join-Path $FixtureRoot ($Stem + '.stdout.txt')
    $Stderr = Join-Path $FixtureRoot ($Stem + '.stderr.txt')
    $CommandLine = ($Arguments | ForEach-Object { Quote-NativeArgument $_ }) -join ' '
    $Info=New-Object Diagnostics.ProcessStartInfo
    $Info.FileName=$Exe; $Info.Arguments=$CommandLine; $Info.UseShellExecute=$false; $Info.CreateNoWindow=$true
    $Info.RedirectStandardOutput=$true; $Info.RedirectStandardError=$true
    $Process=New-Object Diagnostics.Process
    $Process.StartInfo=$Info
    try {
        if (-not $Process.Start()) { throw "Fixture child did not start: $Stem" }
        $OutTask=$Process.StandardOutput.ReadToEndAsync(); $ErrTask=$Process.StandardError.ReadToEndAsync()
        if (-not $Process.WaitForExit($TimeoutSeconds * 1000)) { throw "Fixture child timed out: $Stem" }
        $Process.Refresh()
        $OutText=$OutTask.GetAwaiter().GetResult(); $ErrText=$ErrTask.GetAwaiter().GetResult()
        Write-FixtureFile $Stdout $OutText; Write-FixtureFile $Stderr $ErrText
        return [pscustomobject]@{ exit_code=$Process.ExitCode; stdout=$OutText; stderr=$ErrText }
    }
    finally {
        if (-not $Process.HasExited) { $Process.Kill(); $Process.WaitForExit() }
        $Process.Dispose()
    }
}
function Invoke-FixturePython {
    param([string[]]$Arguments, [string]$Stem)
    $Result = Invoke-FixtureProcess $PythonExePath (@('-B', '-I', $PythonHelper) + $Arguments) $Stem
    if ($Result.exit_code -ne 0) { throw "Fixture Python failed ($Stem): $($Result.stderr)" }
    return $Result.stdout.Trim() | ConvertFrom-Json
}
try {
    [IO.Directory]::CreateDirectory($FixtureRoot) | Out-Null
    $InstallerFull = [IO.Path]::GetFullPath($InstallerPath)
    $PythonFull = [IO.Path]::GetFullPath($PythonExePath)
    Assert-Contract 'explicit_installer_exists' ([IO.File]::Exists($InstallerFull))
    Assert-Contract 'explicit_python_exists' ([IO.File]::Exists($PythonFull))
    $Tokens=$null; $ParseErrors=$null
    $Ast = [Management.Automation.Language.Parser]::ParseFile($InstallerFull, [ref]$Tokens, [ref]$ParseErrors)
    $Summary.parser_errors=@($ParseErrors | ForEach-Object { [ordered]@{ message=$_.Message.Substring(0,[Math]::Min(300,$_.Message.Length)); line=$_.Extent.StartLineNumber; text=$_.Extent.Text.Substring(0,[Math]::Min(80,$_.Extent.Text.Length)) } })
    Assert-Contract 'installer_parse_no_errors' ($ParseErrors.Count -eq 0)
    $Names = @('Get-PithServerTreeInventory', 'Test-PithOwnedApplicationBytecode', 'Clear-PithApplicationBytecode', 'Get-PithSha256')
    $FunctionSources = foreach ($Name in $Names) {
        $Found = @($Ast.FindAll({param($Node) $Node -is [Management.Automation.Language.FunctionDefinitionAst] -and $Node.Name -eq $Name}, $true))
        Assert-Contract "exact_single_helper:$Name" ($Found.Count -eq 1)
        $Found[0].Extent.Text
    }
    $HelperSource = $FunctionSources -join "`r`n"
    # Helpers only; never dot-source the installer body.
    . ([scriptblock]::Create($HelperSource))
    $Summary.installer_sha256 = Get-PithSha256 $InstallerFull
    $Summary.harness_sha256 = Get-PithSha256 $MyInvocation.MyCommand.Path
    $Steps = @($Ast.EndBlock.Statements | Where-Object { $_ -is [Management.Automation.Language.PipelineAst] -and $_.Extent.Text -match '^Write-Step [34] ' })
    Assert-Contract 'exact_step3_step4_boundaries' ($Steps.Count -eq 2 -and $Steps[0].Extent.Text -match '^Write-Step 3 ' -and $Steps[1].Extent.Text -match '^Write-Step 4 ')
    $InstallerText = [IO.File]::ReadAllText($InstallerFull)
    $Step3 = $InstallerText.Substring($Steps[0].Extent.StartOffset, $Steps[1].Extent.StartOffset - $Steps[0].Extent.StartOffset)
    Assert-Contract 'step3_slice_excludes_step4' ($Step3 -notmatch 'Write-Step 4 ')

    $PythonHelper = Join-Path $FixtureRoot 'fixture.py'
    Write-FixtureFile $PythonHelper @'
import functools, hashlib, http.server, importlib.machinery, importlib.util, json, os
from pathlib import Path
import py_compile, struct, sys, zipfile
STAMP = 315561600
OLD = "version = '1.0.7'\ndef behavior(): return 'old'\n"
NEW = "version = '1.0.8'\ndef behavior(): return 'new'\n"
assert len(OLD.encode()) == len(NEW.encode())
def source(root, text, stamp=STAMP):
    root = Path(root)
    for name in ('app/api/server.py', 'pith_client/cli.py', 'scripts/tool.py',
                 'migrations/migration.py', 'integrations/adapter.py', 'pith_mcp.py',
                 'skill_deployer.py'):
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding='utf-8', newline='')
        os.utime(p, (stamp, stamp))
    (root / 'requirements.txt').write_text('# disposable fixture\n', encoding='utf-8')
def load(p):
    namespace = {}
    code = importlib.machinery.SourceFileLoader('fixture_probe', str(p)).get_code('fixture_probe')
    exec(code, namespace)
    return {'version': namespace['version'], 'behavior': namespace['behavior']()}
mode = sys.argv[1]
if mode == 'environment':
    assert sys.version_info[:2] == (3, 12), sys.version
    print(json.dumps({'python': sys.version, 'executable': sys.executable}))
elif mode == 'seed':
    root = Path(sys.argv[2]); stamp = int(sys.argv[3]) if len(sys.argv)>3 else STAMP; source(root, OLD, stamp)
    for p in root.rglob('*.py'):
        py_compile.compile(str(p), doraise=True, invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP)
    p = root / 'app/api/server.py'
    header = Path(importlib.util.cache_from_source(str(p))).read_bytes()[:16]
    magic, flags, mtime, size = struct.unpack('<4sIII', header)
    assert flags == 0 and mtime == stamp and size == len(OLD.encode())
    result = load(p); assert result == {'version':'1.0.7', 'behavior':'old'}
    print(json.dumps(dict(result, timestamp=mtime, size=size, flags=flags)))
elif mode == 'package':
    root = Path(sys.argv[2]); source(root, NEW)
    archive = Path(sys.argv[3]); archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(root.rglob('*')):
            if p.is_file():
                info = zipfile.ZipInfo(p.relative_to(root).as_posix(), (1980,1,1,0,0,0))
                info.compress_type = zipfile.ZIP_DEFLATED
                z.writestr(info, p.read_bytes())
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_name(archive.name + '.sha256').write_text(digest + '  pith-server-latest.zip\n', encoding='ascii')
    print(json.dumps({'sha256':digest, 'source_size':len(NEW.encode())}))
elif mode == 'load':
    p = Path(sys.argv[2]); print(json.dumps(dict(load(p), timestamp=int(p.stat().st_mtime), size=p.stat().st_size)))
elif mode == 'serve':
    root, ready, requests = sys.argv[2:5]
    class Handler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, fmt, *args):
            with open(requests, 'a', encoding='utf-8') as out:
                out.write(fmt % args + '\n')
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(Handler, directory=root))
    Path(ready).write_text(json.dumps({'port':server.server_port}), encoding='utf-8')
    server.serve_forever()
else:
    raise ValueError(mode)
'@
    $EnvProof = Invoke-FixturePython @('environment') 'python-environment'
    $Summary.python = $EnvProof.python

    # T1/T2/T5: concrete files and hash preservation, not only classification.
    $TestHome = Join-Path $FixtureRoot 'helper home'
    $Server = Join-Path $TestHome 'pith-server'
    $Caches = @('app\api\__pycache__\server.cpython-312.pyc', 'app\legacy.pyc', 'pith_client\__pycache__\cli.cpython-312.pyc', 'scripts\tool.pyc', 'migrations\migration.pyc', 'integrations\adapter.pyc', '__pycache__\pith_mcp.cpython-312.opt-1.pyc', '__pycache__\skill_deployer.cpython-311.pyc', 'pith_mcp.pyc')
    $Keep = @('app\api\server.py', 'app\api\__pycache__\user.txt', '.env', 'custom\user.pyc', '__pycache__\user.cpython-312.pyc', 'root-user.pyc')
    $Outside = @('config\openai-tunnel-client.json', 'runtime\python\runtime.pyc', 'venv\library.pyc', 'brain.db')
    $Preserved = @{}
    foreach ($Relative in @($Caches + $Keep)) { Write-FixtureFile (Join-Path $Server $Relative) 'fixture sentinel' }
    foreach ($Relative in $Keep) { $Path=Join-Path $Server $Relative; $Preserved[$Path]=Get-PithSha256 $Path }
    foreach ($Relative in $Outside) { $Path=Join-Path $TestHome $Relative; Write-FixtureFile $Path 'outside sentinel'; $Preserved[$Path]=Get-PithSha256 $Path }
    Assert-Contract 'owned_removal_exact_count' ((Clear-PithApplicationBytecode $TestHome $Server) -eq $Caches.Count)
    foreach ($Relative in $Caches) { Assert-Contract "removed:$Relative" (-not (Test-Path -LiteralPath (Join-Path $Server $Relative))) }
    foreach ($Path in $Preserved.Keys) { Assert-Contract "hash_preserved:$($Path.Substring($TestHome.Length))" ((Get-PithSha256 $Path) -eq $Preserved[$Path]) }
    Assert-Contract 'repeat_zero' ((Clear-PithApplicationBytecode $TestHome $Server) -eq 0)
    Assert-Throws 'wrong_server_rejected' { Clear-PithApplicationBytecode $TestHome $TestHome }
    Assert-Throws 'relative_rejected' { Clear-PithApplicationBytecode '.' '.\pith-server' }
    Assert-Throws 'volume_rejected' { Clear-PithApplicationBytecode 'C:\' 'C:\pith-server' }
    Assert-Throws 'root_relative_rejected' { Clear-PithApplicationBytecode '\fixture' '\fixture\pith-server' }
    Assert-Throws 'drive_relative_rejected' { Clear-PithApplicationBytecode 'C:fixture' 'C:fixture\pith-server' }
    $MissingHome=Join-Path $FixtureRoot 'missing home'
    Assert-Contract 'missing_tree_zero' ((Clear-PithApplicationBytecode $MissingHome (Join-Path $MissingHome 'pith-server')) -eq 0)
    # ADVERSARIAL-FP: valid unowned near-boundary inputs must remain untouched.
    foreach ($Relative in @('application\cache.pyc','app-custom\cache.pyc','custom\app\cache.pyc','__pycache__\pith_mcp_extra.cpython-312.pyc','app\api\source.py')) {
        $Path=Join-Path $Server $Relative; Write-FixtureFile $Path 'near-boundary sentinel'
        Assert-Contract "classifier_preserves:$Relative" (-not (Test-PithOwnedApplicationBytecode $Relative))
        $Preserved[$Path]=Get-PithSha256 $Path
    }
    foreach ($Relative in @('app\..\custom\cache.pyc','app\.\cache.pyc','C:\app\cache.pyc','\app\cache.pyc','app/cache.pyc','app\*.pyc','app\?.pyc',('app\bad'+[char]0+'.pyc'),('app\bad'+[char]10+'.pyc'),'app\\cache.pyc','app\cache.pyc:stream')) {
        Assert-Contract 'classifier_rejects_noncanonical' (-not (Test-PithOwnedApplicationBytecode $Relative))
    }
    Assert-Contract 'classifier_uppercase_valid' (Test-PithOwnedApplicationBytecode 'APP\API\SERVER.PYC')
    Assert-Contract 'classifier_unicode_valid' (Test-PithOwnedApplicationBytecode ('app\'+[char]0x03B1+'.pyc'))
    Assert-Contract 'classifier_numeric_safe' (-not (Test-PithOwnedApplicationBytecode 123))
    Assert-Throws 'classifier_null_binding' { Test-PithOwnedApplicationBytecode $null }
    Assert-Throws 'classifier_empty_binding' { Test-PithOwnedApplicationBytecode '' }
    $Locked=Join-Path $Server 'app\locked.pyc'; Write-FixtureFile $Locked 'locked'
    $Lock=[IO.File]::Open($Locked,'Open','ReadWrite','None')
    try { Assert-Throws 'locked_cleanup_fails' { Clear-PithApplicationBytecode $TestHome $Server } }
    finally { $Lock.Dispose() }
    Assert-Contract 'released_lock_retry' ((Clear-PithApplicationBytecode $TestHome $Server) -eq 1)
    foreach ($Path in $Preserved.Keys) { Assert-Contract "repeat_hash_preserved:$($Path.Substring($TestHome.Length))" ((Get-PithSha256 $Path) -eq $Preserved[$Path]) }

    # T3: real static junction and linked-ancestor boundaries, target hash checked.
    # PS 5.1 New-Item Junction resolves Target as wildcard; use an explicit GUID
    # sibling target without brackets, while retaining brackets in tested paths.
    $External=$FixtureRoot.Replace('[literal]', 'external') + ' target'
    $Summary.external_fixture_target=$External
    $ExternalCache=Join-Path $External 'outside.pyc'; Write-FixtureFile $ExternalCache 'external sentinel'
    $ExternalHash=Get-PithSha256 $ExternalCache
    $Junction=Join-Path $Server 'app\junction'
    $CompoundCache=Join-Path $Server 'app\compound.pyc'; Write-FixtureFile $CompoundCache 'compound sentinel'
    New-Item -ItemType Junction -Path $Junction -Target $External -ErrorAction Stop | Out-Null
    try {
        Assert-Throws 'static_junction_rejected' { Clear-PithApplicationBytecode $TestHome $Server } 'Reparse'
        Assert-Contract 'junction_target_unchanged' ((Get-PithSha256 $ExternalCache) -eq $ExternalHash)
        Assert-Contract 'junction_rejected_before_any_cache_deletion' ([IO.File]::Exists($CompoundCache))
    } finally { [IO.Directory]::Delete($Junction) }
    $AncestorLink=Join-Path $FixtureRoot 'linked ancestor'
    New-Item -ItemType Junction -Path $AncestorLink -Target $External -ErrorAction Stop | Out-Null
    try {
        $LinkedHome=Join-Path $AncestorLink '.pith'
        Assert-Throws 'linked_ancestor_rejected' { Get-PithServerTreeInventory $LinkedHome (Join-Path $LinkedHome 'pith-server') } 'Unsafe'
        Assert-Contract 'ancestor_target_unchanged' ((Get-PithSha256 $ExternalCache) -eq $ExternalHash)
    } finally { [IO.Directory]::Delete($AncestorLink) }
    $Symbolic=Join-Path $Server 'app\symbolic.pyc'
    $SymbolicCreated=$false
    try {
        try { New-Item -ItemType SymbolicLink -Path $Symbolic -Target $ExternalCache -ErrorAction Stop | Out-Null; $SymbolicCreated=$true }
        catch {
            if ($_.Exception.Message -notmatch 'privilege|administrator|not held') { throw }
            $Summary.symlink_skip=$_.Exception.Message
        }
        if ($SymbolicCreated) {
            Assert-Throws 'symlink_file_rejected' { Clear-PithApplicationBytecode $TestHome $Server } 'Reparse'
            Assert-Contract 'symlink_target_hash_preserved' ((Get-PithSha256 $ExternalCache) -eq $ExternalHash)
        }
    } finally { if ($SymbolicCreated) { [IO.File]::Delete($Symbolic) } }

    # T4: actual bounded inventory plus isolated metadata/residual fault injection.
    $CapHome=Join-Path $FixtureRoot 'inventory cap'; $CapServer=Join-Path $CapHome 'pith-server'
    [IO.Directory]::CreateDirectory($CapServer) | Out-Null
    for ($Index=0; $Index -lt 100001; $Index++) { [IO.File]::WriteAllBytes((Join-Path $CapServer ($Index.ToString()+'.txt')), [byte[]]@()) }
    Assert-Throws 'inventory_100001_rejected' { Get-PithServerTreeInventory $CapHome $CapServer } 'safety inventory limit'
    Remove-Item -LiteralPath (Join-Path $CapServer '100000.txt') -Force -ErrorAction Stop
    Assert-Contract 'inventory_exact_100000_allowed' (@(Get-PithServerTreeInventory $CapHome $CapServer).Count -eq 100000)
    $FaultHome=Join-Path $FixtureRoot 'fault home'; $FaultServer=Join-Path $FaultHome 'pith-server'
    $FaultCache=Join-Path $FaultServer 'app\fault.pyc'; Write-FixtureFile $FaultCache 'fault sentinel'
    & {
        function Get-ChildItem { param($LiteralPath,[switch]$Force,$ErrorAction) throw 'injected enumeration denial' }
        Assert-Throws 'enumeration_denial_fails' { Clear-PithApplicationBytecode $FaultHome $FaultServer } 'injected enumeration denial'
        Assert-Contract 'enumeration_denial_no_delete' ([IO.File]::Exists($FaultCache))
    }
    & {
        function Remove-Item {
            param($LiteralPath,[switch]$Force,$ErrorAction)
            Microsoft.PowerShell.Management\Remove-Item -LiteralPath $LiteralPath -Force -ErrorAction Stop
            Write-FixtureFile (Join-Path $FaultServer 'app\residual.pyc') 'injected residual'
        }
        Assert-Throws 'residual_rescan_fails' { Clear-PithApplicationBytecode $FaultHome $FaultServer } 'Residual'
    }

    # T6-T9: source-backed Step 3 in a real script, so MyInvocation retains its path.
    # PS 5.1 Expand-Archive internally wildcard-tests DestinationPath. Its existing
    # bracket-destination limitation is not a source-identity fix. Helper tests
    # above retain brackets; acquisition uses GUID siblings with spaces only.
    $AcquisitionRoot=$FixtureRoot.Replace('[literal]', 'acquisition')
    $Summary.acquisition_fixture_root=$AcquisitionRoot
    $PackageSource=Join-Path $AcquisitionRoot 'package source'
    $HttpRoot=Join-Path $AcquisitionRoot 'http root'
    $Package=Join-Path $HttpRoot 'pith-server-latest.zip'
    $PackageProof=Invoke-FixturePython @('package',$PackageSource,$Package) 'package'
    # ZIP DOS time is interpreted in local time by PS 5.1. Calibrate with the real
    # extractor, then seed eligible old caches at that exact resulting timestamp.
    $Calibration=Join-Path $AcquisitionRoot 'zip timestamp calibration'
    Expand-Archive -LiteralPath $Package -DestinationPath $Calibration -Force -ErrorAction Stop
    $ZipMetadata=Invoke-FixturePython @('load',(Join-Path $Calibration 'app\api\server.py')) 'zip-calibration'
    $Summary.zip_extracted_timestamp=$ZipMetadata.timestamp
    $Ready=Join-Path $FixtureRoot 'http-ready.json'; $RequestLog=Join-Path $FixtureRoot 'http-requests.txt'
    $HttpStdout=Join-Path $FixtureRoot 'http.stdout.txt'; $HttpStderr=Join-Path $FixtureRoot 'http.stderr.txt'
    $HttpArguments=(@('-B','-I',$PythonHelper,'serve',$HttpRoot,$Ready,$RequestLog) | ForEach-Object { Quote-NativeArgument $_ }) -join ' '
    $HttpInfo=New-Object Diagnostics.ProcessStartInfo
    $HttpInfo.FileName=$PythonFull; $HttpInfo.Arguments=$HttpArguments; $HttpInfo.UseShellExecute=$false; $HttpInfo.CreateNoWindow=$true
    $OwnedHttp=New-Object Diagnostics.Process
    $OwnedHttp.StartInfo=$HttpInfo
    if (-not $OwnedHttp.Start()) { throw 'Loopback child did not start' }
    $ReadyDeadline=[DateTime]::UtcNow.AddSeconds(15)
    while (-not [IO.File]::Exists($Ready) -and -not $OwnedHttp.HasExited -and [DateTime]::UtcNow -lt $ReadyDeadline) { Start-Sleep -Milliseconds 100 }
    Assert-Contract 'loopback_server_ready' ([IO.File]::Exists($Ready) -and -not $OwnedHttp.HasExited)
    $Port=([IO.File]::ReadAllText($Ready) | ConvertFrom-Json).port
    $PsExe=(Get-Process -Id $PID).Path
    $ExpectedRoute=@{ distribution='distribution_directory'; adjacent='adjacent_verified_zip'; hosted='hosted_verified_zip' }
    foreach ($Route in @('distribution','adjacent','hosted')) {
        $RouteRoot=Join-Path $AcquisitionRoot ($Route+' route')
        $RouteHome=Join-Path $RouteRoot 'installed home'
        $RouteServer=Join-Path $RouteHome 'pith-server'
        $ExpectedStamp=if ($Route -eq 'distribution') { 315561600 } else { $ZipMetadata.timestamp }
        $Seed=Invoke-FixturePython @('seed',$RouteServer,[string]$ExpectedStamp) ($Route+'-seed')
        Assert-Contract "${Route}:timestamp_cache_seed" ($Seed.flags -eq 0 -and $Seed.timestamp -eq $ExpectedStamp -and $Seed.behavior -eq 'old')
        if ($Route -ne 'distribution') {
            # Real counterfactual: overlay new ZIP bytes without invalidation and
            # prove a fresh normal loader still selects old timestamp bytecode.
            Expand-Archive -LiteralPath $Package -DestinationPath $RouteServer -Force -ErrorAction Stop
            $Collision=Invoke-FixturePython @('load',(Join-Path $RouteServer 'app\api\server.py')) ($Route+'-collision-control')
            Assert-Contract "${Route}:zip_overlay_without_clear_loads_old" ($Collision.version -eq '1.0.7' -and $Collision.behavior -eq 'old' -and $Collision.timestamp -eq $Seed.timestamp -and $Collision.size -eq $Seed.size)
            Assert-Contract "${Route}:zip_overlay_source_is_new" ([IO.File]::ReadAllText((Join-Path $RouteServer 'app\api\server.py')).Contains("version = '1.0.8'"))
        }
        $RouteSentinels=@{}
        foreach ($Relative in @('.env','custom\user.pyc','__pycache__\user.cpython-312.pyc')) {
            $Path=Join-Path $RouteServer $Relative; Write-FixtureFile $Path 'route preservation sentinel'; $RouteSentinels[$Path]=Get-PithSha256 $Path
        }
        foreach ($Relative in $Outside) { $Path=Join-Path $RouteHome $Relative; Write-FixtureFile $Path 'route outside sentinel'; $RouteSentinels[$Path]=Get-PithSha256 $Path }
        $Distribution=Join-Path $RouteRoot 'distribution'
        $FixtureInstaller=Join-Path $Distribution 'scripts\install.ps1'
        if ($Route -eq 'distribution') { Copy-Item -LiteralPath $PackageSource -Destination $Distribution -Recurse -Force -ErrorAction Stop }
        else { [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($FixtureInstaller)) | Out-Null }
        if ($Route -eq 'adjacent') {
            Copy-Item -LiteralPath $Package -Destination (Join-Path ([IO.Path]::GetDirectoryName($FixtureInstaller)) 'pith-server-latest.zip') -Force
            Copy-Item -LiteralPath ($Package+'.sha256') -Destination (Join-Path ([IO.Path]::GetDirectoryName($FixtureInstaller)) 'pith-server-latest.zip.sha256') -Force
        }
        Write-FixtureFile $FixtureInstaller $Step3
        Assert-Contract "${Route}:exact_step3_bytes" ([IO.File]::ReadAllText($FixtureInstaller) -ceq $Step3)
        $ExtractedHelpers=Join-Path $RouteRoot 'helpers.ps1'; Write-FixtureFile $ExtractedHelpers $HelperSource
        $ConfigPath=Join-Path $RouteRoot 'config.json'
        $Config=[ordered]@{ home=$RouteHome; installer=$FixtureInstaller; helpers=$ExtractedHelpers; url="http://127.0.0.1:$Port"; require_local=($Route -ne 'hosted') }
        Write-FixtureFile $ConfigPath ($Config | ConvertTo-Json)
        $Wrapper=Join-Path $RouteRoot 'wrapper.ps1'
        Write-FixtureFile $Wrapper @'
param([Parameter(Mandatory=$true)][string]$ConfigPath)
$ErrorActionPreference='Stop'
$Config=Get-Content -LiteralPath $ConfigPath -Raw | ConvertFrom-Json
. $Config.helpers
function Write-Step { param($Number,$Message) Write-Host "FIXTURE_STEP=$Number" }
function Write-Success { param($Message) Write-Host $Message }
function Write-Error-Custom { param($Message) throw $Message }
$PithHome=$Config.home
$DownloadUrl=$Config.url; $ChecksumUrl=$Config.url
$PithServerFilename='pith-server-latest.zip'; $PithChecksumFilename='pith-server-latest.zip.sha256'
$env:PITH_REQUIRE_LOCAL_PACKAGE=if ($Config.require_local) { '1' } else { '0' }
try {
    & $Config.installer
    Write-Host 'FIXTURE_STEP4_REACHED=1'
    exit 0
} catch {
    [Console]::Error.WriteLine($_.Exception.Message)
    exit 1
}
'@
        for ($Run=1; $Run -le 2; $Run++) {
            $Result=Invoke-FixtureProcess $PsExe @('-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass','-File',$Wrapper,'-ConfigPath',$ConfigPath) ($Route+'-run'+$Run)
            Assert-Contract "${Route}:run${Run}:child_success" ($Result.exit_code -eq 0)
            Assert-Contract "${Route}:run${Run}:correct_branch" ($Result.stdout.Contains('PITH_PACKAGE_SOURCE='+$ExpectedRoute[$Route]))
            Assert-Contract "${Route}:run${Run}:single_cleanup_marker" (([regex]::Matches($Result.stdout,'(?m)^PITH_APPLICATION_BYTECODE_REMOVED=\d+\r?$')).Count -eq 1)
            Assert-Contract "${Route}:run${Run}:step4_boundary" ($Result.stdout.Contains('FIXTURE_STEP4_REACHED=1'))
            Assert-Contract "${Route}:run${Run}:no_owned_cache" (@(Get-PithServerTreeInventory $RouteHome $RouteServer | Where-Object { -not $_.PSIsContainer -and (Test-PithOwnedApplicationBytecode ($_.FullName.Substring($RouteServer.Length+1))) }).Count -eq 0)
            foreach ($Relative in @('app\api\server.py','pith_client\cli.py','scripts\tool.py','migrations\migration.py','integrations\adapter.py','pith_mcp.py','skill_deployer.py')) {
                $Behavior=Invoke-FixturePython @('load',(Join-Path $RouteServer $Relative)) ($Route+'-run'+$Run+'-'+$Relative.Replace('\','-'))
                Assert-Contract "${Route}:run${Run}:collision_metadata:$Relative" ($Behavior.timestamp -eq $Seed.timestamp -and $Behavior.size -eq $Seed.size)
                Assert-Contract "${Route}:run${Run}:fresh_behavior:$Relative" ($Behavior.version -eq '1.0.8' -and $Behavior.behavior -eq 'new')
            }
            foreach ($Directory in @('app','pith_client','scripts','migrations','integrations')) { Assert-Contract "${Route}:run${Run}:no_nested:$Directory" (-not (Test-Path -LiteralPath (Join-Path $RouteServer ($Directory+'\'+$Directory)))) }
            foreach ($Path in $RouteSentinels.Keys) { Assert-Contract "${Route}:run${Run}:preserved:$($Path.Substring($RouteHome.Length))" ((Get-PithSha256 $Path) -eq $RouteSentinels[$Path]) }
        }
        # Lock a standalone cache not overwritten by package acquisition.
        $Locked=Join-Path $RouteServer 'app\locked.pyc'; Write-FixtureFile $Locked 'route lock'
        $Lock=[IO.File]::Open($Locked,'Open','ReadWrite','None')
        try { $Result=Invoke-FixtureProcess $PsExe @('-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass','-File',$Wrapper,'-ConfigPath',$ConfigPath) ($Route+'-locked') }
        finally { $Lock.Dispose() }
        Assert-Contract "${Route}:locked_fails_before_boundary" ($Result.exit_code -ne 0 -and -not $Result.stdout.Contains('FIXTURE_STEP4_REACHED=1') -and -not $Result.stdout.Contains('PITH_APPLICATION_BYTECODE_REMOVED='))
        Assert-Contract "${Route}:locked_cache_retained" ([IO.File]::Exists($Locked))
        # Helpers can clean the fixture after its lock is released; no live path involved.
        Clear-PithApplicationBytecode $RouteHome $RouteServer | Out-Null
        if ($Route -ne 'distribution') {
            $BadChecksum=if ($Route -eq 'adjacent') { Join-Path ([IO.Path]::GetDirectoryName($FixtureInstaller)) 'pith-server-latest.zip.sha256' } else { $Package+'.sha256' }
            $GoodChecksum=[IO.File]::ReadAllText($BadChecksum)
            Write-FixtureFile $BadChecksum (('0'*64)+'  pith-server-latest.zip')
            try { $Result=Invoke-FixtureProcess $PsExe @('-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass','-File',$Wrapper,'-ConfigPath',$ConfigPath) ($Route+'-bad-checksum') }
            finally { Write-FixtureFile $BadChecksum $GoodChecksum }
            Assert-Contract "${Route}:bad_checksum_fails_before_cleanup" ($Result.exit_code -ne 0 -and -not $Result.stdout.Contains('FIXTURE_STEP4_REACHED=1') -and -not $Result.stdout.Contains('PITH_APPLICATION_BYTECODE_REMOVED='))
        }
    }
    $Requests=[IO.File]::ReadAllText($RequestLog)
    Assert-Contract 'hosted_real_zip_get' ($Requests -match 'GET /pith-server-latest.zip HTTP')
    Assert-Contract 'hosted_real_checksum_get' ($Requests -match 'GET /pith-server-latest.zip.sha256 HTTP')
    $Summary.package_sha256=$PackageProof.sha256
    $Summary.status='completed'
}
catch {
    $Summary.status='error'; $Summary.error=$_.Exception.Message; $Summary.error_position=$_.InvocationInfo.PositionMessage
    # Bounded fixture-only counterfactual evidence for deliberately broken candidates.
    if ($RouteServer -and $AcquisitionRoot -and $RouteServer.StartsWith($AcquisitionRoot + '\') -and [IO.File]::Exists((Join-Path $RouteServer 'app\api\server.py'))) {
        try { $Summary.failure_behavior=Invoke-FixturePython @('load',(Join-Path $RouteServer 'app\api\server.py')) 'failed-route-behavior' }
        catch { $Summary.failure_behavior_error=$_.Exception.Message }
    }
}
finally {
    if ($OwnedHttp) {
        if (-not $OwnedHttp.HasExited) { $OwnedHttp.Kill(); $OwnedHttp.WaitForExit() }
        $OwnedHttp.Dispose()
    }
    $Summary.rows=$Rows.ToArray()
    $Summary.finished_utc=[DateTime]::UtcNow.ToString('o')
    # Retain GUID fixture trees (especially failures). No recursive fixture purge.
    Write-FixtureFile ([IO.Path]::GetFullPath($OutputPath)) ($Summary | ConvertTo-Json -Depth 8)
}
if ($Summary.status -ne 'completed') { Write-Error $Summary.error -ErrorAction Continue; exit 1 }
Write-Host "Windows upgrade contract passed: $($Rows.Count) rows. Proof: $OutputPath"
