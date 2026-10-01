# Pith CLI wrapper for Windows
$PithHome = "__PITH_HOME__"
$VenvPath = "__VENV_PATH__"
if ($VenvPath -match '^__[^_]+__$') {
    $VenvPath = "$PithHome\venv"
}
$PithServerPath = "$PithHome\pith-server"
$PythonExe = "$VenvPath\Scripts\python.exe"
$PipExe = "$VenvPath\Scripts\pip.exe"

function Import-PithEnvFile {
    param([string]$EnvFile)

    if (-not (Test-Path $EnvFile)) {
        return
    }

    foreach ($RawLine in Get-Content -Path $EnvFile) {
        $Line = $RawLine.Trim()
        if (-not $Line -or $Line.StartsWith("#") -or $Line -notmatch "=") {
            continue
        }

        $Parts = $Line -split "=", 2
        $Name = $Parts[0].Trim()
        $Value = $Parts[1].Trim().Trim('"').Trim("'")
        if ($Name -notmatch '^[A-Za-z_][A-Za-z0-9_]*$') {
            continue
        }
        if ([Environment]::GetEnvironmentVariable($Name, "Process")) {
            continue
        }
        [Environment]::SetEnvironmentVariable($Name, $Value, "Process")
    }
}

function Import-PithApiKeyFile {
    if ($env:PITH_API_KEY) {
        return
    }

    $ApiKeyFile = "$PithHome\config\api.key"
    if (-not (Test-Path $ApiKeyFile)) {
        return
    }

    $ApiKey = (Get-Content -Path $ApiKeyFile -Raw).Trim()
    if ($ApiKey) {
        $env:PITH_API_KEY = $ApiKey
    }
}

function Resolve-PithDefaultDataDir {
    $Profile = if ($env:PITH_PROFILE) { $env:PITH_PROFILE } else { "default" }
    $DataRoot = Join-Path ([Environment]::GetFolderPath("UserProfile")) "pith-data"
    return (Join-Path $DataRoot $Profile)
}

function Resolve-PithDataDir {
    if ($env:PITH_DATA_DIR) { return $env:PITH_DATA_DIR }
    $EnvFile = "$PithServerPath\.env"
    if (Test-Path $EnvFile) {
        $DataDirLine = Get-Content $EnvFile |
            Where-Object { $_ -match '^PITH_DATA_DIR=' } |
            Select-Object -First 1
        if ($DataDirLine) {
            $ConfiguredDataDir = (($DataDirLine -split '=', 2)[1]).Trim()
            if ($ConfiguredDataDir) { return $ConfiguredDataDir }
        }
    }
    return (Resolve-PithDefaultDataDir)
}

# DEBT-082: Single DB resolution function (pith.db first, brain.db fallback)
function Resolve-DbPath {
    $DataDir = Resolve-PithDataDir
    $PithDb = Join-Path $DataDir "pith.db"
    $BrainDb = Join-Path $DataDir "brain.db"
    if (Test-Path $PithDb) { return $PithDb }
    if (Test-Path $BrainDb) { return $BrainDb }
    return $PithDb
}

function Resolve-PithPort {
    if ($env:PITH_PORT) { return $env:PITH_PORT }
    $EnvFile = "$PithServerPath\.env"
    if (Test-Path $EnvFile) {
        $PortLine = Get-Content $EnvFile |
            Where-Object { $_ -match '^(PITH_PORT|PORT)=' } |
            Select-Object -First 1
        if ($PortLine) {
            return (($PortLine -split '=', 2)[1]).Trim()
        }
    }
    return "8000"
}

$PithNativeApiOperations = @{
    health = @{ method = "GET"; path = "/health"; auth = $false }
    readyz = @{ method = "GET"; path = "/readyz"; auth = $false }
    pith_health = @{ method = "GET"; path = "/pith_health"; auth = $true }
    stats = @{ method = "GET"; path = "/pith_stats"; auth = $true }
    session_start = @{ method = "POST"; path = "/session_start"; auth = $true }
    connection_proof = @{ method = "POST"; path = "/conversation_turn"; auth = $true }
    conversation_turn = @{ method = "POST"; path = "/conversation_turn"; auth = $true }
    checkpoint = @{ method = "POST"; path = "/checkpoint"; auth = $true }
    session_end = @{ method = "POST"; path = "/session_end"; auth = $true }
    session_learn = @{ method = "POST"; path = "/session_learn"; auth = $true }
    surface_activity = @{ method = "GET"; path = "/diagnostics/surface_activity"; auth = $true }
    workstreams = @{ method = "POST"; path = "/pith_threads"; auth = $true }
}
$PithNativeLifecycleEventOperations = @("conversation_turn", "checkpoint", "session_end", "session_learn")

function ConvertTo-PithHashtable {
    param([object]$Value)

    if ($null -eq $Value) {
        return $null
    }
    if ($Value -is [System.Collections.IDictionary]) {
        $Result = [ordered]@{}
        foreach ($Key in $Value.Keys) {
            $Result[$Key] = ConvertTo-PithHashtable -Value $Value[$Key]
        }
        return $Result
    }
    if ($Value -is [System.Management.Automation.PSCustomObject]) {
        $Result = [ordered]@{}
        foreach ($Property in $Value.PSObject.Properties) {
            $Result[$Property.Name] = ConvertTo-PithHashtable -Value $Property.Value
        }
        return $Result
    }
    if (($Value -is [System.Collections.IEnumerable]) -and -not ($Value -is [string])) {
        $Items = New-Object System.Collections.Generic.List[object]
        foreach ($Item in $Value) {
            $Items.Add((ConvertTo-PithHashtable -Value $Item))
        }
        # PowerShell enumerates function output. The unary comma preserves
        # zero-, one-, and many-item JSON arrays at the caller boundary.
        return ,([object[]]$Items.ToArray())
    }
    return $Value
}

function New-PithApiError {
    param(
        [string]$Code,
        [string]$Message,
        [string]$Operation,
        [string]$TransportMode,
        [object]$StatusCode = $null,
        [object]$Body = $null
    )

    $Result = [ordered]@{
        error = $true
        code = $Code
        message = (Redact-PithDiagnosticText -Value $Message)
        operation = $Operation
        transport_mode = $TransportMode
    }
    if ($null -ne $StatusCode) {
        $Result.status_code = $StatusCode
    }
    if ($null -ne $Body) {
        $Result.body = $Body
    }
    return $Result
}

function ConvertTo-PithNativeJson {
    param(
        [object]$Value,
        [int]$Depth = 20
    )

    return (ConvertTo-Json -InputObject $Value -Compress -Depth $Depth)
}

function Write-PithNativeJson {
    param(
        [object]$Value,
        [int]$Depth = 20
    )

    [Console]::Out.WriteLine((ConvertTo-PithNativeJson -Value $Value -Depth $Depth))
}

function Test-PithNativeApiBodyEmpty {
    param([object]$Body)

    if ($null -eq $Body) {
        return $true
    }
    if ($Body -is [System.Collections.IDictionary]) {
        return ([int]$Body.Count -eq 0)
    }
    if ($Body -is [System.Management.Automation.PSCustomObject]) {
        return (@($Body.PSObject.Properties).Count -eq 0)
    }
    return $false
}

function Get-PithApiCommandOptions {
    param([object[]]$RemainingArgs)

    $Options = [ordered]@{
        operation = $null
        stdin_json = $false
        json_file = $null
        base_url = if ($env:PITH_API_URL) { $env:PITH_API_URL.TrimEnd('/') } else { "http://localhost:$($env:PITH_PORT)" }
        timeout_seconds = 8
        use_python = $false
        error_message = $null
    }

    if (($null -eq $RemainingArgs) -or ($RemainingArgs.Count -lt 1)) {
        $Options.use_python = $true
        return $Options
    }

    $Options.operation = [string]$RemainingArgs[0]
    for ($i = 1; $i -lt $RemainingArgs.Count; $i++) {
        $Arg = [string]$RemainingArgs[$i]
        switch ($Arg) {
            "--stdin-json" {
                $Options.stdin_json = $true
            }
            "--json-file" {
                if ($i + 1 -ge $RemainingArgs.Count) {
                    $Options.error_message = "--json-file requires a path"
                    return $Options
                }
                $i += 1
                $Options.json_file = [string]$RemainingArgs[$i]
            }
            "--base-url" {
                if ($i + 1 -ge $RemainingArgs.Count) {
                    $Options.error_message = "--base-url requires a URL"
                    return $Options
                }
                $i += 1
                $Options.base_url = ([string]$RemainingArgs[$i]).TrimEnd('/')
            }
            "--timeout" {
                if ($i + 1 -ge $RemainingArgs.Count) {
                    $Options.error_message = "--timeout requires a value"
                    return $Options
                }
                $i += 1
                $ParsedTimeout = 0.0
                if (-not [double]::TryParse([string]$RemainingArgs[$i], [ref]$ParsedTimeout)) {
                    $Options.error_message = "--timeout must be numeric"
                    return $Options
                }
                $Options.timeout_seconds = [Math]::Max(1, [int][Math]::Ceiling($ParsedTimeout))
            }
            { $_ -in @("--example", "--schema", "--transport-mode") } {
                $Options.use_python = $true
                return $Options
            }
            default {
                $Options.use_python = $true
                return $Options
            }
        }
    }

    if ($Options.stdin_json -and $Options.json_file) {
        $Options.error_message = "--stdin-json and --json-file are mutually exclusive"
    }
    return $Options
}

function Read-PithNativeApiPayload {
    param([object]$Options)

    $RawPayload = $null
    if ($Options.stdin_json) {
        $PipelineInput = @($script:PithCliPipelineInput | ForEach-Object { [string]$_ })
        if ($PipelineInput.Count -gt 0) {
            $RawPayload = $PipelineInput -join [Environment]::NewLine
        }
        else {
            $RawPayload = [Console]::In.ReadToEnd()
        }
    }
    elseif ($Options.json_file) {
        if (-not (Test-Path -LiteralPath $Options.json_file)) {
            throw "JSON file not found: $($Options.json_file)"
        }
        $RawPayload = Get-Content -LiteralPath $Options.json_file -Raw
    }

    if (-not $RawPayload) {
        return [ordered]@{}
    }

    try {
        return (ConvertTo-PithHashtable -Value ($RawPayload | ConvertFrom-Json))
    }
    catch {
        throw "Invalid JSON payload: $($_.Exception.Message)"
    }
}

function Add-PithNativeApiDefaults {
    param(
        [string]$Operation,
        [object]$Payload
    )

    if ($null -eq $Payload) {
        $Payload = [ordered]@{}
    }
    if ((($Operation -eq "conversation_turn") -or ($Operation -eq "connection_proof")) -and ($Payload -is [System.Collections.IDictionary])) {
        if (-not $Payload.Contains("surface_id")) {
            $Payload["surface_id"] = "local_api_cli"
        }
        if (($Operation -eq "connection_proof") -and (-not $Payload.Contains("message"))) {
            $Payload["message"] = "Pith connection proof"
        }
        if (-not $Payload.Contains("extracted_concepts_json")) {
            $Payload["extracted_concepts_json"] = "[]"
        }
    }
    if (($Operation -eq "workstreams") -and ($Payload -is [System.Collections.IDictionary]) -and (-not $Payload.Contains("action")) -and $Payload.Contains("operation")) {
        $Payload["action"] = $Payload["operation"]
    }
    return $Payload
}

function ConvertTo-PithQueryString {
    param([object]$Payload)

    if (($null -eq $Payload) -or -not ($Payload -is [System.Collections.IDictionary]) -or ($Payload.Count -eq 0)) {
        return ""
    }

    $Pairs = New-Object System.Collections.Generic.List[string]
    foreach ($Key in $Payload.Keys) {
        $Value = $Payload[$Key]
        if ($null -eq $Value) {
            continue
        }
        if (($Value -is [System.Collections.IDictionary]) -or (($Value -is [System.Collections.IEnumerable]) -and -not ($Value -is [string]))) {
            $Value = ($Value | ConvertTo-Json -Compress -Depth 8)
        }
        $Pairs.Add(("{0}={1}" -f [Uri]::EscapeDataString([string]$Key), [Uri]::EscapeDataString([string]$Value)))
    }
    if ($Pairs.Count -eq 0) {
        return ""
    }
    return "?" + ($Pairs -join "&")
}

function Get-PithNativeApiHeaders {
    param(
        [string]$Operation,
        [string]$TransportMode,
        [bool]$AuthRequired
    )

    $Headers = @{
        "Content-Type" = "application/json"
        "X-Pith-Transport" = $TransportMode
    }
    if ($AuthRequired) {
        if (-not $env:PITH_API_KEY) {
            throw "PITH_API_KEY unavailable for authenticated local API call"
        }
        $Headers["X-API-Key"] = $env:PITH_API_KEY
    }
    return $Headers
}

function Invoke-PithNativeApiHttp {
    param(
        [string]$ApiUrl,
        [string]$Path,
        [string]$Method,
        [hashtable]$Headers,
        [object]$Payload,
        [int]$TimeoutSeconds
    )

    $LastResult = $null
    foreach ($ProbeApiUrl in (Get-PithDiagnosticApiUrls -ApiUrl $ApiUrl)) {
        $Uri = "$ProbeApiUrl$Path"
        if ($Method -eq "GET") {
            $Uri += (ConvertTo-PithQueryString -Payload $Payload)
        }

        $Request = $null
        $Response = $null
        try {
            $Request = [System.Net.WebRequest]::Create($Uri)
            $Request.Method = $Method
            $Request.Timeout = $TimeoutSeconds * 1000
            $Request.ReadWriteTimeout = $TimeoutSeconds * 1000
            $Request.Proxy = $null
            $Request.ContentType = "application/json"
            foreach ($Name in $Headers.Keys) {
                if ($Name -ne "Content-Type") {
                    $Request.Headers[$Name] = [string]$Headers[$Name]
                }
            }
            if ($Method -ne "GET") {
                $BodyJson = ($Payload | ConvertTo-Json -Compress -Depth 20)
                $BodyBytes = [System.Text.Encoding]::UTF8.GetBytes($BodyJson)
                $Request.ContentLength = $BodyBytes.Length
                $RequestStream = $Request.GetRequestStream()
                try {
                    $RequestStream.Write($BodyBytes, 0, $BodyBytes.Length)
                }
                finally {
                    $RequestStream.Close()
                }
            }
            $Response = $Request.GetResponse()
            $StatusCode = [int]$Response.StatusCode
            $Reader = New-Object System.IO.StreamReader($Response.GetResponseStream())
            try {
                $Text = $Reader.ReadToEnd()
            }
            finally {
                $Reader.Close()
            }
            $Body = if ($Text) { ConvertTo-PithHashtable -Value ($Text | ConvertFrom-Json) } else { [ordered]@{} }
            return [ordered]@{
                ok = (($StatusCode -ge 200) -and ($StatusCode -lt 400))
                reachable = $true
                status_code = $StatusCode
                body = $Body
                message = $null
                api_url = $ProbeApiUrl
            }
        }
        catch [System.Net.WebException] {
            $StatusCode = $null
            $Body = $null
            if ($_.Exception.Response) {
                $StatusCode = [int]$_.Exception.Response.StatusCode
                try {
                    $Reader = New-Object System.IO.StreamReader($_.Exception.Response.GetResponseStream())
                    try {
                        $Text = $Reader.ReadToEnd()
                    }
                    finally {
                        $Reader.Close()
                    }
                    if ($Text) {
                        try { $Body = ConvertTo-PithHashtable -Value ($Text | ConvertFrom-Json) } catch { $Body = $Text }
                    }
                }
                finally {
                    $_.Exception.Response.Close()
                }
            }
            $LastResult = [ordered]@{
                ok = $false
                reachable = ($null -ne $StatusCode)
                status_code = $StatusCode
                body = $Body
                message = $_.Exception.Message
                failure_kind = [string]$_.Exception.Status
                api_url = $ProbeApiUrl
            }
            if ($LastResult.reachable) {
                return $LastResult
            }
        }
        catch {
            $LastResult = [ordered]@{
                ok = $false
                reachable = $false
                status_code = $null
                body = $null
                message = $_.Exception.Message
                failure_kind = "UnhandledException"
                api_url = $ProbeApiUrl
            }
        }
        finally {
            if ($null -ne $Response) {
                $Response.Close()
            }
        }
    }
    return $LastResult
}

function Test-PithRetryableConversationTurnStartup {
    param([object]$Result)

    if ($null -eq $Result) {
        return $true
    }
    if (-not $Result.reachable) {
        return ([string]$Result.failure_kind -in @("ConnectFailure", "ConnectionClosed", "NameResolutionFailure"))
    }
    if ([int]($Result.status_code) -ne 503) {
        return $false
    }
    $Body = $Result.body
    $Detail = ""
    if ($Body -is [System.Collections.IDictionary]) {
        $Detail = [string]$Body["detail"]
        if (-not $Detail) {
            $Detail = ($Body | ConvertTo-Json -Compress -Depth 8)
        }
    }
    elseif ($Body) {
        $Detail = [string]$Body
    }
    $Detail = $Detail.ToLowerInvariant()
    return (($Detail -like "*retrieval initialization*") -or ($Detail -like "*retrieval recovery*") -or ($Detail -like "*server startup*"))
}

function Get-PithPayloadValue {
    param(
        [object]$Payload,
        [string]$Name
    )

    if (($null -ne $Payload) -and ($Payload -is [System.Collections.IDictionary]) -and $Payload.Contains($Name)) {
        return $Payload[$Name]
    }
    return $null
}

function Write-PithNativeTransportEvent {
    param(
        [string]$Operation,
        [object]$Payload,
        [object]$ResultBody,
        [string]$TransportMode,
        [object]$ApiStatus,
        [string]$ApiUrl
    )

    if ($PithNativeLifecycleEventOperations -notcontains $Operation) {
        return
    }

    $Body = if ($ResultBody -is [System.Collections.IDictionary]) { $ResultBody } else { [ordered]@{} }
    $AutoLearned = if ($Body.Contains("auto_learned") -and ($Body["auto_learned"] -is [System.Collections.IDictionary])) { $Body["auto_learned"] } else { $null }
    $SurfaceId = Get-PithPayloadValue -Payload $Payload -Name "surface_id"
    if (-not $SurfaceId) {
        $SurfaceId = if ($Operation -eq "conversation_turn") { "local_api_cli" } else { "local_api_cli" }
    }
    $Entry = [ordered]@{
        ts = [DateTimeOffset]::UtcNow.ToString("o")
        event = "lifecycle_api_call"
        pid = $PID
        api_url = $ApiUrl
        operation = $Operation
        transport_mode = $TransportMode
        surface_id = $SurfaceId
        session_id = Get-PithPayloadValue -Payload $Payload -Name "session_id"
        resolved_session_id = if ($Body.Contains("resolved_session_id")) { $Body["resolved_session_id"] } elseif ($Body.Contains("session_id")) { $Body["session_id"] } else { $null }
        origin_id = if ((Get-PithPayloadValue -Payload $Payload -Name "origin_id")) { Get-PithPayloadValue -Payload $Payload -Name "origin_id" } elseif ($Body.Contains("origin_id")) { $Body["origin_id"] } else { $null }
        workspace_id = if ((Get-PithPayloadValue -Payload $Payload -Name "workspace_id")) { Get-PithPayloadValue -Payload $Payload -Name "workspace_id" } elseif ($Body.Contains("workspace_id")) { $Body["workspace_id"] } else { $null }
        request_id = if ((Get-PithPayloadValue -Payload $Payload -Name "request_id")) { Get-PithPayloadValue -Payload $Payload -Name "request_id" } elseif ($Body.Contains("request_id")) { $Body["request_id"] } else { $null }
        status = if ($Body.Contains("status")) { $Body["status"] } elseif ($Body.Contains("error") -and $Body["error"]) { "error" } else { "ok" }
        api_status = $ApiStatus
        error = [bool]($Body.Contains("error") -and $Body["error"])
        previous_response_present = [bool](Get-PithPayloadValue -Payload $Payload -Name "previous_response")
        previous_message_present = [bool](Get-PithPayloadValue -Payload $Payload -Name "previous_message")
        extracted_concepts_present = [bool](Get-PithPayloadValue -Payload $Payload -Name "extracted_concepts_json")
        auto_learned = [bool]$AutoLearned
        learning_events = if ($Body.Contains("learning_events")) { $Body["learning_events"] } elseif ($AutoLearned -and $AutoLearned.Contains("events")) { $AutoLearned["events"] } else { $null }
        accepted_learning_events = if ($Body.Contains("accepted_learning_events")) { $Body["accepted_learning_events"] } else { $null }
        checkpoint_task_id = if ((Get-PithPayloadValue -Payload $Payload -Name "task_id")) { Get-PithPayloadValue -Payload $Payload -Name "task_id" } elseif ($Body.Contains("task_id")) { $Body["task_id"] } else { $null }
    }

    try {
        $LogPath = Join-Path ([Environment]::GetFolderPath("UserProfile")) ".pith\logs\pith_mcp_transport.jsonl"
        $LogDir = Split-Path -Parent $LogPath
        New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
        Add-Content -LiteralPath $LogPath -Value ($Entry | ConvertTo-Json -Compress -Depth 8) -Encoding UTF8
    }
    catch {
        return
    }
}

function New-PithNativeLifecycleStatusUnavailable {
    param(
        [object]$Payload,
        [string]$TransportMode
    )

    $Diagnostic = Get-PithNativeRuntimeDiagnostic
    return [ordered]@{
        schema_version = "surface_lifecycle_status.v1"
        status = "degraded"
        overall_verdict = "not_observed"
        lifecycle_proof_status = "python_client_unavailable"
        can_claim_context_delivered = $false
        surface_id = Get-PithPayloadValue -Payload $Payload -Name "surface_id"
        selector = $Payload
        transport_mode = $TransportMode
        reporter_kind = "native_windows_python_unavailable"
        python_launch_ok = $Diagnostic.python_launch_ok
        python_launch_error = $Diagnostic.python_launch_error
        backend_health_ok = $Diagnostic.backend_health_ok
        protected_auth_valid = $Diagnostic.protected_auth_valid
        limitations = @("Native lifecycle_status fallback only reports wrapper/runtime/backend/auth capability; it does not prove conversation_turn delivery.")
        operator_action = "Run conversation_turn after Python client access is restored or use native lifecycle operation proof."
    }
}

function New-PithNativeConnectionProof {
    param(
        [object]$Payload,
        [object]$TurnBody,
        [string]$TransportMode,
        [string]$ApiUrl,
        [object]$StatusCode = $null,
        [bool]$RequestOk = $false,
        [object]$ErrorBody = $null
    )

    $Body = if ($TurnBody -is [System.Collections.IDictionary]) { $TurnBody } else { [ordered]@{} }
    $AuthError = $null
    if ($Body.Contains("auth_error")) {
        $AuthError = $Body["auth_error"]
    }
    elseif (($ErrorBody -is [System.Collections.IDictionary]) -and $ErrorBody.Contains("code") -and ($ErrorBody["code"] -eq "AUTH_FAILED")) {
        $AuthError = $ErrorBody
    }
    elseif (($ErrorBody -is [System.Collections.IDictionary]) -and $ErrorBody.Contains("body")) {
        $ErrorPayload = $ErrorBody["body"]
        if ($ErrorPayload -is [System.Collections.IDictionary]) {
            $Detail = ""
            if ($ErrorPayload.Contains("detail")) { $Detail = [string]$ErrorPayload["detail"] }
            elseif ($ErrorPayload.Contains("message")) { $Detail = [string]$ErrorPayload["message"] }
            if (($Detail.ToLowerInvariant() -like "*api key*") -or ($Detail.ToLowerInvariant() -like "*unauthorized*")) {
                $AuthError = $ErrorPayload
            }
        }
    }

    $BindStatus = if ($Body.Contains("bind_status")) { $Body["bind_status"] } else { $null }
    $ResolvedSessionId = if ($Body.Contains("resolved_session_id")) { $Body["resolved_session_id"] } elseif ($Body.Contains("session_id")) { $Body["session_id"] } else { $null }
    $Protocol = if ($Body.Contains("_protocol") -and ($Body["_protocol"] -is [System.Collections.IDictionary])) { $Body["_protocol"] } else { $null }
    $ProtocolSessionActivePresent = ($null -ne $Protocol) -and $Protocol.Contains("session_active")
    if ($ProtocolSessionActivePresent) {
        $SessionActive = [bool]$Protocol["session_active"]
        $SessionActiveSource = "conversation_turn._protocol.session_active"
    }
    else {
        $SessionActive = (($BindStatus -eq "bound") -and [bool]$ResolvedSessionId)
        $SessionActiveSource = "fallback_bind_status_bound_with_resolved_session_id"
    }
    $SameTurn = ($RequestOk -and (-not ($Body.Contains("error") -and $Body["error"])))
    $Connected = ($SameTurn -and ($BindStatus -eq "bound") -and [bool]$ResolvedSessionId -and $SessionActive -and (-not $AuthError))
    $ErrorCode = if (($ErrorBody -is [System.Collections.IDictionary]) -and $ErrorBody.Contains("code")) { $ErrorBody["code"] } else { $null }
    $ErrorMessage = if (($ErrorBody -is [System.Collections.IDictionary]) -and $ErrorBody.Contains("message")) { $ErrorBody["message"] } else { $null }
    $TransportFailureKind = if (($ErrorBody -is [System.Collections.IDictionary]) -and $ErrorBody.Contains("failure_kind")) { $ErrorBody["failure_kind"] } else { $null }

    return [ordered]@{
        schema_version = "pith_connection_proof.v1"
        verdict = if ($Connected) { "connected" } else { "not_connected" }
        same_turn = $SameTurn
        evidence_source = "current_conversation_turn_response"
        transport_mode = $TransportMode
        api_url = $ApiUrl
        status_code = $StatusCode
        surface_id = if ($Body.Contains("surface_id")) { $Body["surface_id"] } else { Get-PithPayloadValue -Payload $Payload -Name "surface_id" }
        origin_id = if ($Body.Contains("origin_id")) { $Body["origin_id"] } else { Get-PithPayloadValue -Payload $Payload -Name "origin_id" }
        workspace_id = if ($Body.Contains("workspace_id")) { $Body["workspace_id"] } else { Get-PithPayloadValue -Payload $Payload -Name "workspace_id" }
        bind_status = $BindStatus
        resolved_session_id = $ResolvedSessionId
        auth_error = $AuthError
        error_code = $ErrorCode
        error_message = $ErrorMessage
        transport_failure_kind = $TransportFailureKind
        session_active = $SessionActive
        session_active_source = $SessionActiveSource
        claim_rule = "connected requires this command's conversation_turn response to bind a session, return resolved_session_id, have session_active=true, and report no auth error; health or bridge reachability alone is insufficient"
    }
}

function Invoke-PithNativeApiCommand {
    param(
        [string]$CommandName,
        [object[]]$RemainingArgs,
        [string]$TransportMode
    )

    $Options = Get-PithApiCommandOptions -RemainingArgs $RemainingArgs
    if ($Options.error_message) {
        Write-PithNativeJson -Value (New-PithApiError -Code "INVALID_ARGUMENT" -Message $Options.error_message -Operation $Options.operation -TransportMode $TransportMode) -Depth 8
        exit 1
    }

    $Operation = $Options.operation
    if (($CommandName -eq "api") -and ($Operation -eq "lifecycle_status") -and (-not $Options.use_python)) {
        $Diagnostic = Get-PithNativeRuntimeDiagnostic
        if (-not $Diagnostic.python_launch_ok) {
            try {
                $Payload = Read-PithNativeApiPayload -Options $Options
            }
            catch {
                $Payload = [ordered]@{ payload_error = $_.Exception.Message }
            }
            Write-PithNativeJson -Value (New-PithNativeLifecycleStatusUnavailable -Payload $Payload -TransportMode $TransportMode) -Depth 8
            exit 0
        }
    }

    if ($Options.use_python -or (-not $PithNativeApiOperations.ContainsKey($Operation))) {
        return $false
    }

    try {
        $Payload = Read-PithNativeApiPayload -Options $Options
        $Payload = Add-PithNativeApiDefaults -Operation $Operation -Payload $Payload
        if ((($Operation -eq "conversation_turn") -or ($Operation -eq "connection_proof")) -and ($Payload -is [System.Collections.IDictionary]) -and $Payload.Contains("current_task_id")) {
            $Diagnostic = Get-PithNativeRuntimeDiagnostic
            if ($Diagnostic.python_launch_ok) {
                return $false
            }
            $Body = New-PithApiError -Code "PYTHON_REQUIRED_FOR_WORKSTREAM_GATE" -Message "conversation_turn with current_task_id requires the Python client workstream activation gate." -Operation $Operation -TransportMode $TransportMode
            $Body.operator_action = "Run pith api workstreams ensure_workstream_activation separately, or restore Python client access before sending current_task_id."
            Write-PithNativeTransportEvent -Operation $Operation -Payload $Payload -ResultBody $Body -TransportMode $TransportMode -ApiStatus "python_required_for_workstream_gate" -ApiUrl $Options.base_url
            Write-PithNativeJson -Value $Body -Depth 8
            exit 1
        }
        $Definition = $PithNativeApiOperations[$Operation]
        $Headers = Get-PithNativeApiHeaders -Operation $Operation -TransportMode $TransportMode -AuthRequired ([bool]$Definition.auth)
    }
    catch {
        Write-PithNativeJson -Value (New-PithApiError -Code "INVALID_PAYLOAD" -Message $_.Exception.Message -Operation $Operation -TransportMode $TransportMode) -Depth 8
        exit 1
    }

    $Attempts = if (($Operation -eq "conversation_turn") -or ($Operation -eq "connection_proof")) { 4 } else { 1 }
    $Result = $null
    for ($Attempt = 0; $Attempt -lt $Attempts; $Attempt++) {
        $Result = Invoke-PithNativeApiHttp -ApiUrl $Options.base_url -Path ([string]$Definition.path) -Method ([string]$Definition.method) -Headers $Headers -Payload $Payload -TimeoutSeconds ([int]$Options.timeout_seconds)
        if (($Attempt -lt ($Attempts - 1)) -and (Test-PithRetryableConversationTurnStartup -Result $Result)) {
            Start-Sleep -Milliseconds ([Math]::Min(10000, [int](500 * [Math]::Pow(2, $Attempt))))
            continue
        }
        break
    }

    if ($null -eq $Result) {
        $Body = New-PithApiError -Code "REQUEST_FAILED" -Message "No local API response." -Operation $Operation -TransportMode $TransportMode
        Write-PithNativeTransportEvent -Operation $Operation -Payload $Payload -ResultBody $Body -TransportMode $TransportMode -ApiStatus "request_failed" -ApiUrl $Options.base_url
        if ($Operation -eq "connection_proof") {
            Write-PithNativeJson -Value (New-PithNativeConnectionProof -Payload $Payload -TurnBody $null -TransportMode $TransportMode -ApiUrl $Options.base_url -StatusCode $null -RequestOk:$false -ErrorBody $Body) -Depth 20
            exit 1
        }
        Write-PithNativeJson -Value $Body -Depth 8
        exit 2
    }

    if (-not $Result.ok) {
        $Code = if ($Result.status_code -eq 401) { "AUTH_FAILED" } elseif ($Result.reachable) { "HTTP_ERROR" } else { "REQUEST_FAILED" }
        $ExitCode = if ($Result.reachable) { 1 } else { 2 }
        $Body = New-PithApiError -Code $Code -Message $Result.message -Operation $Operation -TransportMode $TransportMode -StatusCode $Result.status_code -Body $Result.body
        if ($Result.failure_kind) {
            $Body.failure_kind = $Result.failure_kind
        }
        Write-PithNativeTransportEvent -Operation $Operation -Payload $Payload -ResultBody $Body -TransportMode $TransportMode -ApiStatus $Result.status_code -ApiUrl $Result.api_url
        if ($Operation -eq "connection_proof") {
            Write-PithNativeJson -Value (New-PithNativeConnectionProof -Payload $Payload -TurnBody $Result.body -TransportMode $TransportMode -ApiUrl $Result.api_url -StatusCode $Result.status_code -RequestOk:$false -ErrorBody $Body) -Depth 20
            exit 1
        }
        Write-PithNativeJson -Value $Body -Depth 8
        exit $ExitCode
    }

    if ((($Operation -eq "conversation_turn") -or ($Operation -eq "connection_proof")) -and (Test-PithNativeApiBodyEmpty -Body $Result.body)) {
        $Body = New-PithApiError -Code "EMPTY_RESPONSE" -Message "Local API returned an empty success body for conversation_turn." -Operation $Operation -TransportMode $TransportMode -StatusCode $Result.status_code
        Write-PithNativeTransportEvent -Operation $Operation -Payload $Payload -ResultBody $Body -TransportMode $TransportMode -ApiStatus "empty_response" -ApiUrl $Result.api_url
        if ($Operation -eq "connection_proof") {
            Write-PithNativeJson -Value (New-PithNativeConnectionProof -Payload $Payload -TurnBody $Result.body -TransportMode $TransportMode -ApiUrl $Result.api_url -StatusCode $Result.status_code -RequestOk:$false -ErrorBody $Body) -Depth 20
            exit 1
        }
        Write-PithNativeJson -Value $Body -Depth 8
        exit 2
    }

    Write-PithNativeTransportEvent -Operation $Operation -Payload $Payload -ResultBody $Result.body -TransportMode $TransportMode -ApiStatus $Result.status_code -ApiUrl $Result.api_url
    if ($Operation -eq "connection_proof") {
        $Proof = New-PithNativeConnectionProof -Payload $Payload -TurnBody $Result.body -TransportMode $TransportMode -ApiUrl $Result.api_url -StatusCode $Result.status_code -RequestOk:$true
        Write-PithNativeJson -Value $Proof -Depth 20
        if ($Proof["verdict"] -eq "connected") {
            exit 0
        }
        exit 1
    }
    Write-PithNativeJson -Value $Result.body -Depth 20
    exit 0
}

function Invoke-PithApiCommand {
    param(
        [string]$CommandName,
        [object[]]$RemainingArgs
    )

    $TransportMode = "first_class_api"
    if ($CommandName -eq "api-fallback") {
        $TransportMode = "exec_http_fallback"
    }

    if ($env:PITH_EXEC_FALLBACK_ENABLED -and $env:PITH_EXEC_FALLBACK_ENABLED -ne "1") {
        [ordered]@{
            error = $true
            code = "DISABLED"
            message = "exec HTTP API command disabled"
            transport_mode = $TransportMode
        } | ConvertTo-Json -Compress
        exit 64
    }

    if (Invoke-PithNativeApiCommand -CommandName $CommandName -RemainingArgs $RemainingArgs -TransportMode $TransportMode) {
        exit 0
    }

    $CliArgs = @("-m", "pith_client.cli") + $RemainingArgs + @("--transport-mode", $TransportMode)
    $StdinPayload = $null
    if ($RemainingArgs -contains "--stdin-json") {
        $PipelineInput = @($script:PithCliPipelineInput | ForEach-Object { [string]$_ })
        if ($PipelineInput.Count -gt 0) {
            $StdinPayload = $PipelineInput -join [Environment]::NewLine
        }
        else {
            $StdinPayload = [Console]::In.ReadToEnd()
        }
    }
    Push-Location $PithServerPath
    try {
        if ($null -ne $StdinPayload) {
            $StdinPayload | & $PythonExe @CliArgs
        }
        else {
            & $PythonExe @CliArgs
        }
        $exitCode = if ($null -eq $LASTEXITCODE) { 0 } else { $LASTEXITCODE }
    }
    finally {
        Pop-Location
    }
    exit $exitCode
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

function Remove-PithOwnedPathEntries {
    param([string]$PithBinDir)

    $UserPath = [Environment]::GetEnvironmentVariable("Path", "User")
    $NewUserPath = Update-PithPathValue -PathValue $UserPath -PithBinDir $PithBinDir -PrependPithBin:$false
    if ($NewUserPath -ne $UserPath) {
        [Environment]::SetEnvironmentVariable("Path", $NewUserPath, "User")
    }
}

function Remove-PithProfilePathLines {
    param([string]$PithBinDir)

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
            }
        }
        else {
            Remove-Item -LiteralPath $ProfilePath -Force -ErrorAction SilentlyContinue
        }
    }
}

function Repair-PithRemovalAccess {
    param([string]$TargetPath)

    if (-not (Test-Path $TargetPath)) {
        return
    }

    & attrib.exe -R -S -H "$TargetPath" /S /D 2>$null | Out-Null
    $CurrentIdentity = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
    & icacls.exe "$TargetPath" /grant "${CurrentIdentity}:F" /T /C 2>$null | Out-Null
    & icacls.exe "$TargetPath" /grant "${CurrentIdentity}:(OI)(CI)F" /T /C 2>$null | Out-Null
}

function Start-PithDeferredRemoval {
    param(
        [string]$TargetPath,
        [string]$MarkerPath = ""
    )

    $CleanupScript = Join-Path ([System.IO.Path]::GetTempPath()) "pith-uninstall-$([guid]::NewGuid().ToString('N')).ps1"
    $CleanupContent = @'
param(
    [string]$TargetPath,
    [string]$MarkerPath = ""
)

$ErrorActionPreference = "SilentlyContinue"
Start-Sleep -Seconds 2

for ($Attempt = 1; $Attempt -le 240; $Attempt++) {
    if (-not (Test-Path -LiteralPath $TargetPath)) {
        if ($MarkerPath) {
            Remove-Item -LiteralPath $MarkerPath -Force -ErrorAction SilentlyContinue
        }
        break
    }

    & attrib.exe -R -S -H "$TargetPath" /S /D 2>$null | Out-Null
    $CurrentIdentity = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
    & icacls.exe "$TargetPath" /grant "${CurrentIdentity}:F" /T /C 2>$null | Out-Null
    & icacls.exe "$TargetPath" /grant "${CurrentIdentity}:(OI)(CI)F" /T /C 2>$null | Out-Null

    try {
        Remove-Item -LiteralPath $TargetPath -Recurse -Force -ErrorAction Stop
        break
    }
    catch {
        Start-Sleep -Milliseconds 500
    }
}

if ((-not (Test-Path -LiteralPath $TargetPath)) -and $MarkerPath) {
    Remove-Item -LiteralPath $MarkerPath -Force -ErrorAction SilentlyContinue
}

Remove-Item -LiteralPath $MyInvocation.MyCommand.Path -Force -ErrorAction SilentlyContinue
'@

    Set-Content -Path $CleanupScript -Value $CleanupContent
    $PowerShellExe = "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe"
    if (-not (Test-Path $PowerShellExe)) {
        $PowerShellExe = "powershell.exe"
    }

    $Arguments = "-NoProfile -ExecutionPolicy Bypass -File `"$CleanupScript`" -TargetPath `"$TargetPath`""
    if ($MarkerPath) {
        $Arguments += " -MarkerPath `"$MarkerPath`""
    }
    Start-Process -FilePath $PowerShellExe -ArgumentList $Arguments -WindowStyle Hidden | Out-Null
}

function Test-PithHelpRequest {
    return (($script:PithCliArgs.Count -gt 1) -and (($script:PithCliArgs[1] -eq "-h") -or ($script:PithCliArgs[1] -eq "--help")))
}

function Invoke-PithNativeHttpProbe {
    param(
        [string]$Uri,
        [hashtable]$Headers = @{},
        [int]$TimeoutMs = 1000
    )
    $Request = $null
    $Response = $null
    try {
        $Request = [System.Net.WebRequest]::Create($Uri)
        $Request.Method = "GET"
        $Request.Timeout = $TimeoutMs
        $Request.ReadWriteTimeout = $TimeoutMs
        $Request.Proxy = $null
        foreach ($Name in $Headers.Keys) {
            $Request.Headers[$Name] = [string]$Headers[$Name]
        }
        $Response = $Request.GetResponse()
        $StatusCode = [int]$Response.StatusCode
        return [ordered]@{
            reachable = (($StatusCode -ge 200) -and ($StatusCode -lt 500))
            ok = (($StatusCode -ge 200) -and ($StatusCode -lt 400))
            status_code = $StatusCode
            error_code = $null
            message = $null
            probe_transport = "dotnet_web_request"
        }
    }
    catch [System.Net.WebException] {
        $StatusCode = $null
        if ($_.Exception.Response) {
            $StatusCode = [int]$_.Exception.Response.StatusCode
            $_.Exception.Response.Close()
        }
        if ($null -eq $StatusCode) {
            return Invoke-PithNativeHttpProbeWithPowerShell -Uri $Uri -Headers $Headers -TimeoutMs $TimeoutMs
        }
        $ErrorCode = if ($StatusCode -eq 401) { "AUTH_FAILED" } elseif ($StatusCode) { "HTTP_ERROR" } else { "REQUEST_FAILED" }
        return [ordered]@{
            reachable = ($null -ne $StatusCode)
            ok = $false
            status_code = $StatusCode
            error_code = $ErrorCode
            message = $_.Exception.Message
            probe_transport = "dotnet_web_request"
        }
    }
    catch {
        return Invoke-PithNativeHttpProbeWithPowerShell -Uri $Uri -Headers $Headers -TimeoutMs $TimeoutMs
    }
    finally {
        if ($null -ne $Response) {
            $Response.Close()
        }
    }
}

function Invoke-PithNativeHttpProbeWithPowerShell {
    param(
        [string]$Uri,
        [hashtable]$Headers = @{},
        [int]$TimeoutMs = 1000
    )

    $TimeoutSec = [Math]::Max(1, [int][Math]::Ceiling($TimeoutMs / 1000.0))
    try {
        $Response = Invoke-WebRequest -Uri $Uri -Headers $Headers -UseBasicParsing -TimeoutSec $TimeoutSec -ErrorAction Stop
        $StatusCode = [int]$Response.StatusCode
        return [ordered]@{
            reachable = (($StatusCode -ge 200) -and ($StatusCode -lt 500))
            ok = (($StatusCode -ge 200) -and ($StatusCode -lt 400))
            status_code = $StatusCode
            error_code = $null
            message = $null
            probe_transport = "powershell_invoke_web_request"
        }
    }
    catch {
        $StatusCode = $null
        if ($_.Exception.Response) {
            $StatusCode = [int]$_.Exception.Response.StatusCode
        }
        $ErrorCode = if ($StatusCode -eq 401) { "AUTH_FAILED" } elseif ($StatusCode) { "HTTP_ERROR" } else { "REQUEST_FAILED" }
        return [ordered]@{
            reachable = ($null -ne $StatusCode)
            ok = $false
            status_code = $StatusCode
            error_code = $ErrorCode
            message = $_.Exception.Message
            probe_transport = "powershell_invoke_web_request"
        }
    }
}

function Get-PithDiagnosticApiUrls {
    param([string]$ApiUrl)

    $Urls = New-Object System.Collections.Generic.List[string]
    $PrimaryUrl = $ApiUrl.TrimEnd('/')
    if ($PrimaryUrl) {
        $Urls.Add($PrimaryUrl)
    }

    try {
        $ParsedUrl = [Uri]$PrimaryUrl
        if ($ParsedUrl.IsLoopback -and (($ParsedUrl.Host -eq "localhost") -or ($ParsedUrl.Host -eq "::1"))) {
            $Builder = New-Object System.UriBuilder($ParsedUrl)
            $Builder.Host = "127.0.0.1"
            $LoopbackUrl = $Builder.Uri.AbsoluteUri.TrimEnd('/')
            if ($LoopbackUrl -and -not $Urls.Contains($LoopbackUrl)) {
                $Urls.Add($LoopbackUrl)
            }
        }
    }
    catch {
        return [string[]]$Urls.ToArray()
    }

    return [string[]]$Urls.ToArray()
}

function Invoke-PithNativeHttpProbeWithLoopbackFallback {
    param(
        [string]$ApiUrl,
        [string]$Path,
        [hashtable]$Headers = @{},
        [int]$TimeoutMs = 1000
    )

    $LastProbe = $null
    foreach ($ProbeApiUrl in (Get-PithDiagnosticApiUrls -ApiUrl $ApiUrl)) {
        $Probe = Invoke-PithNativeHttpProbe -Uri "$ProbeApiUrl$Path" -Headers $Headers -TimeoutMs $TimeoutMs
        $Probe["path"] = $Path
        $Probe["api_url"] = $ProbeApiUrl
        $LastProbe = $Probe
        if ($Probe.ok -or $Probe.reachable) {
            return $Probe
        }
    }

    return $LastProbe
}

function Redact-PithDiagnosticText {
    param([string]$Value)

    if (-not $Value) {
        return ""
    }

    $Text = [string]$Value
    $Text = [regex]::Replace(
        $Text,
        '(?i)\b([A-Z0-9_]*(?:API_KEY|TOKEN|SECRET|PASSWORD|PRIVATE_KEY|ACCESS_KEY|AUTH)[A-Z0-9_]*)(\s*[:=]\s*)("[^"]*"|''[^'']*''|\S+)',
        '$1$2<redacted>'
    )
    $Text = [regex]::Replace($Text, '(?i)\b((?:X-API-Key|Authorization):\s*)(Bearer\s+)?\S+', '$1<redacted>')
    $Text = [regex]::Replace($Text, '\b(?:sk-[A-Za-z0-9_-]{12,}|[A-Za-z0-9_-]{40,})\b', '<redacted>')
    if ($Text.Length -gt 240) {
        return "$($Text.Substring(0, 240))..."
    }
    return $Text
}

function Get-PithApiKeyFingerprint {
    param([string]$Value)

    if (-not $Value) {
        return $null
    }
    try {
        $Bytes = [System.Text.Encoding]::UTF8.GetBytes($Value)
        $Sha = [System.Security.Cryptography.SHA256]::Create()
        try {
            $Hash = $Sha.ComputeHash($Bytes)
        }
        finally {
            $Sha.Dispose()
        }
        $Hex = ([System.BitConverter]::ToString($Hash)).Replace("-", "").ToLowerInvariant()
        return "sha256:$($Hex.Substring(0, 12))"
    }
    catch {
        return $null
    }
}

function Get-PithProcessPythonExecutable {
    param([string]$CommandLine)

    if (-not $CommandLine) {
        return ""
    }
    $TrimmedCommandLine = $CommandLine.TrimStart()
    if ($TrimmedCommandLine -match '^"(?<exe>[A-Za-z]:[\\/](?:[^"\\/]+[\\/])*python(?:\.exe)?)"(?:\s|$)') {
        return $Matches["exe"]
    }
    if ($TrimmedCommandLine -match '^(?<exe>[A-Za-z]:[\\/](?:[^\s\\/]+[\\/])*python(?:\.exe)?)(?:\s|$)') {
        return $Matches["exe"]
    }
    return ""
}

function Test-PithServerCommandLine {
    param([string]$CommandLine)

    if (-not $CommandLine) {
        return $false
    }
    return (
        ($CommandLine -match '(?i)(?:^|\s)-m\s+app\.api\.serve(?:\s|$)') -or
        (($CommandLine -match '(?i)\buvicorn(?:\.exe)?\b') -and ($CommandLine -match '(?i)app\.api\.server(?::app)?'))
    )
}

function Test-PithMcpBootstrapCommandLine {
    param([string]$CommandLine)

    if ($CommandLine -match '[\x00-\x1f]') { return $false }
    $PythonPath = Get-PithProcessPythonExecutable -CommandLine $CommandLine
    if (-not $PythonPath -or -not $PithHome) { return $false }
    $PythonToken = '"' + [regex]::Escape($PythonPath) + '"'
    if ($PythonPath -notmatch '\s') { $PythonToken += '|' + [regex]::Escape($PythonPath) }
    $HomeToken = '"' + [regex]::Escape($PithHome.TrimEnd('\')) + '"'
    if ($PithHome -notmatch '\s') { $HomeToken += '|' + [regex]::Escape($PithHome.TrimEnd('\')) }
    $Pattern = '^\s*(?:' + $PythonToken + ')\s+(?:"(?<script>[^"\r\n]+)"|(?<script>[^\s"]+))' +
        '\s+--pith-home\s+(?:' + $HomeToken + ')\s+--surface-id\s+(?:"claude_desktop_mcp"|claude_desktop_mcp)\s*$'
    if ($CommandLine -notmatch $Pattern) { return $false }
    $ScriptPath = $Matches['script']
    $Suffix = '\Claude Extensions\local.mcpb.pith.pith\server\windows_bootstrap.py'
    if ($env:APPDATA -and [string]::Equals($ScriptPath, ($env:APPDATA.TrimEnd('\') + '\Claude' + $Suffix), [StringComparison]::OrdinalIgnoreCase)) {
        return $true
    }
    if ($env:LOCALAPPDATA) {
        $StorePattern = '^' + [regex]::Escape($env:LOCALAPPDATA.TrimEnd('\')) +
            '\\Packages\\Claude_[A-Za-z0-9]+\\LocalCache\\Roaming\\Claude' + [regex]::Escape($Suffix) + '$'
        return ($ScriptPath -match $StorePattern)
    }
    return $false
}

function Test-PithOwnedMcpBootstrapProcess {
    param($Process)

    if (-not $Process -or -not (Test-PithMcpBootstrapCommandLine -CommandLine ([string]$Process.CommandLine))) {
        return $false
    }
    $OwnedPython = $PithHome.TrimEnd('\') + '\runtime\python\python.exe'
    return [string]::Equals((Get-PithProcessPythonExecutable -CommandLine ([string]$Process.CommandLine)), $OwnedPython, [StringComparison]::OrdinalIgnoreCase) -and
        [string]::Equals([string]$Process.ExecutablePath, $OwnedPython, [StringComparison]::OrdinalIgnoreCase)
}

function Get-PithProcessInventory {
    $EscapedPithHome = [regex]::Escape($PithHome)
    $ServerProcesses = @()
    $BridgeProcesses = @()
    $PythonExecutables = New-Object System.Collections.Generic.List[string]
    $ServerPythonExecutables = New-Object System.Collections.Generic.List[string]
    $BridgePythonExecutables = New-Object System.Collections.Generic.List[string]
    try {
        $Processes = @(Get-CimInstance Win32_Process -ErrorAction Stop |
            Where-Object {
                if ($_.CommandLine) {
                    $CandidateCommandLine = [string]$_.CommandLine
                    $ComparableCommandLine = $CandidateCommandLine.Replace('/', '\')
                    $CandidatePython = Get-PithProcessPythonExecutable -CommandLine $CandidateCommandLine
                    if ($CandidatePython) { $CandidatePython = $CandidatePython.Replace('/', '\') }
                    $CandidatePython -and
                        (
                            (($ComparableCommandLine -match $EscapedPithHome) -and ($ComparableCommandLine -match "pith_mcp\.py")) -or
                            (Test-PithMcpBootstrapCommandLine -CommandLine $CandidateCommandLine) -or
                            ((($ComparableCommandLine -match $EscapedPithHome) -or ($CandidatePython -ieq $PythonExe)) -and (Test-PithServerCommandLine -CommandLine $CandidateCommandLine))
                        )
                }
            })
    }
    catch {
        return [ordered]@{
            available = $false
            error = (Redact-PithDiagnosticText -Value $_.Exception.Message)
            expected_python_executable = $PythonExe
            server_process_count = 0
            mcp_bridge_process_count = 0
            python_executables = @()
            server_python_executables = @()
            mcp_bridge_python_executables = @()
            mixed_python_provenance = $false
            benign_venv_redirect_server_pair = $false
            benign_venv_redirect_mcp_bridge_pair = $false
            server_process_overlap_detected = $false
            mcp_bridge_process_overlap_detected = $false
            unexpected_python_provenance_detected = $false
            overlap_detected = $false
            server_processes = @()
            mcp_bridge_processes = @()
        }
    }

    foreach ($Process in $Processes) {
        $CommandLine = [string]$Process.CommandLine
        $PythonPath = Get-PithProcessPythonExecutable -CommandLine $CommandLine
        if ($PythonPath) { $PythonPath = $PythonPath.Replace('/', '\') }
        if ($PythonPath -and -not $PythonExecutables.Contains($PythonPath)) {
            $PythonExecutables.Add($PythonPath)
        }
        $Entry = [ordered]@{
            pid = [int]$Process.ProcessId
            parent_pid = [int]$Process.ParentProcessId
            python_executable = $PythonPath
            mcp_bootstrap = (Test-PithMcpBootstrapCommandLine -CommandLine $CommandLine)
            owned_mcp_bootstrap = (Test-PithOwnedMcpBootstrapProcess -Process $Process)
            command_summary = (Redact-PithDiagnosticText -Value $CommandLine)
        }
        if (($CommandLine -match "pith_mcp\.py") -or (Test-PithMcpBootstrapCommandLine -CommandLine $CommandLine)) {
            $BridgeProcesses += $Entry
            if ($PythonPath -and -not $BridgePythonExecutables.Contains($PythonPath)) {
                $BridgePythonExecutables.Add($PythonPath)
            }
        }
        elseif (Test-PithServerCommandLine -CommandLine $CommandLine) {
            $ServerProcesses += $Entry
            if ($PythonPath -and -not $ServerPythonExecutables.Contains($PythonPath)) {
                $ServerPythonExecutables.Add($PythonPath)
            }
        }
    }

    $MixedPython = ($PythonExecutables.Count -gt 1)
    $BenignVenvRedirectServerPair = $false
    if ($ServerProcesses.Count -eq 2) {
        $VenvServer = $ServerProcesses | Where-Object { $_.python_executable -ieq $PythonExe } | Select-Object -First 1
        if ($VenvServer) {
            $RedirectChild = $ServerProcesses |
                Where-Object { ($_.parent_pid -eq $VenvServer.pid) -and $_.python_executable -and ($_.python_executable -ine $PythonExe) } |
                Select-Object -First 1
            $BenignVenvRedirectServerPair = [bool]$RedirectChild
        }
    }

    $ServerUnexpectedPython = $false
    foreach ($PythonPath in $ServerPythonExecutables) {
        if ($PythonPath -and ($PythonPath -ine $PythonExe)) {
            $ServerUnexpectedPython = $true
        }
    }
    if ($BenignVenvRedirectServerPair) {
        $ServerUnexpectedPython = $false
    }

    $BenignVenvRedirectMcpBridgePair = $false
    $BenignBridgeRedirectChildPids = New-Object System.Collections.Generic.List[int]
    if ($BridgeProcesses.Count -ge 2) {
        $VenvBridgeProcesses = @($BridgeProcesses | Where-Object { $_.python_executable -ieq $PythonExe })
        foreach ($VenvBridge in $VenvBridgeProcesses) {
            $RedirectChildren = @($BridgeProcesses |
                Where-Object { ($_.parent_pid -eq $VenvBridge.pid) -and $_.python_executable -and ($_.python_executable -ine $PythonExe) -and (-not $_.mcp_bootstrap) })
            foreach ($RedirectChild in $RedirectChildren) {
                $BenignVenvRedirectMcpBridgePair = $true
                if (-not $BenignBridgeRedirectChildPids.Contains([int]$RedirectChild.pid)) {
                    $BenignBridgeRedirectChildPids.Add([int]$RedirectChild.pid)
                }
            }
        }
    }

    $BridgeUnexpectedPython = $false
    foreach ($BridgeProcess in $BridgeProcesses) {
        $PythonPath = $BridgeProcess.python_executable
        $UnexpectedBootstrap = $BridgeProcess.mcp_bootstrap -and (-not $BridgeProcess.owned_mcp_bootstrap)
        if ($UnexpectedBootstrap -or ($PythonPath -and ($PythonPath -ine $PythonExe) -and (-not $BridgeProcess.owned_mcp_bootstrap) -and (-not $BenignBridgeRedirectChildPids.Contains([int]$BridgeProcess.pid)))) {
            $BridgeUnexpectedPython = $true
        }
    }

    $ServerOverlap = (($ServerProcesses.Count -gt 1) -and (-not $BenignVenvRedirectServerPair))
    $MultipleMcpBridgesActive = ($BridgeProcesses.Count -gt 1)
    $BridgeOverlap = $false
    $UnexpectedPython = ($ServerUnexpectedPython -or $BridgeUnexpectedPython)
    $Overlap = ($ServerOverlap -or $UnexpectedPython)
    return [ordered]@{
        available = $true
        error = $null
        expected_python_executable = $PythonExe
        server_process_count = $ServerProcesses.Count
        mcp_bridge_process_count = $BridgeProcesses.Count
        python_executables = [string[]]$PythonExecutables.ToArray()
        server_python_executables = [string[]]$ServerPythonExecutables.ToArray()
        mcp_bridge_python_executables = [string[]]$BridgePythonExecutables.ToArray()
        mixed_python_provenance = $MixedPython
        benign_venv_redirect_server_pair = $BenignVenvRedirectServerPair
        benign_venv_redirect_mcp_bridge_pair = $BenignVenvRedirectMcpBridgePair
        server_process_overlap_detected = $ServerOverlap
        mcp_bridge_process_overlap_detected = $BridgeOverlap
        mcp_bridge_multiple_active_detected = $MultipleMcpBridgesActive
        unexpected_python_provenance_detected = $UnexpectedPython
        overlap_detected = $Overlap
        server_processes = $ServerProcesses
        mcp_bridge_processes = $BridgeProcesses
    }
}

function Invoke-PithNativeHealthEvidence {
    param(
        [string]$ApiUrl,
        [int]$TimeoutMs = 1500
    )

    $Probes = @()
    foreach ($Path in @("/healthz", "/health")) {
        $Probe = Invoke-PithNativeHttpProbeWithLoopbackFallback -ApiUrl $ApiUrl -Path $Path -TimeoutMs $TimeoutMs
        $Probe["path"] = $Path
        $Probes += $Probe
        if ($Probe.ok) {
            break
        }
    }

    $SuccessfulProbe = $Probes | Where-Object { $_.ok } | Select-Object -First 1
    $ReachableProbe = $Probes | Where-Object { $_.reachable } | Select-Object -First 1
    $LastProbe = $Probes | Select-Object -Last 1
    return [ordered]@{
        reachable = [bool]$ReachableProbe
        ok = [bool]$SuccessfulProbe
        status_code = if ($SuccessfulProbe) { $SuccessfulProbe.status_code } elseif ($ReachableProbe) { $ReachableProbe.status_code } else { $null }
        error_code = if ($SuccessfulProbe) { $null } else { $LastProbe.error_code }
        message = if ($SuccessfulProbe) { $null } else { $LastProbe.message }
        probes = $Probes
    }
}

function Test-PithHealthz {
    param(
        [int]$Port,
        [int]$TimeoutMs = 1000
    )
    $Probe = Invoke-PithNativeHttpProbe -Uri "http://127.0.0.1:$Port/healthz" -TimeoutMs $TimeoutMs
    return [bool]$Probe.ok
}

function Test-PithConversationReady {
    param(
        [int]$Port,
        [int]$TimeoutSeconds = 2
    )

    $Result = Invoke-PithNativeApiHttp `
        -ApiUrl "http://127.0.0.1:$Port" `
        -Path "/readyz" `
        -Method "GET" `
        -Headers @{} `
        -Payload ([ordered]@{}) `
        -TimeoutSeconds $TimeoutSeconds
    if ((-not $Result.ok) -or (-not ($Result.body -is [System.Collections.IDictionary]))) {
        return $false
    }
    return (
        ([string]$Result.body["process_state"] -eq "running") -and
        ([string]$Result.body["write_state"] -eq "accepting") -and
        ([string]$Result.body["retrieval_state"] -ne "recovering")
    )
}

function Get-PithLogSnapshot {
    param([object[]]$CommandArgs)
    $LineCount = 50
    $RequestedFile = "both"
    $JsonOutput = $false
    for ($Index = 2; $Index -lt $CommandArgs.Count; $Index++) {
        if ($CommandArgs[$Index] -eq "--json") {
            $JsonOutput = $true
        }
        elseif (($CommandArgs[$Index] -eq "--lines") -and (($Index + 1) -lt $CommandArgs.Count)) {
            $LineCount = [int]$CommandArgs[$Index + 1]
            $Index++
        }
        elseif (($CommandArgs[$Index] -eq "--file") -and (($Index + 1) -lt $CommandArgs.Count)) {
            $RequestedFile = $CommandArgs[$Index + 1]
            $Index++
        }
    }

    $LogMap = [ordered]@{
        pith = "$PithHome\logs\server.log"
        err = "$PithHome\logs\server.err.log"
    }
    $Snapshot = [ordered]@{}
    foreach ($Name in @("pith", "err")) {
        if (($RequestedFile -ne "both") -and ($RequestedFile -ne $Name)) {
            continue
        }
        $Path = $LogMap[$Name]
        if (Test-Path -LiteralPath $Path) {
            $Lines = @(Get-Content -LiteralPath $Path -Tail $LineCount -ErrorAction SilentlyContinue |
                ForEach-Object { [string]$_ })
            $Snapshot[$Name] = [string[]]$Lines
        }
        else {
            $Snapshot[$Name] = [string[]]@("<missing: $Path>")
        }
    }

    if ($JsonOutput) {
        $Snapshot | ConvertTo-Json -Depth 4
    }
    else {
        foreach ($Name in $Snapshot.Keys) {
            Write-Host "==== $Name ===="
            $Snapshot[$Name]
        }
    }
}

function Print-PithWrapperHelp {
    param([string]$CommandName)

    switch ($CommandName) {
        "serve" {
            Write-Host "Usage: pith serve"
            Write-Host "  Run the Pith API server in the foreground for a process manager."
        }
        "start" {
            Write-Host "Usage: pith start"
            Write-Host "  Start the Pith API server in the background."
        }
        "stop" {
            Write-Host "Usage: pith stop"
            Write-Host "  Stop the Pith API server for this PITH_HOME/PITH_PORT."
        }
        "restart" {
            Write-Host "Usage: pith restart"
            Write-Host "  Restart the Pith API server while preserving connected MCP clients."
        }
        "status" {
            Write-Host "Usage: pith status [--json]"
            Write-Host "  Show service status from PID, port, health, and readiness checks."
        }
        "health" {
            Write-Host "Usage: pith health [--json]"
            Write-Host "  Check operational health/readiness."
        }
        "logs" {
            Write-Host "Usage: pith logs [snapshot [--json] [--file {pith,err,both}] [--lines N]]"
            Write-Host "  Tail the Pith log, or print a bounded snapshot."
        }
        "import" {
            Write-Host "Usage: pith import [options]"
            Write-Host "  Import conversation exports safely."
        }
        { $_ -in @("search", "concept", "orient", "sessions", "metrics") } {
            Write-Host "Usage: pith $CommandName [options]"
            Write-Host "  Run read-only Pith query command '$CommandName'."
        }
        { $_ -in @("doctor", "clients", "support", "report") } {
            Write-Host "Usage: pith $CommandName [options]"
            Write-Host "  Run support command '$CommandName'."
        }
        { $_ -in @("api", "api-fallback") } {
            Write-Host "Usage: pith $CommandName <api-command> [options]"
            Write-Host "  Run a local Pith API command."
        }
        "backup" {
            Write-Host "Usage: pith backup"
            Write-Host "  Run the configured WAL-safe backup helper."
        }
        "restore" {
            Write-Host "Usage: pith restore <backup_file.db>"
            Write-Host "  Restore a profile database from a backup file."
        }
        "update" {
            Write-Host "Usage: pith update"
            Write-Host "  Run local dependency and embedding update checks."
        }
        "version" {
            Write-Host "Usage: pith version"
            Write-Host "  Show installed Pith and runtime version information."
        }
        "runtime" {
            Write-Host "Usage: pith runtime {status [--json]|repair}"
            Write-Host "  status  Show Python runtime provenance and native diagnostic state"
            Write-Host "  repair  Reinstall the managed Python runtime when eligible"
        }
        "uninstall" {
            Write-Host "Usage: pith uninstall [--yes]"
            Write-Host "  Remove Pith after an interactive confirmation prompt."
        }
        "profiles" {
            Write-Host "Usage: pith profiles"
            Write-Host "  List local Pith profiles."
        }
        "maintenance" {
            Write-Host "Usage: pith maintenance {run|status|install|uninstall}"
            Write-Host "  run [--phases 1,2,3] [--dry-run]  Run maintenance cycle"
            Write-Host "  status                             Show task status"
            Write-Host "  install                            Install optional Windows Scheduled Task"
            Write-Host "  uninstall                          Remove optional Windows Scheduled Task"
        }
        "stats" {
            Write-Host "Usage: pith stats"
            Write-Host "  Show quick knowledge-base statistics from the active profile database."
        }
        "trust" {
            Write-Host "Usage: pith trust [--json] [--question QUESTION] [QUESTION]"
            Write-Host "       pith trust correct --question QUESTION --message MESSAGE [--apply --confirm] [--json]"
            Write-Host "  Inspect or explicitly correct what Pith currently trusts for a question."
        }
        "trust-health" {
            Write-Host "Usage: pith trust-health [--json]"
            Write-Host "  Show local Trust Health evidence and the internal scorecard claim boundary."
        }
        "protocol" {
            Write-Host "Usage: pith protocol"
            Write-Host "  Print Pith cognitive-loop instructions and copy them to clipboard when available."
        }
        default {
            Write-Host "Usage: pith <command>"
            Write-Host "Commands: serve start stop restart status health stats trust trust-health logs search concept orient sessions metrics doctor clients support import api api-fallback backup restore update version report profiles maintenance protocol runtime uninstall"
        }
    }
}

function Get-PithUninstallMarkerPaths {
    return @(
        "$PithHome\config\uninstalling",
        "$PithHome.uninstalling"
    )
}

function Test-PithUninstallMarker {
    foreach ($MarkerPath in (Get-PithUninstallMarkerPaths)) {
        if (Test-Path -LiteralPath $MarkerPath) {
            return $true
        }
    }
    return $false
}

function Set-PithUninstallMarker {
    New-Item -ItemType Directory -Path "$PithHome\config" -Force | Out-Null
    foreach ($MarkerPath in (Get-PithUninstallMarkerPaths)) {
        Set-Content -Path $MarkerPath -Value ((Get-Date).ToUniversalTime().ToString("o")) -Encoding ASCII
    }
}

function Assert-PithNotUninstalling {
    param([string]$ActionName)

    if (-not (Test-PithUninstallMarker)) {
        return
    }
    Write-Host "Pith uninstall is pending or complete for $PithHome." -ForegroundColor Yellow
    Write-Host "Reinstall Pith before running 'pith $ActionName'." -ForegroundColor Yellow
    exit 1
}

function Remove-PithScheduledTasksForUninstall {
    $TaskNames = @(
        "Pith-OpenAI-Tunnel",
        "Pith-Server",
        "Pith-Server-$env:USERNAME",
        "Pith-Daily-Backup",
        "Pith-Backup-3h",
        "Pith-Maintenance"
    )

    foreach ($TaskName in $TaskNames) {
        Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
    }

    $Survivors = @($TaskNames | Where-Object {
        Get-ScheduledTask -TaskName $_ -ErrorAction SilentlyContinue
    })
    if ($Survivors.Count -eq 0) {
        return [PSCustomObject]@{
            success = $true
            elevation_attempted = $false
            survivors = [string[]]@()
            error = $null
        }
    }

    $CleanupScript = Join-Path ([System.IO.Path]::GetTempPath()) "pith-uninstall-tasks-$([guid]::NewGuid().ToString('N')).ps1"
    $FallbackTaskName = "Pith-Server-$env:USERNAME"
    $CleanupContent = @'
param([string]$FallbackTaskName)

$ErrorActionPreference = "Continue"
$TaskNames = @(
    "Pith-OpenAI-Tunnel",
    "Pith-Server",
    $FallbackTaskName,
    "Pith-Daily-Backup",
    "Pith-Backup-3h",
    "Pith-Maintenance"
)
foreach ($TaskName in $TaskNames) {
    Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
}
'@

    try {
        Set-Content -LiteralPath $CleanupScript -Value $CleanupContent -Encoding UTF8 -ErrorAction Stop
        $PowerShellExe = "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe"
        if (-not (Test-Path -LiteralPath $PowerShellExe -PathType Leaf)) {
            $PowerShellExe = "powershell.exe"
        }
        $Arguments = "-NoProfile -ExecutionPolicy Bypass -File `"$CleanupScript`" -FallbackTaskName `"$FallbackTaskName`""
        $Elevated = Start-Process -FilePath $PowerShellExe -Verb RunAs -ArgumentList $Arguments `
            -Wait -PassThru -ErrorAction Stop
        if ($Elevated.ExitCode -ne 0) {
            throw "Elevated scheduled-task cleanup exited with code $($Elevated.ExitCode)."
        }
    }
    catch {
        return [PSCustomObject]@{
            success = $false
            elevation_attempted = $true
            survivors = [string[]]$Survivors
            error = $_.Exception.Message
        }
    }
    finally {
        Remove-Item -LiteralPath $CleanupScript -Force -ErrorAction SilentlyContinue
    }

    $Survivors = @($TaskNames | Where-Object {
        Get-ScheduledTask -TaskName $_ -ErrorAction SilentlyContinue
    })
    return [PSCustomObject]@{
        success = ($Survivors.Count -eq 0)
        elevation_attempted = $true
        survivors = [string[]]$Survivors
        error = if ($Survivors.Count -gt 0) { "Elevated cleanup left scheduled tasks: $($Survivors -join ', ')" } else { $null }
    }
}

function Stop-PithMcpBridgeProcesses {
    $EscapedPithHome = [regex]::Escape($PithHome)
    try {
        $Processes = Get-CimInstance Win32_Process -ErrorAction Stop |
            Where-Object {
                (($_.CommandLine -match "pith_mcp\.py") -and
                    ($_.CommandLine -match $EscapedPithHome)) -or
                (Test-PithOwnedMcpBootstrapProcess -Process $_)
            }
    }
    catch {
        return
    }

    foreach ($Process in $Processes) {
        try {
            Stop-Process -Id $Process.ProcessId -Force -ErrorAction Stop
            Write-Host "Stopped Pith MCP bridge process PID $($Process.ProcessId)"
        }
        catch {
            Write-Host "Could not stop Pith MCP bridge process PID $($Process.ProcessId): $($_.Exception.Message)" -ForegroundColor Yellow
        }
    }
}

function Stop-PithHomeRuntimeProcesses {
    $NormalizedPithHome = $PithHome.TrimEnd('\') + '\'
    try {
        $Processes = Get-CimInstance Win32_Process -ErrorAction Stop |
            Where-Object {
                $_.ProcessId -ne $PID -and
                $_.ExecutablePath -and
                ([string]$_.ExecutablePath).StartsWith(
                    $NormalizedPithHome,
                    [System.StringComparison]::OrdinalIgnoreCase
                )
            }
    }
    catch {
        return
    }

    foreach ($Process in $Processes) {
        try {
            Stop-Process -Id $Process.ProcessId -Force -ErrorAction Stop
            Write-Host "Stopped Pith runtime process PID $($Process.ProcessId)"
        }
        catch {
            Write-Host "Could not stop Pith runtime process PID $($Process.ProcessId): $($_.Exception.Message)" -ForegroundColor Yellow
        }
    }
}

function Get-PithServerProcesses {
    $EscapedPithHome = [regex]::Escape($PithHome)
    try {
        return @(Get-CimInstance Win32_Process -ErrorAction Stop |
            Where-Object {
                $_.CommandLine -and
                (Test-PithServerCommandLine -CommandLine ([string]$_.CommandLine)) -and
                ($_.CommandLine -match $EscapedPithHome)
            })
    }
    catch {
        return @()
    }
}

function Stop-PithServerProcesses {
    $StoppedAny = $false
    foreach ($Process in (Get-PithServerProcesses)) {
        if ($Process.ProcessId -eq $PID) {
            continue
        }
        try {
            Stop-Process -Id $Process.ProcessId -Force -ErrorAction Stop
            $StoppedAny = $true
            Write-Host "Stopped Pith server process PID $($Process.ProcessId)"
        }
        catch {
            Write-Host "Could not stop Pith server process PID $($Process.ProcessId): $($_.Exception.Message)" -ForegroundColor Yellow
        }
    }
    return $StoppedAny
}

function Test-PithOpenAITunnelHealth {
    $TunnelRoot = Join-Path $PithHome "tunnel"
    $HealthUrlFile = Join-Path $TunnelRoot "health.url"
    $PidFile = Join-Path $TunnelRoot "tunnel-client.pid"
    if (
        (-not (Test-Path -LiteralPath $HealthUrlFile -PathType Leaf)) -or
        (-not (Test-Path -LiteralPath $PidFile -PathType Leaf))
    ) {
        return $false
    }

    $TunnelPid = 0
    $PidText = (Get-Content -LiteralPath $PidFile -Raw -ErrorAction SilentlyContinue).Trim()
    if (-not [int]::TryParse($PidText, [ref]$TunnelPid) -or $TunnelPid -le 0) {
        return $false
    }
    $TunnelProcess = Get-Process -Id $TunnelPid -ErrorAction SilentlyContinue
    if (-not $TunnelProcess -or ([string]$TunnelProcess.ProcessName -notlike "tunnel-client*")) {
        return $false
    }

    $BaseUrl = (Get-Content -LiteralPath $HealthUrlFile -Raw -ErrorAction SilentlyContinue).Trim().TrimEnd('/')
    if ($BaseUrl -notmatch '^http://127\.0\.0\.1:\d+$') {
        return $false
    }
    try {
        $Response = Invoke-WebRequest -UseBasicParsing -Uri "$BaseUrl/healthz" -TimeoutSec 2 -ErrorAction Stop
        return ($Response.StatusCode -eq 200)
    }
    catch {
        return $false
    }
}

function Restore-PithOpenAITunnelAfterRestart {
    $TaskName = "Pith-OpenAI-Tunnel"
    $Task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if (-not $Task) {
        return $true
    }
    if (Test-PithOpenAITunnelHealth) {
        return $true
    }

    $ActualUser = ([string]$Task.Principal.UserId).Trim().ToLowerInvariant()
    $CurrentUser = ([string](whoami)).Trim().ToLowerInvariant()
    $CurrentUserAliases = @(
        $CurrentUser,
        ([string]$env:USERNAME).Trim().ToLowerInvariant(),
        ("$env:USERDOMAIN\$env:USERNAME").Trim().ToLowerInvariant(),
        ("$env:COMPUTERNAME\$env:USERNAME").Trim().ToLowerInvariant()
    ) | Where-Object { -not [string]::IsNullOrWhiteSpace($_) } | Select-Object -Unique
    if ($ActualUser -notin $CurrentUserAliases) {
        Write-Host "ChatGPT tunnel task belongs to another principal; refusing to start it." -ForegroundColor Red
        return $false
    }

    try {
        Start-ScheduledTask -TaskName $TaskName -ErrorAction Stop
    }
    catch {
        Write-Host "Could not restart the enrolled ChatGPT tunnel task: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }

    $Deadline = (Get-Date).AddSeconds(30)
    do {
        if (Test-PithOpenAITunnelHealth) {
            Write-Host "ChatGPT tunnel is healthy after Pith restart"
            return $true
        }
        Start-Sleep -Milliseconds 500
    } while ((Get-Date) -lt $Deadline)

    Write-Host "Pith restarted, but the enrolled ChatGPT tunnel did not recover." -ForegroundColor Red
    return $false
}

function Get-PithRuntimeMetaValue {
    param([string]$Name)

    $RuntimeMetaPath = "$PithHome\config\python-runtime.json"
    if (-not (Test-Path $RuntimeMetaPath)) {
        return ""
    }
    try {
        $Meta = Get-Content -Path $RuntimeMetaPath -Raw | ConvertFrom-Json
        $Property = $Meta.PSObject.Properties[$Name]
        if ($Property) {
            return [string]$Property.Value
        }
    }
    catch {
        return ""
    }
    return ""
}

function Get-PithApiKeySource {
    if ([Environment]::GetEnvironmentVariable("PITH_API_KEY", "Process")) {
        return "process_env"
    }

    if (Test-Path "$PithHome\config\api.key") {
        return "config_api_key_file"
    }

    foreach ($EnvPath in @("$PithHome\.env", "$PithServerPath\.env")) {
        if (-not (Test-Path $EnvPath)) {
            continue
        }
        $Line = Get-Content $EnvPath -ErrorAction SilentlyContinue |
            Where-Object { $_ -match '^PITH_API_KEY=' } |
            Select-Object -First 1
        if ($Line) {
            return "env_file"
        }
    }

    return "none"
}

function Get-PithNativeRuntimeDiagnostic {
    $PythonExists = Test-Path $PythonExe
    $PythonLaunchOk = $false
    $PythonExitCode = $null
    $PythonVersion = $null
    $PythonLaunchError = $null
    if ($PythonExists) {
        try {
            $PythonVersion = (& $PythonExe --version 2>&1 | Out-String).Trim()
            $PythonExitCode = if ($null -eq $LASTEXITCODE) { 0 } else { $LASTEXITCODE }
            $PythonLaunchOk = ($PythonExitCode -eq 0)
        }
        catch {
            $PythonLaunchOk = $false
            $PythonLaunchError = $_.Exception.Message
        }
    }

    $RuntimeManagedBy = Get-PithRuntimeMetaValue -Name "managed_by"
    $RuntimeId = Get-PithRuntimeMetaValue -Name "runtime_id"
    $RuntimeSource = Get-PithRuntimeMetaValue -Name "source"
    $RuntimeMetadataPythonExe = Get-PithRuntimeMetaValue -Name "python_executable"
    $RuntimeBasePythonExe = Get-PithRuntimeMetaValue -Name "base_python_executable"
    $RuntimeVenvPath = Get-PithRuntimeMetaValue -Name "venv_path"
    $ApiUrl = if ($env:PITH_API_URL) { $env:PITH_API_URL.TrimEnd('/') } else { "http://localhost:$($env:PITH_PORT)" }
    $HealthEvidence = Invoke-PithNativeHealthEvidence -ApiUrl $ApiUrl -TimeoutMs 1500
    $ApiKeySource = Get-PithApiKeySource
    $ApiKeyPresent = [bool]$env:PITH_API_KEY
    $ApiKeyFingerprint = Get-PithApiKeyFingerprint -Value $env:PITH_API_KEY
    $ProtectedAuthProbe = [ordered]@{
        tested = $false
        reachable = $false
        ok = $false
        status_code = $null
        error_code = "API_KEY_MISSING"
        message = "No API key source is available to this process."
    }
    if ($ApiKeyPresent) {
        $NativeAuthProbe = Invoke-PithNativeHttpProbeWithLoopbackFallback `
            -ApiUrl $ApiUrl `
            -Path "/auth/validate" `
            -Headers @{ "X-API-Key" = $env:PITH_API_KEY } `
            -TimeoutMs 1500
        $ProtectedAuthProbe = [ordered]@{
            tested = $true
            reachable = $NativeAuthProbe.reachable
            ok = $NativeAuthProbe.ok
            status_code = $NativeAuthProbe.status_code
            error_code = $NativeAuthProbe.error_code
            message = $NativeAuthProbe.message
            probe_transport = $NativeAuthProbe.probe_transport
            api_url = $NativeAuthProbe.api_url
        }
    }

    $ProcessInventory = Get-PithProcessInventory
    $StatusReasons = New-Object System.Collections.Generic.List[string]
    $WarningReasons = New-Object System.Collections.Generic.List[string]
    if (-not $PythonExists) {
        $WarningReasons.Add("python_required_operations_python_executable_missing")
    }
    elseif (-not $PythonLaunchOk) {
        $WarningReasons.Add("python_required_operations_python_launch_failed")
    }
    if (-not $HealthEvidence.ok) {
        $StatusReasons.Add("backend_health_failed")
    }
    if (-not $ApiKeyPresent) {
        $StatusReasons.Add("api_key_missing")
    }
    elseif (-not $ProtectedAuthProbe.ok) {
        $StatusReasons.Add("api_auth_failed")
    }
    if ($ProcessInventory.overlap_detected) {
        $StatusReasons.Add("process_overlap_detected")
    }
    $NativeLifecycleApiReady = ($HealthEvidence.ok -and $ProtectedAuthProbe.ok -and (-not $ProcessInventory.overlap_detected))
    $OverallOk = $NativeLifecycleApiReady

    return [ordered]@{
        schema_version = "pith_windows_runtime_diagnostic.v2"
        status = if ($OverallOk) { "ok" } else { "degraded" }
        diagnostic_status_reason = if ($StatusReasons.Count -gt 0) { [string[]]$StatusReasons.ToArray() } else { @("ok") }
        diagnostic_warning_reason = if ($WarningReasons.Count -gt 0) { [string[]]$WarningReasons.ToArray() } else { @() }
        diagnostic_boundary = "Native wrapper/runtime/backend/auth capability evidence only; this does not prove semantic context retrieval or lifecycle enforcement."
        diagnostic_contract = [ordered]@{
            runtime_status_command = "pith runtime status --json"
            conversation_turn_command = "pith api conversation_turn --stdin-json"
            required_status_report_fields = @("bind_status", "resolved_session_id", "session_active", "auth_error")
            field_sources = [ordered]@{
                bind_status = "conversation_turn.bind_status"
                resolved_session_id = "conversation_turn.resolved_session_id"
                session_active = "conversation_turn._protocol.session_active when present; otherwise conversation_turn.bind_status == bound"
                auth_error = "conversation_turn.auth_error or AUTH_FAILED error payload; otherwise null"
            }
            connected_claim_required = @("runtime_status_ok", "conversation_turn_exit_0", "bind_status_bound", "resolved_session_id_present", "session_active_true", "auth_error_absent")
            status_claim_rule = "A surface must not report connected unless runtime status is ok and conversation_turn returns bind_status=bound, resolved_session_id, session_active=true, and no auth_error for that same surface and turn."
            weak_reachability_not_connected = "Bridge transport, host_transport_open, local HTTP health, or background service status alone may only support reachable/service-running; they are not sufficient for a connected claim."
            claude_desktop_chat_parity_rule = "Claude Desktop / Claude Chat remains unproven until a fresh host session completes pith_connection_proof or a same-turn pith_conversation_turn with model-visible fields; MCP config presence or pith_bridge_status is not enough."
            claude_desktop_chat_timeout_remediation = "If Claude Desktop / Claude Chat times out on bridge/status checks, fully quit the app, confirm no stale app-hosted Pith MCP bridge remains, restart the app, then retest with pith_connection_proof when available or pith_conversation_turn when it is the only exposed Pith tool."
        }
        pith_home = $PithHome
        pith_server_path = $PithServerPath
        api_url = $ApiUrl
        wrapper_invoked = $true
        runtime_managed_by = $RuntimeManagedBy
        runtime_id = $RuntimeId
        runtime_source = $RuntimeSource
        runtime_python_executable = $RuntimeMetadataPythonExe
        runtime_base_python_executable = $RuntimeBasePythonExe
        runtime_venv_path = $RuntimeVenvPath
        runtime_provenance_external = ($RuntimeManagedBy -and ($RuntimeManagedBy -ne "pith"))
        expected_python_executable = $PythonExe
        python_executable = $PythonExe
        python_executable_exists = $PythonExists
        python_launch_ok = $PythonLaunchOk
        python_exit_code = $PythonExitCode
        python_version = $PythonVersion
        python_launch_error = $PythonLaunchError
        api_cli_requires_python = $false
        native_lifecycle_api_ready = $NativeLifecycleApiReady
        native_lifecycle_api_operations = [string[]]($PithNativeApiOperations.Keys | Sort-Object)
        api_python_required_ready = $PythonLaunchOk
        api_python_required_operations = @("lifecycle_diagnostic", "lifecycle_status", "list", "trust", "trust_health", "--example", "--schema")
        api_python_required_boundary = "Python remains required for pseudo/local discovery and trust diagnostics, and for conversation_turn payloads that require the workstream activation gate."
        backend_health_reachable = $HealthEvidence.reachable
        backend_health_ok = $HealthEvidence.ok
        backend_health_status_code = $HealthEvidence.status_code
        backend_health_error_code = $HealthEvidence.error_code
        health_probes = $HealthEvidence.probes
        api_key_present = $ApiKeyPresent
        api_key_source = $ApiKeySource
        api_key_fingerprint = $ApiKeyFingerprint
        api_auth_valid = $ProtectedAuthProbe.ok
        protected_auth_valid = $ProtectedAuthProbe.ok
        protected_auth_probe = $ProtectedAuthProbe
        api_auth_probe = $ProtectedAuthProbe
        process_overlap_detected = $ProcessInventory.overlap_detected
        process_inventory = $ProcessInventory
        next_diagnostic_commands = @(
            "pith api conversation_turn --stdin-json",
            "pith api connection_proof --stdin-json",
            "pith api lifecycle_diagnostic --stdin-json",
            "pith restart"
        )
    }
}

function Show-PithRuntimeProvenance {
    param([switch]$JsonOutput)

    if ($JsonOutput) {
        Get-PithNativeRuntimeDiagnostic | ConvertTo-Json -Depth 8
        return
    }

    $RuntimeMetaPath = "$PithHome\config\python-runtime.json"
    if (Test-Path $PythonExe) {
        $PythonVersion = (& $PythonExe --version 2>&1 | Out-String).Trim()
    }
    else {
        $PythonVersion = "not found"
    }
    Write-Host "Python:       $PythonVersion"
    Write-Host "Python exe:   $PythonExe"
    if (-not (Test-Path $RuntimeMetaPath)) {
        Write-Host "Runtime:      unknown (no python-runtime.json)"
        return
    }
    $ManagedBy = Get-PithRuntimeMetaValue -Name "managed_by"
    $RuntimeId = Get-PithRuntimeMetaValue -Name "runtime_id"
    $RuntimeExe = Get-PithRuntimeMetaValue -Name "python_executable"
    $RuntimeSource = Get-PithRuntimeMetaValue -Name "source"
    $RuntimeSha = Get-PithRuntimeMetaValue -Name "sha256"
    Write-Host "Runtime:      $ManagedBy"
    Write-Host "Runtime ID:   $RuntimeId"
    Write-Host "Runtime exe:  $RuntimeExe"
    Write-Host "Runtime src:  $RuntimeSource"
    Write-Host "Runtime sha:  $RuntimeSha"
}

Import-PithEnvFile -EnvFile "$PithHome\.env"
Import-PithEnvFile -EnvFile "$PithServerPath\.env"
Import-PithApiKeyFile

$env:PITH_HOME = $PithHome
$env:PITH_PORT = Resolve-PithPort
if (-not $env:PITH_API_URL) {
    $env:PITH_API_URL = "http://localhost:$($env:PITH_PORT)"
}
$ResolvedPithDataDir = Resolve-PithDataDir
if ($ResolvedPithDataDir) {
    $env:PITH_DATA_DIR = $ResolvedPithDataDir
}

$script:PithCliArgs = $args
$script:PithCliPipelineInput = @($input)
$action = 'status'
if ($args.Count -gt 0) { $action = $args[0] }

if ((Test-PithUninstallMarker) -and ($action -eq "status")) {
    Write-Host "Pith is not running"
    Write-Host "Health: Uninstall pending or complete"
    exit 0
}

switch ($action) {
    "serve" {
        if (Test-PithHelpRequest) { Print-PithWrapperHelp -CommandName "serve"; exit 0 }
        Assert-PithNotUninstalling -ActionName "serve"
        Write-Host "Serving Pith Server on port $($env:PITH_PORT)..."
        Push-Location $PithServerPath
        try {
            & $PythonExe -m app.api.serve
            $exitCode = if ($null -eq $LASTEXITCODE) { 0 } else { $LASTEXITCODE }
        }
        finally {
            Pop-Location
        }
        exit $exitCode
    }
    "start" {
        if (Test-PithHelpRequest) { Print-PithWrapperHelp -CommandName "start"; exit 0 }
        Assert-PithNotUninstalling -ActionName "start"
        $PithPort = Resolve-PithPort
        Write-Host "Starting Pith Server on port $PithPort..."
        New-Item -ItemType Directory -Path "$PithHome\logs" -Force | Out-Null
        $ServerLog = "$PithHome\logs\server.log"
        $ServerErrLog = "$PithHome\logs\server.err.log"
        $ExistingDeadline = (Get-Date).AddSeconds(5)
        $ExistingLive = $false
        while ((Get-Date) -lt $ExistingDeadline) {
            if (Test-PithHealthz -Port $PithPort -TimeoutMs 1000) {
                $ExistingLive = $true
                break
            }
            Start-Sleep -Milliseconds 500
        }
        if ($ExistingLive) {
            $ReadyDeadline = (Get-Date).AddSeconds(120)
            while ((Get-Date) -lt $ReadyDeadline) {
                if (Test-PithConversationReady -Port $PithPort -TimeoutSeconds 2) {
                    Write-Host "Pith is already running and conversation-ready on port $PithPort"
                    exit 0
                }
                Start-Sleep -Milliseconds 500
            }
            Write-Host "Pith is running but did not become conversation-ready within 120 seconds." -ForegroundColor Red
            exit 1
        }
        $ExistingServerProcesses = @(Get-PithServerProcesses)
        if ($ExistingServerProcesses.Count -gt 0) {
            Write-Host "Pith server process exists but health is not reachable; clearing stale processes before start retry..." -ForegroundColor Yellow
            Remove-Item "$PithHome\pith.pid" -ErrorAction SilentlyContinue
            Stop-PithServerProcesses | Out-Null
            Start-Sleep -Seconds 2
            if (Test-PithHealthz -Port $PithPort -TimeoutMs 1000) {
                Write-Host "Pith recovered on port $PithPort after stale process cleanup"
                exit 0
            }
            $RemainingServerProcesses = @(Get-PithServerProcesses)
            if ($RemainingServerProcesses.Count -gt 0) {
                Write-Host "Pith server processes remain after stale cleanup. Run 'pith stop' and retry." -ForegroundColor Red
                exit 1
            }
        }
        $WindowsEmbeddingRuntime = "$PithServerPath\scripts\windows_embedding_runtime.ps1"
        if (-not (Test-Path -LiteralPath $WindowsEmbeddingRuntime -PathType Leaf)) {
            Write-Host "Windows runtime helper missing: $WindowsEmbeddingRuntime" -ForegroundColor Red
            exit 1
        }
        . $WindowsEmbeddingRuntime
        $proc = Start-PithDetachedPythonProcess `
            -PythonExe $PythonExe `
            -WorkingDirectory $PithServerPath `
            -Arguments @('-m', 'app.api.serve') `
            -StdoutPath $ServerLog `
            -StderrPath $ServerErrLog
        Set-Content -Path "$PithHome\pith.pid" -Value $proc.Id -Encoding ASCII
        $ConversationReady = $false
        $StartDeadline = (Get-Date).AddSeconds(120)
        while ((Get-Date) -lt $StartDeadline) {
            if ($proc.HasExited) {
                break
            }
            if (Test-PithConversationReady -Port $PithPort -TimeoutSeconds 2) {
                Start-Sleep -Seconds 1
                if (-not $proc.HasExited -and (Test-PithConversationReady -Port $PithPort -TimeoutSeconds 2)) {
                    $ConversationReady = $true
                    break
                }
            }
            Start-Sleep -Milliseconds 500
        }
        if ($ConversationReady) {
            Write-Host "Pith started successfully and is conversation-ready (PID: $($proc.Id))"
            exit 0
        }
        else {
            if ($proc.HasExited) {
                Write-Host "Pith failed to start. See $ServerErrLog" -ForegroundColor Red
            }
            else {
                Write-Host "Pith process started, but did not become conversation-ready. See $ServerLog" -ForegroundColor Yellow
            }
            if (Test-Path $ServerErrLog) {
                Get-Content $ServerErrLog -Tail 20
            }
            exit 1
        }
    }
    "stop" {
        if (Test-PithHelpRequest) { Print-PithWrapperHelp -CommandName "stop"; exit 0 }
        Write-Host "Stopping Pith server..."
        $PithPort = Resolve-PithPort
        $StoppedAny = $false
        $StopFailed = $false
        if (Test-Path "$PithHome\pith.pid") {
            $pidVal = (Get-Content "$PithHome\pith.pid" -Raw).Trim()
            if ($pidVal -match '^\d+$') {
                & taskkill.exe /PID $pidVal /T /F 2>$null | Out-Null
                if ($LASTEXITCODE -eq 0) {
                    $StoppedAny = $true
                }
                else {
                    $StopFailed = $true
                    Write-Host "Could not stop Pith server PID $pidVal. The process may be elevated or owned by another user." -ForegroundColor Red
                }
            }
            else {
                Remove-Item "$PithHome\pith.pid" -ErrorAction SilentlyContinue
            }
        }
        if (Stop-PithServerProcesses) {
            $StoppedAny = $true
        }
        $PortOwners = @()
        try {
            $PortOwners = @(Get-NetTCPConnection -LocalPort $PithPort -State Listen -ErrorAction SilentlyContinue |
                Select-Object -ExpandProperty OwningProcess -Unique)
        }
        catch {
            $PortOwners = @()
        }
        foreach ($OwnerPid in $PortOwners) {
            if ($OwnerPid -and ($OwnerPid -ne $PID)) {
                & taskkill.exe /PID $OwnerPid /T /F 2>$null | Out-Null
                if ($LASTEXITCODE -eq 0) {
                    $StoppedAny = $true
                }
                else {
                    $StopFailed = $true
                    Write-Host "Could not stop process PID $OwnerPid listening on Pith port $PithPort." -ForegroundColor Red
                }
            }
        }
        $StopDeadline = (Get-Date).AddSeconds(10)
        while ((Get-Date) -lt $StopDeadline) {
            if (-not (Test-PithHealthz -Port $PithPort -TimeoutMs 500)) {
                break
            }
            Start-Sleep -Milliseconds 250
        }
        if (Test-PithHealthz -Port $PithPort -TimeoutMs 500) {
            Write-Host "Pith is still reachable on port $PithPort; stop failed." -ForegroundColor Red
            exit 1
        }
        Remove-Item "$PithHome\pith.pid" -ErrorAction SilentlyContinue
        if ($StoppedAny) {
            Start-Sleep -Milliseconds 500
            Write-Host "Pith stopped"
        } elseif ($StopFailed) {
            exit 1
        } else {
            Write-Host "Pith is not running"
        }
    }
    "restart" {
        if (Test-PithHelpRequest) { Print-PithWrapperHelp -CommandName "restart"; exit 0 }
        Assert-PithNotUninstalling -ActionName "restart"
        & $MyInvocation.MyCommand.Path stop
        $stopExit = if ($null -eq $LASTEXITCODE) { 0 } else { $LASTEXITCODE }
        if ($stopExit -ne 0) {
            Write-Host "Pith restart aborted because the existing server could not be stopped." -ForegroundColor Red
            exit $stopExit
        }
        Start-Sleep -Seconds 1
        & $MyInvocation.MyCommand.Path start
        $restartExit = if ($null -eq $LASTEXITCODE) { 0 } else { $LASTEXITCODE }
        if ($restartExit -ne 0) {
            exit $restartExit
        }
        if (-not (Restore-PithOpenAITunnelAfterRestart)) {
            exit 1
        }
        exit $restartExit
    }
    "status" {
        if (Test-PithHelpRequest) { Print-PithWrapperHelp -CommandName "status"; exit 0 }
        Push-Location $PithServerPath
        try {
            if ($args.Count -gt 1) {
                & $PythonExe -m app.ops.support_cli status @($args[1..($args.Count - 1)])
            } else {
                & $PythonExe -m app.ops.support_cli status
            }
            $statusExit = if ($null -eq $LASTEXITCODE) { 0 } else { $LASTEXITCODE }
        }
        finally {
            Pop-Location
        }
        exit $statusExit
    }
    "health" {
        if (Test-PithHelpRequest) { Print-PithWrapperHelp -CommandName "health"; exit 0 }
        Push-Location $PithServerPath
        if ($args.Count -gt 1) {
            & $PythonExe -m app.ops.health_cli @($args[1..($args.Count - 1)])
        } else {
            & $PythonExe -m app.ops.health_cli
        }
        $exitCode = if ($null -eq $LASTEXITCODE) { 0 } else { $LASTEXITCODE }
        Pop-Location
        exit $exitCode
    }
    "logs" {
        if (Test-PithHelpRequest) { Print-PithWrapperHelp -CommandName "logs"; exit 0 }
        if (($args.Count -gt 1) -and ($args[1] -eq "snapshot")) {
            Get-PithLogSnapshot -CommandArgs $args
        } elseif (Test-Path "$PithHome\logs\server.log") {
            Get-Content "$PithHome\logs\server.log" -Tail 50 -Wait
        } else {
            Write-Host "No logs found"
        }
    }
    "import" {
        if (Test-PithHelpRequest) { Print-PithWrapperHelp -CommandName "import"; exit 0 }
        Push-Location $PithServerPath
        try {
            if ($args.Count -gt 1) {
                & $PythonExe -m app.ops.import_cli @($args[1..($args.Count - 1)])
            } else {
                & $PythonExe -m app.ops.import_cli
            }
            $importExit = if ($null -eq $LASTEXITCODE) { 0 } else { $LASTEXITCODE }
        }
        finally {
            Pop-Location
        }
        exit $importExit
    }
    { $_ -in @("search", "concept", "orient", "sessions", "metrics") } {
        if (Test-PithHelpRequest) { Print-PithWrapperHelp -CommandName $args[0]; exit 0 }
        Push-Location $PithServerPath
        try {
            if ($args.Count -gt 1) {
                & $PythonExe -m app.ops.read_cli $args[0] @($args[1..($args.Count - 1)])
            } else {
                & $PythonExe -m app.ops.read_cli $args[0]
            }
            $readExit = if ($null -eq $LASTEXITCODE) { 0 } else { $LASTEXITCODE }
        }
        finally {
            Pop-Location
        }
        exit $readExit
    }
    { $_ -in @("doctor", "clients", "support") } {
        if (Test-PithHelpRequest) { Print-PithWrapperHelp -CommandName $args[0]; exit 0 }
        Push-Location $PithServerPath
        try {
            if ($args.Count -gt 1) {
                & $PythonExe -m app.ops.support_cli $args[0] @($args[1..($args.Count - 1)])
            } else {
                & $PythonExe -m app.ops.support_cli $args[0]
            }
            $supportExit = if ($null -eq $LASTEXITCODE) { 0 } else { $LASTEXITCODE }
        }
        finally {
            Pop-Location
        }
        exit $supportExit
    }
    { $_ -in @("api", "api-fallback") } {
        if (Test-PithHelpRequest) { Print-PithWrapperHelp -CommandName $args[0]; exit 0 }
        $ApiArgs = @()
        if ($args.Count -gt 1) {
            $ApiArgs = @($args[1..($args.Count - 1)])
        }
        Invoke-PithApiCommand -CommandName $args[0] -RemainingArgs $ApiArgs
    }
    "backup" {
        if (Test-PithHelpRequest) { Print-PithWrapperHelp -CommandName "backup"; exit 0 }
        $SafeBackup = "$PithHome\pith-server\scripts\backup\safe_backup.ps1"
        if (Test-Path $SafeBackup) {
            & powershell -NoProfile -ExecutionPolicy Bypass -File $SafeBackup
            $backupExit = if ($null -eq $LASTEXITCODE) { 0 } else { $LASTEXITCODE }
            exit $backupExit
        } else {
            Write-Host "Backup script not found: $SafeBackup" -ForegroundColor Red
            exit 1
        }
    }
    "restore" {
        if (Test-PithHelpRequest) { Print-PithWrapperHelp -CommandName "restore"; exit 0 }
        Assert-PithNotUninstalling -ActionName "restore"
        if ($args.Count -lt 2) {
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
            $restoreExit = if ($null -eq $LASTEXITCODE) { 0 } else { $LASTEXITCODE }
            exit $restoreExit
        } else {
            Write-Host "WARNING: Database integrity check failed: $IntCheck" -ForegroundColor Red
            Write-Host "Restore aborted. Original database may need manual recovery."
            exit 1
        }
    }
    "update" {
        if (Test-PithHelpRequest) { Print-PithWrapperHelp -CommandName "update"; exit 0 }
        Assert-PithNotUninstalling -ActionName "update"
        Write-Host "Updating Pith..."
        & $MyInvocation.MyCommand.Path stop
        Start-Sleep -Seconds 1
        $CoreReqFile = "$PithHome\logs\core_requirements_windows_update.txt"
        Get-Content -Path "$PithServerPath\requirements.txt" | Where-Object {
            $_ -notmatch '^\s*sentence-transformers\b' -and $_ -notmatch '^\s*torch\b'
        } | Set-Content -Path $CoreReqFile
        & $PipExe install --quiet --upgrade -r $CoreReqFile 2>$null
        $coreUpdateExit = if ($null -eq $LASTEXITCODE) { 0 } else { $LASTEXITCODE }
        if ($coreUpdateExit -ne 0) {
            Write-Host "Core dependency update failed (exit $coreUpdateExit)." -ForegroundColor Red
            exit $coreUpdateExit
        }

        $WindowsEmbeddingRuntime = "$PithServerPath\scripts\windows_embedding_runtime.ps1"
        if (Test-Path $WindowsEmbeddingRuntime) {
            . $WindowsEmbeddingRuntime
            $EmbedResult = Install-PithEmbeddings `
                -PithHome $PithHome `
                -VenvPath $VenvPath `
                -PipExe $PipExe `
                -PythonExe $PythonExe
            if (-not $EmbedResult) {
                Write-Host "  Pith will run with TF-IDF search (fully functional, reduced semantic quality)." -ForegroundColor Yellow
            }
        } else {
            Write-Host "Embedding runtime helper missing. Using TF-IDF search." -ForegroundColor Yellow
            "embeddings=false`nreason=embedding_runtime_script_missing" | Out-File "$PithHome\.install_capabilities"
        }

        & $MyInvocation.MyCommand.Path start
        $updateStartExit = if ($null -eq $LASTEXITCODE) { 0 } else { $LASTEXITCODE }
        if ($updateStartExit -ne 0) {
            Write-Host "Update completed, but Pith failed to restart (exit $updateStartExit)." -ForegroundColor Red
            exit $updateStartExit
        }
        Write-Host "Update complete"
        exit 0
    }
    "uninstall" {
        if (Test-PithHelpRequest) { Print-PithWrapperHelp -CommandName "uninstall"; exit 0 }
        Write-Host "This will remove Pith and all its data from $PithHome"
        $Confirm = ""
        if (($args.Count -gt 1) -and (($args[1] -eq "--yes") -or ($args[1] -eq "-y"))) {
            $Confirm = "yes"
        }
        else {
            $Confirm = Read-Host "Are you sure? (yes/no)"
        }
        if ($Confirm -eq "yes") {
            Set-PithUninstallMarker
            & $MyInvocation.MyCommand.Path stop
            Start-Sleep -Seconds 1
            Stop-PithMcpBridgeProcesses
            Stop-PithHomeRuntimeProcesses
            $TaskCleanup = Remove-PithScheduledTasksForUninstall
            if (-not $TaskCleanup.success) {
                $RemainingTasks = @($TaskCleanup.survivors) -join ", "
                Write-Host "Uninstall could not remove scheduled tasks: $RemainingTasks" -ForegroundColor Red
                if ($TaskCleanup.error) {
                    Write-Host "Scheduled-task cleanup error: $($TaskCleanup.error)" -ForegroundColor Red
                }
                Write-Host "Pith remains in terminal uninstall state. Approve the administrator prompt, then rerun 'pith uninstall --yes'." -ForegroundColor Yellow
                exit 1
            }
            Remove-PithOwnedPathEntries -PithBinDir "$PithHome\bin"
            Remove-PithProfilePathLines -PithBinDir "$PithHome\bin"
            if ($VenvPath -and (Test-Path $VenvPath) -and ($VenvPath.TrimEnd('\') -ine "$PithHome\venv".TrimEnd('\')) -and ($VenvPath -notlike "$PithHome*")) {
                Start-PithDeferredRemoval -TargetPath $VenvPath
            }
            try {
                Start-PithDeferredRemoval -TargetPath $PithHome -MarkerPath "$PithHome.uninstalling"
            }
            catch {
                Write-Host "Uninstall failed: $($_.Exception.Message)" -ForegroundColor Red
                Write-Host "Pith data remains at $PithHome" -ForegroundColor Yellow
                exit 1
            }
            Write-Host "Pith uninstall started. CLI start/restart commands are disabled while data removal finishes in the background."
        } else {
            Write-Host "Uninstall cancelled"
            exit 1
        }
    }
    "maintenance" {
        if (Test-PithHelpRequest) { Print-PithWrapperHelp -CommandName "maintenance"; exit 0 }
        $maintenanceAction = if ($args.Count -gt 1) { $args[1] } else { "run" }
        if ($maintenanceAction -ne "status") {
            Assert-PithNotUninstalling -ActionName "maintenance $maintenanceAction"
        }
        switch ($maintenanceAction) {
            "run" {
                Write-Host "Running maintenance..."
                Push-Location $PithServerPath
                try {
                    if ($args.Count -gt 2) {
                        & $PythonExe -m app.ops.maintenance_cli run @($args[2..($args.Count - 1)])
                    } else {
                        & $PythonExe -m app.ops.maintenance_cli run
                    }
                    $maintenanceExit = if ($null -eq $LASTEXITCODE) { 0 } else { $LASTEXITCODE }
                }
                finally {
                    Pop-Location
                }
                exit $maintenanceExit
            }
            "status" {
                Push-Location $PithServerPath
                try {
                    & $PythonExe -m app.ops.maintenance_cli status
                    $maintenanceExit = if ($null -eq $LASTEXITCODE) { 0 } else { $LASTEXITCODE }
                }
                finally {
                    Pop-Location
                }
                exit $maintenanceExit
            }
            "install" {
                $TaskName = "Pith-Maintenance"
                $PithCmd = "$PithHome\bin\pith.cmd"
                if (-not (Test-Path $PithCmd)) {
                    Write-Host "Cannot install maintenance task: pith.cmd not found at $PithCmd" -ForegroundColor Red
                    exit 1
                }
                $Action = New-ScheduledTaskAction -Execute $PithCmd -Argument "maintenance run"
                $Trigger = New-ScheduledTaskTrigger -Daily -At "3:00AM"
                $Principal = New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
                $Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Hours 2)
                Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Principal $Principal -Settings $Settings -Description "Run Pith maintenance daily at 3:00 AM" -Force -ErrorAction Stop | Out-Null
                Write-Host "Installed optional Windows maintenance task: $TaskName"
            }
            "uninstall" {
                Unregister-ScheduledTask -TaskName "Pith-Maintenance" -Confirm:$false -ErrorAction SilentlyContinue
                Write-Host "Removed optional Windows maintenance task: Pith-Maintenance"
            }
            default {
                Write-Host "Usage: pith maintenance {run|status|install|uninstall}" -ForegroundColor Yellow
                exit 1
            }
        }
    }
    "version" {
        if (Test-PithHelpRequest) { Print-PithWrapperHelp -CommandName "version"; exit 0 }
        $CapFile = "$PithHome\.install_capabilities"
        Write-Host "Pith v__PITH_VERSION__"
        Show-PithRuntimeProvenance
        if (Test-Path $CapFile) {
            Get-Content $CapFile | ForEach-Object { Write-Host "  $_" }
        } else {
            Write-Host "  (no capabilities info - re-run installer)"
        }
    }
    "runtime" {
        if (Test-PithHelpRequest) { Print-PithWrapperHelp -CommandName "runtime"; exit 0 }
        $runtimeAction = if ($args.Count -gt 1) { $args[1] } else { "status" }
        switch ($runtimeAction) {
            "status" {
                $RuntimeJsonOutput = $args -contains "--json"
                Show-PithRuntimeProvenance -JsonOutput:$RuntimeJsonOutput
            }
            "repair" {
                Assert-PithNotUninstalling -ActionName "runtime repair"
                $Installer = "$PithServerPath\scripts\install.ps1"
                if (-not (Test-Path $Installer)) {
                    Write-Host "Runtime repair needs the installer at $Installer" -ForegroundColor Red
                    Write-Host "Run from the extracted beta artifact: `$env:PITH_AUTO_PYTHON=1; `$env:PITH_REPAIR_RUNTIME=1; powershell -ExecutionPolicy Bypass -File scripts\install.ps1 -Force"
                    exit 1
                }
                $ManagedBy = Get-PithRuntimeMetaValue -Name "managed_by"
                if ($ManagedBy -and ($ManagedBy -ne "pith")) {
                    $RuntimeExe = Get-PithRuntimeMetaValue -Name "python_executable"
                    Write-Host "Refusing to repair non-Pith Python runtime: $RuntimeExe" -ForegroundColor Red
                    exit 1
                }
                $env:PITH_AUTO_PYTHON = "1"
                $env:PITH_REPAIR_RUNTIME = "1"
                & powershell -NoProfile -ExecutionPolicy Bypass -File $Installer -Force
                exit $LASTEXITCODE
            }
            default {
                Write-Host "Usage: pith runtime {status|repair}" -ForegroundColor Yellow
                exit 1
            }
        }
    }
    "profiles" {
        if (Test-PithHelpRequest) { Print-PithWrapperHelp -CommandName "profiles"; exit 0 }
        $ProfilesRoot = Join-Path ([Environment]::GetFolderPath("UserProfile")) "pith-data"
        $ActiveProfile = if ($env:PITH_PROFILE) { $env:PITH_PROFILE } else { "default" }
        Write-Host "Available profiles in $ProfilesRoot"
        if (Test-Path $ProfilesRoot) {
            Get-ChildItem -Path $ProfilesRoot -Directory -ErrorAction SilentlyContinue | ForEach-Object {
                $DbSize = ""
                foreach ($DbName in @("pith.db", "brain.db", "data\pith.db", "data\brain.db")) {
                    $Candidate = Join-Path $_.FullName $DbName
                    if (Test-Path $Candidate) {
                        $DbSize = " ($([math]::Round((Get-Item $Candidate).Length / 1MB, 1)) MB)"
                        break
                    }
                }
                $Active = if ($_.Name -eq $ActiveProfile) { " [active]" } else { "" }
                Write-Host "  * $($_.Name)$DbSize$Active"
            }
        }
        else {
            Write-Host "  No profiles found. Create one with: mkdir $ProfilesRoot\myprofile"
        }
    }
    "report" {
        if (Test-PithHelpRequest) { Print-PithWrapperHelp -CommandName "report"; exit 0 }
        Push-Location $PithServerPath
        try {
            if ($args.Count -gt 1) {
                & $PythonExe -m app.ops.support_cli report @($args[1..($args.Count - 1)])
            } else {
                & $PythonExe -m app.ops.support_cli report
            }
            $reportExit = if ($null -eq $LASTEXITCODE) { 0 } else { $LASTEXITCODE }
        }
        finally {
            Pop-Location
        }
        exit $reportExit
    }
    "stats" {
        if (Test-PithHelpRequest) { Print-PithWrapperHelp -CommandName "stats"; exit 0 }
        $DbPath = Resolve-DbPath
        if (-not (Test-Path $DbPath)) {
            Write-Host "No database found at $DbPath" -ForegroundColor Yellow
            Write-Host "Run 'pith start' first to initialize."
            exit 1
        }
        $StatsScript = @'
import os
import sqlite3
import sys

db_path = sys.argv[1]
conn = sqlite3.connect(db_path)
try:
    concepts = conn.execute('SELECT COUNT(*) FROM concepts WHERE status = "active"').fetchone()[0]
    total = conn.execute('SELECT COUNT(*) FROM concepts').fetchone()[0]
    kas = conn.execute('SELECT COUNT(DISTINCT knowledge_area) FROM concepts WHERE status = "active"').fetchone()[0]
    associations = conn.execute('SELECT COUNT(*) FROM associations').fetchone()[0]
    sessions = conn.execute('SELECT COUNT(*) FROM sessions').fetchone()[0]
    db_mb = os.path.getsize(db_path) / (1024 * 1024)
    print("Pith Stats")
    print("==========================")
    print(f"  Concepts:        {concepts:,} active ({total:,} total)")
    print(f"  Knowledge Areas: {kas}")
    print(f"  Associations:    {associations:,}")
    print(f"  Sessions:        {sessions:,}")
    print(f"  Database:        {db_mb:.1f} MB")
    print(f"  Path:            {db_path}")
except Exception as exc:
    print(f"Error reading stats: {exc}")
    sys.exit(1)
finally:
    conn.close()
'@
        $StatsScriptPath = Join-Path ([System.IO.Path]::GetTempPath()) ("pith-stats-{0}.py" -f ([System.Guid]::NewGuid().ToString("N")))
        $StatsExitCode = 1
        try {
            Set-Content -LiteralPath $StatsScriptPath -Value $StatsScript -Encoding UTF8
            & $PythonExe $StatsScriptPath $DbPath
            $StatsExitCode = if ($null -eq $LASTEXITCODE) { 0 } else { $LASTEXITCODE }
        } finally {
            Remove-Item -LiteralPath $StatsScriptPath -Force -ErrorAction SilentlyContinue
        }
        exit $StatsExitCode
    }
    "trust-health" {
        if (Test-PithHelpRequest) { Print-PithWrapperHelp -CommandName "trust-health"; exit 0 }
        Push-Location $PithServerPath
        if ($args.Count -gt 1) {
            & $PythonExe "$PithServerPath\scripts\trust_health_status.py" @($args[1..($args.Count - 1)])
        } else {
            & $PythonExe "$PithServerPath\scripts\trust_health_status.py"
        }
        $trustHealthExit = if ($null -eq $LASTEXITCODE) { 0 } else { $LASTEXITCODE }
        Pop-Location
        exit $trustHealthExit
    }
    "trust" {
        if (Test-PithHelpRequest) { Print-PithWrapperHelp -CommandName "trust"; exit 0 }
        Push-Location $PithServerPath
        if ($args.Count -gt 1) {
            & $PythonExe "$PithServerPath\scripts\trust_control.py" @($args[1..($args.Count - 1)])
        } else {
            & $PythonExe "$PithServerPath\scripts\trust_control.py"
        }
        $trustExit = if ($null -eq $LASTEXITCODE) { 0 } else { $LASTEXITCODE }
        Pop-Location
        exit $trustExit
    }
    "protocol" {
        if (Test-PithHelpRequest) { Print-PithWrapperHelp -CommandName "protocol"; exit 0 }
        $SystemPrompt = "$PithHome\SYSTEM_PROMPT.md"
        if (-not (Test-Path $SystemPrompt)) {
            Write-Host "System prompt not found at $SystemPrompt" -ForegroundColor Red
            Write-Host "Re-run the installer to regenerate it."
            exit 1
        }
        Get-Content -Path $SystemPrompt
        Write-Host ""
        if (Get-Command Set-Clipboard -ErrorAction SilentlyContinue) {
            try {
                Get-Content -Path $SystemPrompt -Raw | Set-Clipboard
                Write-Host "--- Copied to clipboard. Paste into Claude Desktop -> Settings -> General -> Instructions for Claude ---"
            }
            catch {
                Write-Host "--- Clipboard unavailable. Paste the text above into Claude Desktop -> Settings -> General -> Instructions for Claude ---"
            }
        }
        else {
            Write-Host "--- Paste the above into Claude Desktop -> Settings -> General -> Instructions for Claude ---"
        }
    }
    default {
        Write-Host "Pith - Personal Knowledge Server"
        Write-Host ""
        Write-Host "Usage: pith <command>"
        Write-Host ""
        Write-Host "Commands:"
        Write-Host "  start       Start the Pith server"
        Write-Host "  stop        Stop the Pith server"
        Write-Host "  restart     Restart the Pith server"
        Write-Host "  status      Check server status"
        Write-Host "  health      Check operational health/readiness"
        Write-Host "  stats       Show quick knowledge-base statistics"
        Write-Host "  trust"
        Write-Host "              Inspect or explicitly correct what Pith currently trusts"
        Write-Host "  trust-health"
        Write-Host "              Show local Trust Health evidence"
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
        Write-Host "  api         Run first-class local API command"
        Write-Host "  api-fallback"
        Write-Host "              Run exec HTTP fallback API command"
        Write-Host "  backup      Create a WAL-safe backup"
        Write-Host "  restore     Restore from a backup file"
        Write-Host "  update      Update dependencies + embeddings"
        Write-Host "  uninstall   Remove Pith completely"
        Write-Host "  profiles    List local Pith profiles"
        Write-Host "  maintenance {run|status|install|uninstall}"
        Write-Host "                Run/status/install/remove maintenance scheduler"
        Write-Host "  protocol    Print cognitive-loop instructions"
        Write-Host "  runtime     Show or repair Python runtime"
        Write-Host "  version     Show version and capabilities"
        Write-Host "  report      Generate diagnostics report"
    }
}
