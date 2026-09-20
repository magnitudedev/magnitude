param([string]$LAB_INITIALIZATION = $env:LAB_INITIALIZATION)
Remove-Item Env:LAB_INITIALIZATION -ErrorAction SilentlyContinue
if ($PSVersionTable.PSVersion.Major -lt 7) { throw 'Runtime preparation requires the pinned PowerShell 7 executable' }
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
Set-StrictMode -Version Latest
if (-not [Security.Principal.WindowsIdentity]::GetCurrent().IsSystem) { throw 'Runtime preparation requires SYSTEM' }
$config = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($LAB_INITIALIZATION)) | ConvertFrom-Json
$LAB_INITIALIZATION = $null
$os = Get-CimInstance Win32_OperatingSystem
$build = [int]$os.BuildNumber
$client = $os.ProductType -eq 1
$matches = if ($config.distribution.os -eq 'windows') {
  $client -and (($config.distribution.version -eq '10' -and $build -ge 19041 -and $build -lt 22000) -or ($config.distribution.version -eq '11' -and $build -ge 22000))
} else { $config.distribution.os -eq 'windows-server' -and $config.distribution.version -eq '2025' -and -not $client -and $build -eq 26100 }
if (-not $matches -or $config.architecture -ne 'x64' -or $env:PROCESSOR_ARCHITECTURE -ne 'AMD64') { throw 'Runtime image does not match its admitted distribution and architecture' }
if ($config.adminUsername -notmatch '^[a-z][a-z0-9]{1,19}$') { throw 'Invalid worker account' }
$account = Get-LocalUser -Name $config.adminUsername
if (-not $account.Enabled) { throw 'Worker account is disabled' }
$toolsRoot = 'C:\MagnitudeLab\Tools'
if (-not (Test-Path -LiteralPath (Join-Path $toolsRoot 'receipt.json'))) { throw 'Pinned tool installation has not completed' }
$tools = Get-Content -LiteralPath (Join-Path $toolsRoot 'paths.json') -Raw | ConvertFrom-Json
$root = 'C:\MagnitudeLab\Runtime'
$state = 'C:\MagnitudeLab\State'
foreach ($directory in @($root,$state)) {
  if (Test-Path -LiteralPath $directory) { throw 'Runtime preparation requires fresh runtime and state directories' }
  New-Item -ItemType Directory -Path $directory | Out-Null
  $acl = [Security.AccessControl.DirectorySecurity]::new()
  $acl.SetAccessRuleProtection($true,$false)
  $system = [Security.Principal.SecurityIdentifier]::new('S-1-5-18')
  $acl.SetOwner($system)
  $acl.AddAccessRule([Security.AccessControl.FileSystemAccessRule]::new($system,'FullControl','ContainerInherit,ObjectInherit','None','Allow'))
  Set-Acl -LiteralPath $directory -AclObject $acl
}
$archive = Join-Path $root 'runtime.tar.gz'
# The runtime grant is blob-only and temporary. No redirect may carry it to another host.
try {
  $item = $config.runtime
  $url = [Uri]$item.url
  if ($url.Scheme -ne 'https' -or $url.UserInfo -or $url.Fragment -or $item.sha256 -notmatch '^[a-f0-9]{64}$' -or $item.bytes -lt 1 -or $item.bytes -gt 1GB) { throw 'Invalid runtime pin' }
  $request = [Net.HttpWebRequest]::Create($url)
  $request.AllowAutoRedirect = $false
  $request.Timeout = 120000
  $request.ReadWriteTimeout = 120000
  $response = $request.GetResponse()
  try {
    if ([int]$response.StatusCode -ne 200) { throw 'Unexpected runtime response' }
    $stream = $response.GetResponseStream()
    $output = [IO.File]::Open($archive,[IO.FileMode]::CreateNew)
    try {
      $buffer = New-Object byte[] 1048576
      [long]$total = 0
      while (($count = $stream.Read($buffer,0,$buffer.Length)) -gt 0) {
        $total += $count
        if ($total -gt $item.bytes) { throw 'Oversized runtime' }
        $output.Write($buffer,0,$count)
      }
    } finally { $output.Dispose(); $stream.Dispose() }
  } finally { $response.Dispose() }
  if ($total -ne $item.bytes -or (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne $item.sha256) { throw 'Runtime integrity mismatch' }
} catch { throw 'Pinned runtime download failed integrity or transport validation' }
$runtimeDigest = $config.runtime.sha256
$config.runtime = $null
function Invoke-Native([string]$Executable,[string[]]$Arguments) {
  & $Executable @Arguments
  if ($LASTEXITCODE -ne 0) { throw "Preparation command $([IO.Path]::GetFileName($Executable)) exited $LASTEXITCODE" }
}
Invoke-Native (Join-Path $env:WINDIR 'System32\tar.exe') @('-xzf',$archive,'-C',$root)
Remove-Item -LiteralPath $archive
$workspace = Join-Path $root 'runtime'
$rust = [regex]::Match((Get-Content -LiteralPath (Join-Path $workspace 'inference\rust-toolchain.toml') -Raw),'(?m)^channel\s*=\s*"(\d+\.\d+\.\d+)"\s*$')
if (-not $rust.Success) { throw 'Trusted runtime has no pinned Rust channel' }
$hermes = Get-Content -LiteralPath (Join-Path $workspace 'packages\testing-lab\tools\hermes.json') -Raw | ConvertFrom-Json
if ($hermes.repository -ne 'https://github.com/NousResearch/hermes-agent.git' -or $hermes.commit -notmatch '^[a-f0-9]{40}$') { throw 'Invalid Hermes source pin' }
$env:CARGO_HOME = Join-Path $root 'cargo'
$env:RUSTUP_HOME = Join-Path $root 'rustup'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $root 'python'
$prefix = @($tools.PSObject.Properties | Where-Object { $_.Name -ne 'visualStudio' } | ForEach-Object { Split-Path -Parent $_.Value })
$env:PATH = ((@((Join-Path $env:CARGO_HOME 'bin')) + $prefix) -join ';') + ';' + $env:PATH
Invoke-Native $tools.rustup @('-y','--profile','minimal','--default-toolchain',$rust.Groups[1].Value,'--component','clippy,rustfmt','--no-modify-path')
Invoke-Native $tools.uv @('python','install','3.12.13')
Set-Location $workspace
Invoke-Native $tools.bun @('install','--frozen-lockfile','--ignore-scripts')
Invoke-Native $tools.bun @('packages/version/scripts/generate-version.ts')
$hermesRoot = Join-Path $root 'hermes-agent'
Invoke-Native $tools.git @('init','-q',$hermesRoot)
Invoke-Native $tools.git @('-C',$hermesRoot,'remote','add','origin',$hermes.repository)
Invoke-Native $tools.git @('-C',$hermesRoot,'fetch','-q','--depth=1','origin',$hermes.commit)
Invoke-Native $tools.git @('-C',$hermesRoot,'checkout','-q','--detach','FETCH_HEAD')
$commit = & $tools.git -C $hermesRoot rev-parse HEAD
if ($LASTEXITCODE -ne 0 -or $commit.Trim() -ne $hermes.commit) { throw 'Hermes checkout differs from its pin' }
Invoke-Native $tools.uv @('sync','--project',$hermesRoot,'--python','3.12.13','--frozen','--no-dev')
$hermesExecutable = Join-Path $hermesRoot '.venv\Scripts\hermes.exe'
$env:PATH = (Split-Path -Parent $hermesExecutable) + ';' + $env:PATH
Invoke-Native $hermesExecutable @('--version')
Invoke-Native (Join-Path $hermesRoot '.venv\Scripts\python.exe') @('-c',('import importlib.metadata; assert importlib.metadata.version("hermes-agent") == "' + $hermes.version + '"'))
Set-Location (Join-Path $workspace 'packages\testing-lab')
Invoke-Native $tools.node @('-e','const pty=require("node-pty"); const p=pty.spawn(process.env.ComSpec,["/d","/c","exit 0"],{}); p.onExit(e=>process.exit(e.exitCode)); setTimeout(()=>{p.kill();process.exit(1)},10000)')
Set-Location $workspace
Invoke-Native $tools.bun @('-e','await import("./packages/testing-lab/src/outward-worker.ts")')
$launch = [ordered]@{workspace=$workspace;bun=$tools.bun;path=$env:PATH;cargo=$env:CARGO_HOME;rustup=$env:RUSTUP_HOME;node=$tools.node;pi=$tools.pi;opencode=$tools.opencode;hermes=$hermesExecutable;dependencies=$tools.dependencies}
$launch | ConvertTo-Json -Compress | Set-Content -LiteralPath (Join-Path $root 'launch.json') -Encoding UTF8
# Runtime is private to the admitted account and SYSTEM; the separate state receipt stays SYSTEM-owned.
$acl = Get-Acl -LiteralPath $root
$acl.AddAccessRule([Security.AccessControl.FileSystemAccessRule]::new($account.SID,'FullControl','ContainerInherit,ObjectInherit','None','Allow'))
Set-Acl -LiteralPath $root -AclObject $acl
@{schemaVersion=1;runtimeSha256=$runtimeDigest;distribution=$config.distribution;architecture=$config.architecture;userSid=$account.SID.Value;rustVersion=$rust.Groups[1].Value;hermesVersion=$hermes.version} | ConvertTo-Json -Compress | Set-Content -LiteralPath (Join-Path $state 'runtime.json') -Encoding UTF8
Write-Output 'Pinned runtime and native dependencies prepared; desktop readiness is still required'
