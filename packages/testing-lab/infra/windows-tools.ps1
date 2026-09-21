param([Parameter(Mandatory=$true)][string]$ConfigurationFile)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
Set-StrictMode -Version Latest
# Administrator-owned preparation only. No candidate source or run authority is accepted here.
if (-not [Security.Principal.WindowsIdentity]::GetCurrent().IsSystem) { throw 'Windows tooling preparation requires SYSTEM' }
if (-not [Environment]::Is64BitOperatingSystem -or $env:PROCESSOR_ARCHITECTURE -ne 'AMD64') { throw 'Windows tooling requires native x64' }
# Azure's small-disk images retain a 30 GiB OS partition after allocating a larger disk.
# Grow only the existing system partition into its supported free extent before installing tools.
$partition = Get-Partition -DriveLetter C
$supported = Get-PartitionSupportedSize -DriveLetter C
if ($supported.SizeMax -gt ($partition.Size + 1MB)) { Resize-Partition -DriveLetter C -Size $supported.SizeMax }
if ((Get-Volume -DriveLetter C).SizeRemaining -lt 32GB) { throw 'Windows build tooling requires at least 32 GiB free on the system volume' }
$config = Get-Content -LiteralPath $ConfigurationFile -Raw | ConvertFrom-Json
$root = 'C:\MagnitudeLab\Tools'
if (Test-Path -LiteralPath $root) { throw 'Windows tooling requires a fresh directory' }
New-Item -ItemType Directory -Path $root | Out-Null
$downloads = Join-Path $root 'downloads'
New-Item -ItemType Directory -Path $downloads | Out-Null
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
Add-Type -AssemblyName System.IO.Compression.FileSystem
function Get-PinnedTool([string]$Name) {
  Write-Output "Preparing pinned $Name" | Out-Host
  $item = $config.$Name
  $destination = Join-Path $downloads $Name
  $url = [Uri]$item.url
  if ($url.Scheme -ne 'https' -or $url.UserInfo -or $url.Fragment -or $item.sha256 -notmatch '^[a-f0-9]{64}$' -or $item.bytes -lt 1 -or $item.bytes -gt 1GB) { throw "Invalid $Name pin" }
  for ($attempt = 1; $attempt -le 3; $attempt++) {
    $url = [Uri]$item.url
    try {
      for ($redirect = 0; $redirect -le 6; $redirect++) {
        $request = [Net.HttpWebRequest]::Create($url)
        $request.AllowAutoRedirect = $false
        $request.Timeout = 120000
        $request.ReadWriteTimeout = 120000
        $response = $request.GetResponse()
        try {
          $status = [int]$response.StatusCode
          if ($status -ge 300 -and $status -lt 400) {
            $url = [Uri]::new($url, $response.Headers['Location'])
            if ($url.Scheme -ne 'https' -or $url.UserInfo -or $url.Fragment) { throw 'Invalid redirect' }
            continue
          }
          if ($status -ne 200) { throw 'Unexpected status' }
          $stream = $response.GetResponseStream()
          $output = [IO.File]::Open($destination, [IO.FileMode]::CreateNew)
          try {
            $buffer = New-Object byte[] 1048576
            [long]$total = 0
            while (($count = $stream.Read($buffer, 0, $buffer.Length)) -gt 0) {
              $total += $count
              if ($total -gt $item.bytes) { throw 'Oversized download' }
              $output.Write($buffer, 0, $count)
            }
          } finally { $output.Dispose(); $stream.Dispose() }
          if ($total -ne $item.bytes -or (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash.ToLowerInvariant() -ne $item.sha256) { throw 'Integrity mismatch' }
          return $destination
        } finally { $response.Dispose() }
      }
      throw 'Redirect limit'
    } catch {
      $reason = $_.Exception.Message
      Remove-Item -LiteralPath $destination -Force -ErrorAction SilentlyContinue
      if ($reason -in @('Integrity mismatch','Oversized download','Invalid redirect','Unexpected status','Redirect limit')) {
        throw "Pinned $Name download failed integrity or redirect validation"
      }
      if ($attempt -eq 3) { throw "Pinned $Name download failed transport validation after three attempts ($($_.Exception.GetType().Name))" }
      Start-Sleep -Seconds (2 * $attempt)
    }
  }
}
function Expand-Tool([string]$Name) {
  $archive = Get-PinnedTool $Name
  $directory = Join-Path $root $Name
  [IO.Compression.ZipFile]::ExtractToDirectory($archive, $directory)
  return $directory
}
function Find-Tool([string]$Directory, [string]$Name) {
  $matches = @(Get-ChildItem -LiteralPath $Directory -Recurse -File -Filter $Name)
  if ($matches.Count -ne 1) { throw "Expected exactly one $Name in its pinned archive" }
  return $matches[0].FullName
}
function Invoke-Tool([string]$Executable, [string[]]$Arguments) {
  & $Executable @Arguments
  if ($LASTEXITCODE -ne 0) { throw "Tool $([IO.Path]::GetFileName($Executable)) exited $LASTEXITCODE" }
}
$paths = [ordered]@{}
foreach ($entry in @(@('bun','bun.exe'),@('node','node.exe'),@('powershell','pwsh.exe'),@('ninja','ninja.exe'),@('uv','uv.exe'),@('pi','pi.exe'),@('opencode','opencode.exe'),@('cmake','cmake.exe'),@('dependencies','Dependencies.exe'))) {
  $paths[$entry[0]] = Find-Tool (Expand-Tool $entry[0]) $entry[1]
}
# NSIS includes its public launcher and a private Bin copy. Use the public launcher.
$nsisRoot = Expand-Tool 'nsis'
$paths.nsis = Join-Path $nsisRoot 'nsis-3.12\makensis.exe'
if (-not (Test-Path -LiteralPath $paths.nsis -PathType Leaf)) { throw 'Pinned NSIS launcher is missing' }
$gitRoot = Expand-Tool 'git'
$paths.git = Join-Path $gitRoot 'cmd\git.exe'
if (-not (Test-Path -LiteralPath $paths.git -PathType Leaf)) { throw 'MinGit command is missing' }
$tirithRoot = Expand-Tool 'tirith'
$paths.tirith = Find-Tool $tirithRoot 'tirith.exe'
# The upstream Windows archive contains tirith.exe only; its Linux authority binary is not a Windows dependency.
$channel = Get-PinnedTool 'vsChannel'
$bootstrap = Get-PinnedTool 'vsBuildTools'
# The installer requires an .exe suffix; the exact downloaded bytes remain pinned.
$installer = $bootstrap + '.exe'
Move-Item -LiteralPath $bootstrap -Destination $installer
$channelFile = $channel + '.chman'
Move-Item -LiteralPath $channel -Destination $channelFile
$vsRoot = 'C:\MagnitudeLab\VisualStudio'
$arguments = @('--quiet','--wait','--norestart','--nocache','--installPath',$vsRoot,
  '--channelUri',$channelFile,'--installChannelUri',$channelFile,
  '--add','Microsoft.VisualStudio.Workload.VCTools',
  '--add','Microsoft.VisualStudio.Component.VC.Tools.x86.x64',
  '--add','Microsoft.VisualStudio.Component.VC.Llvm.Clang',
  '--add','Microsoft.VisualStudio.Component.VC.CMake.Project',
  '--add','Microsoft.VisualStudio.Component.Windows11SDK.22621')
$process = Start-Process -FilePath $installer -ArgumentList $arguments -Wait -PassThru
if ($process.ExitCode -notin @(0,3010)) { throw "Visual Studio setup exited $($process.ExitCode)" }
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$observed = & $vswhere -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 Microsoft.VisualStudio.Component.VC.Llvm.Clang Microsoft.VisualStudio.Component.VC.CMake.Project -property installationPath
if ($LASTEXITCODE -ne 0 -or @($observed).Count -ne 1 -or $observed.TrimEnd('\') -ne $vsRoot) { throw 'Visual Studio component verification failed' }
if (-not (Test-Path -LiteralPath (Join-Path $vsRoot 'VC\Tools\Llvm\x64\bin\libclang.dll'))) { throw 'Native libclang is missing' }
$paths.visualStudio = $vsRoot
$paths.rustup = Get-PinnedTool 'rustup'
$rustInstaller = Join-Path $downloads 'rustup-init.exe'
Move-Item -LiteralPath $paths.rustup -Destination $rustInstaller
$paths.rustup = $rustInstaller
# Runtime preparation will install the Rust channel from its trusted source snapshot.
# Verify native commands now; do not claim an application build or client-OS qualification.
$env:PATH = (@($paths.GetEnumerator() | Where-Object { $_.Key -ne 'visualStudio' } | ForEach-Object { Split-Path -Parent $_.Value }) -join ';') + ';' + $env:PATH
Invoke-Tool $paths.rustup @('--version')
Invoke-Tool $paths.bun @('--version')
Invoke-Tool $paths.node @('--version')
Invoke-Tool $paths.powershell @('-NoProfile','-NonInteractive','-Command','$PSVersionTable.PSVersion.ToString()')
Invoke-Tool $paths.git @('--version')
Invoke-Tool $paths.cmake @('--version')
Invoke-Tool $paths.ninja @('--version')
Invoke-Tool $paths.nsis @('/VERSION')
Invoke-Tool $paths.uv @('--version')
Invoke-Tool $paths.pi @('--version')
Invoke-Tool $paths.opencode @('--version')
Invoke-Tool $paths.tirith @('--version')
Invoke-Tool $paths.dependencies @('-json','-depth','1','-chain',"$env:SystemRoot\System32\cmd.exe")
$paths | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $root 'paths.json') -Encoding UTF8
@{schemaVersion=1; downloads=$config; paths=$paths; restartRequired=($process.ExitCode -eq 3010)} | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $root 'receipt.json') -Encoding UTF8
Write-Output 'Pinned Windows tool installation and native command checks completed'
