param(
  [Parameter(Mandatory = $true)][string]$Directory,
  [Parameter(Mandatory = $true)][string]$Version,
  [switch]$RequireSignature
)
$ErrorActionPreference = 'Stop'
$Directory = [IO.Path]::GetFullPath($Directory)
$installation = Join-Path $env:LOCALAPPDATA 'Programs\Magnitude'
if (Test-Path -LiteralPath $installation) { throw 'Installed acceptance requires a disposable Windows consumer without Magnitude installed.' }
if ($RequireSignature -and [string]::IsNullOrWhiteSpace($env:MAGNITUDE_WINDOWS_PUBLISHER)) { throw 'Missing expected Windows publisher.' }

function AssertSignature([string]$Path, [switch]$AllowMicrosoftRuntime) {
  if (!$RequireSignature) { return }
  $publisher = $env:MAGNITUDE_WINDOWS_PUBLISHER
  if ($AllowMicrosoftRuntime -and [IO.Path]::GetFileName($Path) -match '^(msvcp140|vcruntime140(_1)?)\.dll$') {
    $publisher = 'Microsoft Windows Software Compatibility Publisher'
  }
  $signature = Get-AuthenticodeSignature -LiteralPath $Path
  if ($signature.Status -ne 'Valid' -or !$signature.TimeStamperCertificate -or
      $signature.SignerCertificate.GetNameInfo([Security.Cryptography.X509Certificates.X509NameType]::SimpleName, $false) -cne $publisher) {
    throw "Invalid publisher signature or timestamp: $Path"
  }
}

$record = Get-Content -Raw -LiteralPath (Join-Path $Directory 'desktop-windows-x64-msvc.artifact.json') | ConvertFrom-Json
if ($record.id -ne 'desktop-windows-x64-msvc' -or $record.filename -ne [IO.Path]::GetFileName($record.filename)) {
  throw 'Unexpected installer artifact identity.'
}
$installer = Join-Path $Directory $record.filename
if ((Get-Item -LiteralPath $installer).Length -ne $record.bytes -or
    (Get-FileHash -LiteralPath $installer -Algorithm SHA256).Hash.ToLowerInvariant() -cne $record.sha256) {
  throw 'Installer artifact bytes changed.'
}
AssertSignature $installer
$process = Start-Process -FilePath $installer -ArgumentList '/S' -Wait -PassThru
if ($process.ExitCode -ne 0) { throw "Installer exited $($process.ExitCode)." }
foreach ($relative in @('Magnitude.exe', 'resources\magnitude.exe', 'resources\magnitude-service.exe', 'resources\desktop-host.node', 'Uninstall Magnitude.exe')) {
  $file = Join-Path $installation $relative
  if (!(Test-Path -LiteralPath $file -PathType Leaf)) { throw "Missing installed file: $relative" }
  AssertSignature $file
}
$registered = Get-ItemProperty 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\MagnitudeDesktop'
if ($registered.DisplayVersion -cne $Version) { throw 'Installed registration version mismatch.' }
$cliVersion = & (Join-Path $installation 'resources\magnitude.exe') --version
if ($LASTEXITCODE -ne 0 -or $cliVersion.Trim() -cne $Version) { throw 'Installed CLI version mismatch.' }

$scratch = Join-Path ([IO.Path]::GetTempPath()) ('magnitude-installed-consumer-' + [guid]::NewGuid())
New-Item -ItemType Directory $scratch | Out-Null
try {
  foreach ($kind in @('cli', 'acn', 'icn-base')) {
    & tar.exe -xzf (Join-Path $Directory "magnitude-$kind-windows-x64-msvc.tar.gz") -C $scratch
    if ($LASTEXITCODE -ne 0) { throw "Could not extract accepted $kind archive." }
  }
  $engine = Join-Path $scratch 'bin\magnitude-inference.exe'
  if (!(Test-Path -LiteralPath $engine -PathType Leaf)) { throw 'Missing accepted inference executable.' }
  AssertSignature $engine
  foreach ($directory in @('runtime', 'backends')) {
    foreach ($library in Get-ChildItem -LiteralPath (Join-Path $scratch $directory) -Filter '*.dll' -File) {
      AssertSignature $library.FullName -AllowMicrosoftRuntime
    }
  }
  foreach ($pair in @(@('magnitude-cli.exe', 'magnitude.exe'), @('magnitude-service.exe', 'magnitude-service.exe'))) {
    $accepted = (Get-FileHash -LiteralPath (Join-Path $scratch "bin\$($pair[0])") -Algorithm SHA256).Hash
    $installed = (Get-FileHash -LiteralPath (Join-Path $installation "resources\$($pair[1])") -Algorithm SHA256).Hash
    if ($accepted -cne $installed) { throw 'Installed executable differs from its accepted archive.' }
  }
} finally {
  Remove-Item -LiteralPath $scratch -Recurse -Force
}
$uninstall = Start-Process -FilePath (Join-Path $installation 'Uninstall Magnitude.exe') -ArgumentList '/S' -Wait -PassThru
if ($uninstall.ExitCode -ne 0) { throw "Uninstaller exited $($uninstall.ExitCode)." }
# NSIS's self-copy completes removal in a separate process.
$deadline = [DateTime]::UtcNow.AddSeconds(30)
while ((Test-Path -LiteralPath $installation) -and [DateTime]::UtcNow -lt $deadline) { Start-Sleep -Milliseconds 200 }
if (Test-Path -LiteralPath $installation) { throw 'Installed payload remained after uninstall.' }
if (Test-Path 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\MagnitudeDesktop') { throw 'Uninstall registration remained.' }
Write-Output 'PASS accepted installer integrity, installed versions and bytes, and native uninstall'
if ($RequireSignature) { Write-Output 'PASS expected publisher and timestamp on installer, installed code, and uninstaller' }
