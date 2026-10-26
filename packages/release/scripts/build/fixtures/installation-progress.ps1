param([string]$Installer)
$ErrorActionPreference = 'Stop'
$parseErrors = $null
$tokens = $null
$tree = [Management.Automation.Language.Parser]::ParseFile($Installer, [ref]$tokens, [ref]$parseErrors)
if ($parseErrors.Count -gt 0) { throw ($parseErrors | Out-String) }
$source = [IO.File]::ReadAllText($Installer)
$start = $source.IndexOf('# Keep status off the success stream.')
$end = $source.IndexOf('function Assert-Publisher', $start)
Invoke-Expression $source.Substring($start, $end - $start)
Start-InstallerProgress 'Downloading app'
$script:InstallerProgress.Interactive = $false
$script:InstallerProgress.Clock = [pscustomobject]@{ Elapsed = [TimeSpan]::FromSeconds(12) }
Update-InstallerProgress 120000000 200000000
Complete-InstallerProgress 'Downloaded app'
Start-InstallerProgress 'Downloading app'
$script:InstallerProgress.Clock = [pscustomobject]@{ Elapsed = [TimeSpan]::FromSeconds(12) }
Update-InstallerProgress 0 200000000
Complete-InstallerProgress 'Finished zero-byte progress check'
Start-InstallerProgress 'Verifying app signature'
$script:InstallerProgress.Clock = [pscustomobject]@{ Elapsed = [TimeSpan]::FromSeconds(12) }
Update-InstallerProgress
Complete-InstallerProgress 'Verified app signature'
$root = Join-Path ([IO.Path]::GetTempPath()) ('magnitude-progress-' + [Guid]::NewGuid().ToString('N'))
$null = New-Item -ItemType Directory $root
try {
  $child = Join-Path $root 'child with spaces.ps1'
  [IO.File]::WriteAllText($child, 'param([string]$Value) if ($Value -cne ''argument with spaces'') { exit 9 }; Start-Sleep -Milliseconds 1200; [Console]::Error.WriteLine(''child output'')')
  $binary = [Diagnostics.Process]::GetCurrentProcess().MainModule.FileName
  Invoke-InstallerProcess $binary @('-NoLogo', '-NoProfile', '-NonInteractive', '-File', $child, 'argument with spaces') 'Installing app' 'Installed app'
  $failed = $false
  try { Invoke-InstallerProcess $binary @('-NoLogo', '-NoProfile', '-NonInteractive', '-Command', 'exit 37') 'Failing install' 'Unexpected success' }
  catch { if ($_.Exception.Message -notlike '*exit code 37*') { throw }; $failed = $true }
  if (-not $failed) { throw 'Native process failure was swallowed.' }

  Add-Type -AssemblyName System.Net.Http
  $compiler = @{}
  if ($PSVersionTable.PSVersion.Major -le 5) { $compiler.ReferencedAssemblies = @("System.dll", "System.Core.dll", [Net.Http.HttpClient].Assembly.Location) }
  Add-Type @compiler -TypeDefinition @'
using System;
using System.IO;
using System.Net;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;
public class InstallerProgressStream : MemoryStream {
  public InstallerProgressStream(byte[] bytes) : base(bytes) {}
  public override async Task<int> ReadAsync(byte[] buffer, int offset, int count, CancellationToken cancellation) {
    await Task.Delay(80, cancellation);
    return Read(buffer, offset, Math.Min(count, 65536));
  }
}
public class InstallerProgressClient {
  private byte[] bytes;
  public InstallerProgressClient(byte[] data) { bytes = data; }
  public Task<HttpResponseMessage> GetAsync(string url, HttpCompletionOption option, CancellationToken cancellation) {
    var response = new HttpResponseMessage(HttpStatusCode.OK);
    response.RequestMessage = new HttpRequestMessage(HttpMethod.Get, url);
    response.Content = new StreamContent(new InstallerProgressStream(bytes));
    response.Content.Headers.ContentLength = bytes.Length;
    return Task.FromResult(response);
  }
}
'@
  $bytes = New-Object byte[] 1048576
  for ($i = 0; $i -lt $bytes.Length; $i++) { $bytes[$i] = $i % 251 }
  $client = [InstallerProgressClient]::new($bytes)
  $download = Join-Path $root 'download with spaces.bin'
  Get-InstallerFile 'https://fixture.invalid/app' $download 2097152 'Downloading app' 'Downloaded app' $bytes.Length
  if ([Convert]::ToBase64String([IO.File]::ReadAllBytes($download)) -cne [Convert]::ToBase64String($bytes)) { throw 'The progress reader changed the download.' }
  $failed = $false
  try { Get-InstallerFile 'https://fixture.invalid/large' (Join-Path $root 'too-large.bin') 1000 'Downloading app' 'Unexpected success' }
  catch { if ($_.Exception.Message -notlike '*exceeds*') { throw }; $failed = $true }
  if (-not $failed) { throw 'Download size protection was bypassed.' }
  $failed = $false
  try { Get-InstallerFile 'http://fixture.invalid/app' (Join-Path $root 'http.bin') 2097152 'Downloading app' 'Unexpected success' }
  catch { if ($_.Exception.Message -notlike '*HTTPS*') { throw }; $failed = $true }
  if (-not $failed) { throw 'HTTPS protection was bypassed.' }
  if ($null -ne $script:InstallerProgress) { throw 'Progress was left active.' }
} finally { Remove-Item $root -Recurse -Force }
Write-Output 'Windows progress checks passed'
