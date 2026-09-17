param([ValidateSet("amd64", "x86")][string]$TargetArchitecture = "amd64")
$ErrorActionPreference = 'Stop'
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
if (-not (Test-Path $vswhere)) { throw 'Visual Studio C++ Build Tools are required.' }
$installation = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $installation) { throw 'No Visual Studio x86/x64 C++ toolchain is installed.' }
$devCommand = Join-Path $installation 'Common7\Tools\VsDevCmd.bat'
$compilerEnvironment = & $env:ComSpec /d /s /c "`"$devCommand`" -arch=$TargetArchitecture -host_arch=amd64 >nul && set"
if ($LASTEXITCODE -ne 0) { throw 'Could not initialize the Visual Studio environment.' }
foreach ($line in $compilerEnvironment) {
  $parts = $line -split '=', 2
  if ($parts.Length -eq 2 -and $parts[0]) { [Environment]::SetEnvironmentVariable($parts[0], $parts[1], 'Process') }
}
