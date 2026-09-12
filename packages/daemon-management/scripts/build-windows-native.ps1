param(
  [Parameter(Mandatory=$true)][string]$Headers,
  [Parameter(Mandatory=$true)][string]$NodeLibrary,
  [Parameter(Mandatory=$true)][string]$Output
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'windows-toolchain.ps1')
$packageRoot = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path (Join-Path $Headers 'node_api.h')) -or -not (Test-Path $NodeLibrary)) { throw 'Node-API headers and an x64 Node import library are required.' }
$Headers = (Resolve-Path $Headers).Path
$NodeLibrary = (Resolve-Path $NodeLibrary).Path
$Output = [IO.Path]::GetFullPath($Output)
$buildRoot = Join-Path ([IO.Path]::GetTempPath()) ('Magnitude native build ' + [Guid]::NewGuid())
New-Item -ItemType Directory $buildRoot | Out-Null
New-Item -ItemType Directory -Force (Split-Path -Parent $Output) | Out-Null
Push-Location $buildRoot
try {
  & cl.exe /nologo /c /W4 /WX /O2 /MT /std:c11 /D_WIN32_WINNT=0x0A00 /DNAPI_VERSION=8 /D_CRT_SECURE_NO_WARNINGS "/I$Headers" `
    (Join-Path $packageRoot 'native\desktop-host.c') (Join-Path $packageRoot 'native\windows-job.c') `
    (Join-Path $packageRoot 'native\windows-pipe.c') (Join-Path $packageRoot 'native\windows-pipe-napi.c') `
    (Join-Path $packageRoot 'native\windows-job-napi.c') (Join-Path $packageRoot 'native\windows-security.c') `
    (Join-Path $packageRoot 'native\windows-process-napi.c') (Join-Path $packageRoot 'native\application-memory.c')
  if ($LASTEXITCODE -ne 0) { throw 'Native Windows source compilation failed.' }
  & cl.exe /nologo /c /W4 /WX /O2 /MT /std:c++17 (Join-Path $packageRoot 'native\windows-delay-load.cc')
  if ($LASTEXITCODE -ne 0) { throw 'Native Windows host binding compilation failed.' }
  & link.exe /nologo /DLL /MACHINE:X64 /DELAYLOAD:node.exe "/OUT:$Output" `
    desktop-host.obj application-memory.obj windows-job.obj windows-job-napi.obj windows-security.obj windows-process-napi.obj windows-pipe.obj windows-pipe-napi.obj windows-delay-load.obj $NodeLibrary delayimp.lib advapi32.lib shell32.lib ole32.lib user32.lib
  if ($LASTEXITCODE -ne 0) { throw 'Native Windows addon linking failed.' }
} finally {
  Pop-Location
  Remove-Item -Recurse -Force $buildRoot
}
