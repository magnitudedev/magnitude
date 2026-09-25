param([Parameter(Mandatory=$true)][string]$Output)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'windows-toolchain.ps1')
$native = Join-Path (Split-Path -Parent $PSScriptRoot) 'native'
$Output = [IO.Path]::GetFullPath($Output)
$buildRoot = Join-Path ([IO.Path]::GetTempPath()) ('Magnitude CLI launcher build ' + [Guid]::NewGuid())
New-Item -ItemType Directory $buildRoot | Out-Null
New-Item -ItemType Directory -Force (Split-Path -Parent $Output) | Out-Null
Push-Location $buildRoot
try {
  & cl.exe /nologo /W4 /WX /O2 /MT /std:c11 /D_WIN32_WINNT=0x0A00 /D_CRT_SECURE_NO_WARNINGS `
    (Join-Path $native 'windows-job.c') (Join-Path $native 'windows-cli-launcher.c') (Join-Path $native 'windows-cli-main.c') `
    "/Fe:$Output" /link /WX /MANIFEST:EMBED "/MANIFESTUAC:level='asInvoker' uiAccess='false'" shell32.lib ole32.lib uuid.lib
  if ($LASTEXITCODE -ne 0) { throw 'Native Windows CLI launcher compilation failed.' }
} finally {
  Pop-Location
  Remove-Item -Recurse -Force $buildRoot
}
