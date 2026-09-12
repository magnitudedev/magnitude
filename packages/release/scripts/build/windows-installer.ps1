param([Parameter(Mandatory=$true)][string]$Output)
$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..\..\..'))
. (Join-Path $projectRoot 'packages\daemon-management\scripts\windows-toolchain.ps1') -TargetArchitecture x86
$Output = [IO.Path]::GetFullPath($Output)
$buildRoot = Join-Path ([IO.Path]::GetTempPath()) ('Magnitude installer build ' + [Guid]::NewGuid())
New-Item -ItemType Directory $buildRoot | Out-Null
New-Item -ItemType Directory -Force (Split-Path -Parent $Output) | Out-Null
Push-Location $buildRoot
try {
  & cl.exe /nologo /c /W4 /WX /O2 /MT /std:c11 /D_WIN32_WINNT=0x0A00 /D_CRT_SECURE_NO_WARNINGS `
    "/I$(Join-Path $projectRoot 'packages\daemon-management\native')" `
    (Join-Path $projectRoot 'packages\release\native\windows-installer.c') `
    (Join-Path $projectRoot 'packages\daemon-management\native\windows-security.c')
  if ($LASTEXITCODE -ne 0) { throw 'Windows installer helper compilation failed.' }
  & link.exe /nologo /WX /DLL /MACHINE:X86 /DYNAMICBASE /NXCOMPAT "/OUT:$Output" `
    "/DEF:$(Join-Path $projectRoot 'packages\release\native\windows-installer.def')" `
    windows-installer.obj windows-security.obj advapi32.lib shell32.lib ole32.lib uuid.lib ntdll.lib
  if ($LASTEXITCODE -ne 0) { throw 'Windows installer helper linking failed.' }
} finally {
  Pop-Location
  Remove-Item -Recurse -Force -LiteralPath $buildRoot
}
