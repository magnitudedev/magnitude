$ErrorActionPreference = 'Stop'
$packageRoot = Split-Path -Parent $PSScriptRoot
$projectRoot = [IO.Path]::GetFullPath((Join-Path $packageRoot '..\..'))
. (Join-Path $projectRoot 'packages\daemon-management\scripts\windows-toolchain.ps1') -TargetArchitecture x86
$testRoot = Join-Path ([IO.Path]::GetTempPath()) ('Magnitude installer tests ' + [Guid]::NewGuid())
New-Item -ItemType Directory $testRoot | Out-Null
Push-Location $testRoot
try {
  $helper = Join-Path $testRoot 'MagnitudeInstallGuard.dll'
  & (Join-Path $PSScriptRoot 'build\windows-installer.ps1') -Output $helper
  & cl.exe /nologo /W4 /WX /O2 /MT /std:c11 /D_WIN32_WINNT=0x0A00 /D_CRT_SECURE_NO_WARNINGS `
    "/I$(Join-Path $projectRoot 'packages\daemon-management\native')" `
    (Join-Path $projectRoot 'packages\daemon-management\native\windows-security.c') `
    (Join-Path $packageRoot 'native\windows-installer.test.c') /Fe:windows-installer-test.exe /link /WX advapi32.lib
  if ($LASTEXITCODE -ne 0) { throw 'Native installer test compilation failed.' }
  & (Join-Path $testRoot 'windows-installer-test.exe') $helper
  if ($LASTEXITCODE -ne 0) { throw 'Native installer configuration preservation failed.' }
} finally {
  Pop-Location
  Remove-Item -Recurse -Force -LiteralPath $testRoot
}
