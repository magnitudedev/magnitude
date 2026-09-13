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
  foreach ($scenario in @('old-moved', 'uncommitted', 'committed')) {
    foreach ($phase in @('setup', 'recover')) {
      & (Join-Path $testRoot 'windows-installer-test.exe') $helper "$phase-$scenario"
      if ($LASTEXITCODE -ne 0) { throw "Native interrupted installation failed: $phase-$scenario" }
    }
  }
  $nsis = Join-Path ${env:ProgramFiles(x86)} 'NSIS\makensis.exe'
  if (!(Test-Path -LiteralPath $nsis)) {
    & choco install nsis --yes --no-progress
    if ($LASTEXITCODE -ne 0) { throw 'NSIS acquisition failed' }
  }
  $env:MAGNITUDE_INSTALLER_TEST_ROOT = $testRoot
  $env:MAGNITUDE_INSTALLER_TEST_NSIS = $nsis
  & bun (Join-Path $PSScriptRoot 'acceptance\build-windows-installer-fixture.ts')
  if ($LASTEXITCODE -ne 0) { throw 'Production NSIS fixture compilation failed' }
  & (Join-Path $PSScriptRoot 'acceptance\test-windows-installer-fixture.ps1') -Root $testRoot
} finally {
  Remove-Item Env:MAGNITUDE_INSTALLER_TEST_ROOT -ErrorAction SilentlyContinue
  Remove-Item Env:MAGNITUDE_INSTALLER_TEST_NSIS -ErrorAction SilentlyContinue
  Pop-Location
  Remove-Item -Recurse -Force -LiteralPath $testRoot
}
