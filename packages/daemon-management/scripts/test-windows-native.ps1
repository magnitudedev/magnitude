param(
  [Parameter(Mandatory=$true)][string]$Headers,
  [Parameter(Mandatory=$true)][string]$NodeLibrary
)
$ErrorActionPreference = 'Stop'
$packageRoot = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'windows-toolchain.ps1')
$testRoot = Join-Path ([IO.Path]::GetTempPath()) ('Magnitude native tests ' + [Guid]::NewGuid())
New-Item -ItemType Directory $testRoot | Out-Null
Push-Location $testRoot
try {
  & cl.exe /nologo /W4 /WX /O2 /std:c11 /D_WIN32_WINNT=0x0A00 /D_CRT_SECURE_NO_WARNINGS `
    (Join-Path $packageRoot 'native\windows-security.c') (Join-Path $packageRoot 'native\windows-security-test.c') /Fe:windows-security-test.exe /link advapi32.lib
  if ($LASTEXITCODE -ne 0) { throw 'Native Windows security compilation failed.' }
  & (Join-Path $testRoot 'windows-security-test.exe')
  if ($LASTEXITCODE -ne 0) { throw 'Native Windows security acceptance failed.' }
  & cl.exe /nologo /W4 /WX /O2 /std:c11 /D_WIN32_WINNT=0x0A00 /D_CRT_SECURE_NO_WARNINGS `
    (Join-Path $packageRoot 'native\windows-job.c') (Join-Path $packageRoot 'native\windows-job-test.c') /Fe:windows-job-test.exe
  if ($LASTEXITCODE -ne 0) { throw 'Native Windows containment compilation failed.' }
  & (Join-Path $testRoot 'windows-job-test.exe')
  if ($LASTEXITCODE -ne 0) { throw 'Native Windows containment acceptance failed.' }
  & cl.exe /nologo /W4 /WX /O2 /std:c11 /D_WIN32_WINNT=0x0A00 /D_CRT_SECURE_NO_WARNINGS `
    (Join-Path $packageRoot 'native\windows-security.c') (Join-Path $packageRoot 'native\windows-pipe.c') (Join-Path $packageRoot 'native\windows-pipe-test.c') /Fe:windows-pipe-test.exe /link advapi32.lib
  if ($LASTEXITCODE -ne 0) { throw 'Native Windows pipe compilation failed.' }
  & (Join-Path $testRoot 'windows-pipe-test.exe')
  if ($LASTEXITCODE -ne 0) { throw 'Native Windows pipe acceptance failed.' }
  $addon = Join-Path $testRoot 'desktop-host.node'
  & (Join-Path $PSScriptRoot 'build-windows-native.ps1') -Headers $Headers -NodeLibrary $NodeLibrary -Output $addon
  $embeddedFixture = Join-Path $testRoot 'windows-embedded.cjs'
  Copy-Item (Join-Path $packageRoot 'src\desktop-native\fixtures\windows-embedded.cjs') $embeddedFixture
  $embeddedExecutable = Join-Path $testRoot 'embedded-native.exe'
  & bun build --compile $embeddedFixture --outfile $embeddedExecutable
  if ($LASTEXITCODE -ne 0) { throw 'Embedded Windows addon fixture compilation failed.' }
  & $embeddedExecutable
  if ($LASTEXITCODE -ne 0) { throw 'Embedded Windows native bootstrap acceptance failed.' }
  $fixture = Join-Path $packageRoot 'src\desktop-native\fixtures\windows-pipe.cjs'
  & node $fixture $addon
  if ($LASTEXITCODE -ne 0) { throw 'Node Windows pipe acceptance failed.' }
  & bun $fixture $addon
  if ($LASTEXITCODE -ne 0) { throw 'Bun Windows pipe acceptance failed.' }
  & bun build --compile $fixture --outfile (Join-Path $testRoot 'compiled-pipe.exe')
  if ($LASTEXITCODE -ne 0) { throw 'Compiled Bun pipe fixture build failed.' }
  & (Join-Path $testRoot 'compiled-pipe.exe') $addon
  if ($LASTEXITCODE -ne 0) { throw 'Compiled Bun Windows pipe acceptance failed.' }

  $projectRoot = Split-Path -Parent (Split-Path -Parent $packageRoot)
  Push-Location $projectRoot
  try {
    & bun -e 'import { buildAcnBinary } from "./packages/release/scripts/build/acn"; await buildAcnBinary("bun-windows-x64")'
    if ($LASTEXITCODE -ne 0) { throw 'Windows service compilation failed.' }
    $service = Join-Path $projectRoot 'bin/magnitude-service.exe'
    $serviceFixture = Join-Path $testRoot 'windows-service.mjs'
    & bun build (Join-Path $packageRoot 'src/desktop-native/fixtures/windows-service.ts') --target=node --format=esm --outfile $serviceFixture
    if ($LASTEXITCODE -ne 0) { throw 'Windows service acceptance fixture compilation failed.' }
    & node $serviceFixture $service $addon
    if ($LASTEXITCODE -ne 0) { throw 'Node owner did not observe terminal service health.' }
    & bun $serviceFixture $service $addon
    if ($LASTEXITCODE -ne 0) { throw 'Bun owner did not observe terminal service health.' }
    & node $serviceFixture $service $addon --withhold-ack
    if ($LASTEXITCODE -ne 0) { throw 'Missing acknowledgement prevented bounded service exit.' }
  } finally { Pop-Location }


} finally {
  Pop-Location
  Remove-Item -Recurse -Force $testRoot
}
