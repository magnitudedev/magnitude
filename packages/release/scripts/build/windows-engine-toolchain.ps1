$ErrorActionPreference = 'Stop'

. (Join-Path $PSScriptRoot '../../../daemon-management/scripts/windows-toolchain.ps1')
$cmakeRoot = Join-Path $env:VSINSTALLDIR 'Common7\IDE\CommonExtensions\Microsoft\CMake'
$env:CMAKE_GENERATOR = 'Ninja'
$env:PATH = "$(Join-Path $cmakeRoot 'CMake\bin');$(Join-Path $cmakeRoot 'Ninja');$env:PATH"
