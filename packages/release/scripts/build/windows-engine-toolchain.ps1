$ErrorActionPreference = 'Stop'

. (Join-Path $PSScriptRoot '../../../daemon-management/scripts/windows-toolchain.ps1')
$cmakeRoot = Join-Path $env:VSINSTALLDIR 'Common7\IDE\CommonExtensions\Microsoft\CMake'
$clang = Join-Path $env:VSINSTALLDIR 'VC\Tools\Llvm\x64\bin\libclang.dll'
if (!(Test-Path -LiteralPath $clang)) { throw 'Visual Studio x64 Clang tools are required for Rust bindings.' }
$env:LIBCLANG_PATH = Split-Path $clang
$env:CMAKE_GENERATOR = 'Ninja'
$env:PATH = "$(Join-Path $cmakeRoot 'CMake\bin');$(Join-Path $cmakeRoot 'Ninja');$env:PATH"
