param(
  [Parameter(Mandatory=$true)][string]$Output,
  [Parameter(Mandatory=$true)][ValidatePattern('^\d+\.\d+\.\d+$')][string]$Version
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'windows-toolchain.ps1')
$native = Join-Path (Split-Path -Parent $PSScriptRoot) 'native'
$Output = [IO.Path]::GetFullPath($Output)
$buildRoot = Join-Path ([IO.Path]::GetTempPath()) ('Magnitude CLI launcher build ' + [Guid]::NewGuid())
New-Item -ItemType Directory $buildRoot | Out-Null
New-Item -ItemType Directory -Force (Split-Path -Parent $Output) | Out-Null
Push-Location $buildRoot
try {
  # Same version resource fields the release build gives the Bun-compiled Magnitude command.
  $numeric = ($Version -split '\.') -join ','
  @"
#include <winver.h>
VS_VERSION_INFO VERSIONINFO
FILEVERSION $numeric,0
PRODUCTVERSION $numeric,0
FILEFLAGSMASK VS_FFI_FILEFLAGSMASK
FILEFLAGS 0
FILEOS VOS_NT_WINDOWS32
FILETYPE VFT_APP
FILESUBTYPE VFT2_UNKNOWN
BEGIN
  BLOCK "StringFileInfo"
  BEGIN
    BLOCK "040904B0"
    BEGIN
      VALUE "CompanyName", "Magnitude AI Inc."
      VALUE "FileDescription", "Magnitude command"
      VALUE "FileVersion", "$Version.0"
      VALUE "InternalName", "magnitude"
      VALUE "LegalCopyright", "Copyright Magnitude AI Inc."
      VALUE "OriginalFilename", "magnitude.exe"
      VALUE "ProductName", "Magnitude"
      VALUE "ProductVersion", "$Version.0"
    END
  END
  BLOCK "VarFileInfo"
  BEGIN
    VALUE "Translation", 0x409, 1200
  END
END
"@ | Set-Content -Encoding ascii 'launcher.rc'
  & rc.exe /nologo /fo launcher.res launcher.rc
  if ($LASTEXITCODE -ne 0) { throw 'Native Windows CLI launcher version resource compilation failed.' }
  & cl.exe /nologo /W4 /WX /O2 /MT /std:c11 /D_WIN32_WINNT=0x0A00 /D_CRT_SECURE_NO_WARNINGS `
    (Join-Path $native 'windows-job.c') (Join-Path $native 'windows-cli-launcher.c') (Join-Path $native 'windows-cli-main.c') launcher.res `
    "/Fe:$Output" /link /WX /MANIFEST:EMBED "/MANIFESTUAC:level='asInvoker' uiAccess='false'" shell32.lib ole32.lib uuid.lib
  if ($LASTEXITCODE -ne 0) { throw 'Native Windows CLI launcher compilation failed.' }
} finally {
  Pop-Location
  Remove-Item -Recurse -Force $buildRoot
}
