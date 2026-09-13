; Per-user fresh installation. Updates require a separate accepted ownership policy.
Unicode true
RequestExecutionLevel user
Name "Magnitude"
OutFile "Magnitude-setup.exe"
Icon "Magnitude.ico"
UninstallIcon "Magnitude.ico"
InstallDir "$LOCALAPPDATA\Programs\Magnitude"
SetCompressor /SOLID lzma
!include "MUI2.nsh"
!include "LogicLib.nsh"
Var Stage
!define MUI_ABORTWARNING
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_LANGUAGE "English"
!define REGKEY "Software\Microsoft\Windows\CurrentVersion\Uninstall\MagnitudeDesktop"
!macro Lease
  InitPluginsDir
  SetOutPath "$PLUGINSDIR"
  File /oname=MagnitudeInstallGuard.dll "MagnitudeInstallGuard.dll"
  System::Call '$PLUGINSDIR\MagnitudeInstallGuard.dll::HoldOwnership() i .r0'
  ${If} $0 != 0
    MessageBox MB_OK|MB_ICONSTOP "Close Magnitude using Quit from its tray menu, then run setup again. Installation access could not be acquired (code $0)." /SD IDOK
    SetErrorLevel 1
    Abort
  ${EndIf}
!macroend
Function .onInit
  SetShellVarContext current
  StrCpy $INSTDIR "$LOCALAPPDATA\Programs\Magnitude"
  !insertmacro Lease
  System::Call '$PLUGINSDIR\MagnitudeInstallGuard.dll::RequireUnusedRegistration(w "$SMPROGRAMS\Magnitude.lnk", w "${REGKEY}") i .r0'
  ${If} $0 != 0
    MessageBox MB_OK|MB_ICONSTOP "An existing shortcut or application registration could not be safely replaced (code $0). It was preserved." /SD IDOK
    SetErrorLevel 1
    Abort
  ${EndIf}
  System::Call '$PLUGINSDIR\MagnitudeInstallGuard.dll::PrepareInstallationDirectory(w "$INSTDIR") i .r0'
  ${If} $0 != 0
    MessageBox MB_OK|MB_ICONSTOP "Setup requires an empty, accessible installation path (code $0). Existing files were preserved. If an earlier removal was interrupted, run $INSTDIR\Uninstall Magnitude.exe to finish it." /SD IDOK
    SetErrorLevel 1
    Abort
  ${EndIf}
FunctionEnd
Function CleanupStage
  StrCmp $Stage "" done
  SetOutPath "$PLUGINSDIR"
  System::Call '$PLUGINSDIR\MagnitudeInstallGuard.dll::CleanupStage() i .r0'
  ${If} $0 == 0
    StrCpy $Stage ""
  ${Else}
    DetailPrint "Temporary extraction could not be removed (code $0). Close files and run setup again."
  ${EndIf}
done:
FunctionEnd
Function .onInstFailed
  Call CleanupStage
FunctionEnd
Section "Magnitude"
  System::Call '$PLUGINSDIR\MagnitudeInstallGuard.dll::CreateStage(w .r9, i 1024) i .r0'
  ${If} $0 != 0
    SetErrorLevel 1
    Abort "Could not create a private installation stage."
  ${EndIf}
  StrCpy $Stage $9
  ClearErrors
  SetOutPath "$Stage"
  IfErrors stageFailed
  SetOverwrite try
@PAYLOAD_FILES@
  IfErrors stageFailed
  SetOutPath "$Stage\resources"
  File /oname=installation-files.txt "installation-files.txt"
  IfErrors stageFailed
  WriteUninstaller "$Stage\Uninstall Magnitude.exe"
  IfErrors stageFailed
  SetOutPath "$PLUGINSDIR"
  Rename "$Stage" "$INSTDIR"
  IfErrors stageFailed
  StrCpy $Stage ""
  ClearErrors
  CreateShortcut "$SMPROGRAMS\Magnitude.lnk" "$INSTDIR\Magnitude.exe"
  IfErrors registrationFailed
  ClearErrors
  WriteRegStr HKCU "${REGKEY}" "DisplayName" "Magnitude"
  IfErrors registrationFailed
  ClearErrors
  WriteRegStr HKCU "${REGKEY}" "Publisher" "Magnitude"
  IfErrors registrationFailed
  ClearErrors
  WriteRegStr HKCU "${REGKEY}" "DisplayVersion" "${MAGNITUDE_VERSION}"
  IfErrors registrationFailed
  ClearErrors
  WriteRegStr HKCU "${REGKEY}" "InstallLocation" "$INSTDIR"
  IfErrors registrationFailed
  ClearErrors
  WriteRegStr HKCU "${REGKEY}" "UninstallString" '$\"$INSTDIR\Uninstall Magnitude.exe$\"'
  IfErrors registrationFailed
  ClearErrors
  WriteRegDWORD HKCU "${REGKEY}" "NoModify" 1
  IfErrors registrationFailed
  ClearErrors
  WriteRegDWORD HKCU "${REGKEY}" "NoRepair" 1
  IfErrors registrationFailed
  Goto installed
registrationFailed:
  SetErrorLevel 1
  Abort "Application registration is incomplete. Run $INSTDIR\Uninstall Magnitude.exe before reinstalling."
stageFailed:
  Call CleanupStage
  SetErrorLevel 1
  Abort "The application could not be staged. The installation path was not replaced."
installed:
SectionEnd
Function un.onInit
  SetShellVarContext current
  StrCpy $INSTDIR "$LOCALAPPDATA\Programs\Magnitude"
  !insertmacro Lease
  System::Call '$PLUGINSDIR\MagnitudeInstallGuard.dll::AcquireRemovalExecutable(w "$INSTDIR\Uninstall Magnitude.exe") i .r0'
  ${If} $0 != 0
    MessageBox MB_OK|MB_ICONSTOP "The installed uninstaller does not match this copy or is in use (code $0). Close applications using these files and retry. Application files were preserved." /SD IDOK
    SetErrorLevel 1
    Abort
  ${EndIf}
FunctionEnd
Section "Uninstall"
  System::Call '$PLUGINSDIR\MagnitudeInstallGuard.dll::RemoveOwnedStartup(w "$INSTDIR\Magnitude.exe") i .r0'
  ${If} $0 != 0
    MessageBox MB_OK|MB_ICONSTOP "Startup registration could not be removed (code $0). Application files were preserved." /SD IDOK
    SetErrorLevel 1
    Abort
  ${EndIf}
@REMOVE_FILES@
@REMOVE_DIRECTORIES@
  ClearErrors
  Delete "$SMPROGRAMS\Magnitude.lnk"
  IfErrors removalFailed
  System::Call '$PLUGINSDIR\MagnitudeInstallGuard.dll::RemoveRegistration(w "${REGKEY}") i .r0'
  ${If} $0 != 0
    Goto removalFailed
  ${EndIf}
  System::Call '$PLUGINSDIR\MagnitudeInstallGuard.dll::RemoveRemovalExecutable() i .r0'
  ${If} $0 != 0
    Goto removalFailed
  ${EndIf}
  Goto removed
removalFailed:
  SetErrorLevel 1
  Abort "Removal is incomplete. Close applications using these files and run this uninstaller again."
removed:
SectionEnd
