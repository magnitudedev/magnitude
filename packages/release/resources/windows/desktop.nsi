; Per-user installation and current-format replacement under the application lease.
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
!include "FileFunc.nsh"
Var Stage
Var PreviousVersion
!define MUI_ABORTWARNING
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_INSTFILES
!define MUI_FINISHPAGE_RUN "$INSTDIR\Magnitude.exe"
!define MUI_FINISHPAGE_RUN_TEXT "Open Magnitude"
!insertmacro MUI_PAGE_FINISH
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_LANGUAGE "English"
!define REGKEY "Software\Microsoft\Windows\CurrentVersion\Uninstall\MagnitudeDesktop"
; Silent runs show no message, so every failure is also appended to a log an administrator can find.
!macro LogFailure Text
  Push $R0
  ClearErrors
  FileOpen $R0 "$TEMP\magnitude-install.log" a
  ${IfNot} ${Errors}
    FileSeek $R0 0 END
    FileWrite $R0 `Magnitude setup failed: ${Text}$\r$\n`
    FileClose $R0
  ${EndIf}
  Pop $R0
!macroend
!macro Fail Text
  !insertmacro LogFailure "${Text}"
  MessageBox MB_OK|MB_ICONSTOP "${Text}" /SD IDOK
  SetErrorLevel 1
  Abort
!macroend
!macro Lease Retry
  InitPluginsDir
  SetOutPath "$PLUGINSDIR"
  File /oname=MagnitudeInstallGuard.dll "MagnitudeInstallGuard.dll"
  System::Call '$PLUGINSDIR\MagnitudeInstallGuard.dll::HoldOwnership() i .r0'
  ${If} $0 != 0
    !insertmacro Fail "Magnitude is still running. Right-click its icon in the notification area, choose Quit Magnitude, then ${Retry} again."
  ${EndIf}
!macroend
Function .onInit
  SetShellVarContext current
  StrCpy $INSTDIR "$LOCALAPPDATA\Programs\Magnitude"
  !insertmacro Lease "run setup"
  StrCpy $PreviousVersion ""
  System::Call '$PLUGINSDIR\MagnitudeInstallGuard.dll::ReadInstallationVersion(w "$INSTDIR", w "${REGKEY}", w .r8, i 1024) i .r0'
  ${If} $0 == 0
    StrCpy $PreviousVersion $8
    System::Call '$PLUGINSDIR\MagnitudeInstallGuard.dll::FlushInstallationRegistration(w "${REGKEY}") i .r0'
    ${If} $0 != 0
      SetErrorLevel 1
      !insertmacro LogFailure "The application registration could not be saved. Existing files were preserved. Run setup again."
      Abort "The application registration could not be saved. Existing files were preserved. Run setup again."
    ${EndIf}
  ${ElseIf} $0 != 2
    !insertmacro Fail "The existing application registration could not be verified (code $0). It was preserved."
  ${EndIf}
  System::Call '$PLUGINSDIR\MagnitudeInstallGuard.dll::RecoverReplacement(w "$INSTDIR", w "$PreviousVersion") i .r0'
  ${If} $0 != 0
    !insertmacro Fail "An interrupted installation could not be recovered (code $0). Its files were preserved. Close applications using those files and run setup again."
  ${EndIf}
  ${If} $PreviousVersion != ""
    System::Call '$PLUGINSDIR\MagnitudeInstallGuard.dll::ValidateOwnedInstallation(w "$INSTDIR", w "$PreviousVersion") i .r0'
    ${If} $0 == 5
      !insertmacro Fail "The installation folder $INSTDIR has permission entries Magnitude didn't set, for example one added for another account or a service. Setup preserved it. Remove the extra entries, or remove the existing installation, then run setup again."
    ${ElseIf} $0 != 0
      !insertmacro Fail "The installed files could not be verified (code $0). Setup preserved them. Remove the existing installation before installing this version."
    ${EndIf}
    Return
  ${EndIf}
  System::Call '$PLUGINSDIR\MagnitudeInstallGuard.dll::RequireUnusedRegistration(w "$SMPROGRAMS\Magnitude.lnk", w "${REGKEY}") i .r0'
  ${If} $0 != 0
    !insertmacro Fail "An existing shortcut or application registration could not be safely replaced (code $0). It was preserved."
  ${EndIf}
  System::Call '$PLUGINSDIR\MagnitudeInstallGuard.dll::PrepareInstallationDirectory(w "$INSTDIR") i .r0'
  ${If} $0 != 0
    !insertmacro Fail "Setup requires an empty, accessible installation path (code $0). Existing files were preserved. If an earlier removal was interrupted, run $INSTDIR\Uninstall Magnitude.exe to finish it."
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
    !insertmacro LogFailure "Could not create a private installation stage."
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
  ${If} $PreviousVersion != ""
    System::Call '$PLUGINSDIR\MagnitudeInstallGuard.dll::BeginReplacement(w "$INSTDIR", w "$PreviousVersion", w "${MAGNITUDE_VERSION}") i .r0'
    ${If} $0 != 0
      Goto stageFailed
    ${EndIf}
    ClearErrors
    WriteRegStr HKCU "${REGKEY}" "DisplayVersion" "${MAGNITUDE_VERSION}"
    IfErrors replacementRegistrationFailed
    System::Call '$PLUGINSDIR\MagnitudeInstallGuard.dll::FlushInstallationRegistration(w "${REGKEY}") i .r0'
    ${If} $0 != 0
      SetErrorLevel 1
      !insertmacro LogFailure "The update registration could not be saved. Installation files were retained for recovery. Run setup again."
      Abort "The update registration could not be saved. Installation files were retained for recovery. Run setup again."
    ${EndIf}
    System::Call '$PLUGINSDIR\MagnitudeInstallGuard.dll::FinishReplacement(w "$PreviousVersion") i .r0'
    ${If} $0 != 0
      DetailPrint "The update is installed. Previous files were retained because cleanup could not complete (code $0). Setup will retry cleanup next time."
    ${EndIf}
    StrCpy $Stage ""
    Goto installed
  ${EndIf}
  Rename "$Stage" "$INSTDIR"
  IfErrors stageFailed
  StrCpy $Stage ""
  ClearErrors
  SetOutPath "$INSTDIR"
  IfErrors registrationFailed
  CreateShortcut "$SMPROGRAMS\Magnitude.lnk" "$INSTDIR\Magnitude.exe"
  IfErrors registrationFailed
  SetOutPath "$PLUGINSDIR"
  ClearErrors
  WriteRegStr HKCU "${REGKEY}" "DisplayName" "Magnitude"
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
replacementRegistrationFailed:
  System::Call '$PLUGINSDIR\MagnitudeInstallGuard.dll::RollbackReplacement() i .r0'
  ${If} $0 != 0
    SetErrorLevel 1
    !insertmacro LogFailure "The update could not finish or restore the old installation. Files were retained for recovery. Run setup again."
    Abort "The update could not finish or restore the old installation. Files were retained for recovery. Run setup again."
  ${EndIf}
  Call CleanupStage
  SetErrorLevel 1
  !insertmacro LogFailure "The update could not be registered. The previous installation was restored."
  Abort "The update could not be registered. The previous installation was restored."
registrationFailed:
  SetOutPath "$PLUGINSDIR"
  SetErrorLevel 1
  !insertmacro LogFailure "Application registration is incomplete. Run $INSTDIR\Uninstall Magnitude.exe before reinstalling."
  Abort "Application registration is incomplete. Run $INSTDIR\Uninstall Magnitude.exe before reinstalling."
stageFailed:
  Call CleanupStage
  SetErrorLevel 1
  !insertmacro LogFailure "Setup could not complete. Installation files were retained for recovery. Close applications using them and run setup again."
  Abort "Setup could not complete. Installation files were retained for recovery. Close applications using them and run setup again."
installed:
  WriteRegStr HKCU "${REGKEY}" "Publisher" "Magnitude AI Inc."
  WriteRegStr HKCU "${REGKEY}" "DisplayIcon" "$INSTDIR\Magnitude.exe"
  ${GetSize} "$INSTDIR" "/S=0K" $0 $1 $2
  WriteRegDWORD HKCU "${REGKEY}" "EstimatedSize" "$0"
  System::Call '$PLUGINSDIR\MagnitudeInstallGuard.dll::InstallCliLauncher(w "$INSTDIR\resources\magnitude-launcher.exe", w "$LOCALAPPDATA\Programs\Magnitude CLI") i .r0'
  ${If} $0 != 0
    SetErrorLevel 1
    !insertmacro LogFailure "Magnitude is installed, but its command could not be published (code $0). Run setup again to repair it."
    Abort "Magnitude is installed, but its command could not be published (code $0). Run setup again to repair it."
  ${EndIf}
  ReadRegStr $9 HKCU "${REGKEY}" "OwnedCliPath"
  ${If} $9 == "$INSTDIR\resources"
    System::Call '$PLUGINSDIR\MagnitudeInstallGuard.dll::ConfigureCliPath(w "$INSTDIR\resources", w "${REGKEY}", i 1) i .r0'
    ${If} $0 != 0
      SetErrorLevel 1
      !insertmacro LogFailure "The previous command registration could not be retired (code $0). Run setup again."
      Abort "The previous command registration could not be retired (code $0). Run setup again."
    ${EndIf}
  ${EndIf}
  System::Call '$PLUGINSDIR\MagnitudeInstallGuard.dll::ConfigureCliPath(w "$LOCALAPPDATA\Programs\Magnitude CLI", w "${REGKEY}", i 0) i .r0'
  ${If} $0 != 0
    SetErrorLevel 1
    !insertmacro LogFailure "Magnitude is installed, but its command could not be added to PATH (code $0). Run setup again to repair command registration."
    Abort "Magnitude is installed, but its command could not be added to PATH (code $0). Run setup again to repair command registration."
  ${EndIf}
  System::Call '$PLUGINSDIR\MagnitudeInstallGuard.dll::RemovePreviousCliCommands(w "$LOCALAPPDATA\Programs\Magnitude CLI", w "$INSTDIR\resources", w .r8, i 1024) i .r0'
  ${If} $0 != 0
    SetErrorLevel 1
    !insertmacro LogFailure "Magnitude is installed, but setup could not replace a previous command: $8 (code $0). Close programs using it and run setup again."
    Abort "Magnitude is installed, but setup could not replace a previous command: $8 (code $0). Close programs using it and run setup again."
  ${EndIf}
  DetailPrint "The magnitude command is available in new terminals."
SectionEnd
Function un.onInit
  SetShellVarContext current
  StrCpy $INSTDIR "$LOCALAPPDATA\Programs\Magnitude"
  !insertmacro Lease "uninstall"
  System::Call '$PLUGINSDIR\MagnitudeInstallGuard.dll::AcquireRemovalExecutable(w "$INSTDIR\Uninstall Magnitude.exe") i .r0'
  ${If} $0 != 0
    !insertmacro Fail "The installed uninstaller does not match this copy or is in use (code $0). Close applications using these files and retry. Application files were preserved."
  ${EndIf}
FunctionEnd
Section "Uninstall"
  System::Call '$PLUGINSDIR\MagnitudeInstallGuard.dll::RetirePreviousForRemoval() i .r0'
  ${If} $0 != 0
    !insertmacro Fail "Previous application files could not be retired (code $0). Close applications using them and run this uninstaller again. The current installation was preserved."
  ${EndIf}
  System::Call '$PLUGINSDIR\MagnitudeInstallGuard.dll::RemoveCliLauncher(w "$LOCALAPPDATA\Programs\Magnitude CLI") i .r0'
  ${If} $0 != 0
    SetErrorLevel 1
    !insertmacro LogFailure "Close Magnitude commands and retry uninstall (code $0). Application files were preserved."
    Abort "Close Magnitude commands and retry uninstall (code $0). Application files were preserved."
  ${EndIf}
  System::Call '$PLUGINSDIR\MagnitudeInstallGuard.dll::ConfigureCliPath(w "$LOCALAPPDATA\Programs\Magnitude CLI", w "${REGKEY}", i 1) i .r0'
  ${If} $0 != 0
    SetErrorLevel 1
    !insertmacro LogFailure "Command registration could not be removed (code $0). Application files were preserved."
    Abort "Command registration could not be removed (code $0). Application files were preserved."
  ${EndIf}
  System::Call '$PLUGINSDIR\MagnitudeInstallGuard.dll::RemoveOwnedStartup(w "$INSTDIR\Magnitude.exe") i .r0'
  ${If} $0 != 0
    !insertmacro Fail "Startup registration could not be removed (code $0). Application files were preserved."
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
  !insertmacro LogFailure "Removal is incomplete. Close applications using these files and run this uninstaller again."
  Abort "Removal is incomplete. Close applications using these files and run this uninstaller again."
removed:
SectionEnd
