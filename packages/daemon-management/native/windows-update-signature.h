#ifndef MAGNITUDE_WINDOWS_UPDATE_SIGNATURE_H
#define MAGNITUDE_WINDOWS_UPDATE_SIGNATURE_H
#include <windows.h>
DWORD magnitude_verify_installer_signature(const WCHAR *path, const WCHAR *organization);
#endif
