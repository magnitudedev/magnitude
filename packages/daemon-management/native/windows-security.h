#ifndef MAGNITUDE_WINDOWS_SECURITY_H
#define MAGNITUDE_WINDOWS_SECURITY_H
#include <windows.h>
/* Caller releases descriptors with LocalFree. */
DWORD magnitude_private_descriptor(BOOL directory, PSECURITY_DESCRIPTOR *descriptor);
/* Returns retained non-inheritable handles; the directory cannot be deleted while owned. */
DWORD magnitude_open_private_lock(const WCHAR *path, HANDLE *file, HANDLE *directory);
/* Validates an already retained directory without creating or repairing permissions. */
DWORD magnitude_validate_private_directory(HANDLE directory);
DWORD magnitude_directory_endpoint(HANDLE directory, WCHAR endpoint[128]);
/* Missing is only reported by the initial directory open; unsupported identity is an error. */
DWORD magnitude_inspect_application_endpoint(const WCHAR *path, WCHAR endpoint[128], BOOL *missing);
#endif
