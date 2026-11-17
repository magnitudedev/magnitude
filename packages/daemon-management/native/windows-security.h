#ifndef MAGNITUDE_WINDOWS_SECURITY_H
#define MAGNITUDE_WINDOWS_SECURITY_H
#include <windows.h>
/* An absolute path in extended-length form: \\?\C:\... for a drive, \\?\UNC\server\share\... for a share, and a path
 * already in \\?\ or \\.\ form kept as it is. Every function below opens its path in this form, so a profile deeper
 * than MAX_PATH works without LongPathsEnabled. Caller frees `extended`. */
DWORD magnitude_extended_from_full(const WCHAR *full, WCHAR **extended);
/* Caller releases descriptors with LocalFree. */
DWORD magnitude_private_descriptor(BOOL directory, PSECURITY_DESCRIPTOR *descriptor);
/* Returns retained non-inheritable handles; the directory cannot be deleted while owned. */
DWORD magnitude_open_private_lock(const WCHAR *path, HANDLE *file, HANDLE *directory);
/* Validates an already retained directory without creating or repairing permissions. */
DWORD magnitude_validate_private_directory(HANDLE directory);
/* Update material is created with explicit current-user ownership and a protected private ACL. */
DWORD magnitude_prepare_private_directory(const WCHAR *path);
/* Under exclusive application/update admission, preserve and retire only a known inherited cache. */
DWORD magnitude_recover_update_directory(const WCHAR *path, BOOL *retired);
DWORD magnitude_create_private_content(const WCHAR *path);
DWORD magnitude_validate_private_content(const WCHAR *path);
DWORD magnitude_directory_endpoint(HANDLE directory, WCHAR endpoint[128]);
/* Missing is only reported by the initial directory open; unsupported identity is an error. */
DWORD magnitude_inspect_application_endpoint(const WCHAR *path, WCHAR endpoint[128], BOOL *missing);
#endif
