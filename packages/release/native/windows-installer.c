#define _WIN32_WINNT 0x0A00
#include <windows.h>
#include <winternl.h>
#include <shlobj.h>
#include <stdio.h>
#include <string.h>
#include <wchar.h>
#include "windows-security.h"

static BOOL leaseHeld = FALSE;
static HANDLE removalExecutable = INVALID_HANDLE_VALUE;
static HANDLE removalDirectory = INVALID_HANDLE_VALUE;
static HANDLE stageDirectory = INVALID_HANDLE_VALUE;

/* Name resolution must reject every reparse point, not only the final component.
   Deletion then acts on the opened object rather than resolving its path again. */
static DWORD open_file_object(HANDLE root, LPCWSTR path, ACCESS_MASK access,
    ULONG share, ULONG options, ULONG objectFlags, HANDLE *output) {
  size_t length = wcslen(path);
  if (!length || length > 32766) return ERROR_BAD_PATHNAME;
  UNICODE_STRING name = {(USHORT)(length * sizeof(WCHAR)), (USHORT)(length * sizeof(WCHAR)), (PWSTR)path};
  OBJECT_ATTRIBUTES attributes;
  InitializeObjectAttributes(&attributes, &name, OBJ_CASE_INSENSITIVE | objectFlags, root, NULL);
  IO_STATUS_BLOCK status;
  NTSTATUS result = NtCreateFile(output, access | SYNCHRONIZE, &attributes, &status,
    NULL, 0, share, FILE_OPEN, options | FILE_OPEN_REPARSE_POINT | FILE_SYNCHRONOUS_IO_NONALERT, NULL, 0);
  return result < 0 ? RtlNtStatusToDosError(result) : ERROR_SUCCESS;
}
static DWORD open_without_reparse(HANDLE root, LPCWSTR path, ACCESS_MASK access,
    ULONG share, ULONG options, HANDLE *output) {
  return open_file_object(root, path, access, share, options, OBJ_DONT_REPARSE, output);
}
static BOOL safe_relative_path(LPCWSTR path) {
  if (!path || !path[0]) return FALSE;
  LPCWSTR segment = path;
  for (LPCWSTR cursor = path;; ++cursor) {
    if (*cursor == L':' || *cursor == L'/' || *cursor == L'*' || *cursor == L'?') return FALSE;
    if (*cursor == L'\\' || !*cursor) {
      size_t length = (size_t)(cursor - segment);
      if (!length || segment[length - 1] == L'.' || segment[length - 1] == L' ') return FALSE;
      if (!*cursor) return TRUE;
      segment = cursor + 1;
    }
  }
}
__declspec(dllexport) DWORD WINAPI RemovePayload(LPCWSTR relative, BOOL directory) {
  if (!leaseHeld || removalDirectory == INVALID_HANDLE_VALUE ||
      removalExecutable == INVALID_HANDLE_VALUE || !safe_relative_path(relative)) return ERROR_INVALID_PARAMETER;
  HANDLE file = INVALID_HANDLE_VALUE;
  DWORD error = open_without_reparse(removalDirectory, relative, DELETE | FILE_READ_ATTRIBUTES,
    FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
    directory ? FILE_DIRECTORY_FILE : FILE_NON_DIRECTORY_FILE, &file);
  if (error) return error == ERROR_FILE_NOT_FOUND || error == ERROR_PATH_NOT_FOUND ? ERROR_SUCCESS : error;
  BY_HANDLE_FILE_INFORMATION info;
  if (!GetFileInformationByHandle(file, &info)) error = GetLastError();
  else if (info.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) error = ERROR_ACCESS_DENIED;
  else {
    FILE_DISPOSITION_INFO remove = {TRUE};
    if (!SetFileInformationByHandle(file, FileDispositionInfo, &remove, sizeof(remove))) error = GetLastError();
  }
  CloseHandle(file);
  return directory && error == ERROR_DIR_NOT_EMPTY ? ERROR_SUCCESS : error;
}

#define STARTUP_NAME L"dev.magnitude.desktop"
#define RUN_KEY L"Software\\Microsoft\\Windows\\CurrentVersion\\Run"
#define APPROVAL_KEY L"Software\\Microsoft\\Windows\\CurrentVersion\\Explorer\\StartupApproved\\Run"

/* Retain deletion access before mutation, rejecting existing/future incompatible
   opens. Final retirement acts on this exact file after other removal succeeds. */
static DWORD open_removal_metadata(LPCWSTR path, HANDLE *output) {
  DWORD error = open_without_reparse(removalDirectory, path, GENERIC_READ | DELETE,
    FILE_SHARE_READ | FILE_SHARE_DELETE, FILE_NON_DIRECTORY_FILE, output);
  if (error) return error;
  BY_HANDLE_FILE_INFORMATION info;
  if (!GetFileInformationByHandle(*output, &info)) return GetLastError();
  if (GetFileType(*output) != FILE_TYPE_DISK || info.nNumberOfLinks != 1 ||
      (info.dwFileAttributes & (FILE_ATTRIBUTE_DIRECTORY | FILE_ATTRIBUTE_REPARSE_POINT | FILE_ATTRIBUTE_READONLY)))
    return ERROR_ACCESS_DENIED;
  return ERROR_SUCCESS;
}
/* The retained installed uninstaller is the removal record. Its bytes must match
   this executing NSIS self-copy before any payload or registration is changed. */
__declspec(dllexport) DWORD WINAPI AcquireRemovalExecutable(LPCWSTR executable) {
  if (!leaseHeld || !executable) return ERROR_INVALID_PARAMETER;
  if (removalExecutable != INVALID_HANDLE_VALUE) return ERROR_ALREADY_EXISTS;
  LPCWSTR leaf = wcsrchr(executable, L'\\');
  if (!leaf || leaf == executable || executable[1] != L':' || executable[2] != L'\\' ||
      wcscmp(leaf + 1, L"Uninstall Magnitude.exe")) return ERROR_BAD_PATHNAME;
  WCHAR directoryPath[32768];
  size_t parentLength = (size_t)(leaf - executable);
  if (parentLength + 5 >= 32768) return ERROR_BAD_PATHNAME;
  wcscpy(directoryPath, L"\\??\\");
  wmemcpy(directoryPath + 4, executable, parentLength);
  directoryPath[parentLength + 4] = 0;
  DWORD error = open_without_reparse(NULL, directoryPath, DELETE | FILE_READ_ATTRIBUTES,
    FILE_SHARE_READ | FILE_SHARE_WRITE, FILE_DIRECTORY_FILE, &removalDirectory);
  if (!error) {
    BY_HANDLE_FILE_INFORMATION info;
    if (!GetFileInformationByHandle(removalDirectory, &info)) error = GetLastError();
    else if (info.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) error = ERROR_ACCESS_DENIED;
  }
  if (!error) error = open_removal_metadata(leaf + 1, &removalExecutable);
  HANDLE current = INVALID_HANDLE_VALUE;
  WCHAR path[32768];
  if (!error) {
    DWORD count = GetModuleFileNameW(NULL, path, 32768);
    if (!count || count >= 32768) error = ERROR_BAD_PATHNAME;
  }
  if (!error) {
    current = CreateFileW(path, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_DELETE,
      NULL, OPEN_EXISTING, FILE_FLAG_OPEN_REPARSE_POINT, NULL);
    if (current == INVALID_HANDLE_VALUE) error = GetLastError();
  }
  if (!error) {
    BY_HANDLE_FILE_INFORMATION installedInfo, currentInfo;
    LARGE_INTEGER installedSize, currentSize;
    if (!GetFileInformationByHandle(removalExecutable, &installedInfo) ||
        !GetFileInformationByHandle(current, &currentInfo) ||
        !GetFileSizeEx(removalExecutable, &installedSize) || !GetFileSizeEx(current, &currentSize))
      error = GetLastError();
    else if (currentInfo.dwFileAttributes & (FILE_ATTRIBUTE_REPARSE_POINT | FILE_ATTRIBUTE_DIRECTORY))
      error = ERROR_ACCESS_DENIED;
    else if (installedInfo.dwVolumeSerialNumber == currentInfo.dwVolumeSerialNumber &&
        installedInfo.nFileIndexHigh == currentInfo.nFileIndexHigh &&
        installedInfo.nFileIndexLow == currentInfo.nFileIndexLow)
      error = ERROR_NOT_SUPPORTED; /* Never mutate when invoked in place with _?=. */
    else if (installedSize.QuadPart != currentSize.QuadPart || installedSize.QuadPart <= 0)
      error = ERROR_INVALID_DATA;
  }
  BYTE installedBytes[16384], currentBytes[16384];
  while (!error) {
    DWORD installedCount = 0, currentCount = 0;
    if (!ReadFile(removalExecutable, installedBytes, sizeof(installedBytes), &installedCount, NULL) ||
        !ReadFile(current, currentBytes, sizeof(currentBytes), &currentCount, NULL)) {
      error = GetLastError(); break;
    }
    if (installedCount != currentCount || memcmp(installedBytes, currentBytes, installedCount)) {
      error = ERROR_INVALID_DATA; break;
    }
    if (!installedCount) break;
  }
  if (current != INVALID_HANDLE_VALUE) CloseHandle(current);
  if (error && removalExecutable != INVALID_HANDLE_VALUE) {
    CloseHandle(removalExecutable); removalExecutable = INVALID_HANDLE_VALUE;
  }
  if (error && removalDirectory != INVALID_HANDLE_VALUE) {
    CloseHandle(removalDirectory); removalDirectory = INVALID_HANDLE_VALUE;
  }
  return error;
}
__declspec(dllexport) DWORD WINAPI RemoveRemovalExecutable(void) {
  if (!leaseHeld || removalExecutable == INVALID_HANDLE_VALUE) return ERROR_INVALID_HANDLE;
  FILE_DISPOSITION_INFO remove = {TRUE};
  if (!SetFileInformationByHandle(removalExecutable, FileDispositionInfo, &remove, sizeof(remove)))
    return GetLastError();
  CloseHandle(removalExecutable); removalExecutable = INVALID_HANDLE_VALUE;
  /* Empty root cleanup is optional: unrelated files keep their directory. */
  SetFileInformationByHandle(removalDirectory, FileDispositionInfo, &remove, sizeof(remove));
  CloseHandle(removalDirectory); removalDirectory = INVALID_HANDLE_VALUE;
  return ERROR_SUCCESS;
}

/* This directory is installer-owned scratch, unlike the published installation.
   Single-component opens are relative to retained handles and never follow links. */
static DWORD clear_scratch(HANDLE directory, unsigned depth) {
  if (depth > 128) return ERROR_DIRECTORY;
  BYTE *buffer = HeapAlloc(GetProcessHeap(), 0, 65536);
  if (!buffer) return ERROR_NOT_ENOUGH_MEMORY;
  DWORD error = ERROR_SUCCESS;
  for (;;) {
    WCHAR name[256]; BOOL found = FALSE;
    FILE_INFO_BY_HANDLE_CLASS query = FileIdBothDirectoryRestartInfo;
    for (;;) {
      if (!GetFileInformationByHandleEx(directory, query, buffer, 65536)) {
        error = GetLastError();
        if (error == ERROR_NO_MORE_FILES) error = ERROR_SUCCESS;
        break;
      }
      FILE_ID_BOTH_DIR_INFO *entry = (FILE_ID_BOTH_DIR_INFO *)buffer;
      for (;;) {
        DWORD length = entry->FileNameLength / sizeof(WCHAR);
        if (entry->FileNameLength % sizeof(WCHAR) || length >= 256) { error = ERROR_BAD_PATHNAME; break; }
        if (!(length == 1 && entry->FileName[0] == L'.') &&
            !(length == 2 && entry->FileName[0] == L'.' && entry->FileName[1] == L'.')) {
          wmemcpy(name, entry->FileName, length); name[length] = 0;
          found = TRUE; break;
        }
        if (!entry->NextEntryOffset) break;
        entry = (FILE_ID_BOTH_DIR_INFO *)((BYTE *)entry + entry->NextEntryOffset);
      }
      if (found || error) break;
      query = FileIdBothDirectoryInfo;
    }
    if (!found || error) break;
    HANDLE child = INVALID_HANDLE_VALUE;
    error = open_file_object(directory, name, DELETE | FILE_READ_ATTRIBUTES | FILE_LIST_DIRECTORY,
      FILE_SHARE_READ | FILE_SHARE_WRITE, FILE_OPEN_FOR_BACKUP_INTENT, 0, &child);
    if (error) break;
    BY_HANDLE_FILE_INFORMATION info;
    if (!GetFileInformationByHandle(child, &info)) error = GetLastError();
    else if ((info.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) &&
             !(info.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT))
      error = clear_scratch(child, depth + 1);
    if (!error) {
      FILE_DISPOSITION_INFO remove = {TRUE};
      if (!SetFileInformationByHandle(child, FileDispositionInfo, &remove, sizeof(remove))) error = GetLastError();
    }
    CloseHandle(child);
    if (error) break;
    /* Restart enumeration after deletion; directory offsets are not stable. */
  }
  HeapFree(GetProcessHeap(), 0, buffer);
  return error;
}
__declspec(dllexport) DWORD WINAPI CleanupStage(void) {
  if (!leaseHeld || stageDirectory == INVALID_HANDLE_VALUE) return ERROR_INVALID_HANDLE;
  return clear_scratch(stageDirectory, 0);
}
/* A fixed, private scratch container makes interrupted extraction recoverable.
   The application lease serializes callers; there is no second owner election. */
__declspec(dllexport) DWORD WINAPI CreateStage(LPWSTR output, DWORD capacity) {
  if (!output || capacity < 1 || capacity > 32768) return ERROR_INVALID_PARAMETER;
  output[0] = 0;
  if (!leaseHeld) return ERROR_INVALID_HANDLE;
  if (stageDirectory != INVALID_HANDLE_VALUE) return ERROR_ALREADY_EXISTS;
  PWSTR local = NULL;
  if (FAILED(SHGetKnownFolderPath(&FOLDERID_LocalAppData, 0, NULL, &local)) || !local)
    return ERROR_PATH_NOT_FOUND;
  WCHAR parent[32768], container[32768], payload[32768];
  int count = swprintf(parent, 32768, L"%ls\\Programs", local);
  CoTaskMemFree(local);
  if (count < 0 || swprintf(container, 32768, L"%ls\\Magnitude-installation-stage", parent) < 0 ||
      swprintf(payload, 32768, L"%ls\\payload", container) < 0 || wcslen(payload) >= capacity)
    return ERROR_BAD_PATHNAME;
  if (!CreateDirectoryW(parent, NULL) && GetLastError() != ERROR_ALREADY_EXISTS) return GetLastError();
  PSECURITY_DESCRIPTOR descriptor = NULL;
  DWORD error = magnitude_private_descriptor(TRUE, &descriptor);
  if (error) return error;
  SECURITY_ATTRIBUTES attributes = {sizeof(attributes), descriptor, FALSE};
  if (!CreateDirectoryW(container, &attributes) && GetLastError() != ERROR_ALREADY_EXISTS) error = GetLastError();
  if (!error) {
    stageDirectory = CreateFileW(container, READ_CONTROL | FILE_READ_ATTRIBUTES | FILE_LIST_DIRECTORY,
      FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, OPEN_EXISTING,
      FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT, NULL);
    if (stageDirectory == INVALID_HANDLE_VALUE) error = GetLastError();
    else error = magnitude_validate_private_directory(stageDirectory);
  }
  if (!error) error = clear_scratch(stageDirectory, 0);
  if (!error && !CreateDirectoryW(payload, &attributes)) error = GetLastError();
  LocalFree(descriptor);
  if (error) {
    if (stageDirectory != INVALID_HANDLE_VALUE) CloseHandle(stageDirectory);
    stageDirectory = INVALID_HANDLE_VALUE;
  } else wcscpy(output, payload);
  return error;
}

__declspec(dllexport) DWORD WINAPI RequireAbsentPath(LPCWSTR path) {
  if (GetFileAttributesW(path) != INVALID_FILE_ATTRIBUTES) return ERROR_ALREADY_EXISTS;
  DWORD error = GetLastError();
  return error == ERROR_FILE_NOT_FOUND || error == ERROR_PATH_NOT_FOUND ? ERROR_SUCCESS : error;
}

/* An empty directory can remain after successful final record removal. Delete
   only that exact empty directory; the kernel refuses any nonempty directory. */
__declspec(dllexport) DWORD WINAPI PrepareInstallationDirectory(LPCWSTR path) {
  if (!leaseHeld || !path || !path[0]) return ERROR_INVALID_PARAMETER;
  HANDLE directory = CreateFileW(path, FILE_READ_ATTRIBUTES | DELETE,
    FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, OPEN_EXISTING,
    FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT, NULL);
  if (directory == INVALID_HANDLE_VALUE) {
    DWORD error = GetLastError();
    return error == ERROR_FILE_NOT_FOUND || error == ERROR_PATH_NOT_FOUND ? ERROR_SUCCESS : error;
  }
  BY_HANDLE_FILE_INFORMATION info;
  DWORD error = ERROR_SUCCESS;
  if (!GetFileInformationByHandle(directory, &info)) error = GetLastError();
  else if (!(info.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) ||
      (info.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT)) error = ERROR_ACCESS_DENIED;
  else {
    FILE_DISPOSITION_INFO remove = {TRUE};
    if (!SetFileInformationByHandle(directory, FileDispositionInfo, &remove, sizeof(remove)))
      error = GetLastError();
  }
  CloseHandle(directory);
  return error;
}

__declspec(dllexport) DWORD WINAPI RequireUnusedRegistration(LPCWSTR shortcut, LPCWSTR registration) {
  if (!leaseHeld || !shortcut || !registration || !registration[0]) return ERROR_INVALID_PARAMETER;
  DWORD error = RequireAbsentPath(shortcut);
  if (error) return error;
  REGSAM views[] = {KEY_WOW64_32KEY, KEY_WOW64_64KEY};
  for (int index = 0; index < 2; ++index) {
    HKEY key;
    LONG result = RegOpenKeyExW(HKEY_CURRENT_USER,
      registration,
      0, KEY_QUERY_VALUE | views[index], &key);
    if (result == ERROR_SUCCESS) { RegCloseKey(key); return ERROR_ALREADY_EXISTS; }
    if (result != ERROR_FILE_NOT_FOUND) return result;
  }
  return ERROR_SUCCESS;
}

/* A retry may follow failure before an uninstall key was created. */
__declspec(dllexport) DWORD WINAPI RemoveRegistration(LPCWSTR keyPath) {
  if (!keyPath || !keyPath[0]) return ERROR_INVALID_PARAMETER;
  LONG error = RegDeleteTreeW(HKEY_CURRENT_USER, keyPath);
  return error == ERROR_FILE_NOT_FOUND ? ERROR_SUCCESS : error;
}

/* Remove only this installation's exact Electron registration. A missing or
   externally replaced command does not establish ownership of approval data. */
__declspec(dllexport) DWORD WINAPI RemoveOwnedStartup(LPCWSTR executable) {
  WCHAR expected[32768], value[32768];
  if (!executable || swprintf(expected, 32768, L"\"%ls\" --background", executable) < 0)
    return ERROR_BAD_PATHNAME;
  HKEY run;
  LONG error = RegOpenKeyExW(HKEY_CURRENT_USER, RUN_KEY, 0, KEY_QUERY_VALUE | KEY_SET_VALUE, &run);
  if (error == ERROR_FILE_NOT_FOUND) return ERROR_SUCCESS;
  if (error) return error;
  DWORD bytes = sizeof(value), type = 0;
  error = RegQueryValueExW(run, STARTUP_NAME, NULL, &type, (BYTE*)value, &bytes);
  if (error == ERROR_FILE_NOT_FOUND || error == ERROR_MORE_DATA) { RegCloseKey(run); return ERROR_SUCCESS; }
  if (error) { RegCloseKey(run); return error; }
  if (type != REG_SZ || bytes < sizeof(WCHAR) || bytes % sizeof(WCHAR) ||
      bytes > sizeof(value) || value[bytes / sizeof(WCHAR) - 1] != 0 ||
      bytes != (wcslen(expected) + 1) * sizeof(WCHAR) || wcscmp(value, expected)) {
    RegCloseKey(run); return ERROR_SUCCESS;
  }
  error = RegDeleteValueW(run, STARTUP_NAME);
  RegCloseKey(run);
  if (error) return error;
  HKEY approval;
  error = RegOpenKeyExW(HKEY_CURRENT_USER, APPROVAL_KEY, 0, KEY_SET_VALUE, &approval);
  if (error == ERROR_FILE_NOT_FOUND) return ERROR_SUCCESS;
  if (error) return error;
  error = RegDeleteValueW(approval, STARTUP_NAME);
  RegCloseKey(approval);
  return error == ERROR_FILE_NOT_FOUND ? ERROR_SUCCESS : error;
}

/* Process-scoped installation lease; never starts or adopts a service. */
__declspec(dllexport) DWORD WINAPI HoldOwnership(void) {
  PWSTR local = NULL;
  HRESULT result = SHGetKnownFolderPath(&FOLDERID_LocalAppData, 0, NULL, &local);
  if (FAILED(result) || !local) return ERROR_PATH_NOT_FOUND;
  WCHAR parent[32768], path[32768];
  if (local[1] != L':' ||
      swprintf(parent, 32768, L"%ls\\Magnitude", local) < 0 ||
      swprintf(path, 32768, L"%ls\\Magnitude\\desktop\\application.lock", local) < 0) {
    CoTaskMemFree(local); return ERROR_BAD_PATHNAME;
  }
  CoTaskMemFree(local);
  if (!CreateDirectoryW(parent, NULL) && GetLastError() != ERROR_ALREADY_EXISTS) return GetLastError();
  HANDLE file, directory;
  DWORD error = magnitude_open_private_lock(path, &file, &directory);
  if (error) return error;
  OVERLAPPED offset = {0};
  if (!LockFileEx(file, LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY, 0, 1, 0, &offset)) {
    error = GetLastError(); CloseHandle(file); CloseHandle(directory); return error;
  }
  leaseHeld = TRUE;
  /* Non-inheritable kernel handles deliberately live until installer process exit.
     DLL unload must not release the lease between NSIS instructions. */
  return ERROR_SUCCESS;
}
