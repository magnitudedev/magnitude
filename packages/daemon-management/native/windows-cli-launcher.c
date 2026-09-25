#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0A00
#endif
#include "windows-cli-launcher.h"
#include "windows-job.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>

static volatile LONG stopping;
static BOOL WINAPI console_control(DWORD event) {
  if (event != CTRL_C_EVENT && event != CTRL_BREAK_EVENT && event != CTRL_CLOSE_EVENT &&
      event != CTRL_LOGOFF_EVENT && event != CTRL_SHUTDOWN_EVENT) return FALSE;
  InterlockedExchange(&stopping, 1);
  return TRUE;
}

static DWORD identity(const WCHAR *path, FILE_ID_INFO *result) {
  HANDLE file = CreateFileW(path, FILE_READ_ATTRIBUTES, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
      NULL, OPEN_EXISTING, FILE_FLAG_OPEN_REPARSE_POINT, NULL);
  if (file == INVALID_HANDLE_VALUE) return GetLastError();
  FILE_ATTRIBUTE_TAG_INFO attributes;
  DWORD failure = ERROR_SUCCESS;
  if (!GetFileInformationByHandleEx(file, FileAttributeTagInfo, &attributes, sizeof(attributes)) ||
      !GetFileInformationByHandleEx(file, FileIdInfo, result, sizeof(*result))) failure = GetLastError();
  else if (GetFileType(file) != FILE_TYPE_DISK ||
      (attributes.FileAttributes & (FILE_ATTRIBUTE_DIRECTORY | FILE_ATTRIBUTE_REPARSE_POINT))) failure = ERROR_ACCESS_DENIED;
  CloseHandle(file);
  return failure;
}

static BOOL append(WCHAR *buffer, size_t *length, WCHAR character) {
  if (*length >= 32766) return FALSE;
  buffer[(*length)++] = character;
  return TRUE;
}

/* Always quote each CRT argument; double slashes preceding quotes or the closing quote. */
static BOOL argument(WCHAR *buffer, size_t *length, const WCHAR *value) {
  if (*length && !append(buffer, length, L' ')) return FALSE;
  if (!append(buffer, length, L'"')) return FALSE;
  while (*value) {
    size_t slashes = 0;
    while (*value == L'\\') { ++slashes; ++value; }
    size_t count = (*value == L'"' || !*value) ? slashes * 2 : slashes;
    while (count--) if (!append(buffer, length, L'\\')) return FALSE;
    if (*value == L'"' && !append(buffer, length, L'\\')) return FALSE;
    if (*value && !append(buffer, length, *value++)) return FALSE;
  }
  return append(buffer, length, L'"');
}

static DWORD standard_handle(DWORD kind, HANDLE *result) {
  HANDLE source = GetStdHandle(kind);
  if (!source || source == INVALID_HANDLE_VALUE) {
    SECURITY_ATTRIBUTES attributes = { sizeof(attributes), NULL, TRUE };
    *result = CreateFileW(L"NUL", kind == STD_INPUT_HANDLE ? GENERIC_READ : GENERIC_WRITE,
        FILE_SHARE_READ | FILE_SHARE_WRITE, &attributes, OPEN_EXISTING, 0, NULL);
    return *result == INVALID_HANDLE_VALUE ? GetLastError() : ERROR_SUCCESS;
  }
  return DuplicateHandle(GetCurrentProcess(), source, GetCurrentProcess(), result, 0, TRUE,
      DUPLICATE_SAME_ACCESS) ? ERROR_SUCCESS : GetLastError();
}

static DWORD run_child(const WCHAR *executable, WCHAR *command, const WCHAR *directory, BOOL serving, DWORD *code) {
  HANDLE standard[3] = { NULL, NULL, NULL };
  const DWORD kinds[3] = { STD_INPUT_HANDLE, STD_OUTPUT_HANDLE, STD_ERROR_HANDLE };
  magnitude_owned_process owned = {0};
  DWORD failure = ERROR_SUCCESS;
  for (size_t index = 0; index < 3; ++index) {
    failure = standard_handle(kinds[index], &standard[index]);
    if (failure) goto cleanup;
  }
  if (InterlockedCompareExchange(&stopping, 0, 0)) { failure = ERROR_CANCELLED; goto cleanup; }
  failure = serving
    ? magnitude_owned_spawn_foreground(executable, command, directory, standard[0], standard[1], standard[2], &owned)
    : magnitude_unowned_spawn_foreground(executable, command, directory, standard[0], standard[1], standard[2], &owned.process);
  if (failure) goto cleanup;
  ULONGLONG cancellation_deadline = 0;
  for (;;) {
    BOOL exited = FALSE;
    failure = magnitude_owned_exit(&owned, &exited, code);
    if (failure || exited) break;
    if (InterlockedCompareExchange(&stopping, 0, 0)) {
      if (!cancellation_deadline) cancellation_deadline = GetTickCount64() + 5000;
      if (GetTickCount64() >= cancellation_deadline) break;
    }
    Sleep(10);
  }
  if (failure) goto cleanup;
  if (!serving) {
    BOOL exited = FALSE;
    failure = magnitude_owned_exit(&owned, &exited, code);
    if (!failure && !exited) {
      if (!TerminateProcess(owned.process, ERROR_CANCELLED)) failure = GetLastError();
      else if (WaitForSingleObject(owned.process, 10000) != WAIT_OBJECT_0) failure = WAIT_TIMEOUT;
    }
    goto cleanup;
  }
  /* A root cannot leave children behind, including on a continuation request. */
  DWORD active;
  failure = magnitude_owned_active(&owned, &active);
  if (failure) goto cleanup;
  if (active) {
    failure = magnitude_owned_terminate(&owned, ERROR_CANCELLED);
    if (failure) goto cleanup;
  }
  ULONGLONG deadline = GetTickCount64() + 10000;
  for (;;) {
    BOOL exited = FALSE;
    failure = magnitude_owned_active(&owned, &active);
    if (!failure) failure = magnitude_owned_exit(&owned, &exited, code);
    if (failure || (!active && exited)) break;
    if (GetTickCount64() >= deadline) { failure = WAIT_TIMEOUT; break; }
    Sleep(10);
  }
cleanup:
  magnitude_owned_close(&owned);
  for (size_t index = 0; index < 3; ++index)
    if (standard[index] && standard[index] != INVALID_HANDLE_VALUE) CloseHandle(standard[index]);
  return failure;
}

DWORD magnitude_cli_run(const WCHAR *executable, int argc, WCHAR **argv, BOOL serving) {
  WCHAR directory[32768], safe_directory[32768], command[32768];
  FILE_ID_INFO before, after;
  DWORD failure = ERROR_SUCCESS, code = 1;
  DWORD length = GetCurrentDirectoryW(32768, directory);
  if (!length || length >= 32768) return 1;
  /* The launcher must not retain a cwd handle in the replaceable application tree. */
  length = GetSystemDirectoryW(safe_directory, 32768);
  if (!length || length >= 32768 || !SetCurrentDirectoryW(safe_directory)) return 1;
  InterlockedExchange(&stopping, 0);
  if (!SetConsoleCtrlHandler(console_control, TRUE)) return 1;
  if (!SetEnvironmentVariableW(L"MAGNITUDE_CLI_LAUNCHER_PROTOCOL", serving ? MAGNITUDE_CLI_LAUNCHER_PROTOCOL : NULL)) {
    failure = GetLastError(); goto cleanup;
  }
  for (int attempt = 0; attempt < 2; ++attempt) {
    failure = identity(executable, &before);
    if (failure) break;
    size_t used = 0;
    if (!argument(command, &used, executable)) { failure = ERROR_BUFFER_OVERFLOW; break; }
    for (int index = 1; index < argc; ++index) {
      if (!argument(command, &used, argv[index])) { failure = ERROR_BUFFER_OVERFLOW; break; }
    }
    if (failure) break;
    command[used] = 0;
    failure = run_child(executable, command, directory, serving, &code);
    if (failure || !serving || code != MAGNITUDE_CLI_CONTINUE || InterlockedCompareExchange(&stopping, 0, 0)) break;
    if (attempt != 0) { failure = ERROR_RETRY; break; }
    failure = identity(executable, &after);
    if (failure) break;
    if (before.VolumeSerialNumber == after.VolumeSerialNumber &&
        !memcmp(&before.FileId, &after.FileId, sizeof(before.FileId))) { failure = ERROR_FILE_INVALID; break; }
  }
cleanup:
  SetConsoleCtrlHandler(console_control, FALSE);
  if (InterlockedCompareExchange(&stopping, 0, 0)) return ERROR_CANCELLED;
  if (failure) {
    fprintf(stderr, "Magnitude could not continue the command (Windows error %lu).\n", failure);
    return 1;
  }
  return code;
}
