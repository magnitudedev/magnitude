#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0A00
#endif
#include "windows-cli-launcher.h"
#include <shlobj.h>
#include <stdio.h>
#include <wchar.h>

/* The desktop's bundled CLI wins; otherwise the standalone version `.magnitude\\cli\\current` names. */
static BOOL resolve(WCHAR *executable) {
  PWSTR local = NULL, profile = NULL;
  BOOL found = FALSE;
  if (SUCCEEDED(SHGetKnownFolderPath(&FOLDERID_LocalAppData, KF_FLAG_DEFAULT, NULL, &local))) {
    int length = swprintf(executable, 32768, L"%ls\\Programs\\Magnitude\\resources\\magnitude.exe", local);
    DWORD attributes = length > 0 && length < 32768 ? GetFileAttributesW(executable) : INVALID_FILE_ATTRIBUTES;
    found = attributes != INVALID_FILE_ATTRIBUTES && !(attributes & FILE_ATTRIBUTE_DIRECTORY);
  }
  CoTaskMemFree(local);
  if (found) return TRUE;
  if (FAILED(SHGetKnownFolderPath(&FOLDERID_Profile, KF_FLAG_DEFAULT, NULL, &profile))) { CoTaskMemFree(profile); return FALSE; }
  WCHAR pointer[32768];
  int length = swprintf(pointer, 32768, L"%ls\\.magnitude\\cli\\current", profile);
  if (length > 0 && length < 32768) {
    HANDLE file = CreateFileW(pointer, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_DELETE, NULL, OPEN_EXISTING, 0, NULL);
    char version[128];
    DWORD read = 0;
    if (file != INVALID_HANDLE_VALUE) {
      if (ReadFile(file, version, sizeof(version) - 1, &read, NULL)) {
        while (read && (version[read - 1] == '\n' || version[read - 1] == '\r' || version[read - 1] == ' ')) --read;
        version[read] = 0;
        BOOL valid = read > 0 && version[0] != '.';
        for (DWORD index = 0; valid && index < read; ++index) {
          char c = version[index];
          valid = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '.' || c == '+' || c == '-';
        }
        if (valid) {
          length = swprintf(executable, 32768, L"%ls\\.magnitude\\cli\\%hs\\magnitude.exe", profile, version);
          DWORD attributes = length > 0 && length < 32768 ? GetFileAttributesW(executable) : INVALID_FILE_ATTRIBUTES;
          found = attributes != INVALID_FILE_ATTRIBUTES && !(attributes & FILE_ATTRIBUTE_DIRECTORY);
        }
      }
      CloseHandle(file);
    }
  }
  CoTaskMemFree(profile);
  return found;
}

int wmain(int argc, WCHAR **argv) {
  /* Root flags take no values. Unknown flags remain the CLI parser's errors. */
  int command = 1;
  while (command < argc && (!wcscmp(argv[command], L"--") || !wcscmp(argv[command], L"--version") ||
      !wcscmp(argv[command], L"-v") || !wcscmp(argv[command], L"--help") || !wcscmp(argv[command], L"-h"))) ++command;
  BOOL serving = command < argc && !wcscmp(argv[command], L"serve");
  return (int)magnitude_cli_run(resolve, argc, argv, serving);
}
