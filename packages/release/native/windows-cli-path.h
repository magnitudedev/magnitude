/* User PATH registration belongs to the installed payload, not its npm predecessor. */
#define CLI_PATH_CAPACITY 32768
static DWORD read_user_path(HKEY key, WCHAR *path, DWORD *type) {
  DWORD bytes = CLI_PATH_CAPACITY * sizeof(WCHAR);
  DWORD error = RegQueryValueExW(key, L"Path", NULL, type, (BYTE *)path, &bytes);
  if (error == ERROR_FILE_NOT_FOUND) { path[0] = 0; *type = REG_EXPAND_SZ; return ERROR_SUCCESS; }
  if (error) return error;
  if ((*type != REG_SZ && *type != REG_EXPAND_SZ) || bytes < sizeof(WCHAR) || bytes % sizeof(WCHAR) ||
      bytes > CLI_PATH_CAPACITY * sizeof(WCHAR) || path[bytes / sizeof(WCHAR) - 1] ||
      (wcslen(path) + 1) * sizeof(WCHAR) != bytes) return ERROR_INVALID_DATA;
  return ERROR_SUCCESS;
}

/* Compare complete entries, never substring-match another application's directory. */
static BOOL cli_path_entry_matches(LPCWSTR start, size_t length, LPCWSTR directory) {
  while (length && (*start == L' ' || *start == L'\t')) { start++; length--; }
  while (length && (start[length - 1] == L' ' || start[length - 1] == L'\t')) length--;
  if (length >= 2 && start[0] == L'"' && start[length - 1] == L'"') { start++; length -= 2; }
  while (length && (start[length - 1] == L'\\' || start[length - 1] == L'/')) length--;
  return length == wcslen(directory) && _wcsnicmp(start, directory, length) == 0;
}

__declspec(dllexport) DWORD WINAPI ConfigureCliPath(LPCWSTR directory, LPCWSTR registration, BOOL remove) {
  if (!leaseHeld || !directory || !*directory || wcschr(directory, L';') || !registration) return ERROR_INVALID_PARAMETER;
  HKEY environment = NULL, installation = NULL;
  DWORD error = RegCreateKeyExW(HKEY_CURRENT_USER, L"Environment", 0, NULL, 0,
    KEY_QUERY_VALUE | KEY_SET_VALUE, NULL, &environment, NULL);
  if (error) return error;
  error = RegOpenKeyExW(HKEY_CURRENT_USER, registration, 0,
    KEY_QUERY_VALUE | KEY_SET_VALUE | KEY_WOW64_32KEY, &installation);
  WCHAR *path = calloc(CLI_PATH_CAPACITY, sizeof(WCHAR));
  WCHAR *next = calloc(CLI_PATH_CAPACITY, sizeof(WCHAR));
  WCHAR *owned = calloc(CLI_PATH_CAPACITY, sizeof(WCHAR));
  DWORD type = 0;
  if (!error && (!path || !next || !owned)) error = ERROR_NOT_ENOUGH_MEMORY;
  if (!error) error = read_user_path(environment, path, &type);
  if (remove && error == ERROR_FILE_NOT_FOUND) { error = ERROR_SUCCESS; goto done; }
  if (!error && remove) {
    error = read_registration_string(installation, L"OwnedCliPath", owned, CLI_PATH_CAPACITY);
    if (error == ERROR_FILE_NOT_FOUND) { error = ERROR_SUCCESS; goto done; }
    if (error) goto done;
    if (_wcsicmp(owned, directory)) { error = ERROR_INVALID_DATA; goto done; }
  }
  if (error) goto done;
  const WCHAR *entry = path, *match = NULL, *end = NULL;
  for (;;) {
    const WCHAR *separator = wcschr(entry, L';');
    size_t length = separator ? (size_t)(separator - entry) : wcslen(entry);
    if (cli_path_entry_matches(entry, length, directory)) { match = entry; end = entry + length; break; }
    if (!separator) break;
    entry = separator + 1;
  }
  if (!remove && match) goto done; /* Do not claim a pre-existing user entry. */
  if (remove) {
    if (!match) {
      error = RegDeleteValueW(installation, L"OwnedCliPath");
      goto done;
    }
    size_t prefix = (size_t)(match - path);
    if (*end == L';') end++;
    else if (prefix) prefix--;
    wmemcpy(next, path, prefix);
    wcscpy(next + prefix, end);
  } else {
    if (wcslen(directory) + wcslen(path) + 2 > CLI_PATH_CAPACITY) { error = ERROR_BUFFER_OVERFLOW; goto done; }
    wcscpy(next, directory);
    if (*path) { wcscat(next, L";"); wcscat(next, path); }
  }
  error = RegSetValueExW(environment, L"Path", 0, type, (const BYTE *)next, (DWORD)((wcslen(next) + 1) * sizeof(WCHAR)));
  if (error) goto done;
  if (remove) error = RegDeleteValueW(installation, L"OwnedCliPath");
  else error = RegSetValueExW(installation, L"OwnedCliPath", 0, REG_SZ,
    (const BYTE *)directory, (DWORD)((wcslen(directory) + 1) * sizeof(WCHAR)));
  if (error) {
    /* A failed ownership write must not leave an unowned PATH change. */
    RegSetValueExW(environment, L"Path", 0, type, (const BYTE *)path, (DWORD)((wcslen(path) + 1) * sizeof(WCHAR)));
  } else {
    DWORD_PTR ignored;
    SendMessageTimeoutW(HWND_BROADCAST, WM_SETTINGCHANGE, 0, (LPARAM)L"Environment",
      SMTO_ABORTIFHUNG, 2000, &ignored);
  }
done:
  free(path); free(next); free(owned);
  if (installation) RegCloseKey(installation);
  RegCloseKey(environment);
  return error;
}

/* Retire previous commands after installing the bundled CLI and registering its PATH. */
__declspec(dllexport) DWORD WINAPI RemovePreviousCliCommands(LPCWSTR directory, LPWSTR conflict, DWORD capacity) {
  if (!leaseHeld || !directory || !conflict || capacity < 2) return ERROR_INVALID_PARAMETER;
  conflict[0] = 0;
  WCHAR *path = calloc(CLI_PATH_CAPACITY, sizeof(WCHAR));
  WCHAR *entry = calloc(CLI_PATH_CAPACITY, sizeof(WCHAR));
  WCHAR *expanded = calloc(CLI_PATH_CAPACITY, sizeof(WCHAR));
  WCHAR *candidate = calloc(CLI_PATH_CAPACITY, sizeof(WCHAR));
  DWORD error = ERROR_SUCCESS;
  if (!path || !entry || !expanded || !candidate) { error = ERROR_NOT_ENOUGH_MEMORY; goto finish; }
  DWORD length = GetEnvironmentVariableW(L"PATH", path, CLI_PATH_CAPACITY);
  if (!length) { error = GetLastError() == ERROR_ENVVAR_NOT_FOUND ? ERROR_SUCCESS : GetLastError(); goto finish; }
  if (length >= CLI_PATH_CAPACITY) { error = ERROR_BUFFER_OVERFLOW; goto finish; }
  LPCWSTR cursor = path;
  for (;;) {
    LPCWSTR separator = wcschr(cursor, L';');
    size_t count = separator ? (size_t)(separator - cursor) : wcslen(cursor);
    while (count && (*cursor == L' ' || *cursor == L'\t')) { cursor++; count--; }
    while (count && (cursor[count - 1] == L' ' || cursor[count - 1] == L'\t')) count--;
    if (count >= 2 && *cursor == L'"' && cursor[count - 1] == L'"') { cursor++; count -= 2; }
    if (count) {
      wmemcpy(entry, cursor, count); entry[count] = 0;
      DWORD needed = ExpandEnvironmentStringsW(entry, expanded, CLI_PATH_CAPACITY);
      if (!needed || needed > CLI_PATH_CAPACITY) { error = needed ? ERROR_BUFFER_OVERFLOW : GetLastError(); goto finish; }
      if (!cli_path_entry_matches(expanded, wcslen(expanded), directory)) {
        static const LPCWSTR extensions[] = {L"", L".exe", L".com", L".cmd", L".bat", L".ps1"};
        for (size_t i = 0; i < sizeof(extensions) / sizeof(extensions[0]); i++) {
          if (swprintf(candidate, CLI_PATH_CAPACITY, L"%ls\\magnitude%ls", expanded, extensions[i]) < 0) { error = ERROR_BUFFER_OVERFLOW; goto finish; }
          DWORD attributes = GetFileAttributesW(candidate);
          if (attributes == INVALID_FILE_ATTRIBUTES) {
            DWORD failure = GetLastError();
            if (failure == ERROR_FILE_NOT_FOUND || failure == ERROR_PATH_NOT_FOUND || failure == ERROR_INVALID_NAME) continue;
            error = failure; goto finish;
          }
          if (attributes & FILE_ATTRIBUTE_DIRECTORY) continue;
          if (wcslen(candidate) >= capacity) { error = ERROR_INSUFFICIENT_BUFFER; goto finish; }
          if (!DeleteFileW(candidate)) { wcscpy(conflict, candidate); error = GetLastError(); goto finish; }
        }
      }
    }
    if (!separator) break;
    cursor = separator + 1;
  }
finish:
  free(path); free(entry); free(expanded); free(candidate);
  return error;
}
