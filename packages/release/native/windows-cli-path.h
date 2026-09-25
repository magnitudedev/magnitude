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
__declspec(dllexport) DWORD WINAPI RemovePreviousCliCommands(LPCWSTR directory, LPCWSTR bundled, LPWSTR conflict, DWORD capacity) {
  if (!leaseHeld || !directory || !bundled || !conflict || capacity < 2) return ERROR_INVALID_PARAMETER;
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
      if (!cli_path_entry_matches(expanded, wcslen(expanded), directory) &&
          !cli_path_entry_matches(expanded, wcslen(expanded), bundled)) {
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

/* The command lives outside the replaceable application tree. Retired images can remain
 * mapped by an existing foreground command; each publication uses a distinct name. */
static DWORD retire_cli_images(HANDLE directory) {
  BYTE buffer[16384];
  FILE_INFO_BY_HANDLE_CLASS query = FileIdBothDirectoryRestartInfo;
  for (;;) {
    if (!GetFileInformationByHandleEx(directory, query, buffer, sizeof(buffer)))
      return GetLastError() == ERROR_NO_MORE_FILES ? ERROR_SUCCESS : GetLastError();
    FILE_ID_BOTH_DIR_INFO *entry = (FILE_ID_BOTH_DIR_INFO *)buffer;
    BOOL removed = FALSE;
    for (;;) {
      WCHAR name[128];
      size_t count = entry->FileNameLength / sizeof(WCHAR);
      if (count < 128) {
        wmemcpy(name, entry->FileName, count); name[count] = 0;
        if ((count == 60 && !wcsncmp(name, L"magnitude-retired-", 18) && !wcscmp(name + 56, L".exe")) ||
            (count == 61 && !wcsncmp(name, L"magnitude-incoming-", 19) && !wcscmp(name + 57, L".exe"))) {
          HANDLE file = INVALID_HANDLE_VALUE;
          DWORD error = open_without_reparse(directory, name, DELETE | FILE_READ_ATTRIBUTES,
            FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, FILE_NON_DIRECTORY_FILE, &file);
          if (!error) {
            FILE_DISPOSITION_INFO disposition = {TRUE};
            removed = SetFileInformationByHandle(file, FileDispositionInfo, &disposition, sizeof(disposition));
            CloseHandle(file);
          }
          if (removed) break;
        }
      }
      if (!entry->NextEntryOffset) break;
      entry = (FILE_ID_BOTH_DIR_INFO *)((BYTE *)entry + entry->NextEntryOffset);
    }
    query = removed ? FileIdBothDirectoryRestartInfo : FileIdBothDirectoryInfo;
  }
}

__declspec(dllexport) DWORD WINAPI InstallCliLauncher(LPCWSTR source, LPCWSTR path) {
  if (!leaseHeld || !source || !path) return ERROR_INVALID_PARAMETER;
  DWORD error = magnitude_prepare_private_directory(path);
  HANDLE directory = INVALID_HANDLE_VALUE, incoming = INVALID_HANDLE_VALUE, current = INVALID_HANDLE_VALUE;
  HANDLE input = INVALID_HANDLE_VALUE;
  WCHAR temporary[32768], retired[128];
  GUID id;
  WCHAR guid[40];
  BOOL moved = FALSE;
  if (!error) error = open_installation_directory(path, &directory);
  if (!error) error = retire_cli_images(directory);
  if (error) goto done;
  if (FAILED(CoCreateGuid(&id)) || !StringFromGUID2(&id, guid, 40) ||
      swprintf(temporary, 32768, L"%ls\\magnitude-incoming-%ls.exe", path, guid) < 0 ||
      swprintf(retired, 128, L"magnitude-retired-%ls.exe", guid) < 0) { error = ERROR_INVALID_DATA; goto done; }
  input = CreateFileW(source, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_FLAG_OPEN_REPARSE_POINT, NULL);
  if (input == INVALID_HANDLE_VALUE) { error = GetLastError(); goto done; }
  BY_HANDLE_FILE_INFORMATION info;
  if (!GetFileInformationByHandle(input, &info)) { error = GetLastError(); goto done; }
  if (info.nNumberOfLinks != 1 || (info.dwFileAttributes & (FILE_ATTRIBUTE_DIRECTORY | FILE_ATTRIBUTE_REPARSE_POINT))) { error = ERROR_INVALID_DATA; goto done; }
  error = magnitude_create_private_content(temporary);
  if (error) goto done;
  incoming = CreateFileW(temporary, GENERIC_WRITE | DELETE, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_FLAG_OPEN_REPARSE_POINT, NULL);
  if (incoming == INVALID_HANDLE_VALUE) { error = GetLastError(); DeleteFileW(temporary); goto done; }
  BYTE buffer[16384]; DWORD read = 0, written = 0;
  for (;;) {
    if (!ReadFile(input, buffer, sizeof(buffer), &read, NULL)) { error = GetLastError(); break; }
    if (!read) break;
    if (!WriteFile(incoming, buffer, read, &written, NULL) || written != read) { error = GetLastError(); if (!error) error = ERROR_WRITE_FAULT; break; }
  }
  if (!error && !FlushFileBuffers(incoming)) error = GetLastError();
  if (error) goto done;
  error = open_without_reparse(directory, L"magnitude.exe", DELETE | FILE_READ_ATTRIBUTES,
    FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, FILE_NON_DIRECTORY_FILE, &current);
  if (error == ERROR_FILE_NOT_FOUND) { current = INVALID_HANDLE_VALUE; error = ERROR_SUCCESS; }
  if (!error && current != INVALID_HANDLE_VALUE) {
    if (!GetFileInformationByHandle(current, &info)) error = GetLastError();
    else if (info.nNumberOfLinks != 1 || (info.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT)) error = ERROR_INVALID_DATA;
    if (!error) { error = rename_directory(current, directory, retired); moved = !error; }
  }
  if (!error) error = rename_directory(incoming, directory, L"magnitude.exe");
  if (error && moved) {
    DWORD rollback = rename_directory(current, directory, L"magnitude.exe");
    if (rollback) error = rollback;
  }
  if (!error && current != INVALID_HANDLE_VALUE) {
    FILE_DISPOSITION_INFO disposition = {TRUE};
    SetFileInformationByHandle(current, FileDispositionInfo, &disposition, sizeof(disposition));
  }
done:
  if (incoming != INVALID_HANDLE_VALUE) {
    if (error) { FILE_DISPOSITION_INFO disposition = {TRUE}; SetFileInformationByHandle(incoming, FileDispositionInfo, &disposition, sizeof(disposition)); }
    CloseHandle(incoming);
  }
  if (current != INVALID_HANDLE_VALUE) CloseHandle(current);
  if (input != INVALID_HANDLE_VALUE) CloseHandle(input);
  if (directory != INVALID_HANDLE_VALUE) CloseHandle(directory);
  return error;
}

__declspec(dllexport) DWORD WINAPI RemoveCliLauncher(LPCWSTR path) {
  if (!leaseHeld || !path) return ERROR_INVALID_PARAMETER;
  HANDLE directory = INVALID_HANDLE_VALUE, file = INVALID_HANDLE_VALUE;
  DWORD error = open_installation_directory(path, &directory);
  if (error == ERROR_FILE_NOT_FOUND || error == ERROR_PATH_NOT_FOUND) return ERROR_SUCCESS;
  if (error) return error;
  error = retire_cli_images(directory);
  BYTE buffer[16384];
  FILE_INFO_BY_HANDLE_CLASS query = FileIdBothDirectoryRestartInfo;
  while (!error) {
    if (!GetFileInformationByHandleEx(directory, query, buffer, sizeof(buffer))) {
      error = GetLastError();
      if (error == ERROR_NO_MORE_FILES) error = ERROR_SUCCESS;
      break;
    }
    FILE_ID_BOTH_DIR_INFO *entry = (FILE_ID_BOTH_DIR_INFO *)buffer;
    for (;;) {
      size_t count = entry->FileNameLength / sizeof(WCHAR);
      BOOL dot = (count == 1 && entry->FileName[0] == L'.') || (count == 2 && !wcsncmp(entry->FileName, L"..", 2));
      if (!dot && (count != 13 || wcsncmp(entry->FileName, L"magnitude.exe", 13))) { error = ERROR_DIR_NOT_EMPTY; break; }
      if (!entry->NextEntryOffset) break;
      entry = (FILE_ID_BOTH_DIR_INFO *)((BYTE *)entry + entry->NextEntryOffset);
    }
    query = FileIdBothDirectoryInfo;
  }
  if (!error) {
    error = open_without_reparse(directory, L"magnitude.exe", DELETE | FILE_READ_ATTRIBUTES,
      FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, FILE_NON_DIRECTORY_FILE, &file);
    if (error == ERROR_FILE_NOT_FOUND) { file = INVALID_HANDLE_VALUE; error = ERROR_SUCCESS; }
    if (!error && file != INVALID_HANDLE_VALUE) {
      BY_HANDLE_FILE_INFORMATION info;
      if (!GetFileInformationByHandle(file, &info)) error = GetLastError();
      else if (info.nNumberOfLinks != 1 || (info.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT)) error = ERROR_INVALID_DATA;
      FILE_DISPOSITION_INFO disposition = {TRUE};
      if (!error && !SetFileInformationByHandle(file, FileDispositionInfo, &disposition, sizeof(disposition))) error = GetLastError();
      CloseHandle(file);
    }
    FILE_DISPOSITION_INFO disposition = {TRUE};
    if (!error && !SetFileInformationByHandle(directory, FileDispositionInfo, &disposition, sizeof(disposition))) error = GetLastError();
  }
  CloseHandle(directory);
  return error;
}
