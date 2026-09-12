#include <windows.h>
#include <stdio.h>
#include <string.h>
#include <wchar.h>

#define RUN_KEY L"Software\\Microsoft\\Windows\\CurrentVersion\\Run"
#define APPROVAL_KEY L"Software\\Microsoft\\Windows\\CurrentVersion\\Explorer\\StartupApproved\\Run"
#define STARTUP_NAME L"dev.magnitude.desktop"

static void require(BOOL condition, const char *message) {
  if (!condition) { fprintf(stderr, "%s (Windows error %lu)\n", message, GetLastError()); ExitProcess(1); }
}
static void set_value(HKEY root, LPCWSTR path, LPCWSTR name, LPCWSTR value) {
  HKEY key;
  require(RegCreateKeyExW(root, path, 0, NULL, 0, KEY_ALL_ACCESS, NULL, &key, NULL) == ERROR_SUCCESS, "create fixture key");
  require(RegSetValueExW(key, name, 0, REG_SZ, (const BYTE *)value,
    (DWORD)((wcslen(value) + 1) * sizeof(WCHAR))) == ERROR_SUCCESS, "write fixture value");
  RegCloseKey(key);
}
static void check_value(HKEY root, LPCWSTR path, LPCWSTR name, LPCWSTR expected) {
  HKEY key;
  require(RegOpenKeyExW(root, path, 0, KEY_QUERY_VALUE, &key) == ERROR_SUCCESS, "open fixture value");
  WCHAR value[1024]; DWORD bytes = sizeof(value), type = 0;
  LONG error = RegQueryValueExW(key, name, NULL, &type, (BYTE *)value, &bytes);
  RegCloseKey(key);
  if (!expected) { require(error == ERROR_FILE_NOT_FOUND, "owned value removed"); return; }
  require(error == ERROR_SUCCESS && type == REG_SZ && bytes == (wcslen(expected) + 1) * sizeof(WCHAR), "preserved value shape");
  require(wcscmp(value, expected) == 0, "preserved value contents");
}

int wmain(int argc, wchar_t **argv) {
  require(argc == 2, "expected absolute helper DLL path");
  HMODULE library = LoadLibraryW(argv[1]); require(library != NULL, "load actual x86 helper DLL");
  typedef DWORD (WINAPI *RemoveStartup)(LPCWSTR);
  RemoveStartup remove_startup;
  FARPROC symbol = GetProcAddress(library, "RemoveOwnedStartup");
  require(symbol != NULL && sizeof(symbol) == sizeof(remove_startup), "resolve undecorated startup export");
  memcpy(&remove_startup, &symbol, sizeof(remove_startup));

  // Override HKCU only in this process. The DLL never sees real startup entries.
  WCHAR fixture[256];
  require(swprintf(fixture, 256, L"Software\\MagnitudeInstallerTest-%lu-%llu",
    GetCurrentProcessId(), GetTickCount64()) > 0, "fixture registry path");
  HKEY original, isolated; DWORD disposition;
  require(RegOpenKeyExW(HKEY_CURRENT_USER, L"", 0, KEY_ALL_ACCESS, &original) == ERROR_SUCCESS, "retain real registry root");
  require(RegCreateKeyExW(original, fixture, 0, NULL, 0, KEY_ALL_ACCESS, NULL, &isolated, &disposition) == ERROR_SUCCESS &&
    disposition == REG_CREATED_NEW_KEY, "create unique registry fixture");
  require(RegOverridePredefKey(HKEY_CURRENT_USER, isolated) == ERROR_SUCCESS, "isolate registry operations");

  LPCWSTR executable = L"C:\\Users\\Fixture User\\Magnitude\\Magnitude.exe";
  LPCWSTR command = L"\"C:\\Users\\Fixture User\\Magnitude\\Magnitude.exe\" --background";
  require(remove_startup(executable) == ERROR_SUCCESS, "absent startup is already removed");
  set_value(isolated, RUN_KEY, STARTUP_NAME, command);
  set_value(isolated, APPROVAL_KEY, STARTUP_NAME, L"owned approval");
  set_value(isolated, RUN_KEY, L"unrelated", L"keep");
  set_value(isolated, APPROVAL_KEY, L"unrelated", L"keep");
  require(remove_startup(executable) == ERROR_SUCCESS, "remove exact spaced-path startup");
  check_value(isolated, RUN_KEY, STARTUP_NAME, NULL);
  check_value(isolated, APPROVAL_KEY, STARTUP_NAME, NULL);
  check_value(isolated, RUN_KEY, L"unrelated", L"keep");
  check_value(isolated, APPROVAL_KEY, L"unrelated", L"keep");

  LPCWSTR replacements[] = { L"\"C:\\Other\\Magnitude.exe\" --background",
    L"\"C:\\Users\\Fixture User\\Magnitude\\Magnitude.exe\" --other", L"malformed" };
  for (size_t index = 0; index < sizeof(replacements) / sizeof(replacements[0]); ++index) {
    set_value(isolated, RUN_KEY, STARTUP_NAME, replacements[index]);
    set_value(isolated, APPROVAL_KEY, STARTUP_NAME, L"keep approval");
    require(remove_startup(executable) == ERROR_SUCCESS, "external replacement is preserved");
    check_value(isolated, RUN_KEY, STARTUP_NAME, replacements[index]);
    check_value(isolated, APPROVAL_KEY, STARTUP_NAME, L"keep approval");
  }
  set_value(isolated, RUN_KEY, STARTUP_NAME, command);
  require(remove_startup(executable) == ERROR_SUCCESS, "remove owned replacement");
  set_value(isolated, APPROVAL_KEY, STARTUP_NAME, L"orphan approval");
  require(remove_startup(executable) == ERROR_SUCCESS, "absent command establishes no approval ownership");
  check_value(isolated, APPROVAL_KEY, STARTUP_NAME, L"orphan approval");

  require(RegOverridePredefKey(HKEY_CURRENT_USER, NULL) == ERROR_SUCCESS, "restore registry root");
  RegCloseKey(isolated);
  require(RegDeleteTreeW(original, fixture) == ERROR_SUCCESS, "remove exact fixture registry tree");
  RegCloseKey(original); FreeLibrary(library);
  puts("PASS actual installer DLL: exact startup removal and external/unrelated/orphan configuration preservation");
  return 0;
}
