#include <windows.h>
#include <stdio.h>
#include <string.h>
#include <wchar.h>
#include "windows-security.h"

#define RUN_KEY L"Software\\Microsoft\\Windows\\CurrentVersion\\Run"
#define APPROVAL_KEY L"Software\\Microsoft\\Windows\\CurrentVersion\\Explorer\\StartupApproved\\Run"
#define STARTUP_NAME L"dev.magnitude.desktop"

static void require(BOOL condition, const char *message) {
  if (!condition) { fprintf(stderr, "%s (Windows error %lu)\n", message, GetLastError()); ExitProcess(1); }
}
static void require_success(DWORD error, const char *message) {
  SetLastError(error); require(error == ERROR_SUCCESS, message);
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

static void write_fixture(LPCWSTR path, LPCWSTR contents) {
  HANDLE file = CreateFileW(path, GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
  require(file != INVALID_HANDLE_VALUE, "create fixture file");
  DWORD bytes = (DWORD)(wcslen(contents) * sizeof(WCHAR)), written = 0;
  require(WriteFile(file, contents, bytes, &written, NULL) && written == bytes, "write fixture file");
  CloseHandle(file);
}
static void check_inventory(HMODULE library) {
  typedef DWORD (WINAPI *ValidateInstallation)(LPCWSTR, LPCWSTR);
  ValidateInstallation validate;
  FARPROC symbol = GetProcAddress(library, "ValidateOwnedInstallation");
  require(symbol != NULL && sizeof(symbol) == sizeof(validate), "resolve inventory validation export");
  memcpy(&validate, &symbol, sizeof(validate));
  WCHAR root[32768], resources[32768], inventory[32768], executable[32768], uninstaller[32768], unexpected[32768];
  require(GetCurrentDirectoryW(32768, root) > 0, "fixture working directory");
  require(wcslen(root) < 32000, "bounded fixture working directory");
  wcscat(root, L"\\owned installation");
  PSECURITY_DESCRIPTOR descriptor = NULL;
  require(magnitude_private_descriptor(TRUE, &descriptor) == ERROR_SUCCESS, "private fixture descriptor");
  SECURITY_ATTRIBUTES attributes = {sizeof(attributes), descriptor, FALSE};
  require(CreateDirectoryW(root, &attributes), "create private installation fixture");
  LocalFree(descriptor);
  swprintf(resources, 32768, L"%ls\\resources", root);
  swprintf(inventory, 32768, L"%ls\\installation-files.txt", resources);
  swprintf(executable, 32768, L"%ls\\Magnitude.exe", root);
  swprintf(uninstaller, 32768, L"%ls\\Uninstall Magnitude.exe", root);
  swprintf(unexpected, 32768, L"%ls\\user notes.txt", root);
  require(CreateDirectoryW(resources, NULL), "create resources fixture");
  LPCWSTR valid = L"\xFEFFmagnitude-installation-v1\n1.2.3\nF\tMagnitude.exe\nF\tUninstall Magnitude.exe\nF\tresources\\installation-files.txt\nD\tresources\n";
  write_fixture(inventory, valid); write_fixture(executable, L"application"); write_fixture(uninstaller, L"uninstaller");
  require(validate(root, L"1.2.3") == ERROR_SUCCESS, "complete owned inventory accepted");
  require(validate(root, L"1.2.4") != ERROR_SUCCESS, "wrong installed version rejected");
  write_fixture(unexpected, L"preserve my notes");
  require(validate(root, L"1.2.3") != ERROR_SUCCESS, "unknown files rejected");
  require(GetFileAttributesW(unexpected) != INVALID_FILE_ATTRIBUTES, "unknown files preserved");
  require(DeleteFileW(unexpected), "remove exact unknown fixture");
  require(DeleteFileW(executable), "remove exact payload fixture");
  require(validate(root, L"1.2.3") != ERROR_SUCCESS, "missing payload rejected");
  require(CreateHardLinkW(executable, uninstaller, NULL), "create hard-link fixture");
  require(validate(root, L"1.2.3") != ERROR_SUCCESS, "hard-linked payload rejected");
  require(DeleteFileW(executable), "remove fixture hard link"); write_fixture(executable, L"application");
  write_fixture(inventory, L"\xFEFFmagnitude-installation-v1\n1.2.3\nF\t..\\outside\n");
  require(validate(root, L"1.2.3") != ERROR_SUCCESS, "path traversal rejected");
  write_fixture(inventory, L"\xFEFFmagnitude-installation-v1\n1.2.3\nF\tMagnitude.exe\nF\tMAGNITUDE.EXE\nF\tUninstall Magnitude.exe\nF\tresources\\installation-files.txt\nD\tresources\n");
  require(validate(root, L"1.2.3") != ERROR_SUCCESS, "case-insensitive duplicate rejected");
  write_fixture(inventory, valid);
  require(validate(root, L"1.2.3") == ERROR_SUCCESS, "valid inventory remains usable after rejection");
  require(DeleteFileW(inventory) && DeleteFileW(executable) && DeleteFileW(uninstaller), "remove exact payload fixtures");
  require(RemoveDirectoryW(resources) && RemoveDirectoryW(root), "remove empty fixture directories");
}

static void fill_installation(LPCWSTR root, LPCWSTR version) {
  WCHAR path[32768], text[1024];
  swprintf(path, 32768, L"%ls\\resources", root);
  require(CreateDirectoryW(path, NULL), "create replacement resources");
  swprintf(path, 32768, L"%ls\\Magnitude.exe", root); write_fixture(path, version);
  swprintf(path, 32768, L"%ls\\Uninstall Magnitude.exe", root); write_fixture(path, version);
  swprintf(path, 32768, L"%ls\\resources\\installation-files.txt", root);
  swprintf(text, 1024, L"\xFEFFmagnitude-installation-v1\n%ls\nF\tMagnitude.exe\nF\tUninstall Magnitude.exe\nF\tresources\\installation-files.txt\nD\tresources\n", version);
  write_fixture(path, text);
}
static void remove_installation_fixture(LPCWSTR root) {
  WCHAR path[32768];
  LPCWSTR files[] = {L"Magnitude.exe", L"Uninstall Magnitude.exe", L"resources\\installation-files.txt"};
  for (size_t index = 0; index < sizeof(files) / sizeof(files[0]); ++index) {
    swprintf(path, 32768, L"%ls\\%ls", root, files[index]); require(DeleteFileW(path), "remove exact installation fixture file");
  }
  swprintf(path, 32768, L"%ls\\resources", root); require(RemoveDirectoryW(path), "remove empty fixture resources");
  require(RemoveDirectoryW(root), "remove empty installation fixture");
}
#define RESOLVE_FUNCTION(library, variable, name) do { \
  FARPROC pointer = GetProcAddress(library, name); \
  require(pointer != NULL && sizeof(pointer) == sizeof(variable), "resolve " name); \
  memcpy(&variable, &pointer, sizeof(variable)); \
} while (0)
static void check_replacement(HMODULE library) {
  DWORD (WINAPI *hold)(void), (WINAPI *rollback)(void);
  DWORD (WINAPI *create_stage)(LPWSTR, DWORD);
  DWORD (WINAPI *begin)(LPCWSTR, LPCWSTR, LPCWSTR);
  DWORD (WINAPI *finish)(LPCWSTR);
  DWORD (WINAPI *validate)(LPCWSTR, LPCWSTR);
  RESOLVE_FUNCTION(library, hold, "HoldOwnership");
  RESOLVE_FUNCTION(library, create_stage, "CreateStage");
  RESOLVE_FUNCTION(library, begin, "BeginReplacement");
  RESOLVE_FUNCTION(library, rollback, "RollbackReplacement");
  RESOLVE_FUNCTION(library, finish, "FinishReplacement");
  RESOLVE_FUNCTION(library, validate, "ValidateOwnedInstallation");
  require(hold() == ERROR_SUCCESS, "hold actual application lease");
  WCHAR payload[32768], root[32768], file[32768];
  require(create_stage(payload, 32768) == ERROR_SUCCESS, "prepare actual private installation stage");
  require(GetCurrentDirectoryW(32768, root) > 0 && wcslen(root) < 32000, "replacement fixture working directory");
  wcscat(root, L"\\replacement installation");
  PSECURITY_DESCRIPTOR descriptor = NULL;
  require(magnitude_private_descriptor(TRUE, &descriptor) == ERROR_SUCCESS, "replacement fixture descriptor");
  SECURITY_ATTRIBUTES attributes = {sizeof(attributes), descriptor, FALSE};
  require(CreateDirectoryW(root, &attributes), "create old private installation"); LocalFree(descriptor);
  fill_installation(root, L"1.2.3"); fill_installation(payload, L"1.2.4");
  require(begin(root, L"1.2.2", L"1.2.4") != ERROR_SUCCESS, "wrong old version cannot replace installation");
  require(validate(root, L"1.2.3") == ERROR_SUCCESS, "rejected replacement preserves old payload");
  require_success(begin(root, L"1.2.3", L"1.2.4"), "publish replacement by native directory handles");
  require_success(rollback(), "roll back unpublished registration");
  require(validate(root, L"1.2.3") == ERROR_SUCCESS, "rollback restores old version");
  require(validate(payload, L"1.2.4") == ERROR_SUCCESS, "rollback retains staged new version");
  require_success(begin(root, L"1.2.3", L"1.2.4"), "retry replacement after rollback");
  require_success(finish(L"1.2.3"), "retire exact old inventory after commit");
  require(validate(root, L"1.2.4") == ERROR_SUCCESS, "committed replacement has complete new inventory");
  swprintf(file, 32768, L"%ls\\Magnitude.exe", root); require(DeleteFileW(file), "remove new fixture executable");
  swprintf(file, 32768, L"%ls\\Uninstall Magnitude.exe", root); require(DeleteFileW(file), "remove new fixture uninstaller");
  swprintf(file, 32768, L"%ls\\resources\\installation-files.txt", root); require(DeleteFileW(file), "remove new fixture inventory");
  swprintf(file, 32768, L"%ls\\resources", root); require(RemoveDirectoryW(file), "remove new fixture resources");
  require(RemoveDirectoryW(root), "remove empty replacement fixture");
}

/* Separate invocations exit with native handles still held, reproducing an
   installer interruption without a test-only production failure switch. */
static void check_interrupted_replacement(HMODULE library, LPCWSTR mode) {
  DWORD (WINAPI *hold)(void), (WINAPI *cleanup)(void);
  DWORD (WINAPI *create_stage)(LPWSTR, DWORD);
  DWORD (WINAPI *begin)(LPCWSTR, LPCWSTR, LPCWSTR);
  DWORD (WINAPI *recover)(LPCWSTR, LPCWSTR);
  DWORD (WINAPI *validate)(LPCWSTR, LPCWSTR);
  RESOLVE_FUNCTION(library, hold, "HoldOwnership");
  RESOLVE_FUNCTION(library, cleanup, "CleanupStage");
  RESOLVE_FUNCTION(library, create_stage, "CreateStage");
  RESOLVE_FUNCTION(library, begin, "BeginReplacement");
  RESOLVE_FUNCTION(library, recover, "RecoverReplacement");
  RESOLVE_FUNCTION(library, validate, "ValidateOwnedInstallation");
  require_success(hold(), "acquire interrupted-installation lease");
  WCHAR root[32768], payload[32768];
  require(GetCurrentDirectoryW(32768, root) > 0 && wcslen(root) < 32000, "recovery fixture working directory");
  wcscat(root, L"\\interrupted installation");
  BOOL setup = !wcsncmp(mode, L"setup-", 6);
  require(setup || !wcsncmp(mode, L"recover-", 8), "recognized interruption mode");
  LPCWSTR scenario = setup ? mode + 6 : mode + 8;
  require(!wcscmp(scenario, L"old-moved") || !wcscmp(scenario, L"uncommitted") || !wcscmp(scenario, L"committed"), "recognized interruption scenario");
  if (setup) {
    require_success(create_stage(payload, 32768), "create interrupted extraction stage");
    PSECURITY_DESCRIPTOR descriptor = NULL;
    require_success(magnitude_private_descriptor(TRUE, &descriptor), "private interrupted fixture descriptor");
    SECURITY_ATTRIBUTES attributes = {sizeof(attributes), descriptor, FALSE};
    require(CreateDirectoryW(root, &attributes), "create interrupted old installation"); LocalFree(descriptor);
    fill_installation(root, L"1.2.3"); fill_installation(payload, L"1.2.4");
    if (!wcscmp(scenario, L"old-moved")) {
      WCHAR previous[32768]; wcscpy(previous, payload);
      WCHAR *leaf = wcsrchr(previous, L'\\'); require(leaf != NULL, "stage has a parent"); wcscpy(leaf + 1, L"previous");
      require(MoveFileExW(root, previous, 0), "interrupt after old directory moved");
    } else require_success(begin(root, L"1.2.3", L"1.2.4"), "interrupt after new directory published");
    ExitProcess(0);
  }
  LPCWSTR registered = !wcscmp(scenario, L"committed") ? L"1.2.4" : L"1.2.3";
  require_success(recover(root, registered), "recover interrupted replacement using registered version");
  require_success(validate(root, registered), "recovery publishes the registered complete version");
  require_success(recover(root, registered), "recovery is repeatable");
  remove_installation_fixture(root);
  require_success(create_stage(payload, 32768), "previous installation is no longer treated as scratch");
  require_success(cleanup(), "clean remaining extraction scratch");
  puts("PASS interrupted installation recovery");
}

int wmain(int argc, wchar_t **argv) {
  require(argc == 2 || argc == 3, "expected absolute helper DLL path and optional interruption mode");
  HMODULE library = LoadLibraryW(argv[1]); require(library != NULL, "load actual x86 helper DLL");
  if (argc == 3) { check_interrupted_replacement(library, argv[2]); return 0; }
  check_inventory(library);
  check_replacement(library);
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
