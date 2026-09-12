/* Native execution is required; cross-compilation alone is not an acceptance receipt. */
#include "windows-process-retirement.h"
#include "windows-job.h"
#include <sddl.h>
#include <stdio.h>
#include <stdlib.h>
#include <wchar.h>

static void require(BOOL condition, const char *message) {
  if (!condition) { fprintf(stderr, "%s (Windows error %lu)\n", message, GetLastError()); ExitProcess(1); }
}
static TOKEN_USER *current_user(void) {
  HANDLE token = NULL; DWORD size = 0;
  require(OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &token), "open caller token");
  require(!GetTokenInformation(token, TokenUser, NULL, 0, &size) && GetLastError() == ERROR_INSUFFICIENT_BUFFER, "size caller SID");
  TOKEN_USER *user = malloc(size); require(user != NULL, "allocate caller SID");
  require(GetTokenInformation(token, TokenUser, user, size, &size), "read caller SID");
  CloseHandle(token); return user;
}
int wmain(int argc, WCHAR **argv) {
  (void)argv;
  if (argc > 1) { Sleep(INFINITE); return 0; }
  WCHAR executable[32768], command[32768];
  require(GetModuleFileNameW(NULL, executable, 32768) != 0, "resolve fixture executable");
  require(swprintf(command, 32768, L"\"%ls\" --sleep", executable) > 0, "encode child command");
  SECURITY_ATTRIBUTES security = { (DWORD)sizeof(security), NULL, TRUE };
  HANDLE input = CreateFileW(L"NUL", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, &security, OPEN_EXISTING, 0, NULL);
  HANDLE output = CreateFileW(L"NUL", GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, &security, OPEN_EXISTING, 0, NULL);
  require(input != INVALID_HANDLE_VALUE && output != INVALID_HANDLE_VALUE, "create fixture stdio");
  magnitude_owned_process child = {0};
  require(magnitude_owned_spawn(executable, command, NULL, input, output, output, &child) == ERROR_SUCCESS, "contain fixture child");
  FILETIME creation;
  require(magnitude_owned_creation(&child, &creation) == ERROR_SUCCESS, "capture creation time");
  TOKEN_USER *user = current_user();
  magnitude_migration_process rejected = {0}, retained = {0};
  FILETIME wrong = creation; wrong.dwLowDateTime ^= 1;
  require(magnitude_migration_open(child.pid, &wrong, executable, user->User.Sid, &rejected) == ERROR_INVALID_DATA && !rejected.process, "reject reused identity");
  require(magnitude_migration_open(child.pid, &creation, L"C:\\wrong-executable.exe", user->User.Sid, &rejected) == ERROR_INVALID_DATA && !rejected.process, "reject wrong executable");
  PSID wrong_user = NULL;
  require(ConvertStringSidToSidW(L"S-1-5-18", &wrong_user), "make different user SID");
  require(!EqualSid(wrong_user, user->User.Sid), "fixture must run as ordinary user");
  require(magnitude_migration_open(child.pid, &creation, executable, wrong_user, &rejected) == ERROR_ACCESS_DENIED && !rejected.process, "reject wrong user");
  LocalFree(wrong_user);
  require(WaitForSingleObject(child.process, 0) == WAIT_TIMEOUT, "rejected identities do not terminate child");
  require(magnitude_migration_open(child.pid, &creation, executable, user->User.Sid, &retained) == ERROR_SUCCESS, "acquire exact migration handle");
  DWORD flags;
  require(GetHandleInformation(retained.process, &flags) && !(flags & HANDLE_FLAG_INHERIT), "retirement handle is not inherited");
  magnitude_migration_close(&retained); magnitude_migration_close(&retained);
  require(WaitForSingleObject(child.process, 0) == WAIT_TIMEOUT, "closing authority does not terminate child");
  require(magnitude_migration_start_retirement(&retained) == ERROR_INVALID_HANDLE, "closed authority cannot terminate");
  require(magnitude_migration_open(child.pid, &creation, executable, user->User.Sid, &retained) == ERROR_SUCCESS, "reacquire exact migration handle");
  require(magnitude_migration_start_retirement(&retained) == ERROR_SUCCESS, "initiate exact retirement");
  require(WaitForSingleObject(child.process, 10000) == WAIT_OBJECT_0, "child exits after exact retirement");
  BOOL exited = FALSE;
  require(magnitude_migration_exited(&retained, &exited) == ERROR_SUCCESS && exited, "retained exit proof");
  require(magnitude_migration_start_retirement(&retained) == ERROR_SUCCESS, "repeated retirement is idempotent");
  magnitude_migration_close(&retained); magnitude_owned_close(&child);
  CloseHandle(input); CloseHandle(output); free(user);
  puts("PASS exact process retirement rejects identity mismatch and retains exit proof"); return 0;
}
