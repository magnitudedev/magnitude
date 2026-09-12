#include "windows-process-retirement.h"
#include <stdlib.h>

static DWORD process_user(HANDLE process, TOKEN_USER **result) {
  HANDLE token = NULL; DWORD bytes = 0, error = ERROR_SUCCESS;
  *result = NULL;
  if (!OpenProcessToken(process, TOKEN_QUERY, &token)) return GetLastError();
  if (GetTokenInformation(token, TokenUser, NULL, 0, &bytes) ||
      GetLastError() != ERROR_INSUFFICIENT_BUFFER || !bytes || bytes > 65536) {
    error = ERROR_INVALID_DATA; goto done;
  }
  *result = malloc(bytes);
  if (!*result) { error = ERROR_NOT_ENOUGH_MEMORY; goto done; }
  if (!GetTokenInformation(token, TokenUser, *result, bytes, &bytes)) {
    error = GetLastError(); free(*result); *result = NULL;
  }
done:
  CloseHandle(token); return error;
}

DWORD magnitude_migration_open(DWORD pid, const FILETIME *creation,
    const WCHAR *executable, PSID user, magnitude_migration_process *result) {
  if (!result) return ERROR_INVALID_PARAMETER;
  result->process = NULL;
  if (!pid || pid == GetCurrentProcessId() || !creation || !executable || !*executable ||
      !user || !IsValidSid(user)) return ERROR_INVALID_PARAMETER;
  HANDLE handle = OpenProcess(SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_TERMINATE, FALSE, pid);
  if (!handle) return GetLastError();
  DWORD error = ERROR_SUCCESS, length = 32768;
  FILETIME actual, exit, kernel, usage;
  WCHAR *image = NULL; TOKEN_USER *owner = NULL, *caller = NULL;
  if (!GetProcessTimes(handle, &actual, &exit, &kernel, &usage)) { error = GetLastError(); goto done; }
  if (CompareFileTime(&actual, creation) != 0) { error = ERROR_INVALID_DATA; goto done; }
  image = malloc((size_t)length * sizeof(WCHAR));
  if (!image) { error = ERROR_NOT_ENOUGH_MEMORY; goto done; }
  if (!QueryFullProcessImageNameW(handle, 0, image, &length)) { error = GetLastError(); goto done; }
  if (CompareStringOrdinal(image, (int)length, executable, -1, TRUE) != CSTR_EQUAL) {
    error = ERROR_INVALID_DATA; goto done;
  }
  error = process_user(handle, &owner); if (error) goto done;
  error = process_user(GetCurrentProcess(), &caller); if (error) goto done;
  if (!EqualSid(owner->User.Sid, user) || !EqualSid(caller->User.Sid, user)) {
    error = ERROR_ACCESS_DENIED; goto done;
  }
  /* OpenProcess returned a non-inheritable handle. Retention prevents PID reuse.
   * Recheck exit only through this same handle; never reopen by PID to terminate.
   */
  result->process = handle; handle = NULL;
done:
  free(image); free(owner); free(caller); if (handle) CloseHandle(handle); return error;
}

DWORD magnitude_migration_exited(const magnitude_migration_process *process, BOOL *exited) {
  if (!process || !process->process || !exited) return ERROR_INVALID_HANDLE;
  DWORD status = WaitForSingleObject(process->process, 0);
  if (status != WAIT_TIMEOUT && status != WAIT_OBJECT_0)
    return status == WAIT_FAILED ? GetLastError() : ERROR_INVALID_DATA;
  *exited = status == WAIT_OBJECT_0; return ERROR_SUCCESS;
}

DWORD magnitude_migration_start_retirement(const magnitude_migration_process *process) {
  BOOL exited = FALSE;
  DWORD error = magnitude_migration_exited(process, &exited);
  if (error || exited) return error;
  if (TerminateProcess(process->process, 1)) return ERROR_SUCCESS;
  error = GetLastError();
  /* Exit can race with TerminateProcess, which then reports access denied. */
  DWORD observed = magnitude_migration_exited(process, &exited);
  return !observed && exited ? ERROR_SUCCESS : error;
}

void magnitude_migration_close(magnitude_migration_process *process) {
  if (process && process->process) { CloseHandle(process->process); process->process = NULL; }
}
