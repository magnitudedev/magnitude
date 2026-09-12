#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0A00
#endif
#include "windows-job.h"
#include <stdlib.h>
#include <string.h>

void magnitude_owned_close(magnitude_owned_process *owned) {
  if (owned->job) CloseHandle(owned->job);
  if (owned->process) CloseHandle(owned->process);
  memset(owned, 0, sizeof(*owned));
}

DWORD magnitude_owned_spawn(const WCHAR *executable, WCHAR *command_line, void *environment,
    HANDLE input, HANDLE output, HANDLE error, magnitude_owned_process *result) {
  magnitude_owned_process owned = {0};
  STARTUPINFOEXW startup = {0};
  PROCESS_INFORMATION process = {0};
  JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits = {0};
  HANDLE standard[] = { input, output, error };
  HANDLE inherited[3];
  SIZE_T inherited_count = 0;
  SIZE_T bytes = 0;
  DWORD failure = ERROR_SUCCESS;
  BOOL attributes_initialized = FALSE;
  if (!result || !executable || !*executable || !command_line || !*command_line)
    return ERROR_INVALID_PARAMETER;
  memset(result, 0, sizeof(*result));
  for (SIZE_T index = 0; index < 3; ++index) {
    DWORD flags;
    if (!standard[index] || standard[index] == INVALID_HANDLE_VALUE ||
        !GetHandleInformation(standard[index], &flags) || !(flags & HANDLE_FLAG_INHERIT))
      return ERROR_INVALID_HANDLE;
    SIZE_T found = 0;
    while (found < inherited_count && inherited[found] != standard[index]) ++found;
    if (found == inherited_count) inherited[inherited_count++] = standard[index];
  }
  owned.job = CreateJobObjectW(NULL, NULL);
  if (!owned.job) return GetLastError();
  limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
  if (!SetInformationJobObject(owned.job, JobObjectExtendedLimitInformation, &limits, (DWORD)sizeof(limits)) ||
      !SetHandleInformation(owned.job, HANDLE_FLAG_INHERIT, 0)) {
    failure = GetLastError(); goto cleanup;
  }
  InitializeProcThreadAttributeList(NULL, 2, 0, &bytes);
  if (GetLastError() != ERROR_INSUFFICIENT_BUFFER || !bytes) {
    failure = ERROR_INVALID_PARAMETER; goto cleanup;
  }
  startup.lpAttributeList = malloc(bytes);
  if (!startup.lpAttributeList) { failure = ERROR_NOT_ENOUGH_MEMORY; goto cleanup; }
  if (!InitializeProcThreadAttributeList(startup.lpAttributeList, 2, 0, &bytes)) {
    failure = GetLastError(); goto cleanup;
  }
  attributes_initialized = TRUE;
  if (!UpdateProcThreadAttribute(startup.lpAttributeList, 0, PROC_THREAD_ATTRIBUTE_JOB_LIST,
          &owned.job, sizeof(owned.job), NULL, NULL) ||
      !UpdateProcThreadAttribute(startup.lpAttributeList, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
          inherited, inherited_count * sizeof(HANDLE), NULL, NULL)) {
    failure = GetLastError(); goto cleanup;
  }
  startup.StartupInfo.cb = (DWORD)sizeof(startup);
  startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
  startup.StartupInfo.hStdInput = input;
  startup.StartupInfo.hStdOutput = output;
  startup.StartupInfo.hStdError = error;
  /* Association occurs inside creation. No create-suspended/assign/resume orphan window. */
  if (!CreateProcessW(executable, command_line, NULL, NULL, TRUE,
      EXTENDED_STARTUPINFO_PRESENT | CREATE_UNICODE_ENVIRONMENT | CREATE_NO_WINDOW,
      environment, NULL, &startup.StartupInfo, &process)) {
    failure = GetLastError(); goto cleanup;
  }
  owned.process = process.hProcess;
  owned.pid = process.dwProcessId;
  CloseHandle(process.hThread);
cleanup:
  if (attributes_initialized) DeleteProcThreadAttributeList(startup.lpAttributeList);
  free(startup.lpAttributeList);
  if (failure) magnitude_owned_close(&owned);
  else *result = owned;
  return failure;
}

/* Identity observation follows acquisition so failure cannot discard the cleanup authority. */
DWORD magnitude_owned_creation(const magnitude_owned_process *owned, FILETIME *creation) {
  FILETIME exited, kernel, user;
  if (!owned || !owned->process || !creation) return ERROR_INVALID_HANDLE;
  return GetProcessTimes(owned->process, creation, &exited, &kernel, &user) ? ERROR_SUCCESS : GetLastError();
}
DWORD magnitude_owned_active(const magnitude_owned_process *owned, DWORD *count) {
  JOBOBJECT_BASIC_ACCOUNTING_INFORMATION information;
  if (!owned || !owned->job || !count) return ERROR_INVALID_HANDLE;
  if (!QueryInformationJobObject(owned->job, JobObjectBasicAccountingInformation,
      &information, (DWORD)sizeof(information), NULL)) return GetLastError();
  *count = information.ActiveProcesses;
  return ERROR_SUCCESS;
}
DWORD magnitude_owned_terminate(const magnitude_owned_process *owned, UINT exit_code) {
  if (!owned || !owned->job) return ERROR_INVALID_HANDLE;
  return TerminateJobObject(owned->job, exit_code) ? ERROR_SUCCESS : GetLastError();
}
DWORD magnitude_owned_exit(const magnitude_owned_process *owned, BOOL *exited, DWORD *exit_code) {
  if (!owned || !owned->process || !exited || !exit_code) return ERROR_INVALID_HANDLE;
  DWORD wait = WaitForSingleObject(owned->process, 0);
  if (wait == WAIT_FAILED) return GetLastError();
  *exited = wait == WAIT_OBJECT_0;
  if (!*exited) return ERROR_SUCCESS;
  return GetExitCodeProcess(owned->process, exit_code) ? ERROR_SUCCESS : GetLastError();
}
DWORD magnitude_owned_validate_current(void) {
  BOOL member = FALSE;
  JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits;
  if (!IsProcessInJob(GetCurrentProcess(), NULL, &member)) return GetLastError();
  if (!member) return ERROR_ACCESS_DENIED;
  if (!QueryInformationJobObject(NULL, JobObjectExtendedLimitInformation, &limits, (DWORD)sizeof(limits), NULL)) return GetLastError();
  DWORD flags = limits.BasicLimitInformation.LimitFlags;
  return (flags & JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE) &&
    !(flags & (JOB_OBJECT_LIMIT_BREAKAWAY_OK | JOB_OBJECT_LIMIT_SILENT_BREAKAWAY_OK)) ? ERROR_SUCCESS : ERROR_ACCESS_DENIED;
}
