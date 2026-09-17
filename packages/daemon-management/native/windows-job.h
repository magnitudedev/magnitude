#ifndef MAGNITUDE_WINDOWS_JOB_H
#define MAGNITUDE_WINDOWS_JOB_H
#include <windows.h>

/* The actual parent owns both handles. Never inherit or duplicate the job into a child. */
typedef struct {
  HANDLE job;
  HANDLE process;
  DWORD pid;
} magnitude_owned_process;

/* command_line is mutable per CreateProcessW. Environment is a UTF-16 double-NUL block,
 * or NULL to inherit the parent environment. Only the three standard handles are inherited.
 * All three must be real inheritable handles; shared stdout/stderr is supported.
 */
DWORD magnitude_owned_spawn(const WCHAR *executable, WCHAR *command_line, void *environment,
    HANDLE input, HANDLE output, HANDLE error, magnitude_owned_process *result);
DWORD magnitude_owned_creation(const magnitude_owned_process *owned, FILETIME *creation);
DWORD magnitude_owned_active(const magnitude_owned_process *owned, DWORD *count);
DWORD magnitude_owned_terminate(const magnitude_owned_process *owned, UINT exit_code);
/* Root exit and complete job retirement are independent observations. */
DWORD magnitude_owned_exit(const magnitude_owned_process *owned, BOOL *exited, DWORD *exit_code);
/* Admission requires immediate-job parent-loss containment and no permitted breakaway. */
DWORD magnitude_owned_validate_current(void);
/* Closing kills remaining descendants; it does not by itself prove they have finished exiting. */
void magnitude_owned_close(magnitude_owned_process *owned);
#endif
