#ifndef MAGNITUDE_WINDOWS_PROCESS_RETIREMENT_H
#define MAGNITUDE_WINDOWS_PROCESS_RETIREMENT_H
#include <windows.h>

/* Migration authority is separate from passive observation and fresh child ownership.
 * The caller must have established ancestry and journaled the exact process first.
 * This capability proves only one process's retirement, never a whole tree's.
 */
typedef struct { HANDLE process; } magnitude_migration_process;
DWORD magnitude_migration_open(DWORD pid, const FILETIME *creation,
    const WCHAR *executable, PSID user, magnitude_migration_process *result);
DWORD magnitude_migration_start_retirement(const magnitude_migration_process *process);
DWORD magnitude_migration_exited(const magnitude_migration_process *process, BOOL *exited);
/* Closing observation does not terminate a process or authorize any new operation. */
void magnitude_migration_close(magnitude_migration_process *process);
#endif
