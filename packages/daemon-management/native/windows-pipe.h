#ifndef MAGNITUDE_WINDOWS_PIPE_H
#define MAGNITUDE_WINDOWS_PIPE_H
#include <windows.h>

typedef struct magnitude_private_pipe magnitude_private_pipe;
/* Creation installs a protected current-user DACL before publishing the endpoint.
 * The first listener rejects a pre-existing endpoint; later instances use the same name.
 */
DWORD magnitude_pipe_create(const WCHAR *name, BOOL first, magnitude_private_pipe **result);
DWORD magnitude_pipe_accept(magnitude_private_pipe *pipe);
DWORD magnitude_pipe_read(magnitude_private_pipe *pipe, void *buffer, DWORD capacity, DWORD *count);
DWORD magnitude_pipe_write(magnitude_private_pipe *pipe, const void *buffer, DWORD length, DWORD *count);
DWORD magnitude_pipe_client_pid(magnitude_private_pipe *pipe, DWORD *pid);
/* Thread-safe cancellation waits for pending OVERLAPPED storage to be released. */
void magnitude_pipe_close(magnitude_private_pipe *pipe);
/* Caller must retain the object until all users, including close waiters, have returned. */
void magnitude_pipe_destroy(magnitude_private_pipe *pipe);
#endif
