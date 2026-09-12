#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0A00
#endif
#include "windows-pipe.h"
#include "windows-security.h"
#include <sddl.h>
#include <stdio.h>
#include <stdlib.h>
#include <wchar.h>

struct magnitude_private_pipe {
  HANDLE handle;
  SRWLOCK lock;
  CONDITION_VARIABLE changed;
  DWORD active;
  BOOL closing;
};
DWORD magnitude_pipe_create(const WCHAR *name, BOOL first, magnitude_private_pipe **result) {
  static const WCHAR prefix[] = L"\\\\.\\pipe\\magnitude-";
  PSECURITY_DESCRIPTOR descriptor = NULL;
  if (!result) return ERROR_INVALID_PARAMETER;
  *result = NULL;
  if (!name || wcsncmp(name, prefix, (sizeof(prefix) / sizeof(WCHAR)) - 1) || wcslen(name) > 240)
    return ERROR_INVALID_NAME;
  DWORD error = magnitude_private_descriptor(FALSE, &descriptor);
  if (error) return error;
  SECURITY_ATTRIBUTES attributes = { (DWORD)sizeof(attributes), descriptor, FALSE };
  HANDLE handle = CreateNamedPipeW(name, PIPE_ACCESS_DUPLEX | FILE_FLAG_OVERLAPPED |
      (first ? FILE_FLAG_FIRST_PIPE_INSTANCE : 0), PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT | PIPE_REJECT_REMOTE_CLIENTS,
      17, 65536, 65536, 0, &attributes);
  error = handle == INVALID_HANDLE_VALUE ? GetLastError() : ERROR_SUCCESS;
  LocalFree(descriptor);
  if (error) return error;
  magnitude_private_pipe *pipe = calloc(1, sizeof(*pipe));
  if (!pipe) { CloseHandle(handle); return ERROR_NOT_ENOUGH_MEMORY; }
  pipe->handle = handle;
  InitializeSRWLock(&pipe->lock); InitializeConditionVariable(&pipe->changed);
  *result = pipe;
  return ERROR_SUCCESS;
}

typedef enum { PIPE_ACCEPT, PIPE_READ, PIPE_WRITE } pipe_operation;
static DWORD perform(magnitude_private_pipe *pipe, pipe_operation operation, void *buffer, DWORD size, DWORD *count) {
  if (!pipe || !count || (operation != PIPE_ACCEPT && (!buffer || !size || size > 65536))) return ERROR_INVALID_PARAMETER;
  *count = 0;
  OVERLAPPED overlapped = {0};
  overlapped.hEvent = CreateEventW(NULL, TRUE, FALSE, NULL);
  if (!overlapped.hEvent) return GetLastError();
  AcquireSRWLockExclusive(&pipe->lock);
  if (pipe->closing) { ReleaseSRWLockExclusive(&pipe->lock); CloseHandle(overlapped.hEvent); return ERROR_OPERATION_ABORTED; }
  ++pipe->active;
  HANDLE handle = pipe->handle;
  /* Issue under the lock: close cannot cancel just before this operation becomes pending. */
  BOOL done = operation == PIPE_ACCEPT ? ConnectNamedPipe(handle, &overlapped)
    : operation == PIPE_READ ? ReadFile(handle, buffer, size, count, &overlapped)
    : WriteFile(handle, buffer, size, count, &overlapped);
  DWORD error = done ? ERROR_SUCCESS : GetLastError();
  ReleaseSRWLockExclusive(&pipe->lock);
  if (error == ERROR_IO_PENDING) error = GetOverlappedResult(handle, &overlapped, count, TRUE) ? ERROR_SUCCESS : GetLastError();
  if ((operation == PIPE_ACCEPT && error == ERROR_PIPE_CONNECTED) ||
      (operation == PIPE_READ && (error == ERROR_BROKEN_PIPE || error == ERROR_NO_DATA))) error = ERROR_SUCCESS;
  CloseHandle(overlapped.hEvent);
  AcquireSRWLockExclusive(&pipe->lock);
  --pipe->active;
  WakeAllConditionVariable(&pipe->changed);
  ReleaseSRWLockExclusive(&pipe->lock);
  return error;
}
DWORD magnitude_pipe_accept(magnitude_private_pipe *pipe) { DWORD count; return perform(pipe, PIPE_ACCEPT, NULL, 0, &count); }
DWORD magnitude_pipe_read(magnitude_private_pipe *pipe, void *buffer, DWORD capacity, DWORD *count) { return perform(pipe, PIPE_READ, buffer, capacity, count); }
DWORD magnitude_pipe_write(magnitude_private_pipe *pipe, const void *buffer, DWORD length, DWORD *count) { return perform(pipe, PIPE_WRITE, (void *)buffer, length, count); }
DWORD magnitude_pipe_client_pid(magnitude_private_pipe *pipe, DWORD *pid) {
  if (!pipe || !pid) return ERROR_INVALID_PARAMETER;
  AcquireSRWLockExclusive(&pipe->lock);
  DWORD error = pipe->closing ? ERROR_OPERATION_ABORTED : GetNamedPipeClientProcessId(pipe->handle, pid) ? ERROR_SUCCESS : GetLastError();
  ReleaseSRWLockExclusive(&pipe->lock);
  return error;
}
void magnitude_pipe_close(magnitude_private_pipe *pipe) {
  if (!pipe) return;
  AcquireSRWLockExclusive(&pipe->lock);
  if (pipe->closing) {
    while (pipe->handle != INVALID_HANDLE_VALUE) SleepConditionVariableSRW(&pipe->changed, &pipe->lock, INFINITE, 0);
    ReleaseSRWLockExclusive(&pipe->lock); return;
  }
  pipe->closing = TRUE;
  CancelIoEx(pipe->handle, NULL);
  while (pipe->active) SleepConditionVariableSRW(&pipe->changed, &pipe->lock, INFINITE, 0);
  CloseHandle(pipe->handle);
  pipe->handle = INVALID_HANDLE_VALUE;
  WakeAllConditionVariable(&pipe->changed);
  ReleaseSRWLockExclusive(&pipe->lock);
}
void magnitude_pipe_destroy(magnitude_private_pipe *pipe) { if (pipe) { magnitude_pipe_close(pipe); free(pipe); } }
