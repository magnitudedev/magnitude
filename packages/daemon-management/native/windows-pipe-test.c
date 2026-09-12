#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0A00
#endif
#include "windows-pipe.h"
#include <aclapi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>

static void require(BOOL value, const char *message) {
  if (!value) { fprintf(stderr, "%s (Windows error %lu)\n", message, GetLastError()); ExitProcess(1); }
}
static void private_acl(HANDLE client) {
  PSECURITY_DESCRIPTOR descriptor = NULL; PACL acl = NULL;
  require(GetSecurityInfo(client, SE_KERNEL_OBJECT, DACL_SECURITY_INFORMATION, NULL, NULL, &acl, NULL, &descriptor) == ERROR_SUCCESS, "inspect pipe DACL");
  SECURITY_DESCRIPTOR_CONTROL control; DWORD revision;
  require(GetSecurityDescriptorControl(descriptor, &control, &revision) && (control & SE_DACL_PROTECTED), "pipe DACL cannot inherit permissions");
  require(acl && acl->AceCount == 1, "exactly one allowed user");
  ACCESS_ALLOWED_ACE *ace = NULL;
  require(GetAce(acl, 0, (void **)&ace) && ace->Header.AceType == ACCESS_ALLOWED_ACE_TYPE, "user allow ACE");
  HANDLE token; DWORD bytes = 0;
  require(OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &token), "read current user");
  GetTokenInformation(token, TokenUser, NULL, 0, &bytes);
  TOKEN_USER *user = malloc(bytes); require(user != NULL, "allocate user information");
  require(GetTokenInformation(token, TokenUser, user, bytes, &bytes) && EqualSid(user->User.Sid, &ace->SidStart), "only current user has pipe access");
  free(user); CloseHandle(token); LocalFree(descriptor);
}
typedef struct {
  magnitude_private_pipe *pipe;
  HANDLE started;
  DWORD operation;
  DWORD error;
} operation;
static DWORD WINAPI operate(void *argument) {
  operation *work = argument; char buffer[65536] = {0}; DWORD count;
  SetEvent(work->started);
  if (work->operation == 0) work->error = magnitude_pipe_accept(work->pipe);
  else if (work->operation == 1) work->error = magnitude_pipe_read(work->pipe, buffer, sizeof(buffer), &count);
  else do { work->error = magnitude_pipe_write(work->pipe, buffer, sizeof(buffer), &count); } while (!work->error);
  return 0;
}
static HANDLE client_for(const WCHAR *name) {
  HANDLE client = CreateFileW(name, GENERIC_READ | GENERIC_WRITE | READ_CONTROL, 0, NULL, OPEN_EXISTING, 0, NULL);
  require(client != INVALID_HANDLE_VALUE, "connect current-user client"); return client;
}
int main(void) {
  WCHAR name[128];
  require(swprintf(name, 128, L"\\\\.\\pipe\\magnitude-test-%lu-%llu", GetCurrentProcessId(), GetTickCount64()) > 0, "format pipe name");
  magnitude_private_pipe *pipe = NULL, *duplicate = NULL;
  require(magnitude_pipe_create(name, TRUE, &pipe) == ERROR_SUCCESS, "create first private listener");
  require(magnitude_pipe_create(name, TRUE, &duplicate) != ERROR_SUCCESS && !duplicate, "cannot replace an existing listener");
  HANDLE client = client_for(name);
  private_acl(client);
  require(magnitude_pipe_accept(pipe) == ERROR_SUCCESS, "accept already-connected client");
  DWORD pid;
  require(magnitude_pipe_client_pid(pipe, &pid) == ERROR_SUCCESS && pid == GetCurrentProcessId(), "observe native client PID");
  DWORD count; char buffer[32];
  require(WriteFile(client, "hello", 5, &count, NULL) && count == 5, "client write");
  require(magnitude_pipe_read(pipe, buffer, sizeof(buffer), &count) == ERROR_SUCCESS && count == 5 && !memcmp(buffer, "hello", 5), "server reads exact bytes");
  require(magnitude_pipe_write(pipe, "reply", 5, &count) == ERROR_SUCCESS && count == 5, "server write");
  require(ReadFile(client, buffer, sizeof(buffer), &count, NULL) && count == 5 && !memcmp(buffer, "reply", 5), "client receives exact bytes");
  CloseHandle(client);
  require(magnitude_pipe_read(pipe, buffer, sizeof(buffer), &count) == ERROR_SUCCESS && count == 0, "peer close becomes EOF");
  magnitude_pipe_close(pipe); magnitude_pipe_close(pipe); magnitude_pipe_destroy(pipe);
  puts("PASS current-user ACL, endpoint ownership, PID identity, duplex bytes and EOF");

  require(magnitude_pipe_create(name, TRUE, &pipe) == ERROR_SUCCESS, "create buffered reply fixture");
  client = client_for(name);
  require(magnitude_pipe_accept(pipe) == ERROR_SUCCESS, "accept buffered reply client");
  char reply[65000]; memset(reply, 0x3d, sizeof(reply));
  require(magnitude_pipe_write(pipe, reply, (DWORD)sizeof(reply), &count) == ERROR_SUCCESS && count == sizeof(reply), "buffer final reply");
  magnitude_pipe_destroy(pipe);
  DWORD received = 0;
  while (received < sizeof(reply)) {
    require(ReadFile(client, buffer, sizeof(buffer), &count, NULL) && count > 0, "read buffered reply after server close");
    for (DWORD index = 0; index < count; ++index) require(buffer[index] == 0x3d, "buffered reply byte survives close");
    received += count;
  }
  require(received == sizeof(reply), "exact buffered reply length");
  CloseHandle(client);
  puts("PASS unread buffered reply survives server close with native client");

  for (DWORD kind = 0; kind < 3; ++kind) for (DWORD attempt = 0; attempt < 30; ++attempt) {
    pipe = NULL; client = INVALID_HANDLE_VALUE;
    require(magnitude_pipe_create(name, TRUE, &pipe) == ERROR_SUCCESS, "reacquire retired endpoint");
    if (kind) { client = client_for(name); require(magnitude_pipe_accept(pipe) == ERROR_SUCCESS, "connect cancellation fixture"); }
    operation work = { pipe, CreateEventW(NULL, TRUE, FALSE, NULL), kind, ERROR_SUCCESS };
    require(work.started != NULL, "operation event");
    HANDLE thread = CreateThread(NULL, 0, operate, &work, 0, NULL);
    require(thread != NULL && WaitForSingleObject(work.started, 10000) == WAIT_OBJECT_0, "operation starts");
    Sleep(attempt % 3);
    magnitude_pipe_close(pipe);
    require(WaitForSingleObject(thread, 10000) == WAIT_OBJECT_0 && work.error == ERROR_OPERATION_ABORTED, "close cancels and joins pending I/O");
    CloseHandle(thread); CloseHandle(work.started);
    if (client != INVALID_HANDLE_VALUE) CloseHandle(client);
    magnitude_pipe_destroy(pipe);
  }
  puts("PASS 90 accept/read/write cancellation races and listener reacquisition");
  require(magnitude_pipe_create(L"\\\\remote\\pipe\\magnitude-test", TRUE, &pipe) == ERROR_INVALID_NAME && !pipe, "reject remote server namespace");
  return 0;
}
