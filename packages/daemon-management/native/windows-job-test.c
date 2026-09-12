/* Native Windows acceptance. Cross-compilation is not an execution receipt. */
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0A00
#endif
#include "windows-job.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <wchar.h>

static WCHAR executable[32768];
static void require(BOOL condition, const char *message) {
  if (!condition) { fprintf(stderr, "%s (Windows error %lu)\n", message, GetLastError()); ExitProcess(1); }
}
static void announce(void) {
  char text[32]; DWORD written;
  int length = snprintf(text, sizeof(text), "%lu\n", GetCurrentProcessId());
  require(WriteFile(GetStdHandle(STD_OUTPUT_HANDLE), text, (DWORD)length, &written, NULL), "announce PID");
}
static void arguments(WCHAR *target, const WCHAR *mode) {
  require(swprintf(target, 32768, L"\"%ls\" %ls", executable, mode) > 0, "format command");
}
static magnitude_owned_process spawn_owned(const WCHAR *mode, HANDLE input, HANDLE output) {
  magnitude_owned_process owned = {0}; WCHAR command[32768];
  arguments(command, mode);
  DWORD error = magnitude_owned_spawn(executable, command, NULL, input, output, output, &owned);
  SetLastError(error); require(error == ERROR_SUCCESS, "atomic job spawn");
  return owned;
}
static HANDLE open_process(DWORD pid) {
  HANDLE process = OpenProcess(SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION, FALSE, pid);
  require(process != NULL, "retain exact process handle"); return process;
}
static DWORD read_pid(HANDLE pipe) {
  char text[32]; size_t length = 0; ULONGLONG deadline = GetTickCount64() + 10000;
  while (GetTickCount64() < deadline) {
    DWORD available, read;
    require(PeekNamedPipe(pipe, NULL, 0, NULL, &available, NULL), "peek child output");
    if (!available) { Sleep(10); continue; }
    char byte;
    require(ReadFile(pipe, &byte, 1, &read, NULL) && read == 1, "read child output");
    if (byte == '\n') { text[length] = 0; DWORD pid = strtoul(text, NULL, 10); require(pid != 0, "valid child PID"); return pid; }
    require(length + 1 < sizeof(text), "bounded child output"); text[length++] = byte;
  }
  require(FALSE, "child output timeout"); return 0;
}
static void retired(magnitude_owned_process *owned) {
  DWORD active = 1, code = 0; BOOL root_exited = FALSE;
  ULONGLONG deadline = GetTickCount64() + 10000;
  while ((active || !root_exited) && GetTickCount64() < deadline) {
    require(magnitude_owned_active(owned, &active) == ERROR_SUCCESS, "query job retirement");
    require(magnitude_owned_exit(owned, &root_exited, &code) == ERROR_SUCCESS, "query root exit during retirement");
    if (active || !root_exited) Sleep(10);
  }
  /* Match the owner contract: job accounting can reach zero before process exit signals. */
  require(active == 0 && root_exited, "all job members retired and root exit observed");
}
int wmain(int argc, WCHAR **argv) {
  require(GetModuleFileNameW(NULL, executable, 32768) != 0, "executable path");
  if (argc > 1) {
    if (!wcscmp(argv[1], L"--probe-handle") && argc == 3) {
      HANDLE unexpected = (HANDLE)(uintptr_t)wcstoull(argv[2], NULL, 16);
      return SetEvent(unexpected) ? 1 : 0;
    }
    if (!wcscmp(argv[1], L"--breakaway")) {
      WCHAR command[32768]; arguments(command, L"--sleep");
      STARTUPINFOW startup = {0}; PROCESS_INFORMATION child = {0}; startup.cb = (DWORD)sizeof(startup);
      if (!CreateProcessW(executable, command, NULL, NULL, FALSE, CREATE_BREAKAWAY_FROM_JOB | CREATE_NO_WINDOW, NULL, NULL, &startup, &child))
        return GetLastError() == ERROR_ACCESS_DENIED ? 0 : 2;
      TerminateProcess(child.hProcess, 1); WaitForSingleObject(child.hProcess, 10000);
      CloseHandle(child.hThread); CloseHandle(child.hProcess); return 1;
    }
    if (!wcscmp(argv[1], L"--sleep")) {
      require(magnitude_owned_validate_current() == ERROR_SUCCESS, "child validates immediate job containment");
      announce(); Sleep(INFINITE); return 0;
    }
    if (!wcscmp(argv[1], L"--nested") || !wcscmp(argv[1], L"--owner")) {
      announce();
      magnitude_owned_process child = spawn_owned(L"--sleep", GetStdHandle(STD_INPUT_HANDLE), GetStdHandle(STD_OUTPUT_HANDLE));
      (void)child; Sleep(INFINITE); return 0;
    }
    if (!wcscmp(argv[1], L"--exit-root")) {
      WCHAR command[32768]; arguments(command, L"--sleep");
      STARTUPINFOW startup = {0}; PROCESS_INFORMATION child = {0}; startup.cb = (DWORD)sizeof(startup);
      startup.dwFlags = STARTF_USESTDHANDLES;
      startup.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
      startup.hStdOutput = startup.hStdError = GetStdHandle(STD_OUTPUT_HANDLE);
      require(CreateProcessW(executable, command, NULL, NULL, TRUE, CREATE_NO_WINDOW, NULL, NULL, &startup, &child), "ordinary descendant inherits job");
      CloseHandle(child.hThread); CloseHandle(child.hProcess); return 0;
    }
    return 2;
  }
  SECURITY_ATTRIBUTES security = { (DWORD)sizeof(security), NULL, TRUE };
  HANDLE input = CreateFileW(L"NUL", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, &security, OPEN_EXISTING, 0, NULL);
  HANDLE reader = NULL, writer = NULL;
  require(input != INVALID_HANDLE_VALUE && CreatePipe(&reader, &writer, &security, 0), "fixture standard handles");
  require(SetHandleInformation(reader, HANDLE_FLAG_INHERIT, 0), "reader must not be inherited");

  magnitude_owned_process nested = spawn_owned(L"--nested", input, writer);
  require(read_pid(reader) == nested.pid, "nested root identity");
  FILETIME creation;
  require(magnitude_owned_creation(&nested, &creation) == ERROR_SUCCESS && (creation.dwLowDateTime || creation.dwHighDateTime), "identity from retained process handle");
  HANDLE grandchild = open_process(read_pid(reader));
  require(magnitude_owned_terminate(&nested, 37) == ERROR_SUCCESS, "terminate nested job");
  retired(&nested);
  require(WaitForSingleObject(grandchild, 0) == WAIT_OBJECT_0, "nested child retired");
  CloseHandle(grandchild); magnitude_owned_close(&nested);
  puts("PASS nested jobs and complete descendant retirement");

  magnitude_owned_process exited = spawn_owned(L"--exit-root", input, writer);
  HANDLE survivor = open_process(read_pid(reader));
  require(WaitForSingleObject(exited.process, 10000) == WAIT_OBJECT_0, "root exit");
  DWORD active, code = 99; BOOL has_exited = FALSE;
  require(magnitude_owned_exit(&exited, &has_exited, &code) == ERROR_SUCCESS && has_exited && code == 0, "root exit status");
  require(magnitude_owned_active(&exited, &active) == ERROR_SUCCESS && active > 0, "root exit is not job retirement");
  magnitude_owned_close(&exited);
  require(WaitForSingleObject(survivor, 10000) == WAIT_OBJECT_0, "closing sole job handle retires surviving descendant");
  CloseHandle(survivor); puts("PASS root exit does not release descendant containment");

  magnitude_owned_process owner = spawn_owned(L"--owner", input, writer);
  require(read_pid(reader) == owner.pid, "owner identity");
  HANDLE protected_child = open_process(read_pid(reader));
  require(TerminateProcess(owner.process, 91), "force owner crash without graceful cleanup");
  require(WaitForSingleObject(protected_child, 10000) == WAIT_OBJECT_0, "crash closes child job handle");
  retired(&owner); CloseHandle(protected_child); magnitude_owned_close(&owner);
  puts("PASS owner crash retires its job without JavaScript");

  HANDLE sentinel = CreateEventW(&security, TRUE, FALSE, NULL);
  require(sentinel != NULL, "create unrelated inheritable handle");
  WCHAR probe[128];
  require(swprintf(probe, 128, L"--probe-handle %llx", (unsigned long long)(uintptr_t)sentinel) > 0, "format handle probe");
  magnitude_owned_process restricted = spawn_owned(probe, input, writer);
  require(WaitForSingleObject(restricted.process, 10000) == WAIT_OBJECT_0, "handle probe completes");
  require(magnitude_owned_exit(&restricted, &has_exited, &code) == ERROR_SUCCESS && has_exited && code == 0 &&
      WaitForSingleObject(sentinel, 0) == WAIT_TIMEOUT, "unlisted inheritable handle stays private");
  retired(&restricted); magnitude_owned_close(&restricted); CloseHandle(sentinel);
  magnitude_owned_process escaping = spawn_owned(L"--breakaway", input, writer);
  require(WaitForSingleObject(escaping.process, 10000) == WAIT_OBJECT_0, "breakaway probe completes");
  require(magnitude_owned_exit(&escaping, &has_exited, &code) == ERROR_SUCCESS && has_exited && code == 0, "breakaway is denied");
  retired(&escaping); magnitude_owned_close(&escaping);
  puts("PASS explicit handle inheritance and denied breakaway");

  magnitude_owned_process rejected = {0}; WCHAR command[32768]; arguments(command, L"--sleep");
  require(magnitude_owned_spawn(executable, command, NULL, reader, writer, writer, &rejected) == ERROR_INVALID_HANDLE && !rejected.job && !rejected.process,
      "reject non-inheritable input before creation");
  require(magnitude_owned_spawn(L"Z:\\magnitude-definitely-missing.exe", command, NULL, input, writer, writer, &rejected) != ERROR_SUCCESS && !rejected.job && !rejected.process,
      "creation failure retains no owned handles");
  puts("PASS invalid handles and creation failure fail closed");
  CloseHandle(input); CloseHandle(reader); CloseHandle(writer); return 0;
}
