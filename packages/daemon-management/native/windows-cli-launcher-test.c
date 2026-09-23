#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0A00
#endif
#include "windows-cli-launcher.h"
#include "windows-job.h"
#include <stdio.h>
#include <stdlib.h>
#include <wchar.h>

static WCHAR executable[32768];
static void require(BOOL condition, const char *message) {
  if (!condition) { fprintf(stderr, "%s (Windows error %lu)\n", message, GetLastError()); ExitProcess(1); }
}
static void announce(void) {
  char text[32]; DWORD written;
  int length = snprintf(text, sizeof(text), "%lu\n", GetCurrentProcessId());
  require(WriteFile(GetStdHandle(STD_OUTPUT_HANDLE), text, (DWORD)length, &written, NULL), "announce identity");
}
static DWORD read_pid(HANDLE pipe) {
  char text[32]; size_t length = 0; ULONGLONG deadline = GetTickCount64() + 10000;
  while (GetTickCount64() < deadline) {
    DWORD available, count;
    require(PeekNamedPipe(pipe, NULL, 0, NULL, &available, NULL), "peek identity");
    if (!available) { Sleep(10); continue; }
    char byte;
    require(ReadFile(pipe, &byte, 1, &count, NULL) && count == 1, "read identity");
    if (byte == '\n') { text[length] = 0; return strtoul(text, NULL, 10); }
    require(length + 1 < sizeof(text), "bounded identity"); text[length++] = byte;
  }
  require(FALSE, "identity timeout"); return 0;
}
static HANDLE retain(DWORD pid) {
  HANDLE result = OpenProcess(SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION, FALSE, pid);
  require(result != NULL, "retain process"); return result;
}
static void containment(BOOL cancel) {
  SECURITY_ATTRIBUTES security = { sizeof(security), NULL, TRUE };
  HANDLE reader, writer;
  require(CreatePipe(&reader, &writer, &security, 0), "create output");
  require(SetHandleInformation(reader, HANDLE_FLAG_INHERIT, 0), "protect reader");
  HANDLE input = CreateFileW(L"NUL", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, &security, OPEN_EXISTING, 0, NULL);
  require(input != INVALID_HANDLE_VALUE, "open stdin");
  STARTUPINFOW startup = {0}; PROCESS_INFORMATION process = {0};
  startup.cb = sizeof(startup); startup.dwFlags = STARTF_USESTDHANDLES;
  startup.hStdInput = input; startup.hStdOutput = writer; startup.hStdError = writer;
  WCHAR command[32768];
  require(swprintf(command, 32768, L"\"%ls\" --run --child", executable) > 0, "format command");
  require(CreateProcessW(executable, command, NULL, NULL, TRUE, CREATE_NEW_PROCESS_GROUP,
      NULL, NULL, &startup, &process), "start launcher fixture");
  CloseHandle(process.hThread); CloseHandle(input); CloseHandle(writer);
  HANDLE child = retain(read_pid(reader));
  HANDLE descendant = retain(read_pid(reader));
  if (cancel) require(GenerateConsoleCtrlEvent(CTRL_BREAK_EVENT, process.dwProcessId), "cancel foreground console group");
  else require(TerminateProcess(process.hProcess, 91), "force launcher death");
  require(WaitForSingleObject(process.hProcess, 20000) == WAIT_OBJECT_0, "launcher exit");
  require(WaitForSingleObject(child, 10000) == WAIT_OBJECT_0, "compiled child retired");
  require(WaitForSingleObject(descendant, 10000) == WAIT_OBJECT_0, "descendant retired");
  DWORD code;
  require(GetExitCodeProcess(process.hProcess, &code) && code == (cancel ? (DWORD)ERROR_CANCELLED : 91U), "launcher exit status");
  CloseHandle(child); CloseHandle(descendant); CloseHandle(process.hProcess); CloseHandle(reader);
}
int wmain(int argc, WCHAR **argv) {
  require(GetModuleFileNameW(NULL, executable, 32768) != 0, "executable location");
  if (argc > 1) {
    if (!wcscmp(argv[1], L"--run")) return (int)magnitude_cli_run(executable, argc - 1, argv + 1);
    if (!wcscmp(argv[1], L"--child") || !wcscmp(argv[1], L"--leaf")) {
      require(magnitude_owned_validate_current() == ERROR_SUCCESS, "child job containment");
      announce();
      if (!wcscmp(argv[1], L"--child")) {
        STARTUPINFOW startup = {0}; PROCESS_INFORMATION child = {0};
        startup.cb = sizeof(startup); startup.dwFlags = STARTF_USESTDHANDLES;
        startup.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
        startup.hStdOutput = GetStdHandle(STD_OUTPUT_HANDLE); startup.hStdError = GetStdHandle(STD_ERROR_HANDLE);
        WCHAR command[32768];
        require(swprintf(command, 32768, L"\"%ls\" --leaf", executable) > 0, "format descendant");
        require(CreateProcessW(executable, command, NULL, NULL, TRUE, 0, NULL, NULL, &startup, &child), "start ordinary descendant");
        CloseHandle(child.hThread); CloseHandle(child.hProcess);
      }
      Sleep(INFINITE); return 0;
    }
    return 2;
  }
  if (!GetConsoleCP()) require(AllocConsole(), "create test console");
  containment(FALSE); puts("PASS launcher death retires child and ordinary descendant");
  containment(TRUE); puts("PASS console cancellation retires foreground tree and preserves cancellation status");
  return 0;
}
