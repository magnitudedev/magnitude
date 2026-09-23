#ifndef MAGNITUDE_WINDOWS_CLI_LAUNCHER_H
#define MAGNITUDE_WINDOWS_CLI_LAUNCHER_H
#include <windows.h>

/* Reserved for a prepared startup update, before acquiring application ownership. */
#define MAGNITUDE_CLI_CONTINUE 75
#define MAGNITUDE_CLI_LAUNCHER_PROTOCOL L"1"

/* Owns a foreground child tree, not the application lock or update transaction. */
DWORD magnitude_cli_run(const WCHAR *executable, int argc, WCHAR **argv);
#endif
