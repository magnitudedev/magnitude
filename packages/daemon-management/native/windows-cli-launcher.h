#ifndef MAGNITUDE_WINDOWS_CLI_LAUNCHER_H
#define MAGNITUDE_WINDOWS_CLI_LAUNCHER_H
#include <windows.h>

/* Reserved for a prepared startup update, before acquiring application ownership. */
#define MAGNITUDE_CLI_CONTINUE 75
#define MAGNITUDE_CLI_LAUNCHER_PROTOCOL L"1"

/* Serving owns a foreground tree; finite commands can launch an independent desktop.
 * Neither path owns the application lock or update transaction. */
DWORD magnitude_cli_run(const WCHAR *executable, int argc, WCHAR **argv, BOOL serving);
#endif
