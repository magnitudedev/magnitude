#ifndef MAGNITUDE_WINDOWS_CLI_LAUNCHER_H
#define MAGNITUDE_WINDOWS_CLI_LAUNCHER_H
#include <windows.h>

/* Reserved for a prepared startup update, before acquiring application ownership. */
#define MAGNITUDE_CLI_CONTINUE 75
#define MAGNITUDE_CLI_LAUNCHER_PROTOCOL L"1"

/* Writes the command to run into `executable` (32768 characters); FALSE when there is none. It is called
 * again after a continuation request, so a standalone update that switched versions runs the new one. */
typedef BOOL (*magnitude_cli_resolver)(WCHAR *executable);

/* Serving owns a foreground tree; finite commands can launch an independent desktop.
 * Neither path owns the application lock or update transaction. */
DWORD magnitude_cli_run(magnitude_cli_resolver resolve, int argc, WCHAR **argv, BOOL serving);
#endif
