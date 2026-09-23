#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0A00
#endif
#include "windows-cli-launcher.h"
#include <shlobj.h>
#include <stdio.h>
#include <wchar.h>

int wmain(int argc, WCHAR **argv) {
  PWSTR local = NULL;
  WCHAR executable[32768];
  HRESULT result = SHGetKnownFolderPath(&FOLDERID_LocalAppData, KF_FLAG_DEFAULT, NULL, &local);
  if (FAILED(result)) {
    fputs("Magnitude could not find its installation directory.\n", stderr);
    return 1;
  }
  int length = swprintf(executable, 32768, L"%ls\\Programs\\Magnitude\\resources\\magnitude.exe", local);
  CoTaskMemFree(local);
  if (length < 0 || length >= 32768) return 1;
  return (int)magnitude_cli_run(executable, argc, argv);
}
