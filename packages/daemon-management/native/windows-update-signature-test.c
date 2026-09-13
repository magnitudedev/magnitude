#include "windows-update-signature.h"
#include <stdio.h>
int wmain(int argc, WCHAR **argv) {
  if (argc != 3) return 2;
  DWORD result = magnitude_verify_installer_signature(argv[1], argv[2]);
  if (result) { fprintf(stderr, "Installer signature rejected (%lu)\n", result); return 1; }
  puts("PASS trusted Windows installer publisher");
  return 0;
}
