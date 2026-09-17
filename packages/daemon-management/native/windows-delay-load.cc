#include <windows.h>
#include <delayimp.h>
#include <string.h>

/* Node-API lives in the actual host (Electron, Node, or compiled Bun), not a separate node.exe. */
static FARPROC WINAPI resolve_host(unsigned int event, DelayLoadInfo *info) {
  if (event != dliNotePreLoadLibrary || _stricmp(info->szDll, "node.exe") != 0) return nullptr;
  return reinterpret_cast<FARPROC>(GetModuleHandleW(nullptr));
}
decltype(__pfnDliNotifyHook2) __pfnDliNotifyHook2 = resolve_host;
