#include <node_api.h>
#include <windows.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <tlhelp32.h>
#include <sddl.h>

typedef struct { HANDLE handle; } observation;
static const napi_type_tag tag = { UINT64_C(0xb9b4f2e63a21464c), UINT64_C(0x96aa6e6d9bb946fe) };
static napi_value failure(napi_env env, DWORD code) {
  char text[96]; napi_value message, error, value;
  snprintf(text, sizeof(text), "Windows process observation failed (%lu)", code);
  napi_create_string_utf8(env, text, NAPI_AUTO_LENGTH, &message);
  napi_create_error(env, NULL, message, &error);
  napi_create_uint32(env, code, &value); napi_set_named_property(env, error, "win32Code", value);
  napi_throw(env, error); return NULL;
}
static void release(observation *item) {
  if (item->handle) { CloseHandle(item->handle); item->handle = NULL; }
}
static void finalize(napi_env env, void *data, void *hint) {
  (void)env; (void)hint; release(data); free(data);
}
static observation *argument(napi_env env, napi_callback_info info) {
  napi_value value; size_t count = 1; bool matches = false; observation *item = NULL;
  if (napi_get_cb_info(env, info, &count, &value, NULL, NULL) != napi_ok || count != 1 ||
      napi_check_object_type_tag(env, value, &tag, &matches) != napi_ok || !matches ||
      napi_unwrap(env, value, (void **)&item) != napi_ok || !item) { failure(env, ERROR_INVALID_HANDLE); return NULL; }
  return item;
}
static napi_value observe(napi_env env, napi_callback_info info) {
  napi_value value, result; size_t count = 1; uint32_t pid; double original;
  if (napi_get_cb_info(env, info, &count, &value, NULL, NULL) != napi_ok || count != 1 ||
      napi_get_value_uint32(env, value, &pid) != napi_ok || !pid ||
      napi_get_value_double(env, value, &original) != napi_ok || original != (double)pid) return failure(env, ERROR_INVALID_PARAMETER);
  // No terminate, set-information, duplication or job rights are acquired.
  HANDLE handle = OpenProcess(SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION, FALSE, pid);
  if (!handle) {
    DWORD error = GetLastError();
    if (error == ERROR_INVALID_PARAMETER) { napi_get_null(env, &result); return result; }
    return failure(env, error);
  }
  observation *item = calloc(1, sizeof(*item));
  if (!item) { CloseHandle(handle); return failure(env, ERROR_NOT_ENOUGH_MEMORY); }
  item->handle = handle;
  if (napi_create_object(env, &result) != napi_ok || napi_type_tag_object(env, result, &tag) != napi_ok ||
      napi_wrap(env, result, item, finalize, NULL, NULL) != napi_ok) {
    release(item); free(item); return failure(env, ERROR_NOT_ENOUGH_MEMORY);
  }
  return result;
}
static napi_value exited(napi_env env, napi_callback_info info) {
  observation *item = argument(env, info); if (!item) return NULL;
  if (!item->handle) return failure(env, ERROR_INVALID_HANDLE);
  DWORD outcome = WaitForSingleObject(item->handle, 0);
  if (outcome != WAIT_TIMEOUT && outcome != WAIT_OBJECT_0) return failure(env, outcome == WAIT_FAILED ? GetLastError() : ERROR_INVALID_DATA);
  napi_value result; napi_get_boolean(env, outcome == WAIT_OBJECT_0, &result); return result;
}
static napi_value details(napi_env env, napi_callback_info info) {
  observation *item = argument(env, info); if (!item) return NULL;
  if (!item->handle) return failure(env, ERROR_INVALID_HANDLE);
  FILETIME creation, exit, kernel, user;
  DWORD pid = GetProcessId(item->handle), error = ERROR_SUCCESS;
  if (!pid || !GetProcessTimes(item->handle, &creation, &exit, &kernel, &user)) return failure(env, GetLastError());
  WCHAR *image = malloc(32768 * sizeof(WCHAR));
  HANDLE token = NULL; TOKEN_USER *owner = NULL; WCHAR *sid = NULL;
  napi_value result = NULL, value; DWORD length = 32768, bytes = 0;
  if (!image) return failure(env, ERROR_NOT_ENOUGH_MEMORY);
  if (!QueryFullProcessImageNameW(item->handle, 0, image, &length) ||
      !OpenProcessToken(item->handle, TOKEN_QUERY, &token)) { error = GetLastError(); goto cleanup; }
  if (GetTokenInformation(token, TokenUser, NULL, 0, &bytes) || GetLastError() != ERROR_INSUFFICIENT_BUFFER || !bytes || bytes > 65536) {
    error = ERROR_INVALID_DATA; goto cleanup;
  }
  owner = malloc(bytes);
  if (!owner) { error = ERROR_NOT_ENOUGH_MEMORY; goto cleanup; }
  if (!GetTokenInformation(token, TokenUser, owner, bytes, &bytes) || !ConvertSidToStringSidW(owner->User.Sid, &sid)) {
    error = GetLastError(); goto cleanup;
  }
  char identity[17];
  snprintf(identity, sizeof(identity), "%08lx%08lx", creation.dwHighDateTime, creation.dwLowDateTime);
  if (napi_create_object(env, &result) != napi_ok ||
      napi_create_uint32(env, pid, &value) != napi_ok || napi_set_named_property(env, result, "pid", value) != napi_ok ||
      napi_create_string_utf8(env, identity, 16, &value) != napi_ok || napi_set_named_property(env, result, "creationTime", value) != napi_ok ||
      napi_create_string_utf16(env, (const char16_t *)image, length, &value) != napi_ok || napi_set_named_property(env, result, "executable", value) != napi_ok ||
      napi_create_string_utf16(env, (const char16_t *)sid, NAPI_AUTO_LENGTH, &value) != napi_ok || napi_set_named_property(env, result, "userSid", value) != napi_ok)
    error = ERROR_NOT_ENOUGH_MEMORY;
cleanup:
  if (sid) LocalFree(sid);
  free(owner); if (token) CloseHandle(token); free(image);
  return error ? failure(env, error) : result;
}
/* A snapshot proves reported ancestry only. Callers separately retain and fence each process. */
static napi_value process_table(napi_env env, napi_callback_info info) {
  (void)info;
  HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
  if (snapshot == INVALID_HANDLE_VALUE) return failure(env, GetLastError());
  PROCESSENTRY32W entry = {0}; entry.dwSize = (DWORD)sizeof(entry);
  DWORD error = ERROR_SUCCESS; uint32_t count = 0;
  napi_value result, row, value;
  if (napi_create_array(env, &result) != napi_ok) { error = ERROR_NOT_ENOUGH_MEMORY; goto cleanup; }
  if (!Process32FirstW(snapshot, &entry)) { error = GetLastError(); goto cleanup; }
  do {
    if (!entry.th32ProcessID) continue; /* System Idle is not a process migration can observe. */
    if (count >= 65536) { error = ERROR_BUFFER_OVERFLOW; goto cleanup; }
    if (napi_create_object(env, &row) != napi_ok ||
        napi_create_uint32(env, entry.th32ProcessID, &value) != napi_ok || napi_set_named_property(env, row, "pid", value) != napi_ok ||
        napi_create_uint32(env, entry.th32ParentProcessID, &value) != napi_ok || napi_set_named_property(env, row, "parentPid", value) != napi_ok ||
        napi_set_element(env, result, count++, row) != napi_ok) { error = ERROR_NOT_ENOUGH_MEMORY; goto cleanup; }
  } while (Process32NextW(snapshot, &entry));
  if (GetLastError() != ERROR_NO_MORE_FILES) error = GetLastError();
cleanup:
  CloseHandle(snapshot);
  return error ? failure(env, error) : result;
}
static napi_value close_observation(napi_env env, napi_callback_info info) {
  observation *item = argument(env, info); if (!item) return NULL;
  release(item); napi_value result; napi_get_undefined(env, &result); return result;
}
void magnitude_register_windows_observers(napi_env env, napi_value exports) {
  napi_property_descriptor methods[] = {
    { "observeProcess", NULL, observe, NULL, NULL, NULL, napi_default, NULL },
    { "observedProcessExited", NULL, exited, NULL, NULL, NULL, napi_default, NULL },
    { "observedProcessDetails", NULL, details, NULL, NULL, NULL, napi_default, NULL },
    { "snapshotProcessParents", NULL, process_table, NULL, NULL, NULL, napi_default, NULL },
    { "releaseObservedProcess", NULL, close_observation, NULL, NULL, NULL, napi_default, NULL },
  };
  napi_define_properties(env, exports, sizeof(methods) / sizeof(methods[0]), methods);
}
