#include <node_api.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <wchar.h>
#include "windows-process-retirement.h"
#include <sddl.h>

static const napi_type_tag tag = { UINT64_C(0x98db3e0325514985), UINT64_C(0xb0cafca246f23502) };
static napi_value failure(napi_env env, DWORD code) {
  char text[96]; napi_value message, error, value;
  snprintf(text, sizeof(text), "Windows migration process failed (%lu)", code);
  napi_create_string_utf8(env, text, NAPI_AUTO_LENGTH, &message);
  napi_create_error(env, NULL, message, &error);
  napi_create_uint32(env, code, &value); napi_set_named_property(env, error, "win32Code", value);
  napi_throw(env, error); return NULL;
}
static void finalize(napi_env env, void *data, void *hint) {
  (void)env; (void)hint; magnitude_migration_close(data); free(data);
}
static magnitude_migration_process *argument(napi_env env, napi_callback_info info) {
  napi_value value; size_t count = 1; bool matches = false; magnitude_migration_process *process = NULL;
  if (napi_get_cb_info(env, info, &count, &value, NULL, NULL) != napi_ok || count != 1 ||
      napi_check_object_type_tag(env, value, &tag, &matches) != napi_ok || !matches ||
      napi_unwrap(env, value, (void **)&process) != napi_ok || !process) {
    failure(env, ERROR_INVALID_HANDLE); return NULL;
  }
  return process;
}
static DWORD string(napi_env env, napi_value value, size_t maximum, WCHAR **result) {
  size_t length = 0, copied = 0; *result = NULL;
  if (napi_get_value_string_utf16(env, value, NULL, 0, &length) != napi_ok || !length || length > maximum)
    return ERROR_INVALID_PARAMETER;
  WCHAR *text = malloc((length + 1) * sizeof(WCHAR));
  if (!text) return ERROR_NOT_ENOUGH_MEMORY;
  if (napi_get_value_string_utf16(env, value, (char16_t *)text, length + 1, &copied) != napi_ok ||
      copied != length || wcslen(text) != length) { free(text); return ERROR_INVALID_PARAMETER; }
  *result = text; return ERROR_SUCCESS;
}
static napi_value acquire(napi_env env, napi_callback_info info) {
  napi_value args[4], result = NULL; size_t count = 4; uint32_t pid; double original;
  if (napi_get_cb_info(env, info, &count, args, NULL, NULL) != napi_ok || count != 4 ||
      napi_get_value_uint32(env, args[0], &pid) != napi_ok || !pid ||
      napi_get_value_double(env, args[0], &original) != napi_ok || original != (double)pid)
    return failure(env, ERROR_INVALID_PARAMETER);
  WCHAR *creation = NULL, *executable = NULL, *sid = NULL; PSID user = NULL;
  magnitude_migration_process *process = NULL; DWORD error = string(env, args[1], 16, &creation);
  if (error) goto done;
  if (wcslen(creation) != 16) { error = ERROR_INVALID_PARAMETER; goto done; }
  ULONGLONG ticks = 0;
  for (size_t i = 0; i < 16; i++) {
    WCHAR c = creation[i];
    if (!((c >= L'0' && c <= L'9') || (c >= L'a' && c <= L'f'))) { error = ERROR_INVALID_PARAMETER; goto done; }
    ticks = (ticks << 4) | (ULONGLONG)(c <= L'9' ? c - L'0' : c - L'a' + 10);
  }
  error = string(env, args[2], 32767, &executable); if (error) goto done;
  error = string(env, args[3], 184, &sid); if (error) goto done;
  if (!ConvertStringSidToSidW(sid, &user)) { error = GetLastError(); goto done; }
  process = calloc(1, sizeof(*process));
  if (!process) { error = ERROR_NOT_ENOUGH_MEMORY; goto done; }
  FILETIME identity = { (DWORD)ticks, (DWORD)(ticks >> 32) };
  error = magnitude_migration_open(pid, &identity, executable, user, process); if (error) goto done;
  if (napi_create_object(env, &result) != napi_ok || napi_type_tag_object(env, result, &tag) != napi_ok ||
      napi_wrap(env, result, process, finalize, NULL, NULL) != napi_ok) { error = ERROR_NOT_ENOUGH_MEMORY; goto done; }
  process = NULL;
done:
  free(creation); free(executable); free(sid); if (user) LocalFree(user);
  if (process) { magnitude_migration_close(process); free(process); }
  return error ? failure(env, error) : result;
}
static napi_value start(napi_env env, napi_callback_info info) {
  magnitude_migration_process *process = argument(env, info); if (!process) return NULL;
  DWORD error = magnitude_migration_start_retirement(process); if (error) return failure(env, error);
  napi_value result; napi_get_undefined(env, &result); return result;
}
static napi_value exited(napi_env env, napi_callback_info info) {
  magnitude_migration_process *process = argument(env, info); if (!process) return NULL;
  BOOL complete = FALSE; DWORD error = magnitude_migration_exited(process, &complete);
  if (error) return failure(env, error);
  napi_value result; napi_get_boolean(env, complete != FALSE, &result); return result;
}
static napi_value close_process(napi_env env, napi_callback_info info) {
  magnitude_migration_process *process = argument(env, info); if (!process) return NULL;
  magnitude_migration_close(process); napi_value result; napi_get_undefined(env, &result); return result;
}
void magnitude_register_windows_retirement(napi_env env, napi_value exports) {
  napi_property_descriptor methods[] = {
    { "acquireMigrationProcess", NULL, acquire, NULL, NULL, NULL, napi_default, NULL },
    { "startMigrationProcessRetirement", NULL, start, NULL, NULL, NULL, napi_default, NULL },
    { "migrationProcessExited", NULL, exited, NULL, NULL, NULL, napi_default, NULL },
    { "releaseMigrationProcess", NULL, close_process, NULL, NULL, NULL, napi_default, NULL },
  };
  napi_define_properties(env, exports, sizeof(methods) / sizeof(methods[0]), methods);
}
