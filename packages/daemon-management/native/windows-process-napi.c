#include <node_api.h>
#include <windows.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

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
static napi_value close_observation(napi_env env, napi_callback_info info) {
  observation *item = argument(env, info); if (!item) return NULL;
  release(item); napi_value result; napi_get_undefined(env, &result); return result;
}
void magnitude_register_windows_observers(napi_env env, napi_value exports) {
  napi_property_descriptor methods[] = {
    { "observeProcess", NULL, observe, NULL, NULL, NULL, napi_default, NULL },
    { "observedProcessExited", NULL, exited, NULL, NULL, NULL, napi_default, NULL },
    { "releaseObservedProcess", NULL, close_observation, NULL, NULL, NULL, napi_default, NULL },
  };
  napi_define_properties(env, exports, sizeof(methods) / sizeof(methods[0]), methods);
}
