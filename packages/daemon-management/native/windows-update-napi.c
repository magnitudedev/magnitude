#include <node_api.h>
#include <stdlib.h>
#include "windows-update-signature.h"

typedef struct {
  napi_async_work work;
  napi_deferred deferred;
  WCHAR *path;
  WCHAR *organization;
  DWORD error;
} signature_check;

static WCHAR *read_string(napi_env env, napi_value value, size_t maximum) {
  size_t length;
  if (napi_get_value_string_utf16(env, value, NULL, 0, &length) != napi_ok || !length || length > maximum) return NULL;
  WCHAR *text = calloc(length + 1, sizeof(WCHAR));
  if (!text) return NULL;
  if (napi_get_value_string_utf16(env, value, (char16_t *)text, length + 1, &length) != napi_ok) {
    free(text); return NULL;
  }
  for (size_t index = 0; index < length; ++index) if (!text[index]) { free(text); return NULL; }
  return text;
}
static void execute(napi_env env, void *data) {
  (void)env;
  signature_check *check = data;
  check->error = magnitude_verify_installer_signature(check->path, check->organization);
}
static void completed(napi_env env, napi_status status, void *data) {
  signature_check *check = data;
  napi_value result;
  if (status != napi_ok || check->error) {
    napi_value message;
    napi_create_string_utf8(env, "The update does not have a trusted Windows signature from the expected publisher.", NAPI_AUTO_LENGTH, &message);
    napi_create_error(env, NULL, message, &result);
    napi_reject_deferred(env, check->deferred, result);
  } else {
    napi_get_undefined(env, &result);
    napi_resolve_deferred(env, check->deferred, result);
  }
  napi_delete_async_work(env, check->work);
  free(check->path); free(check->organization); free(check);
}
static napi_value verify(napi_env env, napi_callback_info info) {
  napi_value args[2], result, name;
  size_t argc = 2;
  if (napi_get_cb_info(env, info, &argc, args, NULL, NULL) != napi_ok || argc != 2) {
    napi_throw_error(env, NULL, "Expected an installer path and publisher organization"); return NULL;
  }
  signature_check *check = calloc(1, sizeof(*check));
  if (!check) { napi_throw_error(env, NULL, "Cannot allocate signature verification"); return NULL; }
  check->path = read_string(env, args[0], 32767);
  check->organization = read_string(env, args[1], 1023);
  if (!check->path || !check->organization) goto failure;
  if (napi_create_promise(env, &check->deferred, &result) != napi_ok ||
      napi_create_string_utf8(env, "Windows installer signature", NAPI_AUTO_LENGTH, &name) != napi_ok ||
      napi_create_async_work(env, NULL, name, execute, completed, check, &check->work) != napi_ok) goto failure;
  if (napi_queue_async_work(env, check->work) != napi_ok) {
    napi_delete_async_work(env, check->work); goto failure;
  }
  return result;
failure:
  free(check->path); free(check->organization); free(check);
  napi_throw_error(env, NULL, "Cannot start Windows installer signature verification"); return NULL;
}
void magnitude_register_windows_updates(napi_env env, napi_value exports) {
  napi_property_descriptor method = {"verifyInstallerSignature", NULL, verify, NULL, NULL, NULL, napi_default, NULL};
  napi_define_properties(env, exports, 1, &method);
}
