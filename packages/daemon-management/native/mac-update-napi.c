#include <node_api.h>
#include <stdlib.h>
#include <string.h>
#include "mac-update-validation.h"

typedef struct {
  napi_async_work work;
  napi_deferred deferred;
  char *arguments[4];
  int valid;
} bundle_check;

static char *read_string(napi_env env, napi_value value, size_t maximum) {
  size_t length;
  if (napi_get_value_string_utf8(env, value, NULL, 0, &length) != napi_ok || !length || length > maximum) return NULL;
  char *text = calloc(length + 1, 1);
  if (!text) return NULL;
  if (napi_get_value_string_utf8(env, value, text, length + 1, &length) != napi_ok || memchr(text, 0, length)) {
    free(text); return NULL;
  }
  return text;
}
static void dispose(bundle_check *check) {
  for (int index = 0; index < 4; index++) free(check->arguments[index]);
  free(check);
}
static void execute(napi_env env, void *data) {
  (void)env;
  bundle_check *check = data;
  check->valid = magnitude_verify_mac_bundle(check->arguments[0], check->arguments[1], check->arguments[2], check->arguments[3]);
}
static void completed(napi_env env, napi_status status, void *data) {
  bundle_check *check = data;
  napi_value result;
  if (status != napi_ok || !check->valid) {
    napi_value message;
    napi_create_string_utf8(env, "The macOS update does not match the required signed application, version or architecture.", NAPI_AUTO_LENGTH, &message);
    napi_create_error(env, NULL, message, &result);
    napi_reject_deferred(env, check->deferred, result);
  } else {
    napi_get_undefined(env, &result);
    napi_resolve_deferred(env, check->deferred, result);
  }
  napi_delete_async_work(env, check->work);
  dispose(check);
}
static napi_value verify(napi_env env, napi_callback_info info) {
  napi_value args[4], result, name;
  size_t argc = 4;
  if (napi_get_cb_info(env, info, &argc, args, NULL, NULL) != napi_ok || argc != 4) {
    napi_throw_error(env, NULL, "Expected bundle path, signature requirement, version and architecture"); return NULL;
  }
  bundle_check *check = calloc(1, sizeof(*check));
  if (!check) { napi_throw_error(env, NULL, "Cannot allocate bundle verification"); return NULL; }
  const size_t limits[] = {4095, 4095, 255, 16};
  for (int index = 0; index < 4; index++) {
    check->arguments[index] = read_string(env, args[index], limits[index]);
    if (!check->arguments[index]) goto failure;
  }
  if (napi_create_promise(env, &check->deferred, &result) != napi_ok ||
      napi_create_string_utf8(env, "Mac application verification", NAPI_AUTO_LENGTH, &name) != napi_ok ||
      napi_create_async_work(env, NULL, name, execute, completed, check, &check->work) != napi_ok) goto failure;
  if (napi_queue_async_work(env, check->work) != napi_ok) {
    napi_delete_async_work(env, check->work); goto failure;
  }
  return result;
failure:
  dispose(check);
  napi_throw_error(env, NULL, "Cannot start macOS bundle verification"); return NULL;
}
void magnitude_register_mac_updates(napi_env env, napi_value exports) {
  napi_property_descriptor method = {"verifyMacBundle", NULL, verify, NULL, NULL, NULL, napi_default, NULL};
  napi_define_properties(env, exports, 1, &method);
}
