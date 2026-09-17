#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0A00
#endif
#include <node_api.h>
#include "windows-pipe.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const napi_type_tag pipe_tag = { UINT64_C(0x3837d064f24f41be), UINT64_C(0x9335bda3062a279a) };
static napi_value failure(napi_env env, const char *message) { napi_throw_error(env, NULL, message); return NULL; }
static napi_value windows_error(napi_env env, DWORD code) {
  char text[96]; napi_value message, error, value;
  snprintf(text, sizeof(text), "Windows pipe operation failed (%lu)", code);
  napi_create_string_utf8(env, text, NAPI_AUTO_LENGTH, &message);
  napi_create_error(env, NULL, message, &error);
  napi_create_uint32(env, code, &value); napi_set_named_property(env, error, "win32Code", value);
  return error;
}
static magnitude_private_pipe *unwrap(napi_env env, napi_value value) {
  bool matches = false; void *data = NULL;
  if (napi_check_object_type_tag(env, value, &pipe_tag, &matches) != napi_ok || !matches ||
      napi_unwrap(env, value, &data) != napi_ok) return NULL;
  return data;
}
static void finalize_pipe(napi_env env, void *data, void *hint) {
  (void)env; (void)hint; magnitude_pipe_destroy(data);
}
static napi_value create_pipe(napi_env env, napi_callback_info info) {
  napi_value args[2], result; size_t argc = 2, length; bool first;
  if (napi_get_cb_info(env, info, &argc, args, NULL, NULL) != napi_ok || argc != 2 ||
      napi_get_value_string_utf16(env, args[0], NULL, 0, &length) != napi_ok || length > 240 ||
      napi_get_value_bool(env, args[1], &first) != napi_ok) return failure(env, "Expected a private pipe name and first-instance flag");
  WCHAR name[241];
  if (napi_get_value_string_utf16(env, args[0], (char16_t *)name, 241, &length) != napi_ok)
    return failure(env, "Invalid private pipe name");
  for (size_t index = 0; index < length; ++index) if (!name[index]) return failure(env, "Pipe names cannot contain NUL");
  magnitude_private_pipe *pipe = NULL;
  DWORD error = magnitude_pipe_create(name, first, &pipe);
  if (error) { napi_throw(env, windows_error(env, error)); return NULL; }
  if (napi_create_object(env, &result) != napi_ok || napi_type_tag_object(env, result, &pipe_tag) != napi_ok ||
      napi_wrap(env, result, pipe, finalize_pipe, NULL, NULL) != napi_ok) {
    magnitude_pipe_destroy(pipe); return failure(env, "Cannot retain native pipe ownership");
  }
  return result;
}
typedef enum { ACCEPT, READ, WRITE, CLOSE } operation;
typedef struct {
  magnitude_private_pipe *pipe;
  operation operation;
  napi_threadsafe_function completion;
  napi_deferred deferred;
  napi_ref owner;
  unsigned char *buffer;
  DWORD size;
  DWORD count;
  DWORD error;
} work;

static void finalize_work(napi_env env, void *data, void *hint) {
  (void)hint; work *item = data;
  if (env) napi_delete_reference(env, item->owner);
  free(item->buffer); free(item);
}
static void complete(napi_env env, napi_value callback, void *context, void *data) {
  (void)callback; (void)data; work *item = context;
  if (!env) return;
  if (item->error) { napi_reject_deferred(env, item->deferred, windows_error(env, item->error)); return; }
  napi_value result;
  if (item->operation == READ) napi_create_buffer_copy(env, item->count, item->buffer, NULL, &result);
  else if (item->operation == CLOSE) napi_get_undefined(env, &result);
  else napi_create_uint32(env, item->count, &result);
  napi_resolve_deferred(env, item->deferred, result);
}
static DWORD WINAPI execute(void *argument) {
  work *item = argument;
  switch (item->operation) {
    case ACCEPT:
      item->error = magnitude_pipe_accept(item->pipe);
      if (!item->error) item->error = magnitude_pipe_client_pid(item->pipe, &item->count);
      break;
    case READ: item->error = magnitude_pipe_read(item->pipe, item->buffer, item->size, &item->count); break;
    case WRITE: item->error = magnitude_pipe_write(item->pipe, item->buffer, item->size, &item->count); break;
    case CLOSE: magnitude_pipe_close(item->pipe); break;
  }
  napi_call_threadsafe_function(item->completion, NULL, napi_tsfn_blocking);
  napi_release_threadsafe_function(item->completion, napi_tsfn_release);
  return 0;
}
static napi_value submit(napi_env env, napi_callback_info info, operation operation) {
  napi_value args[2], promise, resource; size_t argc = 2;
  if (napi_get_cb_info(env, info, &argc, args, NULL, NULL) != napi_ok || argc != (operation == WRITE ? 2u : 1u))
    return failure(env, "Invalid native pipe operation");
  magnitude_private_pipe *pipe = unwrap(env, args[0]);
  if (!pipe) return failure(env, "Expected an owned native pipe");
  void *source = NULL; size_t size = operation == READ ? 65536 : 0;
  if (operation == WRITE && (napi_get_buffer_info(env, args[1], &source, &size) != napi_ok || !size || size > 65536))
    return failure(env, "Native pipe writes require 1 to 65536 bytes");
  work *item = calloc(1, sizeof(*item));
  if (!item) return failure(env, "Cannot allocate native pipe operation");
  item->pipe = pipe; item->operation = operation; item->size = (DWORD)size;
  if (size) {
    item->buffer = malloc(size);
    if (!item->buffer) { free(item); return failure(env, "Cannot allocate native pipe buffer"); }
    if (operation == WRITE) memcpy(item->buffer, source, size);
  }
  if (napi_create_reference(env, args[0], 1, &item->owner) != napi_ok ||
      napi_create_promise(env, &item->deferred, &promise) != napi_ok ||
      napi_create_string_utf8(env, "MagnitudePrivatePipe", NAPI_AUTO_LENGTH, &resource) != napi_ok) {
    if (item->owner) napi_delete_reference(env, item->owner);
    free(item->buffer); free(item); return failure(env, "Cannot retain native pipe operation");
  }
  /* Dedicated native waits cannot exhaust Node's shared worker pool and starve pipe writes. */
  if (napi_create_threadsafe_function(env, NULL, NULL, resource, 1, 1, item, finalize_work, item, complete, &item->completion) != napi_ok) {
    napi_delete_reference(env, item->owner); free(item->buffer); free(item);
    return failure(env, "Cannot create native pipe completion");
  }
  HANDLE thread = CreateThread(NULL, 0, execute, item, 0, NULL);
  if (thread) CloseHandle(thread);
  else {
    item->error = GetLastError();
    napi_call_threadsafe_function(item->completion, NULL, napi_tsfn_nonblocking);
    napi_release_threadsafe_function(item->completion, napi_tsfn_release);
  }
  return promise;
}
static napi_value accept_pipe(napi_env env, napi_callback_info info) { return submit(env, info, ACCEPT); }
static napi_value read_pipe(napi_env env, napi_callback_info info) { return submit(env, info, READ); }
static napi_value write_pipe(napi_env env, napi_callback_info info) { return submit(env, info, WRITE); }
static napi_value close_pipe(napi_env env, napi_callback_info info) { return submit(env, info, CLOSE); }

void magnitude_register_windows_pipes(napi_env env, napi_value exports) {
  napi_property_descriptor methods[] = {
    { "createPrivatePipe", NULL, create_pipe, NULL, NULL, NULL, napi_default, NULL },
    { "acceptPrivatePipe", NULL, accept_pipe, NULL, NULL, NULL, napi_default, NULL },
    { "readPrivatePipe", NULL, read_pipe, NULL, NULL, NULL, napi_default, NULL },
    { "writePrivatePipe", NULL, write_pipe, NULL, NULL, NULL, napi_default, NULL },
    { "closePrivatePipe", NULL, close_pipe, NULL, NULL, NULL, napi_default, NULL },
  };
  napi_define_properties(env, exports, sizeof(methods) / sizeof(methods[0]), methods);
}
