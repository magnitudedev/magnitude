#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0A00
#endif
#include <node_api.h>
#include "windows-job.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>

static const napi_type_tag job_tag = { UINT64_C(0x67bd39c7f05d4e16), UINT64_C(0xb5b1ea75cd60cf0f) };
static napi_value failure(napi_env env, const char *message) { napi_throw_error(env, NULL, message); return NULL; }
static napi_value windows_failure(napi_env env, DWORD code) {
  char text[96]; napi_value message, error, value;
  snprintf(text, sizeof(text), "Windows owned process operation failed (%lu)", code);
  napi_create_string_utf8(env, text, NAPI_AUTO_LENGTH, &message);
  napi_create_error(env, NULL, message, &error);
  napi_create_uint32(env, code, &value); napi_set_named_property(env, error, "win32Code", value);
  napi_throw(env, error); return NULL;
}
static void finalize_job(napi_env env, void *data, void *hint) {
  (void)env; (void)hint; magnitude_owned_close(data); free(data);
}
static magnitude_owned_process *argument(napi_env env, napi_callback_info info) {
  napi_value value; size_t count = 1; bool matches = false; void *data = NULL;
  if (napi_get_cb_info(env, info, &count, &value, NULL, NULL) != napi_ok || count != 1 ||
      napi_check_object_type_tag(env, value, &job_tag, &matches) != napi_ok || !matches ||
      napi_unwrap(env, value, &data) != napi_ok) { failure(env, "Expected an owned Windows process"); return NULL; }
  return data;
}
static WCHAR *string_argument(napi_env env, napi_value value, size_t maximum, bool environment) {
  size_t length;
  if (napi_get_value_string_utf16(env, value, NULL, 0, &length) != napi_ok || !length || length > maximum) return NULL;
  WCHAR *text = calloc(length + 1, sizeof(WCHAR));
  if (!text) return NULL;
  if (napi_get_value_string_utf16(env, value, (char16_t *)text, length + 1, &length) != napi_ok) { free(text); return NULL; }
  if (environment) {
    if (length < 2 || text[length - 1] || text[length - 2]) { free(text); return NULL; }
    for (size_t index = 0; index + 2 < length; ++index) if (!text[index] && !text[index + 1]) { free(text); return NULL; }
  } else {
    for (size_t index = 0; index < length; ++index) if (!text[index]) { free(text); return NULL; }
  }
  return text;
}

static int key_length(const WCHAR *entry) {
  const WCHAR *separator = wcschr(entry + (*entry == L'=' ? 1 : 0), L'=');
  return separator ? (int)(separator - entry) : 0;
}
static int compare_environment(const void *left, const void *right) {
  const WCHAR *a = *(const WCHAR *const *)left, *b = *(const WCHAR *const *)right;
  return CompareStringOrdinal(a, key_length(a), b, key_length(b), TRUE) - CSTR_EQUAL;
}
/* Compare using Windows' own case folding, not the host JavaScript locale or Unicode version. */
static DWORD order_environment(WCHAR *block) {
  size_t count = 0, units = 0;
  while (block[units]) {
    WCHAR *entry = block + units;
    int key = key_length(entry);
    if (!key || (*entry == L'=' && !(key == 3 && entry[2] == L':' &&
        ((entry[1] >= L'A' && entry[1] <= L'Z') || (entry[1] >= L'a' && entry[1] <= L'z'))))) return ERROR_INVALID_PARAMETER;
    units += wcslen(entry) + 1;
    ++count;
  }
  if (!count) return ERROR_SUCCESS;
  WCHAR **entries = calloc(count, sizeof(*entries));
  WCHAR *ordered = calloc(units + 1, sizeof(*ordered));
  if (!entries || !ordered) { free(entries); free(ordered); return ERROR_NOT_ENOUGH_MEMORY; }
  size_t offset = 0;
  for (size_t index = 0; index < count; ++index) { entries[index] = block + offset; offset += wcslen(entries[index]) + 1; }
  qsort(entries, count, sizeof(*entries), compare_environment);
  offset = 0;
  DWORD error = ERROR_SUCCESS;
  for (size_t index = 0; index < count; ++index) {
    if (index && compare_environment(&entries[index - 1], &entries[index]) == 0) { error = ERROR_INVALID_PARAMETER; break; }
    size_t length = wcslen(entries[index]) + 1;
    memcpy(ordered + offset, entries[index], length * sizeof(WCHAR)); offset += length;
  }
  if (!error) memcpy(block, ordered, (units + 1) * sizeof(WCHAR));
  free(entries); free(ordered); return error;
}

static BOOL private_pipe_name(const WCHAR *name) {
  return name && !wcsncmp(name, L"\\\\.\\pipe\\magnitude-", 19);
}

/* Service control uses merged diagnostics; inference needs distinct stdin/stdout/stderr. Both
 * forms inherit only parent-created handles and enter their job atomically. */
static napi_value spawn_with_streams(napi_env env, napi_callback_info info, BOOL separate) {
  napi_value args[6], result; size_t count = 6;
  if (napi_get_cb_info(env, info, &count, args, NULL, NULL) != napi_ok || count != (separate ? 6u : 4u))
    return failure(env, "Expected executable, command line, environment block and private stream pipes");
  WCHAR *executable = string_argument(env, args[0], 32767, false);
  WCHAR *command = string_argument(env, args[1], 32766, false);
  WCHAR *environment = string_argument(env, args[2], 1048576, true);
  WCHAR *input_name = separate ? string_argument(env, args[3], 240, false) : NULL;
  WCHAR *output_name = string_argument(env, args[separate ? 4 : 3], 240, false);
  WCHAR *error_name = separate ? string_argument(env, args[5], 240, false) : NULL;
  if (!executable || !command || !environment || !private_pipe_name(output_name) ||
      (separate && (!private_pipe_name(input_name) || !private_pipe_name(error_name) ||
        !_wcsicmp(input_name, output_name) || !_wcsicmp(input_name, error_name) || !_wcsicmp(output_name, error_name)))) {
    free(executable); free(command); free(environment); free(input_name); free(output_name); free(error_name);
    return failure(env, "Invalid owned Windows process arguments");
  }
  DWORD error = order_environment(environment);
  magnitude_owned_process *owned = NULL;
  if (!error) { owned = calloc(1, sizeof(*owned)); if (!owned) error = ERROR_NOT_ENOUGH_MEMORY; }
  SECURITY_ATTRIBUTES security = { sizeof(security), NULL, TRUE };
  HANDLE input = INVALID_HANDLE_VALUE, output = INVALID_HANDLE_VALUE, diagnostic = INVALID_HANDLE_VALUE;
  if (!error) {
    input = CreateFileW(separate ? input_name : L"NUL", GENERIC_READ,
        separate ? 0 : FILE_SHARE_READ | FILE_SHARE_WRITE, &security, OPEN_EXISTING, 0, NULL);
    if (input == INVALID_HANDLE_VALUE) error = GetLastError();
  }
  if (!error) {
    output = CreateFileW(output_name, GENERIC_WRITE, 0, &security, OPEN_EXISTING, 0, NULL);
    if (output == INVALID_HANDLE_VALUE) error = GetLastError();
  }
  if (!error && separate) {
    diagnostic = CreateFileW(error_name, GENERIC_WRITE, 0, &security, OPEN_EXISTING, 0, NULL);
    if (diagnostic == INVALID_HANDLE_VALUE) error = GetLastError();
  }
  if (!error) error = magnitude_owned_spawn(executable, command, environment, input, output, separate ? diagnostic : output, owned);
  if (input != INVALID_HANDLE_VALUE) CloseHandle(input);
  if (output != INVALID_HANDLE_VALUE) CloseHandle(output);
  if (diagnostic != INVALID_HANDLE_VALUE) CloseHandle(diagnostic);
  free(executable); free(command); free(environment); free(input_name); free(output_name); free(error_name);
  if (error) { if (owned) { magnitude_owned_close(owned); free(owned); } return windows_failure(env, error); }
  // Wrapping precedes identity observation. Failure retains cleanup authority through this point.
  if (napi_create_object(env, &result) != napi_ok || napi_type_tag_object(env, result, &job_tag) != napi_ok ||
      napi_wrap(env, result, owned, finalize_job, NULL, NULL) != napi_ok) {
    magnitude_owned_close(owned); free(owned); return failure(env, "Cannot retain Windows job ownership");
  }
  return result;
}
static napi_value spawn_owned(napi_env env, napi_callback_info info) { return spawn_with_streams(env, info, FALSE); }
static napi_value spawn_owned_pipes(napi_env env, napi_callback_info info) { return spawn_with_streams(env, info, TRUE); }
static napi_value identity(napi_env env, napi_callback_info info) {
  magnitude_owned_process *owned = argument(env, info); if (!owned) return NULL;
  FILETIME creation; DWORD error = magnitude_owned_creation(owned, &creation);
  if (error) return windows_failure(env, error);
  napi_value result, pid, stamp; char text[17];
  snprintf(text, sizeof(text), "%08lx%08lx", creation.dwHighDateTime, creation.dwLowDateTime);
  napi_create_object(env, &result); napi_create_uint32(env, owned->pid, &pid);
  napi_create_string_utf8(env, text, NAPI_AUTO_LENGTH, &stamp);
  napi_set_named_property(env, result, "pid", pid); napi_set_named_property(env, result, "creationTime", stamp);
  return result;
}
static napi_value active(napi_env env, napi_callback_info info) {
  magnitude_owned_process *owned = argument(env, info); if (!owned) return NULL;
  DWORD count, error = magnitude_owned_active(owned, &count); napi_value result;
  if (error) return windows_failure(env, error);
  napi_create_uint32(env, count, &result); return result;
}
static napi_value exited(napi_env env, napi_callback_info info) {
  magnitude_owned_process *owned = argument(env, info); if (!owned) return NULL;
  BOOL done; DWORD code, error = magnitude_owned_exit(owned, &done, &code); napi_value result;
  if (error) return windows_failure(env, error);
  if (done) napi_create_uint32(env, code, &result); else napi_get_null(env, &result);
  return result;
}
static napi_value terminate(napi_env env, napi_callback_info info) {
  magnitude_owned_process *owned = argument(env, info); if (!owned) return NULL;
  DWORD error = magnitude_owned_terminate(owned, 1); napi_value result;
  if (error) return windows_failure(env, error);
  napi_get_undefined(env, &result); return result;
}
static napi_value close_owned(napi_env env, napi_callback_info info) {
  magnitude_owned_process *owned = argument(env, info); if (!owned) return NULL;
  magnitude_owned_close(owned); napi_value result; napi_get_undefined(env, &result); return result;
}
void magnitude_register_windows_jobs(napi_env env, napi_value exports) {
  napi_property_descriptor methods[] = {
    { "spawnOwnedProcess", NULL, spawn_owned, NULL, NULL, NULL, napi_default, NULL },
    { "spawnOwnedProcessWithPipes", NULL, spawn_owned_pipes, NULL, NULL, NULL, napi_default, NULL },
    { "ownedProcessIdentity", NULL, identity, NULL, NULL, NULL, napi_default, NULL },
    { "ownedProcessActiveCount", NULL, active, NULL, NULL, NULL, napi_default, NULL },
    { "ownedProcessExit", NULL, exited, NULL, NULL, NULL, napi_default, NULL },
    { "terminateOwnedProcess", NULL, terminate, NULL, NULL, NULL, napi_default, NULL },
    { "closeOwnedProcess", NULL, close_owned, NULL, NULL, NULL, napi_default, NULL },
  };
  napi_define_properties(env, exports, sizeof(methods) / sizeof(methods[0]), methods);
}
