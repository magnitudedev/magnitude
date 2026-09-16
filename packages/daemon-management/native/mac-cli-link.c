/* A one-shot Authorization Services request made by the app, not an AppleScript process. */
#include <node_api.h>
#include <Security/Authorization.h>
#include <Security/AuthorizationTags.h>
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

typedef struct {
  napi_async_work work;
  napi_deferred deferred;
  char link[PATH_MAX];
  char target[PATH_MAX];
  bool remove;
  OSStatus status;
} cli_operation;

static OSStatus execute_tool(AuthorizationRef authorization, const char *tool, char *const *arguments) {
  FILE *output = NULL;
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
  OSStatus status = AuthorizationExecuteWithPrivileges(authorization, tool, kAuthorizationFlagDefaults, arguments, &output);
#pragma clang diagnostic pop
  /* EOF synchronizes completion without waitpid(-1), which could reap an app-owned child. */
  if (output) {
    char buffer[256];
    while (fread(buffer, 1, sizeof(buffer), output)) {}
    fclose(output);
  }
  return status;
}

static void execute(napi_env env, void *data) {
  (void)env;
  cli_operation *operation = data;
  struct stat info;
  char destination[PATH_MAX];
  if (operation->remove) {
    ssize_t length = readlink(operation->link, destination, sizeof(destination) - 1);
    if (length < 0) return;
    destination[length] = 0;
    if (strcmp(destination, operation->target)) return;
  } else if (!lstat(operation->link, &info) && S_ISDIR(info.st_mode)) {
    operation->status = errAuthorizationDenied;
    return;
  }
  AuthorizationRef authorization = NULL;
  operation->status = AuthorizationCreate(NULL, kAuthorizationEmptyEnvironment, kAuthorizationFlagDefaults, &authorization);
  if (operation->status != errAuthorizationSuccess) return;
  AuthorizationItem right = { kAuthorizationRightExecute, 0, NULL, 0 };
  AuthorizationRights rights = { 1, &right };
  const char *prompt = operation->remove ? "Magnitude wants to remove its command-line tool." : "Magnitude wants to install its command-line tool.";
  AuthorizationItem promptItem = { kAuthorizationEnvironmentPrompt, strlen(prompt), (void *)prompt, 0 };
  AuthorizationEnvironment environment = { 1, &promptItem };
  operation->status = AuthorizationCopyRights(authorization, &rights, &environment,
    kAuthorizationFlagInteractionAllowed | kAuthorizationFlagExtendRights | kAuthorizationFlagPreAuthorize, NULL);
  if (operation->status == errAuthorizationSuccess) {
    if (operation->remove) {
      /* Recheck after the user has responded to the authorization prompt. */
      ssize_t length = readlink(operation->link, destination, sizeof(destination) - 1);
      if (length >= 0) {
        destination[length] = 0;
        if (!strcmp(destination, operation->target)) {
          char *arguments[] = { operation->link, NULL };
          operation->status = execute_tool(authorization, "/bin/rm", arguments);
        }
      }
    } else {
      char parent[PATH_MAX];
      strcpy(parent, operation->link);
      *strrchr(parent, '/') = 0;
      char *mkdirArguments[] = { "-p", parent, NULL };
      operation->status = execute_tool(authorization, "/bin/mkdir", mkdirArguments);
      if (operation->status == errAuthorizationSuccess) {
        char *linkArguments[] = { "-sfn", operation->target, operation->link, NULL };
        operation->status = execute_tool(authorization, "/bin/ln", linkArguments);
      }
    }
  }
  AuthorizationFree(authorization, kAuthorizationFlagDestroyRights);
}

static void complete(napi_env env, napi_status status, void *data) {
  cli_operation *operation = data;
  napi_value value;
  if (status == napi_ok && operation->status == errAuthorizationSuccess) {
    napi_get_undefined(env, &value);
    napi_resolve_deferred(env, operation->deferred, value);
  } else {
    char message[160];
    snprintf(message, sizeof(message), "Command-line tool authorization failed or was cancelled (%d)", (int)operation->status);
    napi_value text;
    napi_create_string_utf8(env, message, NAPI_AUTO_LENGTH, &text);
    napi_create_error(env, NULL, text, &value);
    napi_reject_deferred(env, operation->deferred, value);
  }
  napi_delete_async_work(env, operation->work);
  free(operation);
}

static napi_value configure(napi_env env, napi_callback_info info) {
  napi_value args[3], promise, name;
  size_t count = 3, linkLength = 0, targetLength = 0;
  napi_get_cb_info(env, info, &count, args, NULL, NULL);
  cli_operation *operation = calloc(1, sizeof(*operation));
  if (!operation) { napi_throw_error(env, NULL, "Cannot allocate command installation"); return NULL; }
  if (count != 3 || napi_get_value_string_utf8(env, args[0], operation->link, PATH_MAX, &linkLength) != napi_ok ||
      napi_get_value_string_utf8(env, args[1], operation->target, PATH_MAX, &targetLength) != napi_ok ||
      napi_get_value_bool(env, args[2], &operation->remove) != napi_ok ||
      !linkLength || linkLength >= PATH_MAX - 1 || !targetLength || targetLength >= PATH_MAX - 1 ||
      operation->link[0] != '/' || operation->target[0] != '/' ||
      strcmp(strrchr(operation->link, '/'), "/magnitude") || strcmp(strrchr(operation->target, '/'), "/magnitude")) {
    free(operation); napi_throw_type_error(env, NULL, "Expected absolute magnitude command paths and a removal flag"); return NULL;
  }
  napi_create_promise(env, &operation->deferred, &promise);
  napi_create_string_utf8(env, "Magnitude command installation", NAPI_AUTO_LENGTH, &name);
  if (napi_create_async_work(env, NULL, name, execute, complete, operation, &operation->work) != napi_ok ||
      napi_queue_async_work(env, operation->work) != napi_ok) {
    if (operation->work) napi_delete_async_work(env, operation->work);
    free(operation); napi_throw_error(env, NULL, "Cannot start command installation"); return NULL;
  }
  return promise;
}

void magnitude_register_mac_cli_link(napi_env env, napi_value exports) {
  napi_property_descriptor method = { "configureCliLink", NULL, configure, NULL, NULL, NULL, napi_default, NULL };
  napi_define_properties(env, exports, 1, &method);
}
