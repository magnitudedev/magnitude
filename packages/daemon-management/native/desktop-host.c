/* Process-lifetime primitives. No JavaScript callbacks run from the watchdog. */
#include <node_api.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#ifdef _WIN32
#include <windows.h>
#include <wchar.h>
#include <shlobj.h>
#include "windows-job.h"
#include "windows-security.h"
void magnitude_register_windows_pipes(napi_env env, napi_value exports);
void magnitude_register_windows_jobs(napi_env env, napi_value exports);
void magnitude_register_windows_observers(napi_env env, napi_value exports);
void magnitude_register_windows_updates(napi_env env, napi_value exports);
#else
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

void magnitude_register_application_memory(napi_env env, napi_value exports);
void magnitude_register_machine_identity(napi_env env, napi_value exports);

typedef struct {
#ifdef _WIN32
  HANDLE handle;
  HANDLE directory;
#else
  int fd;
#endif
  int released;
} owner_lock;
static const napi_type_tag lock_tag = { UINT64_C(0x7a8a34a11f004d7b), UINT64_C(0xb8d77013af82c491) };

static napi_value failure(napi_env env, const char *message) {
  napi_throw_error(env, NULL, message);
  return NULL;
}

static void release_lock(owner_lock *lock) {
  if (lock->released) return;
  lock->released = 1;
#ifdef _WIN32
  CloseHandle(lock->handle);
  CloseHandle(lock->directory);
#else
  close(lock->fd);
#endif
}

static void finalize_lock(napi_env env, void *data, void *hint) {
  (void)env; (void)hint;
  owner_lock *lock = data;
  release_lock(lock);
  free(lock);
}

#ifdef _WIN32
static napi_value local_app_data(napi_env env, napi_callback_info info) {
  (void)info;
  /* Query the current user's actual folder; do not infer it from a roaming home or environment. */
  static const GUID folder = { 0xf1b32785, 0x6fba, 0x4fcf, { 0x9d, 0x55, 0x7b, 0x8e, 0x7f, 0x15, 0x70, 0x91 } };
  PWSTR path = NULL;
  HRESULT status = SHGetKnownFolderPath(&folder, 0, NULL, &path);
  if (FAILED(status) || !path) {
    CoTaskMemFree(path);
    return failure(env, "Cannot locate the current user's local application data directory");
  }
  napi_value result;
  napi_status encoded = napi_create_string_utf16(env, (const char16_t *)path, NAPI_AUTO_LENGTH, &result);
  CoTaskMemFree(path);
  return encoded == napi_ok ? result : failure(env, "Cannot encode local application data directory");
}

static WCHAR *private_path(napi_env env, napi_value arg) {
  size_t length;
  if (napi_get_value_string_utf16(env, arg, NULL, 0, &length) != napi_ok || length < 3 || length > 32767) {
    failure(env, "Invalid lock-file path"); return NULL;
  }
  WCHAR *path = calloc(length + 1, sizeof(WCHAR));
  if (!path) { failure(env, "Cannot allocate lock-file path"); return NULL; }
  if (napi_get_value_string_utf16(env, arg, (char16_t *)path, length + 1, &length) != napi_ok) {
    free(path); failure(env, "Invalid lock-file path"); return NULL;
  }
  for (size_t index = 0; index < length; ++index) if (!path[index]) {
    free(path); failure(env, "Lock path cannot contain NUL"); return NULL;
  }
  const WCHAR *drive = length >= 7 && !wcsncmp(path, L"\\\\?\\", 4) ? path + 4 : path;
  if (!(((drive[0] >= L'A' && drive[0] <= L'Z') || (drive[0] >= L'a' && drive[0] <= L'z')) &&
        drive[1] == L':' && (drive[2] == L'\\' || drive[2] == L'/'))) {
    free(path); failure(env, "Ownership lock requires an absolute local drive path"); return NULL;
  }
  WCHAR volume[] = { drive[0], L':', L'\\', 0 };
  UINT kind = GetDriveTypeW(volume);
  if (kind != DRIVE_FIXED && kind != DRIVE_REMOVABLE && kind != DRIVE_RAMDISK) {
    free(path); failure(env, "Application ownership requires a supported local drive"); return NULL;
  }
  return path;
}
static napi_value private_directory(napi_env env, napi_callback_info info) {
  napi_value arg, result; size_t argc = 1;
  if (napi_get_cb_info(env, info, &argc, &arg, NULL, NULL) != napi_ok || argc != 1)
    return failure(env, "Expected a private directory path");
  WCHAR *path = private_path(env, arg);
  if (!path) return NULL;
  DWORD error = magnitude_prepare_private_directory(path);
  free(path);
  if (error) return failure(env, "Unsafe or inaccessible private directory");
  napi_get_undefined(env, &result); return result;
}
static napi_value private_content(napi_env env, napi_callback_info info) {
  napi_value arg, result; size_t argc = 1;
  if (napi_get_cb_info(env, info, &argc, &arg, NULL, NULL) != napi_ok || argc != 1)
    return failure(env, "Expected a private file path");
  WCHAR *path = private_path(env, arg);
  if (!path) return NULL;
  DWORD error = magnitude_validate_private_content(path);
  free(path);
  if (error) return failure(env, "Unsafe or inaccessible private file");
  napi_get_undefined(env, &result); return result;
}
static napi_value create_private_content(napi_env env, napi_callback_info info) {
  napi_value arg, result; size_t argc = 1;
  if (napi_get_cb_info(env, info, &argc, &arg, NULL, NULL) != napi_ok || argc != 1)
    return failure(env, "Expected a new private file path");
  WCHAR *path = private_path(env, arg);
  if (!path) return NULL;
  DWORD error = magnitude_create_private_content(path);
  free(path);
  if (error) return failure(env, "Cannot create private file");
  napi_get_undefined(env, &result); return result;
}
#endif

static napi_value acquire(napi_env env, napi_callback_info info) {
  napi_value arg, result;
  size_t argc = 1;
  if (napi_get_cb_info(env, info, &argc, &arg, NULL, NULL) != napi_ok || argc != 1)
    return failure(env, "Expected an absolute private lock-file path");
  owner_lock *lock = calloc(1, sizeof(*lock));
  if (!lock) return failure(env, "Cannot allocate ownership lock");
#ifdef _WIN32
  WCHAR *path = private_path(env, arg);
  if (!path) { free(lock); return NULL; }
  DWORD opened = magnitude_open_private_lock(path, &lock->handle, &lock->directory);
  free(path);
  if (opened) {
    free(lock); return failure(env, "Unsafe or inaccessible private ownership directory or lock");
  }
  OVERLAPPED overlap = {0};
  if (!LockFileEx(lock->handle, LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY,
      0, 1, 0, &overlap)) {
    DWORD error = GetLastError();
    release_lock(lock); free(lock);
    if (error == ERROR_LOCK_VIOLATION) { napi_get_null(env, &result); return result; }
    return failure(env, "Cannot acquire ownership lock");
  }
#else
  size_t length;
  if (napi_get_value_string_utf8(env, arg, NULL, 0, &length) != napi_ok || !length) {
    free(lock); return failure(env, "Invalid lock-file path");
  }
  char *path = calloc(length + 1, 1);
  if (!path) { free(lock); return failure(env, "Cannot allocate lock-file path"); }
  if (napi_get_value_string_utf8(env, arg, path, length + 1, &length) != napi_ok) {
    free(path); free(lock); return failure(env, "Invalid lock-file path");
  }
  for (size_t index = 0; index < length; ++index) if (!path[index]) {
    free(path); free(lock); return failure(env, "Lock path cannot contain NUL");
  }
  if (path[0] != '/') { free(path); free(lock); return failure(env, "Lock path must be absolute"); }
  lock->fd = open(path, O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW, 0600);
  free(path);
  if (lock->fd < 0) { free(lock); return failure(env, "Cannot open private ownership lock"); }
  struct stat file;
  if (fstat(lock->fd, &file) != 0 || !S_ISREG(file.st_mode) ||
      file.st_uid != getuid() || (file.st_mode & 077) || file.st_nlink != 1) {
    close(lock->fd); free(lock); return failure(env, "Unsafe ownership lock file");
  }
  if (flock(lock->fd, LOCK_EX | LOCK_NB) != 0) {
    int error = errno;
    close(lock->fd); free(lock);
    if (error == EWOULDBLOCK || error == EAGAIN) { napi_get_null(env, &result); return result; }
    return failure(env, "Cannot acquire ownership lock");
  }
#endif
  if (napi_create_object(env, &result) != napi_ok || napi_type_tag_object(env, result, &lock_tag) != napi_ok ||
      napi_wrap(env, result, lock, finalize_lock, NULL, NULL) != napi_ok) {
    release_lock(lock); free(lock); return failure(env, "Cannot retain ownership lock");
  }
  return result;
}

static napi_value release(napi_env env, napi_callback_info info) {
  napi_value arg, result;
  size_t argc = 1;
  void *data;
  bool matches = false;
  if (napi_get_cb_info(env, info, &argc, &arg, NULL, NULL) != napi_ok || argc != 1 ||
      napi_check_object_type_tag(env, arg, &lock_tag, &matches) != napi_ok || !matches ||
      napi_unwrap(env, arg, &data) != napi_ok)
    return failure(env, "Invalid ownership lock");
  release_lock(data);
  napi_get_undefined(env, &result);
  return result;
}

#ifdef _WIN32
static napi_value lock_endpoint(napi_env env, napi_callback_info info) {
  napi_value arg, result; size_t argc = 1; bool matches = false; owner_lock *lock = NULL;
  if (napi_get_cb_info(env, info, &argc, &arg, NULL, NULL) != napi_ok || argc != 1 ||
      napi_check_object_type_tag(env, arg, &lock_tag, &matches) != napi_ok || !matches ||
      napi_unwrap(env, arg, (void **)&lock) != napi_ok || !lock || lock->released)
    return failure(env, "Expected a live ownership lock");
  WCHAR endpoint[128];
  if (magnitude_directory_endpoint(lock->directory, endpoint) != ERROR_SUCCESS)
    return failure(env, "Cannot resolve the owned application directory identity");
  napi_create_string_utf16(env, (char16_t *)endpoint, NAPI_AUTO_LENGTH, &result); return result;
}
static napi_value inspect_endpoint(napi_env env, napi_callback_info info) {
  napi_value arg, result; size_t argc = 1;
  if (napi_get_cb_info(env, info, &argc, &arg, NULL, NULL) != napi_ok || argc != 1)
    return failure(env, "Expected a private application directory");
  WCHAR *path = private_path(env, arg);
  if (!path) return NULL;
  WCHAR endpoint[128]; BOOL missing;
  DWORD error = magnitude_inspect_application_endpoint(path, endpoint, &missing);
  free(path);
  if (error) return failure(env, "Cannot inspect the private application directory identity");
  if (missing) napi_get_null(env, &result);
  else napi_create_string_utf16(env, (char16_t *)endpoint, NAPI_AUTO_LENGTH, &result);
  return result;
}
#endif

#ifndef _WIN32
static int guard_started = 0;
static void *watch_parent(void *argument) {
  int fd = (int)(intptr_t)argument;
  char buffer[64];
  for (;;) {
    ssize_t count = read(fd, buffer, sizeof(buffer));
    if (count > 0 || (count < 0 && errno == EINTR)) continue;
    /* The child must remain its original process-group leader. */
    kill(-getpid(), SIGKILL);
    _exit(91);
  }
}
#endif

static napi_value guard(napi_env env, napi_callback_info info) {
#ifdef _WIN32
  static int guarded = 0;
  napi_value arg, result; size_t argc = 1; int32_t descriptor;
  if (napi_get_cb_info(env, info, &argc, &arg, NULL, NULL) != napi_ok || argc != 1 ||
      napi_get_value_int32(env, arg, &descriptor) != napi_ok || descriptor != 0)
    return failure(env, "Invalid Windows ownership admission");
  if (guarded) return failure(env, "Parent ownership already validated");
  if (magnitude_owned_validate_current() != ERROR_SUCCESS)
    return failure(env, "Windows child requires a containing kill-on-close job without breakaway");
  guarded = 1;
  napi_get_undefined(env, &result); return result;
#else
  napi_value arg, result;
  size_t argc = 1;
  int32_t source;
  if (napi_get_cb_info(env, info, &argc, &arg, NULL, NULL) != napi_ok || argc != 1 ||
      napi_get_value_int32(env, arg, &source) != napi_ok || source < 0)
    return failure(env, "Invalid parent lifetime descriptor");
  if (guard_started) return failure(env, "Parent lifetime guard already installed");
  if (getpid() != getpgrp()) return failure(env, "Owned child must lead its process group");
  struct stat channel;
  if (fstat(source, &channel) != 0 || !(S_ISFIFO(channel.st_mode) || S_ISSOCK(channel.st_mode)))
    return failure(env, "Parent lifetime channel must be a pipe or socket");
  int flags = fcntl(source, F_GETFL);
  if (flags < 0 || (flags & O_NONBLOCK)) return failure(env, "Parent lifetime channel must block");
  int fd = fcntl(source, F_DUPFD_CLOEXEC, 3);
  if (fd < 0) return failure(env, "Cannot retain parent lifetime channel");
  pthread_t thread;
  int error = pthread_create(&thread, NULL, watch_parent, (void *)(intptr_t)fd);
  if (error) { close(fd); return failure(env, "Cannot start native lifetime guard"); }
  guard_started = 1;
  pthread_detach(thread);
  napi_get_undefined(env, &result);
  return result;
#endif
}

#ifdef _WIN32
static napi_value interactive_desktop(napi_env env, napi_callback_info info) {
  (void)info;
  USEROBJECTFLAGS flags;
  DWORD needed = 0;
  WCHAR desktop[256];
  HWINSTA station = GetProcessWindowStation();
  if (!station || !GetUserObjectInformationW(station, UOI_FLAGS, &flags, sizeof(flags), &needed))
    return failure(env, "Cannot inspect the current Windows window station");
  BOOL available = FALSE;
  if (flags.dwFlags & WSF_VISIBLE) {
    HDESK current = GetThreadDesktop(GetCurrentThreadId());
    if (!current || !GetUserObjectInformationW(current, UOI_NAME, desktop, sizeof(desktop), &needed))
      return failure(env, "Cannot inspect the current Windows desktop");
    available = _wcsicmp(desktop, L"Default") == 0;
  }
  /* Inspect our assigned desktop, not the input desktop: a locked session may still
     start its background owner, ready for the user when they unlock it. */
  napi_value result;
  napi_get_boolean(env, available != FALSE, &result);
  return result;
}
#endif

#ifdef __linux__
static napi_value adopt_installation_lease(napi_env env, napi_callback_info info) {
  (void)info;
  struct stat inherited, installed;
  const int descriptor = 9;
  if (fstat(descriptor, &inherited) != 0 ||
      lstat("/var/lib/magnitude-desktop/installation.lock", &installed) != 0 ||
      !S_ISREG(inherited.st_mode) || inherited.st_uid != 0 || (inherited.st_mode & 0222) ||
      inherited.st_dev != installed.st_dev || inherited.st_ino != installed.st_ino)
    return failure(env, "Start Magnitude through its installed desktop launcher");
  if (flock(descriptor, LOCK_SH | LOCK_NB) != 0 ||
      lstat("/var/lib/magnitude-desktop/installing", &installed) == 0 || errno != ENOENT)
    return failure(env, "Magnitude installation is in progress or needs package-manager repair");
  int flags = fcntl(descriptor, F_GETFD);
  if (flags < 0 || fcntl(descriptor, F_SETFD, flags | FD_CLOEXEC) != 0)
    return failure(env, "Cannot retain private installation admission");
  napi_value result;
  napi_get_undefined(env, &result); return result;
}
#endif

static napi_value init(napi_env env, napi_value exports) {
  napi_property_descriptor methods[] = {
    {"acquireLock", NULL, acquire, NULL, NULL, NULL, napi_default, NULL},
    {"releaseLock", NULL, release, NULL, NULL, NULL, napi_default, NULL},
    {"guardParent", NULL, guard, NULL, NULL, NULL, napi_default, NULL},
#ifdef __linux__
    {"adoptInstallationLease", NULL, adopt_installation_lease, NULL, NULL, NULL, napi_default, NULL},
#endif
#ifdef _WIN32
    {"lockEndpoint", NULL, lock_endpoint, NULL, NULL, NULL, napi_default, NULL},
    {"inspectApplicationEndpoint", NULL, inspect_endpoint, NULL, NULL, NULL, napi_default, NULL},
    {"localAppDataDirectory", NULL, local_app_data, NULL, NULL, NULL, napi_default, NULL},
    {"preparePrivateDirectory", NULL, private_directory, NULL, NULL, NULL, napi_default, NULL},
    {"createPrivateContent", NULL, create_private_content, NULL, NULL, NULL, napi_default, NULL},
    {"validatePrivateContent", NULL, private_content, NULL, NULL, NULL, napi_default, NULL},
    {"isInteractiveDesktop", NULL, interactive_desktop, NULL, NULL, NULL, napi_default, NULL},
#endif
  };
  napi_define_properties(env, exports, sizeof(methods) / sizeof(methods[0]), methods);
  #ifdef _WIN32
  magnitude_register_windows_pipes(env, exports);
  magnitude_register_windows_jobs(env, exports);
  magnitude_register_windows_observers(env, exports);
  magnitude_register_windows_updates(env, exports);
  #endif
  magnitude_register_application_memory(env, exports);
  magnitude_register_machine_identity(env, exports);
  return exports;
}
NAPI_MODULE(NODE_GYP_MODULE_NAME, init)
