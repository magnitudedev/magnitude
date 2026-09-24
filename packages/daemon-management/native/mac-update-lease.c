/* Installation admission survives bundle exchange and is never inherited by children. */
#include <node_api.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/acl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

typedef struct {
  int fd, parent;
  char path[PATH_MAX], name[NAME_MAX + 1];
  struct stat identity, parent_identity;
} installation_lease;
static const napi_type_tag lease_tag = { UINT64_C(0x6e1317094ec44eca), UINT64_C(0x972889f17f4195dd) };
static napi_value fail(napi_env env) {
  napi_throw_error(env, NULL, "The macOS installation lease could not be acquired safely."); return NULL;
}
static napi_value nothing(napi_env env) { napi_value value; napi_get_undefined(env, &value); return value; }
static int same(struct stat a, struct stat b) { return a.st_dev == b.st_dev && a.st_ino == b.st_ino; }
static int no_acl(int fd) {
  acl_t acl = acl_get_fd_np(fd, ACL_TYPE_EXTENDED);
  if (!acl) return errno == ENOENT;
  acl_entry_t entry;
  int empty = acl_valid(acl) == 0 && acl_get_entry(acl, ACL_FIRST_ENTRY, &entry) == -1 && errno == EINVAL;
  acl_free(acl); return empty;
}
static void release(installation_lease *lease) {
  if (lease->fd >= 0) { close(lease->fd); lease->fd = -1; }
  if (lease->parent >= 0) { close(lease->parent); lease->parent = -1; }
}
static void finalize(napi_env env, void *data, void *hint) {
  (void)env; (void)hint; release(data); free(data);
}
static installation_lease *unwrap(napi_env env, napi_value value) {
  bool tagged = false; void *data = NULL;
  if (napi_check_object_type_tag(env, value, &lease_tag, &tagged) != napi_ok || !tagged ||
      napi_unwrap(env, value, &data) != napi_ok) return NULL;
  return data;
}
static int current(installation_lease *lease) {
  struct stat parent, named, retained;
  return lease->fd >= 0 && lease->parent >= 0 &&
    lstat(lease->path, &parent) == 0 && S_ISDIR(parent.st_mode) && same(parent, lease->parent_identity) &&
    parent.st_uid == lease->parent_identity.st_uid && !(parent.st_mode & S_IWOTH) &&
    fstatat(lease->parent, lease->name, &named, AT_SYMLINK_NOFOLLOW) == 0 && same(named, lease->identity) &&
    fstat(lease->fd, &retained) == 0 && same(named, retained) && S_ISREG(retained.st_mode) &&
    retained.st_uid == lease->identity.st_uid && retained.st_nlink == 1 && retained.st_size == 0 &&
    (retained.st_mode & 07777) == 0644 && no_acl(lease->fd);
}
static napi_value acquire(napi_env env, napi_callback_info info) {
  napi_value args[2], result; size_t argc = 2, length; bool exclusive;
  char supplied[PATH_MAX], bundle[NAME_MAX + 1]; struct stat app, opened;
  if (napi_get_cb_info(env, info, &argc, args, NULL, NULL) != napi_ok || argc != 2 ||
      napi_get_value_string_utf8(env, args[0], NULL, 0, &length) != napi_ok || !length || length >= sizeof(supplied) ||
      napi_get_value_string_utf8(env, args[0], supplied, sizeof(supplied), &length) != napi_ok ||
      memchr(supplied, 0, length) || supplied[0] != '/' ||
      napi_get_value_bool(env, args[1], &exclusive) != napi_ok) return fail(env);
  char *slash = strrchr(supplied, '/');
  if (!slash || !slash[1] || strlen(slash + 1) + sizeof("..installation.lock") > sizeof(bundle) ||
      !strcmp(slash + 1, ".") || !strcmp(slash + 1, "..")) return fail(env);
  strcpy(bundle, slash + 1);
  if (slash == supplied) slash[1] = 0; else *slash = 0;
  installation_lease *lease = calloc(1, sizeof(*lease));
  if (!lease) return fail(env);
  lease->fd = lease->parent = -1;
  if (!realpath(supplied, lease->path)) goto failed;
  lease->parent = open(lease->path, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
  if (lease->parent < 0 || fstat(lease->parent, &lease->parent_identity) ||
      (lease->parent_identity.st_mode & S_IWOTH) ||
      fstatat(lease->parent, bundle, &app, AT_SYMLINK_NOFOLLOW) || !S_ISDIR(app.st_mode)) goto failed;
  snprintf(lease->name, sizeof(lease->name), ".%s.installation.lock", bundle);
  lease->fd = openat(lease->parent, lease->name, O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC);
  if (lease->fd < 0 && errno == ENOENT && app.st_uid == geteuid()) {
    lease->fd = openat(lease->parent, lease->name, O_RDWR | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0644);
    if (lease->fd >= 0) {
      if (fchmod(lease->fd, 0644) || fsync(lease->fd) || fsync(lease->parent)) goto failed;
    } else if (errno == EEXIST) {
      lease->fd = openat(lease->parent, lease->name, O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC);
    }
  }
  if (lease->fd < 0 || fstat(lease->fd, &lease->identity) || lease->identity.st_uid != app.st_uid ||
      !current(lease)) goto failed;
  if (flock(lease->fd, (exclusive ? LOCK_EX : LOCK_SH) | LOCK_NB)) {
    if (errno != EWOULDBLOCK && errno != EAGAIN) goto failed;
    release(lease); free(lease); napi_get_null(env, &result); return result;
  }
  if (!current(lease) || fstatat(lease->parent, bundle, &opened, AT_SYMLINK_NOFOLLOW) || !same(app, opened)) goto failed;
  if (napi_create_object(env, &result) != napi_ok || napi_type_tag_object(env, result, &lease_tag) != napi_ok ||
      napi_wrap(env, result, lease, finalize, NULL, NULL) != napi_ok) goto failed;
  return result;
failed:
  release(lease); free(lease); return fail(env);
}
static napi_value close_lease(napi_env env, napi_callback_info info) {
  napi_value arg; size_t argc = 1;
  if (napi_get_cb_info(env, info, &argc, &arg, NULL, NULL) != napi_ok || argc != 1) return fail(env);
  installation_lease *lease = unwrap(env, arg);
  if (!lease) return fail(env);
  release(lease); return nothing(env);
}
static napi_value validate(napi_env env, napi_callback_info info) {
  napi_value arg; size_t argc = 1;
  if (napi_get_cb_info(env, info, &argc, &arg, NULL, NULL) != napi_ok || argc != 1) return fail(env);
  installation_lease *lease = unwrap(env, arg);
  return lease && current(lease) ? nothing(env) : fail(env);
}
void magnitude_register_mac_update_lease(napi_env env, napi_value exports) {
  napi_property_descriptor methods[] = {
    {"acquireMacUpdateLease", NULL, acquire, NULL, NULL, NULL, napi_default, NULL},
    {"releaseMacUpdateLease", NULL, close_lease, NULL, NULL, NULL, napi_default, NULL},
    {"validateMacUpdateLease", NULL, validate, NULL, NULL, NULL, napi_default, NULL},
  };
  napi_define_properties(env, exports, sizeof(methods) / sizeof(methods[0]), methods);
}
