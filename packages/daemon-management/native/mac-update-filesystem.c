/* Bounded synchronous operations for the finite installer process, under installation exclusion. */
#include <node_api.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/acl.h>
#include <sys/stat.h>
#include <unistd.h>

#define RECORD_LIMIT 16384
typedef struct { int fd, private_directory; char path[PATH_MAX]; struct stat identity; } update_directory;
static const napi_type_tag directory_tag = { UINT64_C(0xd5226c719d3447df), UINT64_C(0x9cfb56b6e2ae72a8) };
static napi_value fail(napi_env env) {
  napi_throw_error(env, NULL, "The macOS update filesystem operation could not be completed safely."); return NULL;
}
static napi_value nothing(napi_env env) { napi_value value; napi_get_undefined(env, &value); return value; }
static int same(struct stat a, struct stat b) { return a.st_dev == b.st_dev && a.st_ino == b.st_ino; }
static int string(napi_env env, napi_value value, char *output, size_t capacity) {
  size_t length;
  return napi_get_value_string_utf8(env, value, NULL, 0, &length) == napi_ok && length && length < capacity &&
    napi_get_value_string_utf8(env, value, output, capacity, &length) == napi_ok && !memchr(output, 0, length);
}
static int leaf(napi_env env, napi_value value, char *output) {
  return string(env, value, output, NAME_MAX + 1) && strcmp(output, ".") && strcmp(output, "..") &&
    !strchr(output, '/') && !strchr(output, '\\') && !strchr(output, ':');
}
static void identity_text(struct stat st, char *text) {
  snprintf(text, 64, "%llu:%llu", (unsigned long long)(uint32_t)st.st_dev, (unsigned long long)st.st_ino);
}
static int no_extended_acl(int fd) {
  acl_t acl = acl_get_fd_np(fd, ACL_TYPE_EXTENDED);
  if (!acl) return errno == ENOENT;
  acl_entry_t entry;
  int empty = acl_valid(acl) == 0 && acl_get_entry(acl, ACL_FIRST_ENTRY, &entry) == -1 && errno == EINVAL;
  acl_free(acl);
  return empty;
}
static void release(update_directory *directory) {
  if (directory->fd >= 0) { close(directory->fd); directory->fd = -1; }
}
static void finalize(napi_env env, void *data, void *hint) {
  (void)env; (void)hint; release(data); free(data);
}
static update_directory *unwrap(napi_env env, napi_value value, int active) {
  bool tagged = false;
  void *data = NULL;
  if (napi_check_object_type_tag(env, value, &directory_tag, &tagged) != napi_ok || !tagged ||
      napi_unwrap(env, value, &data) != napi_ok || !data) return NULL;
  update_directory *directory = data;
  if (active) {
    struct stat current;
    if (directory->fd < 0 || lstat(directory->path, &current) || !S_ISDIR(current.st_mode) ||
        !same(current, directory->identity) || current.st_uid != directory->identity.st_uid ||
        (current.st_mode & S_IWOTH) || (directory->private_directory &&
          ((current.st_mode & 077) || !no_extended_acl(directory->fd)))) return NULL;
  }
  return directory;
}
static napi_value acquire(napi_env env, napi_callback_info info) {
  napi_value args[2], result; size_t argc = 2;
  char path[PATH_MAX]; bool private_directory;
  struct stat supplied;
  if (napi_get_cb_info(env, info, &argc, args, NULL, NULL) != napi_ok || argc != 2 ||
      !string(env, args[0], path, sizeof(path)) || path[0] != '/' ||
      napi_get_value_bool(env, args[1], &private_directory) != napi_ok ||
      lstat(path, &supplied) || !S_ISDIR(supplied.st_mode)) return fail(env);
  update_directory *directory = calloc(1, sizeof(*directory));
  if (!directory) return fail(env);
  directory->fd = -1;
  directory->private_directory = private_directory;
  if (!realpath(path, directory->path)) goto failed;
  directory->fd = open(directory->path, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
  if (directory->fd < 0 || fstat(directory->fd, &directory->identity) || !same(supplied, directory->identity) ||
      (directory->identity.st_uid != geteuid() && directory->identity.st_uid != 0) ||
      (directory->identity.st_mode & S_IWOTH)) goto failed;
  if (private_directory && (directory->identity.st_uid != geteuid() ||
      (directory->identity.st_mode & 077) || !no_extended_acl(directory->fd))) goto failed;
  char identity[64]; napi_value encoded, encoded_path;
  identity_text(directory->identity, identity);
  if (napi_create_string_utf8(env, identity, NAPI_AUTO_LENGTH, &encoded) != napi_ok ||
      napi_create_string_utf8(env, directory->path, NAPI_AUTO_LENGTH, &encoded_path) != napi_ok) goto failed;
  napi_property_descriptor properties[] = {
    {"identity", NULL, NULL, NULL, NULL, encoded, napi_default, NULL},
    {"path", NULL, NULL, NULL, NULL, encoded_path, napi_default, NULL},
  };
  if (napi_create_object(env, &result) != napi_ok || napi_define_properties(env, result, 2, properties) != napi_ok ||
      napi_type_tag_object(env, result, &directory_tag) != napi_ok ||
      napi_wrap(env, result, directory, finalize, NULL, NULL) != napi_ok) goto failed;
  return result;
failed:
  release(directory); free(directory); return fail(env);
}
static napi_value close_directory(napi_env env, napi_callback_info info) {
  napi_value arg; size_t argc = 1;
  if (napi_get_cb_info(env, info, &argc, &arg, NULL, NULL) != napi_ok || argc != 1) return fail(env);
  update_directory *directory = unwrap(env, arg, 0);
  if (!directory) return fail(env);
  release(directory); return nothing(env);
}
static napi_value sync_directory(napi_env env, napi_callback_info info) {
  napi_value arg; size_t argc = 1;
  if (napi_get_cb_info(env, info, &argc, &arg, NULL, NULL) != napi_ok || argc != 1) return fail(env);
  update_directory *directory = unwrap(env, arg, 1);
  if (!directory || fsync(directory->fd)) return fail(env);
  return nothing(env);
}
static int sync_tree_contents(int fd, unsigned depth, unsigned *entries, uint64_t *bytes) {
  if (depth > 64) return 0;
  int duplicate = dup(fd);
  if (duplicate < 0) return 0;
  DIR *directory = fdopendir(duplicate);
  if (!directory) { close(duplicate); return 0; }
  int valid = 1;
  struct dirent *entry;
  for (;;) {
    errno = 0;
    entry = readdir(directory);
    if (!entry) { if (errno) valid = 0; break; }
    if (!strcmp(entry->d_name, ".") || !strcmp(entry->d_name, "..")) continue;
    struct stat before, opened;
    if (++*entries > 65536 || fstatat(fd, entry->d_name, &before, AT_SYMLINK_NOFOLLOW)) { valid = 0; break; }
    if (S_ISLNK(before.st_mode)) continue;
    if ((!S_ISDIR(before.st_mode) && !S_ISREG(before.st_mode)) || before.st_uid != geteuid()) { valid = 0; break; }
    int child = openat(fd, entry->d_name, O_RDONLY | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK |
      (S_ISDIR(before.st_mode) ? O_DIRECTORY : 0));
    if (child < 0) { valid = 0; break; }
    valid = fstat(child, &opened) == 0 && same(before, opened);
    if (valid && S_ISDIR(before.st_mode)) valid = sync_tree_contents(child, depth + 1, entries, bytes);
    else if (valid) {
      const uint64_t limit = UINT64_C(8) * 1024 * 1024 * 1024;
      valid = opened.st_nlink == 1 && opened.st_size >= 0 && (uint64_t)opened.st_size <= limit - *bytes;
      if (valid) { *bytes += (uint64_t)opened.st_size; valid = fsync(child) == 0; }
    }
    close(child);
    if (!valid) break;
  }
  closedir(directory);
  return valid && fsync(fd) == 0;
}
static napi_value sync_tree(napi_env env, napi_callback_info info) {
  napi_value args[3]; size_t argc = 3;
  char name[NAME_MAX + 1], expected[64], actual[64]; struct stat before, opened, after;
  if (napi_get_cb_info(env, info, &argc, args, NULL, NULL) != napi_ok || argc != 3 ||
      !leaf(env, args[1], name) || !string(env, args[2], expected, sizeof(expected))) return fail(env);
  update_directory *directory = unwrap(env, args[0], 1);
  if (!directory || !directory->private_directory || fstatat(directory->fd, name, &before, AT_SYMLINK_NOFOLLOW) ||
      !S_ISDIR(before.st_mode) || before.st_uid != geteuid()) return fail(env);
  identity_text(before, actual);
  if (strcmp(actual, expected)) return fail(env);
  int fd = openat(directory->fd, name, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
  if (fd < 0) return fail(env);
  unsigned entries = 0; uint64_t bytes = 0;
  int valid = fstat(fd, &opened) == 0 && same(before, opened) && sync_tree_contents(fd, 0, &entries, &bytes) &&
    fstatat(directory->fd, name, &after, AT_SYMLINK_NOFOLLOW) == 0 && same(before, after) && fsync(directory->fd) == 0;
  close(fd);
  return valid ? nothing(env) : fail(env);
}
static int remove_tree_contents(int fd, unsigned depth, unsigned *entries) {
  if (depth > 64) return 0;
  int duplicate = dup(fd);
  if (duplicate < 0) return 0;
  DIR *directory = fdopendir(duplicate);
  if (!directory) { close(duplicate); return 0; }
  int valid = 1;
  for (;;) {
    errno = 0;
    struct dirent *entry = readdir(directory);
    if (!entry) { if (errno) valid = 0; break; }
    if (!strcmp(entry->d_name, ".") || !strcmp(entry->d_name, "..")) continue;
    struct stat before, opened, current;
    if (++*entries > 65536 || fstatat(fd, entry->d_name, &before, AT_SYMLINK_NOFOLLOW)) { valid = 0; break; }
    if (S_ISDIR(before.st_mode)) {
      int child = openat(fd, entry->d_name, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
      if (child < 0) { valid = 0; break; }
      valid = fstat(child, &opened) == 0 && same(before, opened) && remove_tree_contents(child, depth + 1, entries);
      close(child);
      if (!valid) break;
    }
    if (fstatat(fd, entry->d_name, &current, AT_SYMLINK_NOFOLLOW) || !same(before, current) ||
        unlinkat(fd, entry->d_name, S_ISDIR(before.st_mode) ? AT_REMOVEDIR : 0)) { valid = 0; break; }
  }
  closedir(directory);
  return valid && fsync(fd) == 0;
}
/* Only a terminal transaction authorizes this operation; the private journal remains intact. */
static napi_value remove_tree(napi_env env, napi_callback_info info) {
  napi_value args[3]; size_t argc = 3;
  char name[NAME_MAX + 1], expected[64], actual[64]; struct stat before, opened, after;
  if (napi_get_cb_info(env, info, &argc, args, NULL, NULL) != napi_ok || argc != 3 ||
      !leaf(env, args[1], name) || !string(env, args[2], expected, sizeof(expected))) return fail(env);
  update_directory *directory = unwrap(env, args[0], 1);
  if (!directory || !directory->private_directory || fstatat(directory->fd, name, &before, AT_SYMLINK_NOFOLLOW) ||
      !S_ISDIR(before.st_mode)) return fail(env);
  identity_text(before, actual);
  if (strcmp(actual, expected)) return fail(env);
  int fd = openat(directory->fd, name, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
  if (fd < 0) return fail(env);
  unsigned entries = 0;
  int valid = fstat(fd, &opened) == 0 && same(before, opened) && remove_tree_contents(fd, 0, &entries) &&
    fstatat(directory->fd, name, &after, AT_SYMLINK_NOFOLLOW) == 0 && same(before, after) &&
    unlinkat(directory->fd, name, AT_REMOVEDIR) == 0 && fsync(directory->fd) == 0;
  close(fd);
  return valid ? nothing(env) : fail(env);
}
static napi_value inspect(napi_env env, napi_callback_info info) {
  napi_value args[2], result; size_t argc = 2;
  char name[NAME_MAX + 1], identity[64]; struct stat st;
  if (napi_get_cb_info(env, info, &argc, args, NULL, NULL) != napi_ok || argc != 2 || !leaf(env, args[1], name)) return fail(env);
  update_directory *directory = unwrap(env, args[0], 1);
  if (!directory) return fail(env);
  if (fstatat(directory->fd, name, &st, AT_SYMLINK_NOFOLLOW)) {
    if (errno != ENOENT) return fail(env);
    napi_get_null(env, &result); return result;
  }
  if (!S_ISDIR(st.st_mode)) return fail(env);
  identity_text(st, identity);
  if (napi_create_string_utf8(env, identity, NAPI_AUTO_LENGTH, &result) != napi_ok) return fail(env);
  return result;
}
static napi_value read_record(napi_env env, napi_callback_info info) {
  napi_value arg, result; size_t argc = 1;
  if (napi_get_cb_info(env, info, &argc, &arg, NULL, NULL) != napi_ok || argc != 1) return fail(env);
  update_directory *directory = unwrap(env, arg, 1);
  if (!directory || !directory->private_directory) return fail(env);
  int fd = openat(directory->fd, "transaction.json", O_RDONLY | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK);
  if (fd < 0) {
    if (errno != ENOENT) return fail(env);
    napi_get_null(env, &result); return result;
  }
  struct stat st;
  char bytes[RECORD_LIMIT + 1]; size_t length = 0;
  int valid = fstat(fd, &st) == 0 && S_ISREG(st.st_mode) && st.st_nlink == 1 && st.st_uid == geteuid() &&
    !(st.st_mode & 077) && st.st_size > 0 && st.st_size <= RECORD_LIMIT && no_extended_acl(fd);
  while (valid && length < sizeof(bytes)) {
    ssize_t count = read(fd, bytes + length, sizeof(bytes) - length);
    if (count < 0 && errno == EINTR) continue;
    if (count < 0) { valid = 0; break; }
    if (!count) break;
    length += (size_t)count;
  }
  close(fd);
  if (!valid || length != (size_t)st.st_size || length > RECORD_LIMIT ||
      napi_create_buffer_copy(env, length, bytes, NULL, &result) != napi_ok) return fail(env);
  return result;
}
static napi_value remove_record(napi_env env, napi_callback_info info) {
  napi_value args[2]; size_t argc = 2, length; void *bytes; bool buffer = false;
  if (napi_get_cb_info(env, info, &argc, args, NULL, NULL) != napi_ok || argc != 2 ||
      napi_is_buffer(env, args[1], &buffer) != napi_ok || !buffer ||
      napi_get_buffer_info(env, args[1], &bytes, &length) != napi_ok || !length || length > RECORD_LIMIT) return fail(env);
  update_directory *directory = unwrap(env, args[0], 1);
  if (!directory || !directory->private_directory) return fail(env);
  int fd = openat(directory->fd, "transaction.json", O_RDONLY | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK);
  if (fd < 0) return fail(env);
  struct stat before, named;
  char actual[RECORD_LIMIT + 1]; size_t read_length = 0;
  int valid = fstat(fd, &before) == 0 && S_ISREG(before.st_mode) && before.st_uid == geteuid() &&
    before.st_nlink == 1 && !(before.st_mode & 077) && before.st_size == (off_t)length && no_extended_acl(fd);
  while (valid && read_length < sizeof(actual)) {
    ssize_t count = read(fd, actual + read_length, sizeof(actual) - read_length);
    if (count < 0 && errno == EINTR) continue;
    if (count < 0) { valid = 0; break; }
    if (!count) break;
    read_length += (size_t)count;
  }
  valid = valid && read_length == length && !memcmp(actual, bytes, length) &&
    fstatat(directory->fd, "transaction.json", &named, AT_SYMLINK_NOFOLLOW) == 0 && same(before, named) &&
    unlinkat(directory->fd, "transaction.json", 0) == 0 && fsync(directory->fd) == 0 && fcntl(fd, F_FULLFSYNC) == 0;
  close(fd);
  return valid ? nothing(env) : fail(env);
}
static napi_value write_record(napi_env env, napi_callback_info info) {
  napi_value args[2]; size_t argc = 2, length; void *bytes; bool buffer = false;
  if (napi_get_cb_info(env, info, &argc, args, NULL, NULL) != napi_ok || argc != 2 ||
      napi_is_buffer(env, args[1], &buffer) != napi_ok || !buffer ||
      napi_get_buffer_info(env, args[1], &bytes, &length) != napi_ok || !length || length > RECORD_LIMIT) return fail(env);
  update_directory *directory = unwrap(env, args[0], 1);
  if (!directory || !directory->private_directory) return fail(env);
  int previous = openat(directory->fd, "transaction.json", O_RDONLY | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK);
  if (previous >= 0) {
    struct stat st;
    int valid = fstat(previous, &st) == 0 && S_ISREG(st.st_mode) && st.st_uid == geteuid() &&
      st.st_nlink == 1 && !(st.st_mode & 077) && st.st_size > 0 && st.st_size <= RECORD_LIMIT && no_extended_acl(previous);
    close(previous);
    if (!valid) return fail(env);
  } else if (errno != ENOENT) return fail(env);
  char temporary[64];
  snprintf(temporary, sizeof(temporary), ".transaction-%08x%08x.tmp", arc4random(), arc4random());
  int fd = openat(directory->fd, temporary, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
  if (fd < 0) return fail(env);
  int valid = 1, renamed = 0; size_t written = 0;
  while (written < length) {
    ssize_t count = write(fd, (const char *)bytes + written, length - written);
    if (count < 0 && errno == EINTR) continue;
    if (count <= 0) { valid = 0; break; }
    written += (size_t)count;
  }
  if (valid) valid = fsync(fd) == 0 && fcntl(fd, F_FULLFSYNC) == 0;
  if (valid) { renamed = renameat(directory->fd, temporary, directory->fd, "transaction.json") == 0; valid = renamed; }
  if (valid) valid = fsync(directory->fd) == 0 && fcntl(fd, F_FULLFSYNC) == 0;
  close(fd);
  if (!renamed) unlinkat(directory->fd, temporary, 0);
  return valid ? nothing(env) : fail(env);
}
static napi_value exchange(napi_env env, napi_callback_info info) {
  napi_value args[6]; size_t argc = 6;
  char left_name[NAME_MAX + 1], right_name[NAME_MAX + 1], expected_left[64], expected_right[64], actual[64];
  struct stat left_st, right_st, observed;
  if (napi_get_cb_info(env, info, &argc, args, NULL, NULL) != napi_ok || argc != 6 ||
      !leaf(env, args[1], left_name) || !leaf(env, args[4], right_name) ||
      !string(env, args[2], expected_left, sizeof(expected_left)) ||
      !string(env, args[5], expected_right, sizeof(expected_right))) return fail(env);
  update_directory *left = unwrap(env, args[0], 1), *right = unwrap(env, args[3], 1);
  if (!left || !right || !right->private_directory ||
      fstatat(left->fd, left_name, &left_st, AT_SYMLINK_NOFOLLOW) || !S_ISDIR(left_st.st_mode) ||
      fstatat(right->fd, right_name, &right_st, AT_SYMLINK_NOFOLLOW) || !S_ISDIR(right_st.st_mode) ||
      same(left_st, right_st) || left_st.st_dev != right_st.st_dev) return fail(env);
  identity_text(left_st, actual); if (strcmp(actual, expected_left)) return fail(env);
  identity_text(right_st, actual); if (strcmp(actual, expected_right)) return fail(env);
  if (renameatx_np(left->fd, left_name, right->fd, right_name, RENAME_SWAP) ||
      fstatat(left->fd, left_name, &observed, AT_SYMLINK_NOFOLLOW) || !same(observed, right_st) ||
      fstatat(right->fd, right_name, &observed, AT_SYMLINK_NOFOLLOW) || !same(observed, left_st) ||
      fsync(left->fd) || fsync(right->fd)) return fail(env);
  return nothing(env);
}
void magnitude_register_mac_update_filesystem(napi_env env, napi_value exports) {
  napi_property_descriptor methods[] = {
    {"openMacUpdateDirectory", NULL, acquire, NULL, NULL, NULL, napi_default, NULL},
    {"closeMacUpdateDirectory", NULL, close_directory, NULL, NULL, NULL, napi_default, NULL},
    {"syncMacUpdateDirectory", NULL, sync_directory, NULL, NULL, NULL, napi_default, NULL},
    {"syncMacUpdateTree", NULL, sync_tree, NULL, NULL, NULL, napi_default, NULL},
    {"removeMacUpdateTree", NULL, remove_tree, NULL, NULL, NULL, napi_default, NULL},
    {"inspectMacUpdateDirectory", NULL, inspect, NULL, NULL, NULL, napi_default, NULL},
    {"readMacUpdateRecord", NULL, read_record, NULL, NULL, NULL, napi_default, NULL},
    {"removeMacUpdateRecord", NULL, remove_record, NULL, NULL, NULL, napi_default, NULL},
    {"writeMacUpdateRecord", NULL, write_record, NULL, NULL, NULL, napi_default, NULL},
    {"exchangeMacUpdateDirectories", NULL, exchange, NULL, NULL, NULL, napi_default, NULL},
  };
  napi_define_properties(env, exports, sizeof(methods) / sizeof(methods[0]), methods);
}
