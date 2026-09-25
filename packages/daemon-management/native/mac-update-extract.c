/* This standalone process extracts only into an empty private staging directory.
   The transaction owner authenticates and retains the archive before invoking it. */
#include "vendor/libarchive/archive.h"
#include "vendor/libarchive/archive_entry.h"
#include <CommonCrypto/CommonDigest.h>
#include <sys/acl.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define ENTRY_LIMIT 65536
#define TABLE_SIZE (ENTRY_LIMIT * 2)
#define EXPANDED_LIMIT (UINT64_C(8) * 1024 * 1024 * 1024)
typedef struct { char *path; int symlink; } extracted_entry;
static extracted_entry entries[TABLE_SIZE];

static int safe_components(const char *path, int directory) {
  if (!path || !*path || strlen(path) >= PATH_MAX || strchr(path, '\\') || strchr(path, ':')) return 0;
  const char *part = path;
  for (const char *cursor = path; ; cursor++) {
    if (*cursor && *cursor != '/') continue;
    size_t length = (size_t)(cursor - part);
    if (!length || (length == 1 && part[0] == '.') || (length == 2 && part[0] == '.' && part[1] == '.')) return 0;
    if (!*cursor || (directory && !cursor[1])) return 1;
    part = cursor + 1;
  }
}
static int remember(const char *path, int symlink) {
  uint32_t hash = 2166136261u;
  for (const unsigned char *cursor = (const unsigned char *)path; *cursor; cursor++) hash = (hash ^ *cursor) * 16777619u;
  size_t slot = hash % TABLE_SIZE;
  while (entries[slot].path) {
    if (!strcmp(entries[slot].path, path)) return 0;
    slot = (slot + 1) % TABLE_SIZE;
  }
  entries[slot].path = strdup(path);
  entries[slot].symlink = symlink;
  return entries[slot].path != NULL;
}
static int no_extended_acl(int fd) {
  acl_t acl = acl_get_fd_np(fd, ACL_TYPE_EXTENDED);
  if (!acl) return errno == ENOENT;
  acl_entry_t entry;
  int empty = acl_valid(acl) == 0 && acl_get_entry(acl, ACL_FIRST_ENTRY, &entry) == -1 && errno == EINVAL;
  acl_free(acl); return empty;
}
/* Hash the same retained descriptor the parser consumes; path substitution cannot change its input. */
static int verify_archive(int fd, const char *expected, uint64_t bytes) {
  struct stat before, after;
  if (fstat(fd, &before) || !S_ISREG(before.st_mode) || before.st_uid != geteuid() || before.st_nlink != 1 ||
      (before.st_mode & 022) || before.st_size < 0 || (uint64_t)before.st_size != bytes || !no_extended_acl(fd) ||
      lseek(fd, 0, SEEK_SET) != 0) return 0;
  CC_SHA256_CTX hash;
  if (!CC_SHA256_Init(&hash)) return 0;
  unsigned char buffer[65536], digest[CC_SHA256_DIGEST_LENGTH];
  uint64_t total = 0;
  for (;;) {
    ssize_t count = read(fd, buffer, sizeof(buffer));
    if (count < 0 && errno == EINTR) continue;
    if (count < 0 || (uint64_t)count > bytes - total) return 0;
    if (!count) break;
    if (!CC_SHA256_Update(&hash, buffer, (CC_LONG)count)) return 0;
    total += (uint64_t)count;
  }
  char actual[CC_SHA256_DIGEST_LENGTH * 2 + 1];
  if (total != bytes || !CC_SHA256_Final(digest, &hash)) return 0;
  for (size_t index = 0; index < sizeof(digest); index++) snprintf(actual + index * 2, 3, "%02x", digest[index]);
  return !strcmp(actual, expected) && fstat(fd, &after) == 0 && before.st_size == after.st_size &&
    before.st_mtimespec.tv_sec == after.st_mtimespec.tv_sec && before.st_mtimespec.tv_nsec == after.st_mtimespec.tv_nsec &&
    before.st_ctimespec.tv_sec == after.st_ctimespec.tv_sec && before.st_ctimespec.tv_nsec == after.st_ctimespec.tv_nsec &&
    lseek(fd, 0, SEEK_SET) == 0;
}
static int empty_private_directory(int fd) {
  struct stat st;
  if (fstat(fd, &st) || !S_ISDIR(st.st_mode) || st.st_uid != geteuid() || (st.st_mode & 077) || !no_extended_acl(fd)) return 0;
  int copy = dup(fd);
  if (copy < 0) return 0;
  DIR *directory = fdopendir(copy);
  if (!directory) { close(copy); return 0; }
  int empty = 1;
  struct dirent *entry;
  errno = 0;
  while ((entry = readdir(directory))) {
    if (strcmp(entry->d_name, ".") && strcmp(entry->d_name, "..")) { empty = 0; break; }
  }
  if (errno) empty = 0;
  closedir(directory);
  return empty;
}
static int validate_links(void) {
  char root[PATH_MAX], target[PATH_MAX];
  if (!realpath("Magnitude.app", root)) return 0;
  size_t length = strlen(root);
  for (size_t index = 0; index < TABLE_SIZE; index++) {
    if (!entries[index].path || !entries[index].symlink) continue;
    if (!realpath(entries[index].path, target) || strncmp(target, root, length) || target[length] != '/') return 0;
  }
  return 1;
}
static int extract(int source) {
  struct archive *reader = archive_read_new(), *writer = archive_write_disk_new();
  int success = 0;
  uint64_t expanded = 0;
  size_t count = 0;
  if (!reader || !writer) goto done;
  if (archive_read_support_format_zip_seekable(reader) != ARCHIVE_OK ||
      archive_read_set_format_option(reader, "zip", "mac-ext", "1") != ARCHIVE_OK ||
      archive_read_open_fd(reader, source, 65536) != ARCHIVE_OK ||
      archive_write_disk_set_options(writer, ARCHIVE_EXTRACT_PERM | ARCHIVE_EXTRACT_TIME |
        ARCHIVE_EXTRACT_MAC_METADATA | ARCHIVE_EXTRACT_XATTR | ARCHIVE_EXTRACT_SECURE_SYMLINKS |
        ARCHIVE_EXTRACT_SECURE_NODOTDOT | ARCHIVE_EXTRACT_SECURE_NOABSOLUTEPATHS) != ARCHIVE_OK) goto done;
  struct archive_entry *entry;
  int status;
  while ((status = archive_read_next_header(reader, &entry)) == ARCHIVE_OK) {
    const char *name = archive_entry_pathname(entry);
    mode_t type = archive_entry_filetype(entry);
    int directory = type == AE_IFDIR, symlink = type == AE_IFLNK;
    if (++count > ENTRY_LIMIT || !safe_components(name, directory) ||
        (strncmp(name, "Magnitude.app/", 14) && strcmp(name, "Magnitude.app")) ||
        (!directory && !strcmp(name, "Magnitude.app")) ||
        (!directory && !symlink && type != AE_IFREG) || archive_entry_hardlink(entry) ||
        archive_entry_is_encrypted(entry) || (archive_entry_perm(entry) & 07000)) goto done;
    if (symlink && !safe_components(archive_entry_symlink(entry), 0)) goto done;
    char path[PATH_MAX];
    memcpy(path, name, strlen(name) + 1);
    size_t length = strlen(path);
    if (path[length - 1] == '/') path[length - 1] = 0;
    if (!remember(path, symlink)) goto done;
    struct stat previous;
    if (lstat(path, &previous) == 0) {
      if (!directory || !S_ISDIR(previous.st_mode)) goto done;
    } else if (errno != ENOENT) goto done;
    la_int64_t size = archive_entry_size(entry);
    if (size < 0 || (uint64_t)size > EXPANDED_LIMIT - expanded) goto done;
    if (archive_write_header(writer, entry) != ARCHIVE_OK) goto done;
    const void *bytes;
    size_t length_read;
    la_int64_t offset;
    uint64_t copied = 0;
    while ((status = archive_read_data_block(reader, &bytes, &length_read, &offset)) == ARCHIVE_OK) {
      if (offset < 0 || (uint64_t)offset != copied || length_read > EXPANDED_LIMIT - expanded ||
          length_read > (uint64_t)size - copied) goto done;
      if (archive_write_data_block(writer, bytes, length_read, offset) != ARCHIVE_OK) goto done;
      copied += length_read;
      expanded += length_read;
    }
    if (status != ARCHIVE_EOF || (type == AE_IFREG && copied != (uint64_t)size) ||
        archive_write_finish_entry(writer) != ARCHIVE_OK) goto done;
  }
  if (status != ARCHIVE_EOF || !count || archive_read_has_encrypted_entries(reader) > 0 ||
      archive_write_close(writer) != ARCHIVE_OK || !validate_links()) goto done;
  success = 1;
done:
  if (!success) fprintf(stderr, "The update archive cannot be extracted safely.\n");
  if (writer) archive_write_free(writer);
  if (reader) archive_read_free(reader);
  for (size_t index = 0; index < TABLE_SIZE; index++) free(entries[index].path);
  return success;
}
int main(int argc, char **argv) {
  if (argc != 5 || argv[1][0] != '/' || argv[2][0] != '/' || strlen(argv[3]) != 64 ||
      strspn(argv[3], "0123456789abcdef") != 64 || !*argv[4] || strlen(argv[4]) > 16 ||
      strspn(argv[4], "0123456789") != strlen(argv[4])) {
    fprintf(stderr, "Expected archive, private staging directory, authenticated digest and byte count.\n"); return 1;
  }
  uint64_t expected_bytes = strtoull(argv[4], NULL, 10);
  if (!expected_bytes || expected_bytes > EXPANDED_LIMIT) return 1;
  int source = open(argv[1], O_RDONLY | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK);
  int destination = open(argv[2], O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
  int valid = source >= 0 && destination >= 0 && empty_private_directory(destination) &&
    verify_archive(source, argv[3], expected_bytes) && fchdir(destination) == 0 && extract(source) &&
    verify_archive(source, argv[3], expected_bytes);
  if (destination >= 0) close(destination);
  if (source >= 0) close(source);
  return valid ? 0 : 1;
}
