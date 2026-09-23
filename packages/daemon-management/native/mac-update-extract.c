/* This standalone process extracts only into an empty private staging directory.
   The transaction owner authenticates and retains the archive before invoking it. */
#include "vendor/libarchive/archive.h"
#include "vendor/libarchive/archive_entry.h"
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
static int empty_private_directory(int fd) {
  struct stat st;
  if (fstat(fd, &st) || !S_ISDIR(st.st_mode) || st.st_uid != geteuid() || (st.st_mode & 077)) return 0;
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
  if (argc != 3 || argv[1][0] != '/' || argv[2][0] != '/') {
    fprintf(stderr, "Expected archive and private staging directory paths.\n"); return 1;
  }
  int source = open(argv[1], O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
  int destination = open(argv[2], O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
  struct stat st;
  int valid = source >= 0 && destination >= 0 && fstat(source, &st) == 0 && S_ISREG(st.st_mode) &&
    empty_private_directory(destination) && fchdir(destination) == 0 && extract(source);
  if (destination >= 0) close(destination);
  if (source >= 0) close(source);
  return valid ? 0 : 1;
}
