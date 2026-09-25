/* Isolated mechanical acceptance; this fixture does not authorize application installation. */
#include <assert.h>
#include <copyfile.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#define CHECK(call) do { if (!(call)) { perror(#call); abort(); } } while (0)
typedef struct { dev_t device; ino_t inode; } Identity;
typedef struct { Identity previous, replacement; int committed; } Journal;

static Identity identity(int parent, const char *name) {
  struct stat st;
  CHECK(fstatat(parent, name, &st, AT_SYMLINK_NOFOLLOW) == 0);
  CHECK(S_ISDIR(st.st_mode));
  return (Identity){st.st_dev, st.st_ino};
}
static int equal(Identity a, Identity b) { return a.device == b.device && a.inode == b.inode; }
static int matches_directory(int parent, const char *name, Identity expected) {
  struct stat st;
  return fstatat(parent, name, &st, AT_SYMLINK_NOFOLLOW) == 0 && S_ISDIR(st.st_mode) &&
    equal((Identity){st.st_dev, st.st_ino}, expected);
}
static void persist(int fd, const Journal *journal) {
  CHECK(pwrite(fd, journal, sizeof(*journal), 0) == sizeof(*journal));
  CHECK(fsync(fd) == 0);
  CHECK(fcntl(fd, F_FULLFSYNC) == 0);
}
/* The journal alone cannot decide whether an exchange happened. Never replay the swap. */
static int classify(int parent, const Journal *journal) {
  if (equal(journal->previous, journal->replacement)) return -1;
  if (matches_directory(parent, "Installed.app", journal->previous) &&
      matches_directory(parent, "Staged.app", journal->replacement)) return 0;
  if (matches_directory(parent, "Installed.app", journal->replacement) &&
      matches_directory(parent, "Staged.app", journal->previous)) return 1;
  return -1;
}
static int open_directory(int parent, const char *name) {
  int fd = openat(parent, name, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
  CHECK(fd >= 0);
  return fd;
}
static void copy_executable(int parent, const char *name, const char *self) {
  int bundle = open_directory(parent, name);
  int source = open(self, O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
  int target = openat(bundle, "program", O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0700);
  CHECK(source >= 0 && target >= 0);
  CHECK(fcopyfile(source, target, NULL, COPYFILE_DATA) == 0);
  CHECK(fsync(target) == 0);
  CHECK(fcntl(target, F_FULLFSYNC) == 0);
  CHECK(fsync(bundle) == 0);
  CHECK(close(source) == 0 && close(target) == 0 && close(bundle) == 0);
}
static void remove_bundle(int parent, const char *name) {
  int bundle = open_directory(parent, name);
  CHECK(unlinkat(bundle, "program", 0) == 0);
  CHECK(close(bundle) == 0);
  CHECK(unlinkat(parent, name, AT_REMOVEDIR) == 0);
}
static void create_bundles(int parent, const char *self) {
  CHECK(mkdirat(parent, "Installed.app", 0700) == 0);
  CHECK(mkdirat(parent, "Staged.app", 0700) == 0);
  copy_executable(parent, "Installed.app", self);
  copy_executable(parent, "Staged.app", self);
  CHECK(fsync(parent) == 0);
}
static void crash_boundary(int parent, const char *self, int boundary) {
  create_bundles(parent, self);
  Journal journal = {identity(parent, "Installed.app"), identity(parent, "Staged.app"), 0};
  int record = openat(parent, "journal", O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
  CHECK(record >= 0);
  pid_t child = fork(); CHECK(child >= 0);
  if (child == 0) {
    persist(record, &journal); CHECK(fsync(parent) == 0);
    if (boundary == 0) raise(SIGKILL);
    CHECK(renameatx_np(parent, "Staged.app", parent, "Installed.app", RENAME_SWAP) == 0);
    if (boundary == 1) raise(SIGKILL);
    CHECK(fsync(parent) == 0);
    if (boundary == 2) raise(SIGKILL);
    journal.committed = 1; persist(record, &journal);
    raise(SIGKILL); _exit(99);
  }
  int status; CHECK(waitpid(child, &status, 0) == child);
  CHECK(WIFSIGNALED(status) && WTERMSIG(status) == SIGKILL);
  CHECK(pread(record, &journal, sizeof(journal), 0) == sizeof(journal));
  CHECK(classify(parent, &journal) == (boundary == 0 ? 0 : 1));
  /* Recovery may repeat, but it must never exchange a second time. */
  if (classify(parent, &journal) == 1 && !journal.committed) {
    journal.committed = 1; persist(record, &journal);
  }
  CHECK(classify(parent, &journal) == (boundary == 0 ? 0 : 1));
  CHECK(classify(parent, &journal) == (boundary == 0 ? 0 : 1));
  CHECK(close(record) == 0 && unlinkat(parent, "journal", 0) == 0);
  remove_bundle(parent, "Installed.app"); remove_bundle(parent, "Staged.app");
}
static void unknown_identity(int parent, const char *self) {
  create_bundles(parent, self);
  Journal journal = {identity(parent, "Installed.app"), identity(parent, "Staged.app"), 0};
  CHECK(renameat(parent, "Installed.app", parent, "Retained.app") == 0);
  CHECK(classify(parent, &journal) == -1);
  CHECK(symlinkat("Retained.app", parent, "Installed.app") == 0);
  CHECK(classify(parent, &journal) == -1);
  CHECK(unlinkat(parent, "Installed.app", 0) == 0);
  CHECK(mkdirat(parent, "Installed.app", 0700) == 0);
  CHECK(classify(parent, &journal) == -1);
  CHECK(unlinkat(parent, "Installed.app", AT_REMOVEDIR) == 0);
  CHECK(renameat(parent, "Retained.app", parent, "Installed.app") == 0);
  CHECK(classify(parent, &journal) == 0);
  journal.replacement = journal.previous;
  CHECK(classify(parent, &journal) == -1);
  remove_bundle(parent, "Installed.app"); remove_bundle(parent, "Staged.app");
  puts("PASS missing, symlinked, substituted and ambiguous identities refuse recovery");
}
static int continue_exec(int argc, char **argv) {
  CHECK(argc == 8);
  CHECK(getpid() == (pid_t)strtol(argv[2], NULL, 10));
  int lock = (int)strtol(argv[3], NULL, 10), transient = (int)strtol(argv[4], NULL, 10);
  CHECK(fcntl(lock, F_GETFD) >= 0);
  errno = 0; CHECK(fcntl(transient, F_GETFD) == -1 && errno == EBADF);
  CHECK(strcmp(argv[5], "literal spaces and Unicode π") == 0);
  CHECK(strcmp(getenv("MAGNITUDE_PROBE_CONTEXT"), "context with spaces π") == 0);
  char cwd[PATH_MAX]; CHECK(getcwd(cwd, sizeof(cwd)) != NULL && strcmp(cwd, argv[6]) == 0);
  struct stat executable; CHECK(stat(argv[0], &executable) == 0);
  CHECK(executable.st_ino == (ino_t)strtoull(argv[7], NULL, 10));
  int contender = open("installation.lock", O_RDWR | O_CLOEXEC | O_NOFOLLOW); CHECK(contender >= 0);
  errno = 0; CHECK(flock(contender, LOCK_EX | LOCK_NB) == -1 && errno == EWOULDBLOCK);
  CHECK(close(lock) == 0);
  CHECK(flock(contender, LOCK_EX | LOCK_NB) == 0);
  CHECK(close(contender) == 0);
  CHECK(fcntl(STDIN_FILENO, F_GETFD) >= 0 && fcntl(STDOUT_FILENO, F_GETFD) >= 0 && fcntl(STDERR_FILENO, F_GETFD) >= 0);
  puts("PASS same-PID exec, invocation context, inherited exclusion and close-on-exec isolation");
  return 0;
}
static int start_exec(int argc, char **argv) {
  CHECK(argc == 3);
  CHECK(chdir(argv[2]) == 0);
  int parent = open(".", O_RDONLY | O_DIRECTORY | O_CLOEXEC); CHECK(parent >= 0);
  int lock = open("installation.lock", O_RDWR | O_CREAT | O_EXCL, 0600); CHECK(lock >= 0);
  CHECK(flock(lock, LOCK_EX | LOCK_NB) == 0);
  int transient = open(".", O_RDONLY | O_DIRECTORY | O_CLOEXEC); CHECK(transient >= 0);
  struct stat replacement; CHECK(fstatat(parent, "Staged.app/program", &replacement, 0) == 0);
  CHECK(renameatx_np(parent, "Staged.app", parent, "Installed.app", RENAME_SWAP) == 0);
  CHECK(fsync(parent) == 0);
  char pid[32], inherited[32], closed[32], inode[32], executable[PATH_MAX];
  snprintf(pid, sizeof(pid), "%d", getpid()); snprintf(inherited, sizeof(inherited), "%d", lock);
  snprintf(closed, sizeof(closed), "%d", transient); snprintf(inode, sizeof(inode), "%llu", (unsigned long long)replacement.st_ino);
  CHECK(snprintf(executable, sizeof(executable), "%s/Installed.app/program", argv[2]) < (int)sizeof(executable));
  CHECK(setenv("MAGNITUDE_PROBE_CONTEXT", "context with spaces π", 1) == 0);
  execl(executable, executable, "--continue", pid, inherited, closed, "literal spaces and Unicode π", argv[2], inode, NULL);
  perror("exec continuation"); return 1;
}
int main(int argc, char **argv) {
  if (argc > 1 && strcmp(argv[1], "--continue") == 0) return continue_exec(argc, argv);
  if (argc > 1 && strcmp(argv[1], "--start") == 0) return start_exec(argc, argv);
  CHECK(argc == 1);
  char self[PATH_MAX]; CHECK(realpath(argv[0], self) != NULL);
  char root[] = "/tmp/magnitude-mac-update-XXXXXX"; CHECK(mkdtemp(root) != NULL);
  char canonical[PATH_MAX]; CHECK(realpath(root, canonical) != NULL);
  int parent = open(root, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC); CHECK(parent >= 0);
  for (int boundary = 0; boundary < 4; boundary++) crash_boundary(parent, self, boundary);
  puts("PASS process-loss recovery at four journal/exchange boundaries without reverse exchange");
  unknown_identity(parent, self);
  create_bundles(parent, self);
  pid_t child = fork(); CHECK(child >= 0);
  if (child == 0) {
    char executable[PATH_MAX]; CHECK(snprintf(executable, sizeof(executable), "%s/Installed.app/program", canonical) < (int)sizeof(executable));
    execl(executable, executable, "--start", canonical, NULL); perror("exec installed fixture"); _exit(1);
  }
  int status; CHECK(waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0);
  remove_bundle(parent, "Installed.app"); remove_bundle(parent, "Staged.app");
  CHECK(unlinkat(parent, "installation.lock", 0) == 0);
  CHECK(close(parent) == 0 && rmdir(root) == 0);
  puts("PASS macOS mechanical update primitives; signed bundle and power-loss acceptance remain separate");
  return 0;
}
