/* A transient command group whose lifetime is bounded by its parent's pipe.
 * fd 3: parent lifetime; fd 4: command exit status. Neither reaches exec.
 * Stay alive after command exit until the parent has drained output and closes fd 3.
 */
#include <errno.h>
#include <poll.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

static void retire(void) { kill(-getpid(), SIGKILL); _exit(91); }
static int channel(int fd) {
  struct stat status;
  return fstat(fd, &status) == 0 && (S_ISFIFO(status.st_mode) || S_ISSOCK(status.st_mode));
}
static void parent_check(int timeout) {
  struct pollfd descriptor = { .fd = 3, .events = POLLIN };
  int result = poll(&descriptor, 1, timeout);
  if (result < 0) { if (errno != EINTR) retire(); return; }
  if (result > 0) {
    char ignored[64];
    if (descriptor.revents & (POLLERR | POLLNVAL | POLLHUP)) retire();
    if ((descriptor.revents & POLLIN) && read(3, ignored, sizeof(ignored)) <= 0) retire();
  }
}
int main(int argc, char **argv) {
  if (argc < 2 || getpid() != getpgrp() || !channel(3) || !channel(4)) return 125;
  parent_check(0);
  pid_t child = fork();
  if (child < 0) return 125;
  if (child == 0) {
    close(3); close(4);
    execvp(argv[1], argv + 1);
    _exit(127);
  }
  close(0); close(1); close(2);
  signal(SIGPIPE, SIG_IGN);
  int completed = 0;
  for (;;) {
    parent_check(50);
    if (!completed) {
      int status;
      pid_t observed = waitpid(child, &status, WNOHANG);
      if (observed == child) {
        int code = WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
        char result[16];
        int length = snprintf(result, sizeof(result), "%d\n", code);
        if (write(4, result, (size_t)length) != length) retire();
        close(4);
        completed = 1;
      } else if (observed < 0 && errno != EINTR) retire();
    }
  }
}
