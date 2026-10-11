"""Keep an engine process group subordinate to its benchmark owner, including owner loss."""

import os
import signal
import subprocess
import sys
import threading
import time


def main():
    owner = int(sys.argv[1])
    # The runner gives this process a new session before execution.
    if os.getpgrp() != os.getpid():
        raise RuntimeError("engine supervisor must own its process group")

    def watch():
        while os.getppid() == owner:
            time.sleep(0.25)
        os.killpg(os.getpgrp(), signal.SIGKILL)

    threading.Thread(target=watch, daemon=True).start()
    child = subprocess.Popen(sys.argv[2:])
    # A group TERM reaches the child as well. Remain available to reap it.
    signal.signal(signal.SIGTERM, lambda *_: None)
    signal.signal(signal.SIGINT, lambda *_: None)
    return child.wait()


if __name__ == "__main__":
    raise SystemExit(main())
