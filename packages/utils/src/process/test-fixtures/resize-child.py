#!/usr/bin/env python3

import os
import signal


def report_size(_signal, _frame):
    size = os.get_terminal_size(1)
    print(f"SIZE {size.columns} {size.lines}", flush=True)


signal.signal(signal.SIGWINCH, report_size)
print("READY", flush=True)
signal.pause()
while True:
    signal.pause()
