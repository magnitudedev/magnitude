---
"@magnitudedev/cli": patch
---

commit: ff42cdb2
author: @thrgreenwald

Fix installation, update and reliability problems found in release testing.

- Linux packages install on RHEL 9 and other systems with glibc 2.34, and the Ubuntu and Debian package no longer pulls a KDE desktop onto headless machines.
- The Linux desktop app no longer crashes when it restarts to apply a setting.
- On Windows, the app window comes back after Restart to update, and the Magnitude app and service show Magnitude's name, publisher and version in firewall prompts, Task Manager and file properties.
- On a Mac, another account's leftover installation lock no longer stops the app from opening, and installing says plainly when another account owns the app or left an update behind.
- An update that installs but never starts is reported as failed by the desktop app too, and isn't installed again automatically. `magnitude update status` and Settings show a failed update as failed, update errors are readable sentences, and `magnitude update check` prints its result. Linux updates wait for another package manager to finish instead of failing.
- A failed inference engine download is reported in one plain sentence instead of an internal error, and when port 10100 is taken, Magnitude says whether another account's Magnitude or another program holds it.
- Cancelling a model download removes its partial files, and the free-space check counts a model and its separate draft together. A model folder Magnitude can't write to is refused, and an unusable saved folder no longer stops Magnitude from starting.
- `magnitude status` shows the address the service actually listens on, and `magnitude connections list` shows agents you never connected as not connected.
