---
"@magnitudedev/cli": patch
---

author: @thrgreenwald

Install Magnitude with one command on every platform, with clearer progress and fewer failed installs.

- The README and installation guides now use the install script on macOS, Linux and Windows.
- The installer shows each download's speed and time remaining, redraws without blank frames, and opens the desktop app when it's done.
- Slow connections no longer fail a download after 10 minutes; a download fails only when no data arrives for 60 seconds, and says so.
- On Windows, silent setup works again, and the installer asks you to quit a running Magnitude before downloading instead of failing at the end.
- On macOS before 15, the installer says Magnitude needs macOS 15 or later instead of installing an app that won't open.
- Ctrl-C, a closed terminal or a killed installer stops the download at once and cleans up after itself.
- On Linux without systemd, such as containers or WSL, the installer no longer offers server mode.
