---
"@magnitudedev/cli": patch
---

- Add `magnitude serve` to run inference without a desktop window on macOS, Windows, and Linux. Opening Desktop takes over from the foreground server and reports why it stopped.
- Replace the `magnitude service` commands with `magnitude serve` and `magnitude status`. Model, catalog, hardware, and connection commands require an existing Desktop or server instead of starting one automatically. Startup errors identify which application must be stopped.
- Share application updates between Desktop and the CLI. A running server can prepare updates without being interrupted; prepared updates install at the next startup. The `magnitude update` commands support checking, downloading, inspecting, installing, and discarding updates while Desktop is closed. Failed installations require an explicit retry.
- Fix Windows update preparation when the update folder has inherited permissions, and improve update recovery and command continuation. Existing affected releases still require a manual installer to receive the fix.
- Add shell and PowerShell installation scripts for the complete application, including its CLI.
- Improve `magnitude app open` during Desktop takeover. On Windows, clicking the tray icon opens Desktop, and sharper tray icons adapt to the system's light or dark theme.
- Add a remote server guide and update network access, CLI, and installation documentation.
