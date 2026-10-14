---
"@magnitudedev/cli": patch
---

commit: 33907ee5
author: @thrgreenwald

- Run Magnitude as a server. `magnitude server setup` makes it a background service that starts at boot, keeps running after you log out, restarts if it crashes, updates itself, and serves browsers and agents on other computers. `magnitude server remove` stops it and keeps your models and settings.
  - Linux: a systemd service running as a dedicated `magnitude` system account, with its data in `/var/lib/magnitude`. On Fedora and other firewalld systems, setup opens port 10100 for you.
  - macOS: a background service that runs as you, starts before anyone logs in, and keeps using your models and settings in `~/.magnitude`.
  - Windows: the remote server guide shows how to run `magnitude serve` as a Windows service with NSSM.
- Install from a terminal: `curl -fsSL https://magnitude.dev/install.sh | sh` on macOS and Linux, or `irm https://magnitude.dev/install.ps1 | iex` in Windows PowerShell. On Linux the installer asks once, up front, whether to run Magnitude as a server (default no, skipped automatically after 60 seconds or when there's no terminal), then installs without further questions. It doesn't ask for a password when your account already has password-free sudo.
- Servers update themselves: a downloaded update installs when Magnitude is idle, never in the middle of a request, and the server comes back on the new version on its own. If an update fails to install, the server keeps running the current version and won't keep retrying it; a newer release installs normally, and `magnitude update check` retries on demand.
- Updates never install at startup any more on any platform: Magnitude always starts right away, and tells you when an update is ready (Restart in the desktop app, or `magnitude update install`).
- On Linux desktops, installing an update with Restart no longer asks for your password.
- When Magnitude runs as a server, `magnitude serve` and `magnitude status` show how to reach it: the address on this computer, the addresses other devices can use, and the address for apps and agents. The network access key is shown once during setup and never in logs or status.
- If you open the desktop app while the server is running, it explains that the two can't run together and lets you keep the server (and open it in your browser) or stop it and use the app, without a password on Linux.
- `magnitude connections connect <agent>` sets up the agent for the person running the command, so connecting an agent on a server configures your own account rather than the service's.
- `magnitude serve` on Windows keeps running when another user logs off the machine.
- Fixes: update reports now say which version a computer updated from, and successful updates installed from the terminal are reported; installing on a Mac no longer fails when your shell profile can't be edited; a failed reinstall no longer leaves the server stopped.
