---
"@magnitudedev/cli": patch
---

author: @thrgreenwald

More reliable Linux server setup and removal.

- Network access is on from the server's first start, so a `magnitude server setup` that's interrupted still leaves a server other computers can reach; running setup again shows its access key.
- `magnitude server remove` closes the firewalld port that setup opened.
- If the installer is interrupted while setting up the server, it says how to finish: `magnitude server setup`.
