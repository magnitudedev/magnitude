---
"@magnitudedev/cli": patch
---

commit: d066f740
author: @thrgreenwald

- Fix agent checkpoints and undo failing on Linux systems whose default file permissions are group-writable, such as Ubuntu.
- Purging the Debian package removes Magnitude's installation state, and the login item no longer points at a removed application.
- If an unfinished package change keeps Magnitude from starting on Linux, opening it from the application menu now shows a notice with repair steps instead of doing nothing.
- `pacman -Qkk` reports no permission mismatch for the Arch package.
- `nohup magnitude serve &` keeps running after you log out of an SSH session.
