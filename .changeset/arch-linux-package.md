---
"@magnitudedev/cli": patch
---

commit: 3fe2c30b
author: @thrgreenwald

- Magnitude is available for Arch Linux, Omarchy, and other Arch-based distributions (x64) as a pacman package. Install it with `sudo pacman -U ./magnitude-desktop.pkg.tar.zst` or the install script, and in-app updates install through pacman. As with the Debian and RPM packages, pacman refuses to upgrade or remove Magnitude while it is running.
