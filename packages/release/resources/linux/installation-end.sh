# Retain the lock inode across upgrades, removal and reinstall.
exec 9</var/lib/magnitude-desktop/installation.lock
flock -x -n 9
rm -f /var/lib/magnitude-desktop/installing
