# The package manager owns this directory; it is not per-user service state.
install -d -m 755 /var/lib/magnitude-desktop
touch /var/lib/magnitude-desktop/installation.lock
chmod 444 /var/lib/magnitude-desktop/installation.lock
exec 9</var/lib/magnitude-desktop/installation.lock
flock -x -n 9 || {
  echo 'Quit Magnitude for all users before changing its installation.' >&2
  exit 1
}
touch /var/lib/magnitude-desktop/installing
