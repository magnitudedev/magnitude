# The package manager owns this directory; it is not per-user service state.
install -d -m 755 /var/lib/magnitude-desktop
touch /var/lib/magnitude-desktop/installation.lock
chmod 444 /var/lib/magnitude-desktop/installation.lock
exec 9</var/lib/magnitude-desktop/installation.lock
flock -x -n 9 || {
  if systemctl is-active --quiet magnitude.service 2>/dev/null; then
    echo 'The Magnitude server is running. To remove Magnitude, run `magnitude server remove` first; to upgrade or reinstall it, stop the server with `sudo systemctl stop magnitude` first.' >&2
  else
    echo 'Quit Magnitude for all users before changing its installation.' >&2
  fi
  exit 1
}
touch /var/lib/magnitude-desktop/installing
