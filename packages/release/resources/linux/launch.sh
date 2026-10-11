#!/bin/sh
set -eu
gate=/var/lib/magnitude-desktop
unavailable() {
  echo "$1" >&2
  # Menu and login launches have no terminal: show the same guidance as a desktop notification. Only
  # system tools run here; the installation itself may be incomplete.
  if [ ! -t 2 ]; then
    if command -v notify-send >/dev/null 2>&1; then
      notify-send --app-name=Magnitude --icon=magnitude-desktop "Magnitude can't start" "$1" >/dev/null 2>&1 || true
    elif command -v gdbus >/dev/null 2>&1; then
      gdbus call --session --dest org.freedesktop.Notifications --object-path /org/freedesktop/Notifications \
        --method org.freedesktop.Notifications.Notify Magnitude 0 magnitude-desktop "Magnitude can't start" "$1" '[]' '{}' -1 >/dev/null 2>&1 || true
    fi
  fi
  exit 75
}
repair='Magnitude needs package-manager repair: a package change did not finish. Reinstall the magnitude-desktop package, then start Magnitude again.'
# A failed redirection on exec ends a POSIX shell before any fallback runs.
[ -r "$gate/installation.lock" ] || unavailable "$repair"
exec 9<"$gate/installation.lock"
flock -s -n 9 || unavailable 'Magnitude is being installed, upgraded or removed. Start it again after the package manager finishes.'
[ ! -e "$gate/installing" ] || unavailable "$repair"
# Electron adopts this shared descriptor as close-on-exec before starting children.
exec /usr/lib/magnitude-desktop/magnitude "$@"
