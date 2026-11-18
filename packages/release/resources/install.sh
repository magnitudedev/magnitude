#!/bin/sh
set -eu

# Release generation supplies publisher identity and the metadata origin.
origin='@MAGNITUDE_INSTALL_ORIGIN@'
apple_team='@MAGNITUDE_APPLE_TEAM@'
publisher_key='@MAGNITUDE_PUBLISHER_KEY@'
channel=stable
destination=/Applications/Magnitude.app
linux_desktop=/usr/lib/magnitude-desktop/resources/magnitude
server_marker=/etc/magnitude/server
fail() { [ -z "${progress_pid:-}" ] || stop_progress; printf '%s\n' "$*" >&2; exit 1; }
usage='Run `curl -fsSL https://magnitude.dev/install.sh | sh` for the CLI only, or `curl -fsSL https://magnitude.dev/install.sh | sh -s -- --desktop` for the desktop app.'
# No options installs the CLI only; --desktop installs the desktop app, which includes the CLI.
desktop=no
case "$#:${1:-}" in
  0:) ;;
  1:--desktop) desktop=yes ;;
  *) fail "install.sh does not accept: $*. $usage" ;;
esac
case "$origin" in https://*) ;; *) fail 'The installation script has no release origin.' ;; esac
case "$(uname -m)" in arm64|aarch64) arch=arm64 ;; x86_64) arch=x64 ;; *) fail 'This architecture is not supported.' ;; esac
umask 077
scratch=$(mktemp -d "${TMPDIR:-/tmp}/magnitude-install.XXXXXXXX") \
  || fail "Could not create a temporary directory in ${TMPDIR:-/tmp}. Check that it exists, is writable and has free space."
keepalive=''
stopped_server=''
privilege=''
progress_pid=''
download_pid=''
staging=''
progress_terminal=no
if [ -t 2 ] && [ "${TERM:-}" != dumb ]; then progress_terminal=yes; fi
# A server stopped for its upgrade is started again however the installer ends.
restore_server() {
  [ -n "$stopped_server" ] || return 0
  stopped_server=''
  if ${privilege:+$privilege -n} systemctl start magnitude.service; then
    printf '%s\n' 'The Magnitude server was started again.' >&2
  else
    printf '%s\n' 'The Magnitude server could not be started again. Start it with: sudo systemctl start magnitude' >&2
  fi
}
# A hung-up terminal fails every write; set -e must not stop the cleanup at its first message.
cleanup() {
  set +e
  [ -z "$download_pid" ] || kill "$download_pid" 2>/dev/null
  stop_progress
  restore_server
  [ -z "$keepalive" ] || kill "$keepalive" 2>/dev/null || true
  [ -z "$staging" ] || rm -rf "$staging"
  rm -rf "$scratch"
}
trap cleanup EXIT
# Ends the line ^C left, so the shell prompt starts on its own line.
trap 'printf "\n" >&2; exit 130' INT
trap 'exit 143' TERM HUP
scratch=$(cd "$scratch" && pwd -P)
# Status goes to stderr so stdout stays usable by scripts. Only a terminal is redrawn.
stop_progress() {
  [ -n "$progress_pid" ] || return 0
  kill "$progress_pid" 2>/dev/null || true
  wait "$progress_pid" 2>/dev/null || true
  progress_pid=''
  [ "$progress_terminal" != yes ] || printf '\r\033[K' >&2
}
render_progress() {
  awk -v label="$1" -v elapsed="$2" -v received="$3" -v total="$4" -v downloading="$5" -v width="${6:-${COLUMNS:-80}}" '
    BEGIN {
      if (!downloading) {
        split("| / - \\" , frames, " ")
        printf "  %s %s · %ds", frames[elapsed % 4 + 1], label, elapsed
        exit
      }
      speed = elapsed > 0 ? received / elapsed : 0
      if (total > 0) {
        percent = int(received * 100 / total); if (percent > 100) percent = 100
        bar = ""; for (i = 0; i < 12; i++) bar = bar (i < int(percent * 12 / 100) ? "=" : "-")
        if (width >= 75) printf "  [%s] ", bar
        else printf "  "
        printf "%d%%  %.1f/%.1f MB", percent, received / 1000000, total / 1000000
      } else printf "  %.1f MB", received / 1000000
      if (speed > 0) {
        printf "  %.1f MB/s", speed / 1000000
        if (total > received) printf "  ~%ds left", int((total - received) / speed + 0.999)
        else if (total > 0) printf "  0s left"
      } else printf "  estimating time remaining"
    }'
}
start_progress() {
  stop_progress
  progress_label=$1
  progress_file=${2:-}
  progress_bytes=${3:-0}
  progress_started=$(date +%s)
  if [ "$progress_terminal" = yes ]; then printf '  %s...' "$progress_label" >&2
  else printf '%s...\n' "$progress_label" >&2; fi
  (
    progress_last_log=0
    # $$ stays the installer in this subshell: the loop ends with it even when it is killed without its exit trap.
    while sleep 1 </dev/null >/dev/null 2>&1 && kill -0 $$ 2>/dev/null; do
      progress_elapsed=$(( $(date +%s) - progress_started ))
      if [ "$progress_terminal" != yes ] && [ "$((progress_elapsed - progress_last_log))" -lt 10 ]; then continue; fi
      progress_last_log=$progress_elapsed
      progress_received=0
      if [ -n "$progress_file" ] && [ -f "$progress_file" ]; then progress_received=$(wc -c < "$progress_file" | tr -d ' '); fi
      if [ "$progress_terminal" = yes ]; then
        progress_width=${COLUMNS:-80}
        progress_dimensions=$(stty size <&2 2>/dev/null) || progress_dimensions=''
        case "${progress_dimensions##* }" in ''|*[!0-9]*|0|1) ;; *) progress_width=${progress_dimensions##* } ;; esac
        # Build the frame before touching the terminal. Erase only its leftover tail, in the same
        # write as the replacement text; clearing first leaves a visibly blank line between frames.
        progress_frame=$(render_progress "$progress_label" "$progress_elapsed" "$progress_received" "$progress_bytes" "$progress_file" "$progress_width")
        if [ "${#progress_frame}" -ge "$progress_width" ]; then
          progress_frame=$(printf '%s' "$progress_frame" | cut -c "1-$((progress_width - 1))")
        fi
        printf '\r%s\033[K' "$progress_frame" >&2
      elif [ -z "$progress_file" ]; then
        printf '%s · %ss elapsed\n' "$progress_label" "$progress_elapsed" >&2
      else
        progress_frame=$(render_progress "$progress_label" "$progress_elapsed" "$progress_received" "$progress_bytes" "$progress_file")
        printf '%s · %s\n' "$progress_label" "$progress_frame" >&2
      fi
    done
  ) </dev/null >/dev/null &
  progress_pid=$!
}
finish_progress() {
  stop_progress
  printf '✓ %s\n' "$1" >&2
}
run_step() {
  step_pending=$1 step_complete=$2
  shift 2
  start_progress "$step_pending"
  if "$@" >"$scratch/step-output.log" 2>"$scratch/step-error.log"; then
    finish_progress "$step_complete"
    cat "$scratch/step-output.log"
    cat "$scratch/step-error.log" >&2
  else
    step_status=$?
    stop_progress
    cat "$scratch/step-output.log"
    cat "$scratch/step-error.log" >&2
    printf '%s failed.\n' "$step_pending" >&2
    return "$step_status"
  fi
}
download() {
  if [ "$#" -ge 4 ]; then start_progress "$4" "$2" "$3"; fi
  # Slow connections may take as long as they need; only a connection that stops delivering data fails.
  # curl runs in the background so a signal interrupts the wait, and cleanup stops curl, at once.
  curl --fail --silent --show-error --location --proto '=https' --proto-redir '=https' \
    --connect-timeout 15 --speed-limit 1 --speed-time 60 --max-filesize "$3" --output "$2" "$1" 2>"$scratch/download-error.log" &
  download_pid=$!
  if wait "$download_pid"; then
    download_pid=''
    if [ "$#" -ge 4 ]; then
      progress_duration=$(( $(date +%s) - progress_started ))
      progress_size=$(awk -v bytes="$(wc -c < "$2" | tr -d ' ')" 'BEGIN { printf "%.1f MB", bytes / 1000000 }')
      finish_progress "$5 · $progress_size · ${progress_duration}s"
    fi
    return 0
  else download_status=$? download_pid=''; fi
  stop_progress
  cat "$scratch/download-error.log" >&2
  case "$download_status" in
    22) fail 'The download server answered with an error. Try again later; if it keeps failing, this release may have been withdrawn.' ;;
    23) fail "Could not save the download in $scratch. Check that there is free disk space." ;;
    63) fail 'The download is larger than the release says.' ;;
    28) fail 'The download timed out: the connection could not be made or stopped delivering data. Check the network connection and run the installer again.' ;;
    *) fail 'The download failed. Check the network connection and try again.' ;;
  esac
}
# The landing server counts each request and answers with the publisher-signed offer for this target.
offer() {
  start_progress 'Checking latest release'
  download "$origin/api/installer?os=$1&arch=$arch&package=$2&offer=1" "$scratch/offer.json" 16384
}
# Answers come from the terminal, never from the script's piped stdin.
has_terminal() { (: </dev/tty) 2>/dev/null; }
# The installed Magnitude version without the package revision, or nothing when it is not installed.
installed_version() {
  case "$package" in
    deb) version=$(dpkg-query -W -f '${Version}' magnitude-desktop 2>/dev/null) || version='' ;;
    rpm) version=$(rpm -q --qf '%{VERSION}-%{RELEASE}' magnitude-desktop 2>/dev/null) || version='' ;;
    *) version=$(pacman -Q magnitude-desktop 2>/dev/null) && version=${version#* } || version='' ;;
  esac
  printf '%s' "${version%-*}"
}
# What the package step did, from the versions installed before and after it.
installed_summary() {
  if [ -z "$after" ]; then printf '%s\n' 'Magnitude was installed.'
  elif [ "$after" = "$before" ]; then printf '%s\n' "Magnitude $after is already up to date."
  elif [ -n "$before" ]; then printf '%s\n' "Magnitude was upgraded from $before to $after."
  else printf '%s\n' "Magnitude $after was installed."
  fi
}
# Open only in the installing user's graphical session, never through SSH or as root.
open_desktop() {
  [ -z "${SSH_CONNECTION:-}${SSH_CLIENT:-}${SSH_TTY:-}" ] || return 0
  [ "$(id -u)" -ne 0 ] || return 0
  case "$1" in
    macos)
      /bin/launchctl print "gui/$(id -u)" >/dev/null 2>&1 || return 0
      desktop_command="$destination/Contents/Resources/magnitude"
      desktop_fallback='Open Magnitude from Applications.' ;;
    linux)
      [ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ] || return 0
      desktop_command=magnitude
      desktop_fallback='Open Magnitude from your applications menu.' ;;
  esac
  start_progress 'Opening Magnitude'
  if "$desktop_command" app open >"$scratch/open-app.log" 2>&1; then
    finish_progress 'Opened Magnitude'
  else
    stop_progress
    printf '%s\n' "Magnitude is installed, but could not open automatically. $desktop_fallback" >&2
  fi
}
# The app declares macOS 15 as its minimum; earlier systems refuse to open it.
require_macos() {
  macos_version=$(/usr/bin/sw_vers -productVersion)
  case "$macos_version" in [0-9].*|1[0-4].*) fail "Magnitude needs macOS 15 or later, and this Mac has macOS $macos_version." ;; esac
  case "$apple_team" in *[!A-Z0-9]*|'') fail 'The installation script has no Apple publisher identity.' ;; esac
  [ "${#apple_team}" -eq 10 ] || fail 'Invalid Apple publisher identity.'
}
# Every Linux build requires this glibc; refuse before asking anything rather than fail later.
require_glibc() {
  glibc_minimum=2.34
  glibc=$(getconf GNU_LIBC_VERSION 2>/dev/null) || glibc=''
  case "$glibc" in
    'glibc '[0-9]*.[0-9]*)
      glibc=${glibc#glibc }
      major=${glibc%%.*} minor=${glibc#*.}
      minor=${minor%%[!0-9]*}
      [ "$major" -gt 2 ] || { [ "$major" -eq 2 ] && [ "${minor:-0}" -ge "${glibc_minimum#2.}" ]; } \
        || fail "Magnitude needs glibc $glibc_minimum or later, and this system has glibc $glibc. Ubuntu 22.04, Debian 12, RHEL 9 and newer releases are supported." ;;
    *) fail "Magnitude needs a Linux distribution with glibc $glibc_minimum or later." ;;
  esac
}
# Reads the offer for a macOS package. Its integrity rests on HTTPS from the origin; what it names is
# checked against the download's checksum and then the publisher's Apple code signature.
read_mac_offer() {
  offer darwin "$1"
  url=$(/usr/bin/plutil -extract download raw -o - "$scratch/offer.json")
  case "$url" in https://github.com/magnitudedev/magnitude/releases/download/*) ;; *) fail 'Unexpected application download location.' ;; esac
  bytes=$(/usr/bin/plutil -extract release.bytes raw -o - "$scratch/offer.json")
  digest=$(/usr/bin/plutil -extract release.sha256 raw -o - "$scratch/offer.json")
  case "$bytes" in ''|*[!0-9]*) fail 'Invalid application size.' ;; esac
  [ "$bytes" -gt 0 ] && [ "$bytes" -le 2147483648 ] || fail 'Application download exceeds the installation limit.'
  case "$digest" in *[!a-f0-9]*|'') fail 'Invalid application checksum.' ;; esac
  [ "${#digest}" -eq 64 ] || fail 'Invalid application checksum.'
  version=$(/usr/bin/plutil -extract release.version raw -o - "$scratch/offer.json")
  case "$version" in ''|.*|*[!0-9A-Za-z.+-]*) fail 'Invalid application version.' ;; esac
  [ "${#version}" -le 96 ] || fail 'Invalid application version.'
}
# Fetches the offer for a Linux package and verifies its publisher signature with openssl alone, then sets
# its fields. The offer is the landing server's exact encoding, so anything else is refused.
read_linux_offer() {
  offer linux "$1"
  [ "$(wc -l < "$scratch/offer.json" | tr -d ' ')" -le 1 ] || fail 'Application release verification failed: Invalid installation offer'
  fields=$(sed -nE 's#^\{"release":\{"version":"([0-9A-Za-z.+-]{1,96})","bytes":([1-9][0-9]{0,9}),"sha256":"([a-f0-9]{64})","signature":"([A-Za-z0-9+/]{86}==)"\},"download":"(https://github\.com/magnitudedev/magnitude/releases/download/[^"\\ ]{1,1900})"\}$#\1 \2 \3 \4 \5#p' "$scratch/offer.json")
  [ -n "$fields" ] || fail 'Application release verification failed: Invalid installation offer'
  set -- "$1" $fields
  version=$2 bytes=$3 digest=$4 signature=$5 url=$6
  printf '%s' "$version" | grep -Eq '^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(-[0-9A-Za-z.-]+)?(\+[0-9A-Za-z.-]+)?$' \
    || fail 'Application release verification failed: Invalid release version'
  # The stable channel takes no prerelease.
  case "${version%%+*}" in *-*) fail 'Application release verification failed: Release does not match the selected channel' ;; esac
  [ "$bytes" -le 2147483648 ] || fail 'Application release verification failed: Invalid application size'
  case "$url" in *%2[fF]*|*%5[cC]*|*%00*) fail 'Application release verification failed: Unexpected application download location' ;; esac
  case "${url#https://github.com/magnitudedev/magnitude/releases/download/}" in */?*) ;; *) fail 'Application release verification failed: Unexpected application download location' ;; esac
  printf '%s' "$publisher_key" | openssl base64 -d -A > "$scratch/publisher.pem" 2>/dev/null \
    || fail 'Application release verification failed: Invalid publisher key'
  printf '%s' "$signature" | openssl base64 -d -A > "$scratch/signature" 2>/dev/null \
    || fail 'Application release verification failed: Invalid publisher signature'
  [ "$(wc -c < "$scratch/signature" | tr -d ' ')" -eq 64 ] || fail 'Application release verification failed: Invalid publisher signature'
  printf 'magnitude-update-release-v1\nlinux\n%s\n%s\n%s\n%s\n%s\n' "$arch" "$1" "$version" "$bytes" "$digest" > "$scratch/signed"
  openssl pkeyutl -verify -pubin -inkey "$scratch/publisher.pem" -rawin -in "$scratch/signed" -sigfile "$scratch/signature" >/dev/null 2>&1 \
    || fail 'Application release verification failed: Invalid publisher signature'
}
sha256_of() {
  if [ "$(uname -s)" = Darwin ]; then /usr/bin/shasum -a 256 "$1" | cut -d ' ' -f 1
  else sha256sum "$1" | cut -d ' ' -f 1; fi
}
verify_download() {
  start_progress 'Verifying download checksum'
  actual_bytes=$(wc -c < "$1" | tr -d ' ')
  [ "$actual_bytes" = "$bytes" ] || fail 'The application download is incomplete.'
  [ "$(sha256_of "$1")" = "$digest" ] || fail 'The application checksum does not match.'
  finish_progress 'Verified download checksum'
}
# Replaces a symlink in one rename. Plain `mv` would move the new link into the directory an old link names.
replace_link() {
  ln -s "$1" "$2.new-$$"
  if [ "$(uname -s)" = Darwin ]; then mv -fh "$2.new-$$" "$2"
  else mv -fT "$2.new-$$" "$2"; fi
}

# The desktop app owns the magnitude command wherever it is installed.
desktop_installed() {
  case "$1" in
    darwin) [ -d "$destination" ] || [ -d "$HOME/Applications/Magnitude.app" ] ;;
    linux)
      [ -e "$linux_desktop" ] && return 0
      case "$(dpkg-query -W -f '${Status}' magnitude-desktop 2>/dev/null)" in *' installed') return 0 ;; esac
      rpm -q magnitude-desktop >/dev/null 2>&1 && return 0
      pacman -Q magnitude-desktop >/dev/null 2>&1 ;;
  esac
}
# Files a CLI-only release holds; anything else in the archive is refused.
cli_files='magnitude magnitude-service desktop-host.node update-configuration.json update-trust.json'
cli_optional_files='magnitude-command LICENSE'
check_cli_files() {
  for entry in "$1"/* "$1"/.[!.]* "$1"/..?*; do
    [ -e "$entry" ] || [ -L "$entry" ] || continue
    name=${entry##*/}
    case " $cli_files $cli_optional_files " in *" $name "*) ;; *) printf 'The CLI download contains an unexpected entry: %s\n' "$name" >&2; return 1 ;; esac
    if [ -L "$entry" ] || [ ! -f "$entry" ]; then printf 'The CLI download entry %s is not a regular file.\n' "$name" >&2; return 1; fi
  done
  for name in $cli_files; do
    [ -f "$1/$name" ] || { printf 'The CLI download is missing %s.\n' "$name" >&2; return 1; }
  done
}
# Every executable must carry the publisher's Developer ID signature, which notarization requires.
check_cli_signatures() {
  requirement="=anchor apple generic and certificate leaf[subject.OU] = \"$apple_team\" and certificate leaf[field.1.2.840.113635.100.6.1.13] exists"
  for name in magnitude magnitude-service desktop-host.node magnitude-command; do
    [ -f "$1/$name" ] || continue
    /usr/bin/codesign --verify --strict --all-architectures -R "$requirement" "$1/$name" || {
      printf '%s does not carry the Magnitude publisher signature.\n' "$name" >&2; return 1; }
  done
}

# Installs into ~/.magnitude/cli/<version>, with `current` naming the version and ~/.magnitude/bin/magnitude
# linking through it. Everything is in the account's own folders: no sudo and no package manager.
install_cli() {
  printf '%s\n' 'Installing the Magnitude CLI only (for servers and headless use). For the desktop app, press Ctrl-C and rerun with --desktop.' >&2
  printf '\nInstalling the Magnitude CLI\n%s · %s\n\n' "$2" "$arch" >&2
  [ -n "${HOME:-}" ] && [ -d "$HOME" ] || fail 'HOME is not set to an existing directory.'
  if desktop_installed "$1"; then
    printf '%s\n' 'The Magnitude desktop app is installed and already includes the magnitude command. Nothing was changed.'
    exit 0
  fi
  if [ "$1" = darwin ]; then
    require_macos
    read_mac_offer cli-only
  else
    require_glibc
    for tool in curl tar gzip openssl; do
      command -v "$tool" >/dev/null 2>&1 || fail "Install $tool before running this installer."
    done
    read_linux_offer cli-only
  fi
  cli_root="$HOME/.magnitude/cli"
  command_directory="$HOME/.magnitude/bin"
  previous=$(readlink "$cli_root/current" 2>/dev/null) || previous=''
  previous=${previous##*/}
  if [ "$previous" = "$version" ] && [ "$("$cli_root/$version/magnitude" --version 2>/dev/null || true)" = "$version" ]; then
    finish_progress "Found Magnitude CLI $version"
    updated=no
  else
    finish_progress "Found Magnitude CLI $version${previous:+ (installed: $previous)}"
    download "$url" "$scratch/magnitude-cli.tar.gz" "$bytes" 'Downloading CLI' 'Downloaded CLI'
    verify_download "$scratch/magnitude-cli.tar.gz"
    mkdir -p "$cli_root"
    staging="$cli_root/.staging-$version"
    rm -rf "$staging"
    mkdir "$staging"
    # Both tars refuse members that climb out of the folder; only regular top-level files are kept.
    if [ "$1" = darwin ]; then tar_command=/usr/bin/tar; else tar_command=tar; fi
    run_step 'Extracting CLI' 'Extracted CLI' "$tar_command" -xzf "$scratch/magnitude-cli.tar.gz" -C "$staging" --no-same-owner
    run_step 'Checking CLI files' 'Checked CLI files' check_cli_files "$staging"
    [ "$1" != darwin ] || run_step 'Verifying CLI signatures' 'Verified CLI signatures' check_cli_signatures "$staging"
    start_progress 'Checking CLI version'
    reported=$("$staging/magnitude" --version 2>"$scratch/version-error.log") || { stop_progress; cat "$scratch/version-error.log" >&2; fail 'The downloaded magnitude command does not run on this system.'; }
    [ "$reported" = "$version" ] || fail "The downloaded magnitude command reports version $reported, not $version."
    finish_progress "Checked CLI version $version"
    rm -rf "${cli_root:?}/$version"
    mv "$staging" "$cli_root/$version"
    staging=''
    replace_link "$version" "$cli_root/current"
    # Keep the version a running serve may still use; remove the rest, and leftovers of interrupted runs.
    for entry in "$cli_root"/* "$cli_root"/.staging-* "$cli_root"/.current-*; do
      [ -e "$entry" ] || [ -L "$entry" ] || continue
      case "${entry##*/}" in current|"$version"|"$previous") ;; *) rm -rf "$entry" ;; esac
    done
    updated=yes
  fi
  mkdir -p "$command_directory"
  [ ! -d "$command_directory/magnitude" ] || [ -L "$command_directory/magnitude" ] || fail "$command_directory/magnitude is a directory; remove it and run the installer again."
  # Relative, so the link follows `current` through updates.
  replace_link ../cli/current/magnitude "$command_directory/magnitude"
  "$cli_root/$version/magnitude" _install-cli 2>"$scratch/path-error.log" || {
    cat "$scratch/path-error.log" >&2
    printf '%s\n' 'To use the magnitude command in a terminal, add ~/.magnitude/bin to your PATH.' >&2
  }
  if [ "$updated" = yes ]; then
    printf '✓ Installed the Magnitude CLI %s (CLI only, no desktop app)\n\n' "$version"
    printf '%s\n%s\n' 'Next: run `magnitude docs` for how to use `magnitude serve` and how to' \
      'keep it running in the background at login on macOS, Linux and Windows.'
  else
    printf '%s\n' "The Magnitude CLI $version is already up to date (CLI only, no desktop app)."
  fi
  case ":$PATH:" in *":$command_directory:"*) ;; *) printf '%s\n' 'Open a new terminal to use the magnitude command.' ;; esac
}

# The desktop app replaces a CLI-only install: its command is the one shells and services run from now on.
absorb_cli_install() {
  rm -rf "$HOME/.magnitude/cli"
  [ "$1" = linux ] || return 0
  link="$HOME/.magnitude/bin/magnitude"
  if [ -L "$link" ] || [ -f "$link" ]; then replace_link /usr/bin/magnitude "$link"; fi
}

install_desktop_macos() {
  printf '\nInstalling Magnitude\nmacOS · %s\n\n' "$arch" >&2
  require_macos
  read_mac_offer mac-zip
  finish_progress "Found Magnitude $version"
  download "$url" "$scratch/magnitude.zip" "$bytes" 'Downloading app' 'Downloaded app'
  verify_download "$scratch/magnitude.zip"
  mkdir "$scratch/bootstrap"
  # bsdtar's default extraction rejects traversal and writes through archive symlinks.
  run_step 'Extracting app' 'Extracted app' /usr/bin/tar -xf "$scratch/magnitude.zip" -C "$scratch/bootstrap" --no-same-owner
  app="$scratch/bootstrap/Magnitude.app"
  run_step 'Verifying app signature' 'Verified app signature' /usr/bin/codesign --verify --deep --strict -R "=anchor apple generic and identifier \"dev.magnitude.desktop\" and certificate leaf[subject.OU] = \"$apple_team\" and certificate leaf[field.1.2.840.113635.100.6.1.13] exists" "$app"
  run_step 'Checking macOS approval' 'Passed macOS approval' /usr/sbin/spctl --assess --type execute "$app"
  /usr/bin/plutil -create xml1 "$scratch/request.plist"
  /usr/bin/plutil -insert bundle -string "$destination" "$scratch/request.plist"
  /usr/bin/plutil -insert archive -string "$scratch/magnitude.zip" "$scratch/request.plist"
  /usr/bin/plutil -insert channel -string "$channel" "$scratch/request.plist"
  /usr/bin/plutil -insert offer -json "$(cat "$scratch/offer.json")" "$scratch/request.plist"
  /usr/bin/plutil -convert json -o "$scratch/request.json" "$scratch/request.plist"
  run_step 'Installing app and registering CLI' 'Installed app and registered CLI' \
    "$app/Contents/Resources/magnitude" _install-mac-application "$(cat "$scratch/request.json")"
  absorb_cli_install macos
  printf '%s\n' 'Magnitude was installed.'
  open_desktop macos
  printf '%s\n' 'Open a new terminal to use the magnitude command.'
}

install_desktop_linux() {
  printf '\nInstalling Magnitude\nLinux · %s\n\n' "$arch" >&2
  require_glibc
  if command -v apt-get >/dev/null 2>&1; then package=deb
  elif command -v dnf >/dev/null 2>&1; then package=rpm
  elif command -v pacman >/dev/null 2>&1; then package=pacman
  else fail 'This Linux distribution requires apt, dnf or pacman.'; fi
  [ "$package" != pacman ] || [ "$arch" = x64 ] || fail 'Arch Linux packages are available for x86-64 only.'
  for tool in openssl curl; do
    command -v "$tool" >/dev/null 2>&1 || fail "Install $tool before running this installer."
  done

  # 1. Get sudo up front and keep it fresh, so the person can walk away once it is granted.
  # Running a command, never `sudo -v`: with sudo's default verifypw=all, -v asks for a password
  # whenever any matching rule needs one, even from accounts that also have NOPASSWD.
  if [ "$(id -u)" -ne 0 ]; then
    command -v sudo >/dev/null 2>&1 || fail 'Installing Magnitude needs sudo. Install it, or run this installer as root.'
    if sudo -n true 2>/dev/null; then :
    elif has_terminal; then sudo true </dev/tty || fail 'Installing Magnitude needs an administrator account (one that can use sudo). Run this installer as an administrator.'
    else fail 'Installing Magnitude needs an administrator account (one that can use sudo), and there is no terminal to ask for a password. Run it in a terminal as an administrator, or as an administrator with passwordless sudo.'
    fi
    privilege=sudo
    # Detached from the terminal so the installer's session ends when it does.
    ( while sleep 50 && kill -0 $$ 2>/dev/null; do sudo -n true 2>/dev/null || exit 0; done ) </dev/null >/dev/null 2>&1 &
    keepalive=$!
  fi

  # 2. Download, verify and install the package.
  read_linux_offer "$package"
  finish_progress "Found Magnitude $version · $package"
  download "$url" "$scratch/magnitude.$package" "$bytes" 'Downloading app' 'Downloaded app'
  verify_download "$scratch/magnitude.$package"
  # Fresh cloud images ship without apt's package lists, so the package's dependencies would not resolve.
  if [ "$package" = deb ]; then
    printf 'Refreshing package lists...\n' >&2
    $privilege apt-get update || printf '%s\n' 'Could not refresh the package lists; installing anyway.' >&2
  fi
  # The scratch directory is private; apt reads the already verified package as its _apt account.
  if [ "$package" = deb ]; then chmod 755 "$scratch"; chmod 644 "$scratch/magnitude.deb"; fi
  before=$(installed_version)
  # A server set up by an earlier release holds the installation lock the package checks, so it is
  # upgraded stopped. Its next start converts it to a background `magnitude serve`.
  if [ -f "$server_marker" ] && systemctl is-active --quiet magnitude.service; then
    stopped_server=1
    $privilege systemctl stop magnitude.service
  fi
  printf 'Installing Magnitude with the package manager...\n' >&2
  if [ "$package" = deb ]; then $privilege env DEBIAN_FRONTEND=noninteractive apt-get install -y -o DPkg::Lock::Timeout=300 "$scratch/magnitude.deb"
  elif [ "$package" = rpm ]; then $privilege dnf install -y "$scratch/magnitude.rpm"
  else $privilege pacman -U --noconfirm "$scratch/magnitude.pacman"
  fi || fail 'The Magnitude package could not be installed; see the package manager output above.'
  restart=$stopped_server
  stopped_server=''
  after=$(installed_version)
  absorb_cli_install linux
  installed_summary

  # 3. Start a server this installer stopped; one an administrator stopped stays stopped.
  if [ -n "$restart" ]; then
    $privilege systemctl start magnitude.service
    printf '%s\n' 'The Magnitude server is restarting. Run `magnitude status` to check on it.'
  elif [ -f "$server_marker" ]; then
    printf '%s\n' 'The server was not running and was left stopped. Start it with: sudo systemctl start magnitude'
  else
    open_desktop linux
  fi
}

case "$(uname -s)" in
  Darwin) if [ "$desktop" = yes ]; then install_desktop_macos; else install_cli darwin macOS; fi ;;
  Linux) if [ "$desktop" = yes ]; then install_desktop_linux; else install_cli linux Linux; fi ;;
  *) fail 'This operating system is not supported.' ;;
esac
