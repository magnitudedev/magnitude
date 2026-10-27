#!/bin/sh
set -eu

# Release generation supplies publisher identity and the metadata origin.
origin='@MAGNITUDE_INSTALL_ORIGIN@'
apple_team='@MAGNITUDE_APPLE_TEAM@'
publisher_key='@MAGNITUDE_PUBLISHER_KEY@'
channel=stable
destination=/Applications/Magnitude.app
server_marker=/etc/magnitude/server
mac_server=/Library/LaunchDaemons/dev.magnitude.server.plist
fail() { [ -z "${progress_pid:-}" ] || stop_progress; printf '%s\n' "$*" >&2; exit 1; }
[ "$#" -eq 0 ] || fail 'install.sh takes no options. Run: curl -fsSL https://magnitude.dev/install.sh | sh'
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
# Server setup marks the server configured before it prints the network access key, so a later run of
# this installer only restarts it; after an interrupted setup, running setup again shows the key.
setting_up=''
unfinished_setup() {
  [ -n "$setting_up" ] || return 0
  stop_progress
  printf '%s\n' 'Server setup was interrupted. To finish it, run: magnitude server setup' >&2
}
# A hung-up terminal fails every write; set -e must not stop the cleanup at its first message.
cleanup() { set +e; [ -z "$download_pid" ] || kill "$download_pid" 2>/dev/null; stop_progress; restore_server; [ -z "$keepalive" ] || kill "$keepalive" 2>/dev/null || true; rm -rf "$scratch"; }
trap cleanup EXIT
# Ends the line ^C left, so the shell prompt starts on its own line.
trap 'printf "\n" >&2; unfinished_setup; exit 130' INT
trap 'unfinished_setup; exit 143' TERM HUP
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

case "$(uname -s)" in
  Darwin)
    printf '\nInstalling Magnitude\nmacOS · %s\n\n' "$arch" >&2
    # The server's LaunchDaemon holds the app and installs its own updates when idle.
    [ ! -e "$mac_server" ] || fail 'Magnitude runs as a server on this Mac and installs its own updates when idle. To reinstall it now, run `magnitude server remove`, then this installer, then `magnitude server setup`.'
    # The app declares macOS 15 as its minimum; earlier systems refuse to open it.
    macos_version=$(/usr/bin/sw_vers -productVersion)
    case "$macos_version" in [0-9].*|1[0-4].*) fail "Magnitude needs macOS 15 or later, and this Mac has macOS $macos_version." ;; esac
    case "$apple_team" in *[!A-Z0-9]*|'') fail 'The installation script has no Apple publisher identity.' ;; esac
    [ "${#apple_team}" -eq 10 ] || fail 'Invalid Apple publisher identity.'
    offer darwin mac-zip
    url=$(/usr/bin/plutil -extract download raw -o - "$scratch/offer.json")
    case "$url" in https://github.com/magnitudedev/magnitude/releases/download/*) ;; *) fail 'Unexpected application download location.' ;; esac
    bytes=$(/usr/bin/plutil -extract release.bytes raw -o - "$scratch/offer.json")
    digest=$(/usr/bin/plutil -extract release.sha256 raw -o - "$scratch/offer.json")
    case "$bytes" in ''|*[!0-9]*) fail 'Invalid application size.' ;; esac
    [ "$bytes" -gt 0 ] && [ "$bytes" -le 2147483648 ] || fail 'Application download exceeds the installation limit.'
    version=$(/usr/bin/plutil -extract release.version raw -o - "$scratch/offer.json")
    case "$version" in ''|*[!0-9A-Za-z.+-]*) fail 'Invalid application version.' ;; esac
    [ "${#version}" -le 96 ] || fail 'Invalid application version.'
    finish_progress "Found Magnitude $version"
    download "$url" "$scratch/magnitude.zip" "$bytes" 'Downloading app' 'Downloaded app'
    start_progress 'Verifying download checksum'
    actual_bytes=$(wc -c < "$scratch/magnitude.zip" | tr -d ' ')
    [ "$actual_bytes" = "$bytes" ] || fail 'The application download is incomplete.'
    actual_digest=$(/usr/bin/shasum -a 256 "$scratch/magnitude.zip" | cut -d ' ' -f 1)
    [ "$actual_digest" = "$digest" ] || fail 'The application checksum does not match.'
    finish_progress 'Verified download checksum'
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
    printf '%s\n' 'Magnitude was installed.'
    open_desktop macos
    printf '%s\n' 'Open a new terminal to use the magnitude command.'
    ;;
  Linux)
    printf '\nInstalling Magnitude\nLinux · %s\n\n' "$arch" >&2
    # Every package requires this glibc; refuse before asking anything rather than fail in the package manager.
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
    if command -v apt-get >/dev/null 2>&1; then package=deb
    elif command -v dnf >/dev/null 2>&1; then package=rpm
    elif command -v pacman >/dev/null 2>&1; then package=pacman
    else fail 'This Linux distribution requires apt, dnf or pacman.'; fi
    [ "$package" != pacman ] || [ "$arch" = x64 ] || fail 'Arch Linux packages are available for x86-64 only.'
    for tool in python3 openssl curl timeout; do
      command -v "$tool" >/dev/null 2>&1 && continue
      # Arch packages python3 as python.
      [ "$tool:$package" != python3:pacman ] || fail 'Install python (pacman -S python) before running this installer.'
      fail "Install $tool before running this installer."
    done

    # 1. Ask first, so the person can walk away after answering and authorizing.
    if [ -f "$server_marker" ]; then answer=configured
    # Server mode runs under systemd; without it (containers, WSL without systemd) there is nothing to offer.
    elif [ ! -d /run/systemd/system ] || ! has_terminal; then answer=no
    else
      printf '%s\n%s ' 'Run Magnitude as a server? It starts on boot, runs without the desktop app,' \
        'and you can use it from a browser on another computer. [y/N]' >/dev/tty
      # --foreground keeps the reader in the terminal's process group, so it may read the terminal.
      # End of input (Ctrl-D) reads as an empty answer, which means No.
      if line=$(timeout --foreground 60 sh -c 'IFS= read -r line </dev/tty || printf "\n" >/dev/tty; printf "%s" "$line"'); then
        case "$line" in [Yy]|[Yy][Ee][Ss]) answer=yes ;; *) answer=no ;; esac
      else
        status=$?
        [ "$status" -eq 124 ] || exit "$status"
        printf '\n' >/dev/tty
        answer=timeout
      fi
    fi

    # 2. Get sudo up front and keep it fresh, instead of waiting at a password prompt later.
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

    # 3. Download, verify and install the package.
    offer linux "$package"
    python3 - "$scratch" "$arch" "$package" "$channel" "$publisher_key" <<'PY'
import base64, json, pathlib, re, subprocess, sys, urllib.parse
root, arch, package, channel, key = sys.argv[1:]
root = pathlib.Path(root)
def unique(pairs):
    result = {}
    for name, value in pairs:
        if name in result: raise ValueError('Duplicate metadata field')
        result[name] = value
    return result
try:
    offer = json.loads((root / 'offer.json').read_text(), object_pairs_hook=unique)
    if set(offer) != {'release', 'download'}: raise ValueError('Invalid installation offer')
    release = offer['release']
    if set(release) != {'version', 'bytes', 'sha256', 'signature'}: raise ValueError('Invalid release metadata')
    version, size, digest = release['version'], release['bytes'], release['sha256']
    if not isinstance(version, str) or len(version) > 96 or not re.fullmatch(r'(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(?:-([0-9A-Za-z.-]+))?(?:\+[0-9A-Za-z.-]+)?', version):
        raise ValueError('Invalid release version')
    prerelease = version.split('+')[0].split('-', 1)
    selected = prerelease[1].split('.')[0] if len(prerelease) == 2 else 'stable'
    allowed = {'stable': {'stable'}, 'beta': {'stable', 'beta'}, 'alpha': {'stable', 'beta', 'alpha'}}
    if selected not in allowed[channel]: raise ValueError('Release does not match the selected channel')
    if type(size) is not int or not 0 < size <= 2147483648: raise ValueError('Invalid application size')
    if not isinstance(digest, str) or not re.fullmatch('[a-f0-9]{64}', digest): raise ValueError('Invalid application checksum')
    url = offer['download']
    if not isinstance(url, str) or len(url) > 2048: raise ValueError('Invalid download URL')
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != 'https' or parsed.netloc != 'github.com' or parsed.query or parsed.fragment or not re.fullmatch(r'/magnitudedev/magnitude/releases/download/.+/[^/]+', parsed.path) or re.search(r'%(2f|5c|00)', parsed.path, re.I) or any(ord(c) <= 32 for c in url):
        raise ValueError('Unexpected application download location')
    signature = base64.b64decode(release['signature'], validate=True)
    if len(signature) != 64 or base64.b64encode(signature).decode() != release['signature']: raise ValueError('Invalid publisher signature')
    (root / 'publisher.pem').write_bytes(base64.b64decode(key, validate=True))
    (root / 'signature').write_bytes(signature)
    (root / 'signed').write_bytes(f'magnitude-update-release-v1\nlinux\n{arch}\n{package}\n{version}\n{size}\n{digest}\n'.encode())
    subprocess.run(['openssl', 'pkeyutl', '-verify', '-pubin', '-inkey', str(root / 'publisher.pem'), '-rawin', '-in', str(root / 'signed'), '-sigfile', str(root / 'signature')], check=True, stdout=subprocess.DEVNULL)
    for name, value in [('url', url), ('bytes', str(size)), ('digest', digest), ('version', version)]: (root / name).write_text(value)
except (ValueError, TypeError, KeyError, OSError, subprocess.SubprocessError) as error:
    sys.exit('Application release verification failed: ' + str(error))
PY
    bytes=$(cat "$scratch/bytes")
    version=$(cat "$scratch/version")
    finish_progress "Found Magnitude $version · $package"
    download "$(cat "$scratch/url")" "$scratch/magnitude.$package" "$bytes" 'Downloading app' 'Downloaded app'
    start_progress 'Verifying download checksum'
    actual_bytes=$(wc -c < "$scratch/magnitude.$package" | tr -d ' ')
    [ "$actual_bytes" = "$bytes" ] || fail 'The application download is incomplete.'
    actual_digest=$(sha256sum "$scratch/magnitude.$package" | cut -d ' ' -f 1)
    [ "$actual_digest" = "$(cat "$scratch/digest")" ] || fail 'The application checksum does not match.'
    finish_progress 'Verified download checksum'
    # Fresh cloud images ship without apt's package lists, so the package's dependencies would not resolve.
    if [ "$package" = deb ]; then
      printf 'Refreshing package lists...\n' >&2
      $privilege apt-get update || printf '%s\n' 'Could not refresh the package lists; installing anyway.' >&2
    fi
    # The scratch directory is private; apt reads the already verified package as its _apt account.
    if [ "$package" = deb ]; then chmod 755 "$scratch"; chmod 644 "$scratch/magnitude.deb"; fi
    before=$(installed_version)
    # A running server holds the installation lock the package checks, so upgrade it stopped.
    if [ "$answer" = configured ] && systemctl is-active --quiet magnitude.service; then
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

    # 4. Set up the server, reusing the sudo session from step 2, or say how to later.
    case "$answer" in
      configured)
        installed_summary
        # Only a server this installer stopped is started; one an administrator stopped stays stopped.
        if [ -n "$restart" ]; then
          $privilege systemctl start magnitude.service
          run_step 'Waiting for server to become ready' 'Server is ready' /usr/lib/magnitude-desktop/resources/magnitude _server-wait-ready \
            || fail 'The package was installed, but the server did not become ready. Check: sudo journalctl -u magnitude -n 100 --no-pager'
          printf '%s\n' 'The server restarted and is ready. Run `magnitude status` to check on it.'
        else printf '%s\n' 'The server was not running and was left stopped. Start it with: sudo systemctl start magnitude'
        fi ;;
      yes)
        installed_summary
        if [ -z "$privilege" ] && [ -n "${SUDO_USER:-}" ]; then
          printf '%s\n' "To set up the server, run as $SUDO_USER: magnitude server setup"
        else
          setting_up=1
          run_step 'Setting up the persistent server service' 'Server is ready' magnitude server setup </dev/null
          setting_up=''
        fi ;;
      *)
        installed_summary
        open_desktop linux ;;
    esac
    ;;
  *) fail 'This operating system is not supported.' ;;
esac
