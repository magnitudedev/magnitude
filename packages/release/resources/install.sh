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
fail() { printf '%s\n' "$*" >&2; exit 1; }
[ "$#" -eq 0 ] || fail 'install.sh takes no options. Run: curl -fsSL https://magnitude.dev/install.sh | sh'
case "$origin" in https://*) ;; *) fail 'The installation script has no release origin.' ;; esac
case "$(uname -m)" in arm64|aarch64) arch=arm64 ;; x86_64) arch=x64 ;; *) fail 'This architecture is not supported.' ;; esac
umask 077
scratch=$(mktemp -d "${TMPDIR:-/tmp}/magnitude-install.XXXXXXXX")
keepalive=''
cleanup() { [ -z "$keepalive" ] || kill "$keepalive" 2>/dev/null || true; rm -rf "$scratch"; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP
scratch=$(cd "$scratch" && pwd -P)
download() {
  curl --fail --silent --show-error --location --proto '=https' --proto-redir '=https' \
    --connect-timeout 15 --max-time 600 --max-filesize "$3" --output "$2" "$1"
}
# The landing server counts each request and answers with the publisher-signed offer for this target.
offer() { download "$origin/api/installer?os=$1&arch=$arch&package=$2&offer=1" "$scratch/offer.json" 16384; }
# Answers come from the terminal, never from the script's piped stdin.
has_terminal() { (: </dev/tty) 2>/dev/null; }

case "$(uname -s)" in
  Darwin)
    # The server's LaunchDaemon holds the app and installs its own updates when idle.
    [ ! -e "$mac_server" ] || fail 'Magnitude runs as a server on this Mac and installs its own updates when idle. To reinstall it now, run `magnitude server remove`, then this installer, then `magnitude server setup`.'
    case "$apple_team" in *[!A-Z0-9]*|'') fail 'The installation script has no Apple publisher identity.' ;; esac
    [ "${#apple_team}" -eq 10 ] || fail 'Invalid Apple publisher identity.'
    offer darwin mac-zip
    url=$(/usr/bin/plutil -extract download raw -o - "$scratch/offer.json")
    case "$url" in https://github.com/magnitudedev/magnitude/releases/download/*) ;; *) fail 'Unexpected application download location.' ;; esac
    bytes=$(/usr/bin/plutil -extract release.bytes raw -o - "$scratch/offer.json")
    digest=$(/usr/bin/plutil -extract release.sha256 raw -o - "$scratch/offer.json")
    case "$bytes" in ''|*[!0-9]*) fail 'Invalid application size.' ;; esac
    [ "$bytes" -gt 0 ] && [ "$bytes" -le 2147483648 ] || fail 'Application download exceeds the installation limit.'
    download "$url" "$scratch/magnitude.zip" "$bytes"
    actual_bytes=$(wc -c < "$scratch/magnitude.zip" | tr -d ' ')
    [ "$actual_bytes" = "$bytes" ] || fail 'The application download is incomplete.'
    actual_digest=$(/usr/bin/shasum -a 256 "$scratch/magnitude.zip" | cut -d ' ' -f 1)
    [ "$actual_digest" = "$digest" ] || fail 'The application checksum does not match.'
    mkdir "$scratch/bootstrap"
    # bsdtar's default extraction rejects traversal and writes through archive symlinks.
    /usr/bin/tar -xf "$scratch/magnitude.zip" -C "$scratch/bootstrap" --no-same-owner
    app="$scratch/bootstrap/Magnitude.app"
    /usr/bin/codesign --verify --deep --strict -R "=anchor apple generic and identifier \"dev.magnitude.desktop\" and certificate leaf[subject.OU] = \"$apple_team\" and certificate leaf[field.1.2.840.113635.100.6.1.13] exists" "$app"
    /usr/sbin/spctl --assess --type execute "$app"
    /usr/bin/plutil -create xml1 "$scratch/request.plist"
    /usr/bin/plutil -insert bundle -string "$destination" "$scratch/request.plist"
    /usr/bin/plutil -insert archive -string "$scratch/magnitude.zip" "$scratch/request.plist"
    /usr/bin/plutil -insert channel -string "$channel" "$scratch/request.plist"
    /usr/bin/plutil -insert offer -json "$(cat "$scratch/offer.json")" "$scratch/request.plist"
    /usr/bin/plutil -convert json -o "$scratch/request.json" "$scratch/request.plist"
    "$app/Contents/Resources/magnitude" _install-mac-application "$(cat "$scratch/request.json")"
    printf '%s\n' 'Magnitude was installed. Open it from Applications, or run `magnitude server setup` to run it as a server.'
    ;;
  Linux)
    for tool in python3 openssl curl timeout; do command -v "$tool" >/dev/null 2>&1 || fail "Install $tool before running this installer."; done
    if command -v apt-get >/dev/null 2>&1; then package=deb
    elif command -v dnf >/dev/null 2>&1; then package=rpm
    elif command -v pacman >/dev/null 2>&1; then package=pacman
    else fail 'This Linux distribution requires apt, dnf or pacman.'; fi
    [ "$package" != pacman ] || [ "$arch" = x64 ] || fail 'Arch Linux packages are available for x86-64 only.'

    # 1. Ask first, so the person can walk away after answering and authorizing.
    if [ -f "$server_marker" ]; then answer=configured
    elif ! has_terminal; then answer=no
    else
      printf '%s\n%s ' 'Run Magnitude as a server? It starts on boot, runs without the desktop app,' \
        'and you can use it from a browser on another computer. [y/N]' >/dev/tty
      # --foreground keeps the reader in the terminal's process group, so it may read the terminal.
      if line=$(timeout --foreground 60 sh -c 'IFS= read -r line </dev/tty && printf "%s" "$line"'); then
        case "$line" in [Yy]|[Yy][Ee][Ss]) answer=yes ;; *) answer=no ;; esac
      else
        status=$?
        [ "$status" -eq 124 ] || exit "$status"
        printf '\n' >/dev/tty
        answer=timeout
      fi
    fi

    # 2. Get sudo up front and keep it fresh, instead of waiting at a password prompt later.
    if [ "$(id -u)" -eq 0 ]; then privilege=''
    else
      command -v sudo >/dev/null 2>&1 || fail 'Installing Magnitude needs sudo. Install it, or run this installer as root.'
      if has_terminal; then sudo -v </dev/tty || fail 'Installing Magnitude needs administrator access.'
      else sudo -n -v 2>/dev/null || fail 'Installing Magnitude needs administrator access, and there is no terminal to ask for a password. Run it in a terminal, or allow passwordless sudo.'
      fi
      privilege=sudo
      # Detached from the terminal so the installer's session ends when it does.
      ( while sleep 50; do sudo -n -v 2>/dev/null || exit 0; done ) </dev/null >/dev/null 2>&1 &
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
    for name, value in [('url', url), ('bytes', str(size)), ('digest', digest)]: (root / name).write_text(value)
except (ValueError, TypeError, KeyError, OSError, subprocess.SubprocessError) as error:
    sys.exit('Application release verification failed: ' + str(error))
PY
    bytes=$(cat "$scratch/bytes")
    download "$(cat "$scratch/url")" "$scratch/magnitude.$package" "$bytes"
    actual_bytes=$(wc -c < "$scratch/magnitude.$package" | tr -d ' ')
    [ "$actual_bytes" = "$bytes" ] || fail 'The application download is incomplete.'
    actual_digest=$(sha256sum "$scratch/magnitude.$package" | cut -d ' ' -f 1)
    [ "$actual_digest" = "$(cat "$scratch/digest")" ] || fail 'The application checksum does not match.'
    # A running server holds the installation lock the package checks, so upgrade it stopped.
    [ "$answer" != configured ] || $privilege systemctl stop magnitude.service
    if [ "$package" = deb ]; then $privilege env DEBIAN_FRONTEND=noninteractive apt-get install -y "$scratch/magnitude.deb"
    elif [ "$package" = rpm ]; then $privilege dnf install -y "$scratch/magnitude.rpm"
    else $privilege pacman -U --noconfirm "$scratch/magnitude.pacman"; fi

    # 4. Set up the server, reusing the sudo session from step 2, or say how to later.
    case "$answer" in
      configured)
        $privilege systemctl start magnitude.service
        printf '%s\n' 'Magnitude was upgraded and the server restarted. Run `magnitude status` to check on it.' ;;
      yes)
        if [ -z "$privilege" ] && [ -n "${SUDO_USER:-}" ]; then
          printf '%s\n' "Magnitude was installed. To set up the server, run as $SUDO_USER: magnitude server setup"
        else magnitude server setup </dev/null
        fi ;;
      timeout) printf '%s\n' 'Magnitude was installed.' 'No answer, skipping. To set it up later: magnitude server setup' ;;
      *) printf '%s\n' 'Magnitude was installed.' 'Skipped. To set it up later: magnitude server setup' ;;
    esac
    ;;
  *) fail 'This operating system is not supported.' ;;
esac
