#!/bin/sh
set -eu

# Release generation supplies publisher identity and the metadata origin.
origin='@MAGNITUDE_INSTALL_ORIGIN@'
apple_team='@MAGNITUDE_APPLE_TEAM@'
publisher_key='@MAGNITUDE_PUBLISHER_KEY@'
channel=stable
destination=/Applications/Magnitude.app
destination_set=false
fail() { printf '%s\n' "$*" >&2; exit 1; }
while [ "$#" -gt 0 ]; do
  case "$1" in
    --help|-h) printf '%s\n' 'Usage: install.sh [--channel stable|beta|alpha] [--destination /Applications/Magnitude.app]'; exit 0 ;;
    --channel) [ "$#" -ge 2 ] || fail 'Missing channel'; channel=$2; shift 2 ;;
    --destination) [ "$#" -ge 2 ] || fail 'Missing destination'; destination=$2; destination_set=true; shift 2 ;;
    *) fail "Unknown installation option: $1" ;;
  esac
done
case "$channel" in stable|beta|alpha) ;; *) fail 'Channel must be stable, beta, or alpha.' ;; esac
case "$origin" in https://*) ;; *) fail 'The installation script has no release origin.' ;; esac
case "$(uname -m)" in arm64|aarch64) arch=arm64 ;; x86_64) arch=x64 ;; *) fail 'This architecture is not supported.' ;; esac
umask 077
scratch=$(mktemp -d "${TMPDIR:-/tmp}/magnitude-install.XXXXXXXX")
trap 'rm -rf "$scratch"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP
download() {
  curl --fail --silent --show-error --location --proto '=https' --proto-redir '=https' \
    --connect-timeout 15 --max-time 600 --max-filesize "$3" --output "$2" "$1"
}

case "$(uname -s)" in
  Darwin)
    case "$apple_team" in *[!A-Z0-9]*|'') fail 'The installation script has no Apple publisher identity.' ;; esac
    [ "${#apple_team}" -eq 10 ] || fail 'Invalid Apple publisher identity.'
    case "$destination" in /*.app) ;; *) fail 'The destination must be an absolute .app path.' ;; esac
    download "$origin/install/$channel/darwin-$arch-mac-zip.json" "$scratch/offer.json" 16384
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
    /usr/bin/plutil -create json "$scratch/request.json"
    /usr/bin/plutil -insert bundle -string "$destination" "$scratch/request.json"
    /usr/bin/plutil -insert archive -string "$scratch/magnitude.zip" "$scratch/request.json"
    /usr/bin/plutil -insert channel -string "$channel" "$scratch/request.json"
    /usr/bin/plutil -insert offer -json "$(cat "$scratch/offer.json")" "$scratch/request.json"
    "$app/Contents/Resources/magnitude" _install-mac-application "$(cat "$scratch/request.json")"
    ;;
  Linux)
    [ "$destination_set" = false ] || fail 'Linux installation paths are owned by the package manager.'
    for tool in python3 openssl curl; do command -v "$tool" >/dev/null 2>&1 || fail "Install $tool before running this installer."; done
    if command -v apt-get >/dev/null 2>&1; then package=deb
    elif command -v dnf >/dev/null 2>&1; then package=rpm
    else fail 'This Linux distribution requires apt or dnf.'; fi
    download "$origin/install/$channel/linux-$arch-$package.json" "$scratch/offer.json" 16384
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
    if [ "$(id -u)" -eq 0 ]; then privilege=''; else privilege=sudo; fi
    if [ "$package" = deb ]; then $privilege apt-get install -y "$scratch/magnitude.deb"
    else $privilege dnf install -y "$scratch/magnitude.rpm"; fi
    printf '%s\n' 'Magnitude was installed. Run magnitude serve to start the server.'
    ;;
  *) fail 'This operating system is not supported.' ;;
esac
