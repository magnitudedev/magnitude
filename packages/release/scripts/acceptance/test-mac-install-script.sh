#!/usr/bin/env bash
# Runs only against a prepared, signed fixture and an unused installation destination.
set -euo pipefail
umask 077
test "$#" -eq 5 || { echo 'Usage: test-mac-install-script.sh HOSTING BUNDLE VERSION PROFILE STATE' >&2; exit 1; }
hosting=$1
bundle=$2
version=$3
export MAGNITUDE_DEV_DATA_DIR=$4
export MAGNITUDE_DESKTOP_STATE_DIR=$5
test ! -e "$bundle"
root=$(mktemp -d "$hosting/mac-script-evidence.XXXXXXXX")
echo "Mac script evidence: $root"
export HOME="$root/home"
export CURL_HOME="$root/curl"
export CURL_CA_BUNDLE="$root/tls.crt"
export PATH=/usr/bin:/bin:/usr/sbin:/sbin
unset ZDOTDIR
mkdir "$HOME" "$CURL_HOME"
printf '%s\n' 'connect-to = "github.com:443:127.0.0.1:18443"' 'noproxy = "*"' > "$CURL_HOME/.curlrc"
cat > "$root/tls.conf" <<'CONFIG'
[req]
distinguished_name=subject
x509_extensions=extensions
prompt=no
[subject]
CN=localhost
[extensions]
subjectAltName=DNS:localhost,DNS:github.com
CONFIG
openssl req -x509 -newkey rsa:2048 -nodes -days 1 -config "$root/tls.conf" \
  -keyout "$root/tls.key" -out "$root/tls.crt" > "$root/tls.log" 2>&1
server_pid=''
offer="$hosting/install/stable/darwin-arm64-mac-zip.json"
cp "$offer" "$root/offer-original.json"
cleanup() {
  cp "$root/offer-original.json" "$offer"
  if test -n "$server_pid"; then kill -TERM "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true; fi
}
trap cleanup EXIT
python3 - "$root" "$hosting" > "$root/https.log" 2>&1 <<'PY' &
import functools, http.server, pathlib, ssl, sys
root = pathlib.Path(sys.argv[1])
server = http.server.ThreadingHTTPServer(('127.0.0.1', 18443), functools.partial(http.server.SimpleHTTPRequestHandler, directory=sys.argv[2]))
context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
context.load_cert_chain(root / 'tls.crt', root / 'tls.key')
server.socket = context.wrap_socket(server.socket, server_side=True)
(root / 'listening').touch()
server.serve_forever()
PY
server_pid=$!
for ((attempt=0; attempt<100; attempt++)); do
  test -f "$root/listening" && break
  kill -0 "$server_pid"
  sleep 0.1
done
test -f "$root/listening"
curl --fail --silent --show-error https://localhost:18443/install.sh > "$root/install.sh"
for pass in fresh repeat; do
  bash "$root/install.sh" --destination "$bundle" > "$root/$pass-install.log" 2>&1
  test "$("$HOME/.magnitude/bin/magnitude" --version)" = "$version"
  "$HOME/.magnitude/bin/magnitude" status > "$root/$pass-status.txt"
  grep -Eq 'Runtime[[:space:]]+Stopped' "$root/$pass-status.txt"
  /usr/bin/codesign --verify --deep --strict "$bundle"
  /usr/bin/xcrun stapler validate "$bundle"
  /usr/sbin/spctl --assess --type execute "$bundle"
done
/bin/zsh -lic 'command -v magnitude; magnitude --version' > "$root/shell-registration.txt"
grep -Fx "$HOME/.magnitude/bin/magnitude" "$root/shell-registration.txt"
grep -Fx "$version" "$root/shell-registration.txt"
python3 - "$offer" <<'PY'
import base64, json, pathlib, sys
path = pathlib.Path(sys.argv[1])
offer = json.loads(path.read_text())
offer['release']['signature'] = base64.b64encode(bytes(64)).decode()
path.write_text(json.dumps(offer))
PY
if bash "$root/install.sh" --destination "$bundle" > "$root/rejected-install.log" 2>&1; then
  echo 'Invalid publisher signature was accepted' >&2; exit 1
fi
test "$("$HOME/.magnitude/bin/magnitude" --version)" = "$version"
"$HOME/.magnitude/bin/magnitude" status > "$root/final-status.txt"
grep -Eq 'Runtime[[:space:]]+Stopped' "$root/final-status.txt"
printf '%s\n' 'Signed HTTPS fresh/repeat installation, shell registration and invalid-signature rejection passed' | tee "$root/result.txt"
cp "$root/result.txt" "$hosting/mac-script-result.txt"
