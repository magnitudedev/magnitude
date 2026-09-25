#!/usr/bin/env bash
# Disposable native Linux VM acceptance; the package must not already be installed.
set -euo pipefail
umask 077
test "$#" -ge 3 && test "$#" -le 4 || { echo 'Usage: test-linux-install-script.sh PACKAGE VERSION ICN_INSTALLATION [PREPARED_HOSTING]' >&2; exit 1; }
artifact=$(realpath "$1")
version=$2
export MAGNITUDE_ICN_PATH=$(realpath "$3")
if command -v dpkg-query >/dev/null && dpkg-query -W -f='${Status}' magnitude-desktop 2>/dev/null | grep -q 'install ok installed'; then
  echo 'Remove the test installation before running this acceptance.' >&2; exit 1
fi
if command -v rpm >/dev/null && rpm -q magnitude-desktop >/dev/null 2>&1; then
  echo 'Remove the test installation before running this acceptance.' >&2; exit 1
fi
sudo -n true
root=$(mktemp -d "${TMPDIR:-/tmp}/magnitude-script-acceptance.XXXXXXXX")
echo "Acceptance output: $root"
repository=$(cd "$(dirname "$0")/../../../.." && pwd)
if test "$#" -eq 4; then
  cp -a "$(realpath "$4")" "$root/hosting"
else
  "${BUN:-bun}" "$repository/packages/release/scripts/acceptance/prepare-linux-script-fixture.ts" "$artifact" "$root/hosting" "$version"
fi
openssl req -x509 -newkey rsa:2048 -nodes -days 1 -subj '/CN=localhost' \
  -addext 'subjectAltName=DNS:localhost,DNS:github.com' -keyout "$root/tls.key" -out "$root/tls.crt" > "$root/tls.log" 2>&1
export CURL_HOME="$root/curl"
mkdir "$CURL_HOME"
printf '%s\n' 'connect-to = "github.com:443:127.0.0.1:18443"' 'noproxy = "*"' > "$CURL_HOME/.curlrc"
export CURL_CA_BUNDLE="$root/tls.crt"
server_pid=''
owner_pid=''
cleanup() {
  if test -n "$owner_pid"; then kill -TERM "$owner_pid" 2>/dev/null || true; wait "$owner_pid" 2>/dev/null || true; fi
  if test -n "$server_pid"; then kill -TERM "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true; fi
}
trap cleanup EXIT
python3 - "$root" > "$root/https.log" 2>&1 <<'PY' &
import functools, http.server, pathlib, ssl, sys
root = pathlib.Path(sys.argv[1])
handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(root / 'hosting'))
server = http.server.ThreadingHTTPServer(('127.0.0.1', 18443), handler)
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
bash "$root/hosting/install.sh" > "$root/install.log" 2>&1
export MAGNITUDE_DEV_DATA_DIR="$root/profile"
export MAGNITUDE_DESKTOP_STATE_DIR="$root/state"
export MAGNITUDE_DEV_PORT=11237
/usr/bin/magnitude status > "$root/initial-status.txt"
grep -Eq 'Runtime[[:space:]]+Stopped' "$root/initial-status.txt"
test "$(/usr/bin/magnitude --version)" = "$version"
/usr/bin/magnitude serve > "$root/serve.log" 2>&1 &
owner_pid=$!
for ((attempt=0; attempt<300; attempt++)); do
  kill -0 "$owner_pid"
  /usr/bin/magnitude status > "$root/status.txt"
  if grep -Eq 'Runtime[[:space:]]+Ready' "$root/status.txt" && grep -Eq 'Owner[[:space:]]+Headless' "$root/status.txt"; then break; fi
  sleep 0.2
done
grep -Eq 'Runtime[[:space:]]+Ready' "$root/status.txt"
grep -Eq 'Owner[[:space:]]+Headless' "$root/status.txt"
/usr/bin/magnitude models status > "$root/models.txt"
/usr/bin/magnitude hardware > "$root/hardware.txt"
kill -TERM "$owner_pid"
wait "$owner_pid"
owner_pid=''
bash "$root/hosting/install.sh" > "$root/repeat-install.log" 2>&1
/usr/bin/magnitude status > "$root/final-status.txt"
grep -Eq 'Runtime[[:space:]]+Stopped' "$root/final-status.txt"
printf '%s\n' 'HTTPS script installation, foreground serve, CLI queries, graceful shutdown and repeat installation passed' | tee "$root/result.txt"
