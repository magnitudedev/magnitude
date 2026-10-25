#!/bin/sh
set -eu
root=$HOME/installer-audit-next
export CURL_HOME=$root/curl
export CURL_CA_BUNDLE=$root/fixture-tls.crt
mkdir -p "$CURL_HOME"
printf '%s\n' 'connect-to = "github.com:443:127.0.0.1:18443"' 'noproxy = "*"' > "$CURL_HOME/.curlrc"
# This disposable VM's app was installed by this audit; start the candidate from a clean destination.
test ! -e /Library/LaunchDaemons/dev.magnitude.server.plist
sudo rm -rf /Applications/Magnitude.app
python3 - "$root" > "$root/https.log" 2>&1 <<'PY' &
import functools,http.server,pathlib,ssl,sys
r=pathlib.Path(sys.argv[1]);s=http.server.ThreadingHTTPServer(('127.0.0.1',18443),functools.partial(http.server.SimpleHTTPRequestHandler,directory=str(r/'hosting/mac-zip')))
c=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER);c.load_cert_chain(r/'fixture-tls.crt',r/'fixture-tls.key');s.socket=c.wrap_socket(s.socket,server_side=True)
(r/'listening').touch();s.serve_forever()
PY
server=$!
trap 'kill -TERM "$server" 2>/dev/null || true; wait "$server" 2>/dev/null || true' EXIT
for n in $(seq 1 100); do [ ! -f "$root/listening" ] || break; sleep .1; done
test -f "$root/listening"
for pass in fresh repeat; do
 echo INSTALL_PASS="$pass"
 curl -fsSL https://localhost:18443/install.sh | sh > "$root/$pass-install.log" 2>&1
 cat "$root/$pass-install.log"
 "$HOME/.magnitude/bin/magnitude" --version
 /usr/bin/codesign --verify --deep --strict /Applications/Magnitude.app
 /usr/sbin/spctl --assess --type execute /Applications/Magnitude.app
done
shasum -a 256 /Applications/Magnitude.app/Contents/Resources/magnitude > "$root/cli-before-rejection.sha256"
cp "$root/hosting/mac-zip/api/installer" "$root/offer-original.json"
python3 - "$root/hosting/mac-zip/api/installer" <<'PY'
import pathlib,json,base64,sys
p=pathlib.Path(sys.argv[1]);o=json.loads(p.read_text());o['release']['signature']=base64.b64encode(bytes(64)).decode();p.write_text(json.dumps(o))
PY
if curl -fsSL https://localhost:18443/install.sh | sh > "$root/rejected-signature.log" 2>&1; then echo 'ERROR: invalid signature accepted'; exit 1; fi
cat "$root/rejected-signature.log"
shasum -a 256 -c "$root/cli-before-rejection.sha256"
cp "$root/offer-original.json" "$root/hosting/mac-zip/api/installer"
/bin/zsh -lic 'command -v magnitude; magnitude --version'
printf 'RESULT=next-mac-signed-fresh-repeat-signature-refusal-passed\n'
