/** Root-owned receipt is written only after the complete trusted setup succeeds. */
export const azureInitializationWait = (directory = "/var/lib/magnitude-lab") => {
  const root = directory.replaceAll("'", "'\\''")
  return [
  "#!/bin/sh", "set -eu", "umask 077",
  `mkdir -p '${root}'`,
  "status=0",
  `cloud-init status --wait --format json > '${root}/cloud-init-status.json' || status=$?`,
  // cloud-init documents 2 as completed with recoverable errors. Preserve them and
  // require our own final receipt; an unrelated platform warning cannot hide setup failure.
  'case "$status" in 0|2) ;; *) exit "$status" ;; esac',
  `test "$(cat '${root}/ready')" = ready`,
  `test -s '${root}/runtime.json'`,
  ].join("\n")
}

/** Keep the native error before Python's potentially enormous inline-command traceback. */
export const azureInitializationDiagnosticsScript = (log = "/var/log/cloud-init-output.log") => `python3 - <<'LAB_DIAGNOSTICS'
import base64,json,pathlib,subprocess
path=pathlib.Path(base64.b64decode('${Buffer.from(log).toString("base64")}').decode())
try:
 with path.open('rb') as stream:
  stream.seek(0,2)
  stream.seek(max(0,stream.tell()-65536))
  text=stream.read(65536).decode('utf-8','replace')
 marker=text.rfind('Traceback (most recent call last):')
 if marker>=0:text=text[:marker]
 print('[native preparation output]')
 print(text[-2200:])
except OSError:print('Preparation output log is unavailable')
try:
 result=subprocess.run(['cloud-init','status','--long','--format','json'],capture_output=True,text=True,timeout=15)
 state=json.loads(result.stdout)
 print('[cloud-init status]')
 print(json.dumps({key:state.get(key) for key in ['status','extended_status','errors']})[:1000])
except (OSError,ValueError,subprocess.TimeoutExpired):print('Cloud-init status is unavailable')
LAB_DIAGNOSTICS`
