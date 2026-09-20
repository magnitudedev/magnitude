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
