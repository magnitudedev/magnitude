#!/bin/sh
set -eu
root=$HOME/installer-audit-next
sh "$GITHUB_WORKSPACE/scripts/installer-audit-mac-check.sh" 2>&1 | tee "$root/acceptance-install.log"
grep -q 'RESULT=next-mac-signed-fresh-repeat-signature-refusal-passed' "$root/acceptance-install.log"
python3 "$GITHUB_WORKSPACE/scripts/installer-audit-unix-runtime.py" --inference > "$root/acceptance-runtime.log" 2>&1
cat "$root/acceptance-runtime.log"
python3 "$GITHUB_WORKSPACE/scripts/installer-audit-unix-api.py" > "$root/acceptance-api.log" 2>&1
cat "$root/acceptance-api.log"
