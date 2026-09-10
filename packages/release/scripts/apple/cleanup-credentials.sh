#!/bin/bash
set -euo pipefail
if [[ -n "${APPLE_RELEASE_KEYCHAIN:-}" ]]; then
  /usr/bin/security delete-keychain "$APPLE_RELEASE_KEYCHAIN"
  rmdir "$(dirname "$APPLE_RELEASE_KEYCHAIN")"
fi
