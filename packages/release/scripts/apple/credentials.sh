#!/bin/bash
# Called only by trusted jobs in the protected apple-release environment. Never enable xtrace.
set -euo pipefail
umask 077
: "${RUNNER_TEMP:?}" "${GITHUB_ENV:?}" "${APPLE_DEVELOPER_ID_P12_BASE64:?}"
: "${APPLE_DEVELOPER_ID_P12_PASSWORD:?}" "${APPLE_NOTARY_API_KEY_P8:?}"
: "${APPLE_NOTARY_KEY_ID:?}" "${APPLE_NOTARY_ISSUER_ID:?}" "${APPLE_TEAM_ID:?}" "${APPLE_SIGNING_IDENTITY:?}"
credential_dir="$(mktemp -d "$RUNNER_TEMP/magnitude-apple.XXXXXX")"
keychain="$credential_dir/release.keychain-db"
keychain_password="$(openssl rand -hex 32)"
trap 'rm -f "$credential_dir/certificate.p12" "$credential_dir/notary.p8" "$credential_dir/DeveloperIDG2CA.cer"' EXIT
printf '%s' "$APPLE_DEVELOPER_ID_P12_BASE64" | /usr/bin/base64 --decode > "$credential_dir/certificate.p12"
printf '%s' "$APPLE_NOTARY_API_KEY_P8" > "$credential_dir/notary.p8"
/usr/bin/security create-keychain -p "$keychain_password" "$keychain"
printf 'APPLE_RELEASE_KEYCHAIN=%s\n' "$keychain" >> "$GITHUB_ENV"
/usr/bin/security set-keychain-settings -lut 21600 "$keychain"
/usr/bin/security unlock-keychain -p "$keychain_password" "$keychain"
/usr/bin/security import "$credential_dir/certificate.p12" -k "$keychain" -P "$APPLE_DEVELOPER_ID_P12_PASSWORD" -T /usr/bin/codesign -T /usr/bin/security > /dev/null
/usr/bin/curl --fail --silent --show-error --location https://www.apple.com/certificateauthority/DeveloperIDG2CA.cer -o "$credential_dir/DeveloperIDG2CA.cer"
/usr/bin/security import "$credential_dir/DeveloperIDG2CA.cer" -k "$keychain" > /dev/null
/usr/bin/security set-key-partition-list -S apple-tool:,apple:,codesign: -s -k "$keychain_password" "$keychain" > /dev/null
/usr/bin/security list-keychains -d user -s "$keychain"
/usr/bin/xcrun notarytool store-credentials magnitude-release --keychain "$keychain" \
  --key "$credential_dir/notary.p8" --key-id "$APPLE_NOTARY_KEY_ID" --issuer "$APPLE_NOTARY_ISSUER_ID" > /dev/null
printf 'APPLE_TEAM_ID=%s\nAPPLE_SIGNING_IDENTITY=%s\n' "$APPLE_TEAM_ID" "$APPLE_SIGNING_IDENTITY" >> "$GITHUB_ENV"
