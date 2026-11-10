#!/usr/bin/env bash
# Smoke test of one Magnitude image on this host's architecture. Starts it the way the docs do,
# with a volume and a network key, then checks readiness, the API's key handling, a catalog
# download, one chat completion, and a clean stop. Needs docker and curl; downloads a ~1 GB model.
#
#   test-docker.sh IMAGE [VERSION]
#
# IMAGE is any reference docker can run (a local tag, repo:tag or repo@sha256:...). With VERSION,
# the installed CLI must report exactly that version. MAGNITUDE_SMOKE_GPU_FLAGS adds run flags,
# such as "--gpus all"; MAGNITUDE_SMOKE_KEEP=1 keeps the container and volume for inspection.
set -euo pipefail
test "$#" -ge 1 && test "$#" -le 2 || { echo 'Usage: test-docker.sh IMAGE [VERSION]' >&2; exit 2; }
image=$1
version=${2:-}
model=minicpm5-1b:gguf:q4
name="magnitude-smoke-$$"
volume="$name-data"
key="$(head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n')"
started=$(date +%s)

log() { printf '[%4ss] %s\n' "$(( $(date +%s) - started ))" "$*"; }
fail() {
  log "FAIL: $*"
  docker exec "$name" magnitude status 2>&1 | sed 's/^/  status: /' || true
  docker logs --tail 200 "$name" 2>&1 | sed 's/^/  log: /' || true
  docker exec "$name" tail -n 100 /data/.magnitude/logs/inference.log 2>&1 | sed 's/^/  inference: /' || true
  exit 1
}
cleanup() {
  if test "${MAGNITUDE_SMOKE_KEEP:-}" = 1; then log "Kept container $name and volume $volume"; return; fi
  docker rm --force "$name" >/dev/null 2>&1 || true
  docker volume rm "$volume" >/dev/null 2>&1 || true
}
trap cleanup EXIT

# The published port arrives from the bridge, not loopback, so the service treats it as another device.
# The smoke test never contacts magnitude.dev.
# shellcheck disable=SC2086
docker run --detach --name "$name" \
  --add-host magnitude.dev:127.0.0.1 \
  --mount "type=volume,source=$volume,target=/data" \
  --env "MAGNITUDE_NETWORK_KEY=$key" \
  --publish 127.0.0.1::10100 \
  ${MAGNITUDE_SMOKE_GPU_FLAGS:-} \
  "$image" >/dev/null
port=$(docker port "$name" 10100/tcp | head -n 1 | sed 's/.*://')
origin="http://127.0.0.1:$port"
log "Started $name from $image on $origin"

code() { curl --silent --output /dev/null --write-out '%{http_code}' --max-time 30 "$@" || true; }

health=''
for ((i = 0; i < 100; i++)); do
  test "$(docker inspect --format '{{.State.Running}}' "$name")" = true || fail "container exited with $(docker inspect --format '{{.State.ExitCode}}' "$name")"
  health=$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{end}}' "$name")
  test "$health" = healthy && break
  sleep 3
done
test "$health" = healthy || fail "not healthy within 5 minutes (last: ${health:-no healthcheck})"
log "Healthy"

status=$(docker exec "$name" magnitude status) || fail 'magnitude status failed'
grep -Eq 'Runtime[[:space:]]+Ready' <<<"$status" || fail "status is not Ready: $status"
test "$(docker exec "$name" id -u)" = 1000 || fail 'the service does not run as uid 1000'
test "$(docker exec "$name" cat /usr/lib/magnitude-desktop/resources/update-package.json)" = '{"format":"docker"}' \
  || fail 'update-package.json is not the docker format'
if test -n "$version"; then
  installed=$(docker exec "$name" magnitude --version | tr -d '[:space:]')
  test "$installed" = "$version" || fail "installed version is $installed, expected $version"
fi
log "Ready as uid 1000, docker package${version:+, version $version}"

test "$(code "$origin/health")" = 200 || fail '/health is not 200 from the host'
test "$(code "$origin/inference/v1/models")" = 401 || fail '/inference/v1/models without the key is not 401'
test "$(code -H "Authorization: Bearer $key" "$origin/inference/v1/models")" = 200 || fail '/inference/v1/models with the key is not 200'
test "$(code -H 'Host: rebind.example' "$origin/inference/v1/models")" = 421 || fail 'an unknown Host without the key is not 421'
# Other containers address it by names such as a compose service's; the key admits any of them.
test "$(code -H 'Host: magnitude:10100' -H "Authorization: Bearer $key" "$origin/inference/v1/models")" = 200 \
  || fail 'a service name with the key is not 200'
log "Key and Host checks passed"

docker exec "$name" magnitude models pull "$model" || fail "models pull $model failed"
for ((i = 0; ; i++)); do
  curl --silent --fail --max-time 30 -H "Authorization: Bearer $key" "$origin/inference/v1/models" | grep -q "\"$model\"" && break
  test "$i" -lt 360 || fail "$model was not installed within 30 minutes"
  if ((i % 12 == 0)); then docker exec "$name" magnitude models show "$model" 2>&1 | head -n 5 | sed 's/^/  /' || true; fi
  sleep 5
done
log "Downloaded $model"

response=$(curl --silent --show-error --fail --max-time 900 \
  -H "Authorization: Bearer $key" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with the single word ready.\"}],\"max_tokens\":32,\"reasoning_effort\":\"none\"}" \
  "$origin/inference/v1/chat/completions") || fail 'chat completion failed'
grep -Eq '"completion_tokens":[1-9]' <<<"$response" || fail "chat completion produced no tokens: $response"
log "Chat completion answered"

docker stop --time 30 "$name" >/dev/null
exit_code=$(docker inspect --format '{{.State.ExitCode}}' "$name")
test "$exit_code" = 0 || fail "docker stop exited with $exit_code"
log "Stopped cleanly. Docker smoke test passed for $image"
