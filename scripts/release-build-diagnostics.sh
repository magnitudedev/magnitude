#!/usr/bin/env bash
set -euo pipefail

snapshot() {
  echo "::group::backend resource diagnostics $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  df -h / "${GITHUB_WORKSPACE:-$PWD}"
  df -i / "${GITHUB_WORKSPACE:-$PWD}"
  if command -v free >/dev/null 2>&1; then
    free -h
  elif command -v vm_stat >/dev/null 2>&1; then
    vm_stat
  fi
  for metric in memory.current memory.peak memory.events; do
    if [[ -r "/sys/fs/cgroup/${metric}" ]]; then
      echo "--- cgroup ${metric}"
      cat "/sys/fs/cgroup/${metric}"
    fi
  done
  echo "--- highest-RSS processes"
  if [[ "$(uname -s)" == "Darwin" ]]; then
    ps -axo pid=,ppid=,rss=,%cpu=,stat=,comm=,command= |
      sort -nr -k3 | head -n 15 | cut -c1-240 || true
  else
    ps -eo pid,ppid,rss,%cpu,stat,comm,args --sort=-rss |
      head -n 16 | cut -c1-240 || true
  fi
  echo "::endgroup::"
}

monitor() {
  trap 'exit 0' INT TERM
  while true; do
    snapshot || true
    sleep "${MAGNITUDE_DIAGNOSTIC_INTERVAL_SECONDS:-60}" &
    wait $!
  done
}

case "${1:-}" in
  once)
    snapshot || true
    ;;
  run)
    shift
    if [[ "${1:-}" == "--" ]]; then shift; fi
    if [[ $# -eq 0 ]]; then
      echo "usage: release-build-diagnostics.sh run -- <command> [args...]" >&2
      exit 2
    fi
    monitor &
    diagnostics_pid=$!
    cleanup() {
      kill "$diagnostics_pid" 2>/dev/null || true
      wait "$diagnostics_pid" 2>/dev/null || true
    }
    trap cleanup EXIT INT TERM
    set +e
    "$@"
    status=$?
    set -e
    snapshot || true
    exit "$status"
    ;;
  *)
    echo "usage: release-build-diagnostics.sh once | run -- <command> [args...]" >&2
    exit 2
    ;;
esac
