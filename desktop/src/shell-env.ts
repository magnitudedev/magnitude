/**
 * Login shell env resolution — spec §13.2
 *
 * The VS Code pattern with two-phase probe, sentinel var, UUID markers,
 * env -0 parsing, 10s timeout. Resolves the login shell environment in
 * the Electron main process BEFORE spawning the daemon so the ACN
 * subprocess inherits the correct PATH and environment variables.
 */
import { spawnSync } from "node:child_process"
import * as os from "node:os"

/**
 * Inherit the login shell environment into process.env.
 * Skips on Windows (env is in registry) and if launched from CLI
 * (env already correct — check for sentinel env var).
 */
export function inheritLoginShellEnv(): void {
  if (process.platform === "win32") return
  if (process.env.MAGNITUDE_LAUNCHED_FROM_CLI) return

  const shell = getUserShell()
  if (!shell) return

  const env = loadShellEnv(shell)
  if (env) {
    // Shell env as base, app env overrides (so explicit app vars take precedence)
    const merged = { ...env, ...process.env }
    Object.assign(process.env, merged)
  }
}

function getUserShell(): string | null {
  if (process.env.SHELL) return process.env.SHELL
  try {
    const info = os.userInfo()
    if (info.shell) return info.shell
  } catch {}
  if (process.platform === "darwin") return "/bin/zsh"
  if (process.platform === "linux") return "/bin/bash"
  return null
}

function loadShellEnv(shell: string): Record<string, string> | null {
  const name = shell.split("/").pop()?.toLowerCase() ?? ""
  // Nushell doesn't support POSIX -il flags — skip, fall back to process.env
  if (name === "nu" || name === "nu.exe") return null

  // Try interactive login first (most complete — sources both .zprofile AND .zshrc)
  const interactive = probeShellEnv(shell, ["-ilc"])
  if (interactive) return interactive

  // Fall back to login only (sources .zprofile) if interactive timed out
  return probeShellEnv(shell, ["-lc"])
}

function probeShellEnv(shell: string, flags: string[]): Record<string, string> | null {
  const marker = `__MAGNITUDE_ENV_${Date.now()}__`
  const command = `echo '${marker}'; env -0; echo '${marker}'`

  try {
    const result = spawnSync(shell, [...flags, command], {
      encoding: "utf8",
      timeout: 10_000,
      env: {
        ...process.env,
        ELECTRON_RUN_AS_NODE: "1",
        ELECTRON_NO_ATTACH_CONSOLE: "1",
        MAGNITUDE_RESOLVING_ENV: "1",
      },
    })

    if (result.error || result.status !== 0) return null

    const start = result.stdout.indexOf(marker)
    const end = result.stdout.lastIndexOf(marker)
    if (start === -1 || end === -1 || start === end) return null

    const envPart = result.stdout.slice(start + marker.length, end).trim()
    const env: Record<string, string> = {}

    for (const entry of envPart.split("\0")) {
      const idx = entry.indexOf("=")
      if (idx > 0) {
        const key = entry.slice(0, idx)
        if (!key.startsWith("MAGNITUDE_RESOLVING") && !key.startsWith("ELECTRON_")) {
          env[key] = entry.slice(idx + 1)
        }
      }
    }

    return env
  } catch {
    return null
  }
}
