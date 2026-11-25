import { FileSystem } from "@effect/platform"
import { Data, Effect } from "effect"
import { execFileSync } from "node:child_process"
import { chmodSync, mkdtempSync, rmSync, writeFileSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { valid as validSemver } from "semver"
import { isPrereleaseVersion } from "../release-channel"
import { renderUnixInstallationScript, renderWindowsInstallationScript } from "./installation-scripts"

export class SyncInstallationScriptsError extends Data.TaggedError("SyncInstallationScriptsError")<{
  readonly message: string
}> {}

export interface SyncInstallationScriptsOptions {
  readonly landingRepo: string
  readonly branch?: string
  readonly version: string
  readonly sourceCommit: string
  readonly origin: string
  readonly appleTeam: string
  readonly windowsPublisher: string
  readonly publicKey: string
  readonly push?: boolean
  readonly authorName?: string
  readonly authorEmail?: string
}

export type SyncInstallationScriptsResult =
  | { readonly status: "skipped-prerelease"; readonly changed: readonly [] }
  | { readonly status: "up-to-date"; readonly changed: readonly [] }
  | { readonly status: "synced"; readonly changed: readonly ("install.sh" | "install.ps1")[]; readonly commit: string }

const fail = (message: string) => Effect.fail(new SyncInstallationScriptsError({ message }))

const runCommand = (command: string, args: readonly string[], cwd?: string) =>
  Effect.try({
    try: () => execFileSync(command, args, { cwd, encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] }).trim(),
    catch: (error) => {
      const detail = error instanceof Error && "stderr" in error && typeof error.stderr === "string" && error.stderr.trim()
        ? error.stderr.trim()
        : error instanceof Error
          ? error.message
          : String(error)
      return new SyncInstallationScriptsError({ message: `${command} ${args.join(" ")} failed: ${detail}` })
    },
  })

const validateUnixShellSyntax = (shell: string) =>
  Effect.try({
    try: () => {
      const dir = mkdtempSync(join(tmpdir(), "magnitude-install-check-"))
      try {
        const script = join(dir, "install.sh")
        writeFileSync(script, shell, { mode: 0o755 })
        execFileSync("sh", ["-n", script], { encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] })
      } finally {
        rmSync(dir, { recursive: true, force: true })
      }
    },
    catch: (error) => {
      const detail = error instanceof Error && "stderr" in error && typeof error.stderr === "string" && error.stderr.trim()
        ? error.stderr.trim()
        : error instanceof Error
          ? error.message
          : String(error)
      return new SyncInstallationScriptsError({ message: `Rendered install.sh failed sh -n syntax check: ${detail}` })
    },
  })

/**
 * Renders `install.sh` and `install.ps1` from this repository and syncs them into the checked-out
 * landing repository's `public/` directory on stable releases. Safe to re-run: identical files are
 * a clean no-op with no commit or push.
 */
export const syncInstallationScripts = (options: SyncInstallationScriptsOptions) =>
  Effect.gen(function* () {
    if (validSemver(options.version) === null) {
      return yield* fail(`Invalid release version: ${options.version}`)
    }
    if (!/^[a-f0-9]{40}$/.test(options.sourceCommit)) {
      return yield* fail(`Invalid source commit SHA: ${options.sourceCommit}`)
    }
    if (isPrereleaseVersion(options.version)) {
      return { status: "skipped-prerelease", changed: [] } as const satisfies SyncInstallationScriptsResult
    }

    const fs = yield* FileSystem.FileSystem
    const branch = options.branch ?? "main"
    const push = options.push ?? true
    const authorName = options.authorName ?? "github-actions[bot]"
    const authorEmail = options.authorEmail ?? "41898282+github-actions[bot]@users.noreply.github.com"

    const shell = yield* renderUnixInstallationScript({
      origin: options.origin,
      appleTeam: options.appleTeam,
      publicKey: options.publicKey,
    })
    const powershell = yield* renderWindowsInstallationScript({
      origin: options.origin,
      publisher: options.windowsPublisher,
    })
    if (shell.includes("@MAGNITUDE_")) {
      return yield* fail("Rendered install.sh still contains an unreplaced @MAGNITUDE_ placeholder")
    }
    yield* validateUnixShellSyntax(shell)

    const publicDir = join(options.landingRepo, "public")
    if (!(yield* fs.exists(publicDir))) {
      return yield* fail(`Landing repository public/ directory does not exist: ${publicDir}`)
    }

    const shellPath = join(publicDir, "install.sh")
    const powershellPath = join(publicDir, "install.ps1")
    const currentShell = (yield* fs.exists(shellPath)) ? yield* fs.readFileString(shellPath) : undefined
    const currentPowershell = (yield* fs.exists(powershellPath)) ? yield* fs.readFileString(powershellPath) : undefined

    const changed: ("install.sh" | "install.ps1")[] = []
    if (currentShell !== shell) {
      yield* fs.writeFileString(shellPath, shell, { mode: 0o755 })
      yield* Effect.sync(() => chmodSync(shellPath, 0o755))
      changed.push("install.sh")
    }
    if (currentPowershell !== powershell) {
      yield* fs.writeFileString(powershellPath, powershell, { mode: 0o644 })
      changed.push("install.ps1")
    }

    if (changed.length === 0) {
      return { status: "up-to-date", changed: [] } as const satisfies SyncInstallationScriptsResult
    }

    yield* runCommand("git", ["add", "--", "public/install.sh", "public/install.ps1"], options.landingRepo)
    const stagedDiff = yield* runCommand(
      "git",
      ["diff", "--cached", "--name-only", "--", "public/install.sh", "public/install.ps1"],
      options.landingRepo,
    )
    if (!stagedDiff) {
      return { status: "up-to-date", changed: [] } as const satisfies SyncInstallationScriptsResult
    }

    const message = [
      `Sync install.sh and install.ps1 for v${options.version}`,
      "",
      `Rendered from magnitudedev/magnitude@${options.sourceCommit}.`,
    ].join("\n")

    yield* runCommand(
      "git",
      ["-c", `user.name=${authorName}`, "-c", `user.email=${authorEmail}`, "commit", "-m", message, "--", "public/install.sh", "public/install.ps1"],
      options.landingRepo,
    )
    const commit = yield* runCommand("git", ["rev-parse", "HEAD"], options.landingRepo)
    if (push) {
      yield* runCommand("git", ["push", "origin", `HEAD:refs/heads/${branch}`], options.landingRepo)
    }
    return { status: "synced", changed, commit } as const satisfies SyncInstallationScriptsResult
  })
