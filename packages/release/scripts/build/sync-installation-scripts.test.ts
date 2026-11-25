import { NodeContext } from "@effect/platform-node"
import { Effect } from "effect"
import { execFileSync } from "node:child_process"
import { generateKeyPairSync } from "node:crypto"
import { mkdirSync, mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { afterEach, describe, expect, it } from "vitest"
import { syncInstallationScripts } from "./sync-installation-scripts"

const publicKey = generateKeyPairSync("ed25519").publicKey.export({ type: "spki", format: "pem" }).toString()
const sourceCommit = "a".repeat(40)
const tempRoots: string[] = []

const git = (cwd: string, ...args: string[]) =>
  execFileSync("git", args, { cwd, encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] }).trim()

const createLandingFixture = (initial?: { readonly sh?: string; readonly ps1?: string }) => {
  const root = mkdtempSync(join(tmpdir(), "magnitude-landing-sync-"))
  tempRoots.push(root)
  const remote = join(root, "remote.git")
  const clone = join(root, "landing")
  git(root, "init", "--bare", "--initial-branch=main", remote)
  git(root, "clone", remote, clone)
  mkdirSync(join(clone, "public"), { recursive: true })
  writeFileSync(join(clone, "public", "install.sh"), initial?.sh ?? "#!/bin/sh\necho old\n", { mode: 0o755 })
  writeFileSync(join(clone, "public", "install.ps1"), initial?.ps1 ?? "# old\n")
  git(clone, "add", ".")
  git(clone, "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-m", "Initial landing commit")
  git(clone, "push", "-u", "origin", "main")
  return { root, remote, clone }
}

const runSync = (options: Parameters<typeof syncInstallationScripts>[0]) =>
  Effect.runPromise(syncInstallationScripts(options).pipe(Effect.provide(NodeContext.layer)))

const baseOptions = (landingRepo: string, overrides: Partial<Parameters<typeof syncInstallationScripts>[0]> = {}) => ({
  landingRepo,
  version: "0.3.3",
  sourceCommit,
  origin: "https://magnitude.dev",
  appleTeam: "ABCDEFGHIJ",
  windowsPublisher: "Magnitude AI Inc.",
  publicKey,
  ...overrides,
})

afterEach(() => {
  for (const root of tempRoots.splice(0)) rmSync(root, { recursive: true, force: true })
})

describe("syncInstallationScripts", () => {
  it("renders, validates, commits, and pushes updated install scripts on a stable release, and is idempotent", async () => {
    const { remote, clone } = createLandingFixture()
    const initialRemoteHead = git(remote, "rev-parse", "refs/heads/main")

    const first = await runSync(baseOptions(clone))
    expect(first.status).toBe("synced")
    if (first.status !== "synced") throw new Error("expected synced")
    expect(first.changed).toEqual(["install.sh", "install.ps1"])

    const remoteHead = git(remote, "rev-parse", "refs/heads/main")
    expect(remoteHead).toBe(first.commit)
    expect(remoteHead).not.toBe(initialRemoteHead)
    expect(git(remote, "log", "-1", "--pretty=%B", "refs/heads/main")).toContain("Sync install.sh and install.ps1 for v0.3.3")
    expect(git(remote, "log", "-1", "--pretty=%B", "refs/heads/main")).toContain(`magnitudedev/magnitude@${sourceCommit}`)

    const syncedSh = git(remote, "show", "refs/heads/main:public/install.sh")
    const syncedPs1 = git(remote, "show", "refs/heads/main:public/install.ps1")
    expect(syncedSh).toContain("origin='https://magnitude.dev'")
    expect(syncedSh).toContain("apple_team='ABCDEFGHIJ'")
    expect(syncedSh).not.toContain("@MAGNITUDE_")
    expect(syncedPs1).toContain("$origin = 'https://magnitude.dev'")
    expect(syncedPs1).toContain("$publisher = 'Magnitude AI Inc.'")
    expect(statSync(join(clone, "public", "install.sh")).mode & 0o111).not.toBe(0)

    // Re-running against the already-synced repo is a clean no-op with zero extra commits.
    const second = await runSync(baseOptions(clone))
    expect(second).toEqual({ status: "up-to-date", changed: [] })
    expect(git(remote, "rev-parse", "refs/heads/main")).toBe(remoteHead)
  })

  it("skips prerelease versions without touching landing files or pushing", async () => {
    const { remote, clone } = createLandingFixture()
    const initialRemoteHead = git(remote, "rev-parse", "refs/heads/main")

    for (const version of ["0.3.4-alpha.1", "0.3.4-beta.0"]) {
      const result = await runSync(baseOptions(clone, { version }))
      expect(result).toEqual({ status: "skipped-prerelease", changed: [] })
    }
    expect(readFileSync(join(clone, "public", "install.sh"), "utf8")).toBe("#!/bin/sh\necho old\n")
    expect(git(remote, "rev-parse", "refs/heads/main")).toBe(initialRemoteHead)
  })

  it("commits locally without pushing when push is false (dry run)", async () => {
    const { remote, clone } = createLandingFixture()
    const initialRemoteHead = git(remote, "rev-parse", "refs/heads/main")

    const result = await runSync(baseOptions(clone, { push: false }))
    expect(result.status).toBe("synced")
    expect(git(clone, "rev-parse", "HEAD")).not.toBe(initialRemoteHead)
    expect(git(remote, "rev-parse", "refs/heads/main")).toBe(initialRemoteHead)
  })

  it("rejects invalid versions and commit SHAs", async () => {
    const { clone } = createLandingFixture()
    await expect(runSync(baseOptions(clone, { version: "not-a-version" }))).rejects.toThrow("Invalid release version")
    await expect(runSync(baseOptions(clone, { sourceCommit: "deadbeef" }))).rejects.toThrow("Invalid source commit SHA")
  })
})
