import { FetchHttpClient } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { Effect, Option } from "effect"
import { afterEach, describe, expect, it } from "vitest"
import {
  installMethodFromEnvironment,
  isDevelopmentVersion,
  isNewerVersion,
  makeCliUpdater,
  updateActionFor,
  updateCommandString,
} from "./cli-update"
import { currentHost } from "@magnitudedev/release"

const roots: string[] = []

afterEach(async () => {
  await Promise.all(roots.splice(0).map((root) =>
    rm(root, { recursive: true, force: true })
  ))
})

describe("CLI updater", () => {
  it("maps launcher provenance to Codex-style global update commands", () => {
    expect(installMethodFromEnvironment({ MAGNITUDE_MANAGED_BY_NPM: "1" }))
      .toBe("npm")
    expect(installMethodFromEnvironment({ MAGNITUDE_MANAGED_BY_BUN: "1" }))
      .toBe("bun")
    expect(installMethodFromEnvironment({ MAGNITUDE_MANAGED_BY_PNPM: "1" }))
      .toBe("pnpm")
    expect(installMethodFromEnvironment({})).toBe("other")

    expect(updateCommandString(Option.getOrThrow(updateActionFor("npm"))))
      .toBe("npm install -g @magnitudedev/cli")
    expect(updateCommandString(Option.getOrThrow(updateActionFor("bun"))))
      .toBe("bun install -g @magnitudedev/cli")
    expect(updateCommandString(Option.getOrThrow(updateActionFor("pnpm"))))
      .toBe("pnpm add -g @magnitudedev/cli")
    expect(Option.isNone(updateActionFor("other"))).toBe(true)
  })

  it("uses semantic version ordering and disables development versions", () => {
    expect(isNewerVersion("0.0.1-alpha.35", "0.0.1-alpha.34")).toBe(true)
    expect(isNewerVersion("0.0.1-alpha.34", "0.0.1-alpha.35")).toBe(false)
    expect(isNewerVersion("not-a-version", "0.0.1-alpha.35")).toBe(false)
    expect(isDevelopmentVersion("0.0.1-alpha.35+dev.abc.1")).toBe(true)
    expect(isDevelopmentVersion("0.0.1-alpha.35")).toBe(false)
  })

  it("executes update actions directly and reports a nonzero status", async () => {
    const root = await mkdtemp(join(tmpdir(), "magnitude-updater-command-"))
    roots.push(root)
    const updater = await Effect.runPromise(
      makeCliUpdater({
        currentVersion: "1.0.0",
        dataDir: root,
        environment: { MAGNITUDE_MANAGED_BY_NPM: "1" },
      }).pipe(Effect.provide([BunContext.layer, FetchHttpClient.layer])),
    )

    await expect(Effect.runPromise(updater.runUpdate({
      method: "npm",
      command: process.execPath,
      args: ["-e", "process.exit(0)"],
    }))).resolves.toBeUndefined()

    const failure = await Effect.runPromise(Effect.flip(updater.runUpdate({
      method: "npm",
      command: process.execPath,
      args: ["-e", "process.exit(7)"],
    })))
    expect(failure).toMatchObject({
      _tag: "UpdateCommandFailed",
      reason: "exited with status 7",
    })
  })

  it("refreshes in the background, verifies the native release, and honors dismissal", async () => {
    const root = await mkdtemp(join(tmpdir(), "magnitude-updater-"))
    roots.push(root)
    let latestVersion = "0.0.1-alpha.35"
    const server = Bun.serve({
      port: 0,
      fetch(request) {
        const pathname = new URL(request.url).pathname
        if (pathname === "/registry") {
          return Response.json({
            latest: latestVersion,
            alpha: "0.0.1-alpha.0",
          })
        }
        return Response.json({
          schemaVersion: 2,
          version: latestVersion,
          acnRevision: 1,
          tag: `@magnitudedev/cli@${latestVersion}`,
          sourceCommit: "a".repeat(40),
          artifacts: [{
            id: `cli-${currentHost()}`,
            kind: "cli",
            host: currentHost(),
            filename: "magnitude-cli.tar.gz",
            bytes: 1,
            sha256: "b".repeat(64),
          }],
        })
      },
    })

    try {
      const updater = await Effect.runPromise(
        makeCliUpdater({
          currentVersion: "0.0.1-alpha.34",
          dataDir: root,
          environment: { MAGNITUDE_MANAGED_BY_NPM: "1" },
          npmPackageUrl: `${server.url}registry`,
          releaseBaseUrl: server.url.toString(),
        }).pipe(Effect.provide([BunContext.layer, FetchHttpClient.layer])),
      )

      expect(Option.isNone(await Effect.runPromise(updater.getUpgradeVersion)))
        .toBe(true)

      const cachePath = join(root, "version.json")
      await expect.poll(async () => {
        try {
          return JSON.parse(await readFile(cachePath, "utf8")).latestVersion
        } catch {
          return null
        }
      }).toBe(latestVersion)

      expect(Option.getOrNull(
        await Effect.runPromise(updater.getUpgradeVersion),
      )).toBe(latestVersion)

      await Effect.runPromise(updater.dismissVersion(latestVersion))
      expect(Option.isNone(await Effect.runPromise(updater.getUpgradeVersion)))
        .toBe(true)

      await writeFile(cachePath, JSON.stringify({
        latestVersion,
        lastCheckedAt: new Date(0).toISOString(),
      }))
      latestVersion = "0.0.1-alpha.36"
      expect(Option.isNone(await Effect.runPromise(updater.getUpgradeVersion)))
        .toBe(true)
      await expect.poll(async () => {
        try {
          return JSON.parse(await readFile(cachePath, "utf8")).latestVersion
        } catch {
          return null
        }
      }).toBe(latestVersion)
      expect(Option.getOrNull(
        await Effect.runPromise(updater.getUpgradeVersion),
      )).toBe(latestVersion)
    } finally {
      server.stop(true)
    }
  })

  it("never caches a registry target without verifying its native release", async () => {
    const root = await mkdtemp(join(tmpdir(), "magnitude-updater-readiness-"))
    roots.push(root)
    let releaseRequests = 0
    const server = Bun.serve({
      port: 0,
      fetch(request) {
        if (new URL(request.url).pathname === "/registry") {
          return Response.json({ latest: "1.5.0" })
        }
        releaseRequests += 1
        return new Response("missing", { status: 400 })
      },
    })

    try {
      const newerUpdater = await Effect.runPromise(
        makeCliUpdater({
          currentVersion: "2.0.0",
          dataDir: root,
          environment: { MAGNITUDE_MANAGED_BY_NPM: "1" },
          npmPackageUrl: `${server.url}registry`,
          releaseBaseUrl: server.url.toString(),
        }).pipe(Effect.provide([BunContext.layer, FetchHttpClient.layer])),
      )
      expect(Option.isNone(
        await Effect.runPromise(newerUpdater.getUpgradeVersion),
      )).toBe(true)
      await expect.poll(() => releaseRequests).toBe(1)

      const olderUpdater = await Effect.runPromise(
        makeCliUpdater({
          currentVersion: "1.0.0",
          dataDir: root,
          environment: { MAGNITUDE_MANAGED_BY_NPM: "1" },
          npmPackageUrl: `${server.url}registry`,
          releaseBaseUrl: server.url.toString(),
        }).pipe(Effect.provide([BunContext.layer, FetchHttpClient.layer])),
      )
      expect(Option.isNone(
        await Effect.runPromise(olderUpdater.getUpgradeVersion),
      )).toBe(true)
    } finally {
      server.stop(true)
    }
  })

  it("does not refresh when startup checks are disabled", async () => {
    const root = await mkdtemp(join(tmpdir(), "magnitude-updater-disabled-"))
    roots.push(root)
    await writeFile(
      join(root, "config.json"),
      JSON.stringify({ checkForUpdateOnStartup: false }),
    )
    let requests = 0
    const server = Bun.serve({
      port: 0,
      fetch() {
        requests += 1
        return Response.json({ latest: "99.0.0" })
      },
    })

    try {
      const updater = await Effect.runPromise(
        makeCliUpdater({
          currentVersion: "1.0.0",
          dataDir: root,
          environment: { MAGNITUDE_MANAGED_BY_NPM: "1" },
          npmPackageUrl: server.url.toString(),
          releaseBaseUrl: server.url.toString(),
        }).pipe(Effect.provide([BunContext.layer, FetchHttpClient.layer])),
      )
      expect(Option.isNone(await Effect.runPromise(updater.getUpgradeVersion)))
        .toBe(true)
      expect(requests).toBe(0)
    } finally {
      server.stop(true)
    }
  })
})
