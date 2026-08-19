import { FetchHttpClient } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { Effect, Exit, Option, Scope } from "effect"
import { afterEach, describe, expect, it } from "vitest"
import {
  isDevelopmentVersion,
  isNewerVersion,
  makeCliUpdater,
} from "./updater"
import { currentHost } from "@magnitudedev/release"

const roots: string[] = []

const runScoped = <A, E>(
  scope: Scope.CloseableScope,
  effect: Effect.Effect<A, E, Scope.Scope>,
) => Effect.runPromise(
  effect.pipe(Effect.provideService(Scope.Scope, scope)),
)

afterEach(async () => {
  await Promise.all(roots.splice(0).map((root) =>
    rm(root, { recursive: true, force: true })
  ))
})

describe("CLI updater", () => {
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
        environment: { MAGNITUDE_MANAGED_BY: "npm" },
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

  it("checks on every launch, verifies the native release, and honors dismissal", async () => {
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

    const scope = await Effect.runPromise(Scope.make())
    try {
      const updater = await Effect.runPromise(
        makeCliUpdater({
          currentVersion: "0.0.1-alpha.34",
          dataDir: root,
          environment: { MAGNITUDE_MANAGED_BY: "npm" },
          npmPackageUrl: `${server.url}registry`,
          releaseBaseUrl: server.url.toString(),
        }).pipe(Effect.provide([BunContext.layer, FetchHttpClient.layer])),
      )

      // Cold cache: nothing known, this launch's check delivers the answer.
      const first = await runScoped(scope, updater.discover)
      expect(Option.isNone(first.known)).toBe(true)
      expect(Option.getOrNull(await Effect.runPromise(first.fresh)))
        .toBe(latestVersion)

      // The cache keeps only the last known answer.
      const cachePath = join(root, "state", "version.json")
      expect(JSON.parse(await readFile(cachePath, "utf8")))
        .toEqual({ latestVersion })

      // Next launch knows the answer without the network.
      const second = await runScoped(scope, updater.discover)
      expect(Option.getOrNull(second.known)).toBe(latestVersion)

      // Dismissal suppresses both the known and the fresh answer.
      await Effect.runPromise(updater.dismissVersion(latestVersion))
      const third = await runScoped(scope, updater.discover)
      expect(Option.isNone(third.known)).toBe(true)
      expect(Option.isNone(await Effect.runPromise(third.fresh))).toBe(true)

      // A newer version is eligible again despite the dismissal.
      latestVersion = "0.0.1-alpha.36"
      const fourth = await runScoped(scope, updater.discover)
      expect(Option.isNone(fourth.known)).toBe(true)
      expect(Option.getOrNull(await Effect.runPromise(fourth.fresh)))
        .toBe(latestVersion)

      // A successful check is authoritative: a rolled-back registry retracts
      // the cached offer instead of falling back to it.
      await writeFile(cachePath, JSON.stringify({ latestVersion: "0.0.1-alpha.40" }))
      latestVersion = "0.0.1-alpha.34"
      const fifth = await runScoped(scope, updater.discover)
      expect(Option.getOrNull(fifth.known)).toBe("0.0.1-alpha.40")
      expect(Option.isNone(await Effect.runPromise(fifth.fresh))).toBe(true)
    } finally {
      await Effect.runPromise(Scope.close(scope, Exit.void))
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

    const scope = await Effect.runPromise(Scope.make())
    try {
      const updater = await Effect.runPromise(
        makeCliUpdater({
          currentVersion: "1.0.0",
          dataDir: root,
          environment: { MAGNITUDE_MANAGED_BY: "npm" },
          npmPackageUrl: `${server.url}registry`,
          releaseBaseUrl: server.url.toString(),
        }).pipe(Effect.provide([BunContext.layer, FetchHttpClient.layer])),
      )
      const discovery = await runScoped(scope, updater.discover)
      expect(Option.isNone(await Effect.runPromise(discovery.fresh))).toBe(true)
      expect(releaseRequests).toBe(1)

      // The failed readiness check never wrote a cache entry.
      const next = await runScoped(scope, updater.discover)
      expect(Option.isNone(next.known)).toBe(true)
    } finally {
      await Effect.runPromise(Scope.close(scope, Exit.void))
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

    const scope = await Effect.runPromise(Scope.make())
    try {
      const updater = await Effect.runPromise(
        makeCliUpdater({
          currentVersion: "1.0.0",
          dataDir: root,
          environment: { MAGNITUDE_MANAGED_BY: "npm" },
          npmPackageUrl: server.url.toString(),
          releaseBaseUrl: server.url.toString(),
        }).pipe(Effect.provide([BunContext.layer, FetchHttpClient.layer])),
      )
      const discovery = await runScoped(scope, updater.discover)
      expect(Option.isNone(discovery.known)).toBe(true)
      expect(Option.isNone(await Effect.runPromise(discovery.fresh))).toBe(true)
      expect(requests).toBe(0)
    } finally {
      await Effect.runPromise(Scope.close(scope, Exit.void))
      server.stop(true)
    }
  })
})
