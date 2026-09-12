import { FetchHttpClient } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { mkdtemp, rm } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { Effect, Option } from "effect"
import { afterEach, describe, expect, it } from "vitest"
import {
  isDevelopmentVersion,
  makeCliUpdater,
} from "./updater"
import { currentHost } from "@magnitudedev/release"

const roots: string[] = []

afterEach(async () => {
  await Promise.all(roots.splice(0).map((root) =>
    rm(root, { recursive: true, force: true })
  ))
})

describe("CLI updater", () => {
  it("recognizes development versions", () => {
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

  it("selects by release channel and admits only the client's channels", async () => {
    const tags = {
      latest: "1.0.0",
      beta: "1.1.0-beta.2",
      alpha: "1.2.0-alpha.1",
    }
    const server = Bun.serve({
      port: 0,
      fetch(request) {
        const pathname = decodeURIComponent(new URL(request.url).pathname)
        if (pathname === "/registry") return Response.json(tags)
        const version = /cli@([^/]+)/.exec(pathname)?.[1]
        if (!version) return new Response("not found", { status: 404 })
        return Response.json({
          schemaVersion: 2,
          version,
          acnRevision: 1,
          rpc: { version: 1, fingerprint: "a".repeat(64) },
          plugins: [{ host: "pi", name: "@magnitudedev/pi-extension", version: "0.0.1", rpcVersion: 1, contentFingerprint: "a".repeat(64), filename: "pi.tgz", integrity: "sha512-dGVzdA==" }],
          tag: `@magnitudedev/cli@${version}`,
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

    const updaterOn = async (currentVersion: string) => {
      const dataDir = await mkdtemp(join(tmpdir(), "magnitude-updater-channel-"))
      roots.push(dataDir)
      return Effect.runPromise(
        makeCliUpdater({
          currentVersion,
          dataDir,
          environment: { MAGNITUDE_MANAGED_BY: "npm" },
          npmPackageUrl: `${server.url}registry`,
          releaseBaseUrl: server.url.toString(),
        }).pipe(Effect.provide([BunContext.layer, FetchHttpClient.layer])),
      )
    }

    try {
      // Stable clients follow only stable releases.
      const stable = await updaterOn("0.9.0")
      expect(Option.getOrNull(await Effect.runPromise(stable.updateTarget)))
        .toBe("1.0.0")

      // Beta clients follow stable and beta, never alpha.
      const beta = await updaterOn("1.0.0-beta.1")
      expect(Option.getOrNull(await Effect.runPromise(beta.updateTarget)))
        .toBe("1.1.0-beta.2")

      // Alpha clients follow everything; the highest admissible version wins.
      const alpha = await updaterOn("1.0.0-alpha.5")
      expect(Option.getOrNull(await Effect.runPromise(alpha.updateTarget)))
        .toBe("1.2.0-alpha.1")

    } finally {
      server.stop(true)
    }
  })

  it("rejects a registry target without a matching native release", async () => {
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
      const updater = await Effect.runPromise(
        makeCliUpdater({
          currentVersion: "1.0.0",
          dataDir: root,
          environment: { MAGNITUDE_MANAGED_BY: "npm" },
          npmPackageUrl: `${server.url}registry`,
          releaseBaseUrl: server.url.toString(),
        }).pipe(Effect.provide([BunContext.layer, FetchHttpClient.layer])),
      )
      expect(Option.isNone(await Effect.runPromise(updater.updateTarget))).toBe(true)
      expect(releaseRequests).toBe(1)
    } finally {
      server.stop(true)
    }
  })

  it("does no network work until an explicit check and reports registry failure", async () => {
    const root = await mkdtemp(join(tmpdir(), "magnitude-updater-explicit-"))
    roots.push(root)
    let requests = 0
    const server = Bun.serve({
      port: 0,
      fetch() {
        requests += 1
        return new Response("unavailable", { status: 503 })
      },
    })
    try {
      const updater = await Effect.runPromise(makeCliUpdater({
        currentVersion: "1.0.0",
        dataDir: root,
        environment: { MAGNITUDE_MANAGED_BY: "npm" },
        npmPackageUrl: server.url.toString(),
        releaseBaseUrl: server.url.toString(),
      }).pipe(Effect.provide([BunContext.layer, FetchHttpClient.layer])))
      expect(requests).toBe(0)
      const failure = await Effect.runPromise(Effect.flip(updater.updateTarget))
      expect(failure).toMatchObject({
        _tag: "UpdateDiscoveryFailed",
        stage: "registry",
        reason: "npm registry returned HTTP 503",
      })
      expect(requests).toBe(1)
    } finally {
      server.stop(true)
    }
  })
})
