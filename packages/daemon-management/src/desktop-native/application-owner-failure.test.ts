import { Effect, Option } from "effect"
import { existsSync } from "node:fs"
import { mkdtemp, rm, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { describe, expect, it } from "vitest"
import { acquireApplicationMaintenance, applicationOwnershipHeld } from "./application-owner"
import { DataFolderUnusable } from "./application-control"
import { NativeHost, NativeHostUnavailable } from "./index"

const refusing = NativeHost.of({
  requireInteractiveDesktop: Effect.void, isServiceSession: Effect.succeed(false), localAppDataDirectory: Effect.succeed(""),
  acquireOwnership: () => Effect.fail(new NativeHostUnavailable({ message: "Error: Unsafe or inaccessible private ownership directory or lock" })),
  guardParent: () => Effect.void, ownedEndpoint: () => Effect.succeed(""), inspectEndpoint: () => Effect.succeed(Option.none()),
})

describe("an ownership lock that can't be opened", () => {
  it("names the data folder, so startup explains it instead of telling people to reinstall", async () => {
    const root = await mkdtemp(join(tmpdir(), "magnitude-owner-"))
    const directory = join(root, "a".repeat(40), "application")
    try {
      const failure = await Effect.runPromise(acquireApplicationMaintenance(directory).pipe(
        Effect.flip, Effect.scoped, Effect.provideService(NativeHost, refusing)))
      expect(failure).toBeInstanceOf(DataFolderUnusable)
      if (!(failure instanceof DataFolderUnusable)) return
      expect(failure.reason).toContain(directory)
      expect(failure.reason).not.toContain("Unsafe or inaccessible")
      expect(failure.message).toContain(directory)
      expect(failure.message).toContain("Unsafe or inaccessible private ownership directory or lock")
      expect(failure.message).not.toContain("Error:")
    } finally { await rm(root, { recursive: true, force: true }) }
  })
})

describe("applicationOwnershipHeld", () => {
  it("skips probing when application.lock is absent and reports lock contention when present", async () => {
    const root = await mkdtemp(join(tmpdir(), "magnitude-held-"))
    let probes = 0
    let held = true
    const host = NativeHost.of({
      requireInteractiveDesktop: Effect.void, isServiceSession: Effect.succeed(false), localAppDataDirectory: Effect.succeed(""),
      acquireOwnership: () => Effect.sync(() => { probes++; return held ? Option.none() : Option.some({}) }),
      guardParent: () => Effect.void, ownedEndpoint: () => Effect.succeed(""), inspectEndpoint: () => Effect.succeed(Option.none()),
    })
    try {
      expect(await Effect.runPromise(applicationOwnershipHeld(root).pipe(Effect.provideService(NativeHost, host)))).toBe(false)
      expect(probes).toBe(0)
      expect(existsSync(join(root, "application.lock"))).toBe(false)

      await writeFile(join(root, "application.lock"), "")
      expect(await Effect.runPromise(applicationOwnershipHeld(root).pipe(Effect.provideService(NativeHost, host)))).toBe(true)
      expect(probes).toBe(1)

      held = false
      expect(await Effect.runPromise(applicationOwnershipHeld(root).pipe(Effect.provideService(NativeHost, host)))).toBe(false)
      expect(probes).toBe(2)

      expect(await Effect.runPromise(applicationOwnershipHeld(root).pipe(Effect.provideService(NativeHost, refusing)))).toBe(false)
    } finally { await rm(root, { recursive: true, force: true }) }
  })
})
