import { Effect } from "effect"
import { mkdtempSync, rmSync, writeFileSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { afterEach, describe, expect, it } from "vitest"
import { acquireUpdateInstallationLease, isUpdateInstallationActive } from "./update-installation-lease"
import { nativeHostLayerFromLoader } from "./index"

const folders: string[] = []
afterEach(() => { for (const folder of folders.splice(0)) rmSync(folder, { recursive: true, force: true }) })

// Exercise the actual scoped native adapter, rather than giving the lease a second lock implementation.
describe("update installation exclusion", () => {
  it("holds the helper lease until its scope exits and releases a passive probe immediately", async () => {
    const state = mkdtempSync(join(tmpdir(), "update-lease-"))
    folders.push(state)
    let held = false
    let probes = 0
    const layer = nativeHostLayerFromLoader(() => ({
      acquireLock: (path: string) => { expect(path).toBe(join(state, "update-installation.lock")); probes += 1; if (held) return null; held = true; return {} },
      releaseLock: () => { held = false },
    }))
    await Effect.runPromise(Effect.gen(function* () {
      expect(yield* isUpdateInstallationActive(state)).toBe(false)
      expect(probes).toBe(0)
      writeFileSync(join(state, "update-installation.lock"), "")
      expect(yield* isUpdateInstallationActive(state)).toBe(false)
      expect(held).toBe(false)
      yield* Effect.scoped(Effect.gen(function* () {
        yield* acquireUpdateInstallationLease(state)
        expect(held).toBe(true)
        expect(yield* isUpdateInstallationActive(state)).toBe(true)
        expect((yield* Effect.scoped(acquireUpdateInstallationLease(state)).pipe(Effect.either))._tag).toBe("Left")
        expect(held).toBe(true)
      }))
      expect(held).toBe(false)
    }).pipe(Effect.provide(layer)))
  })

  it("reports no installation for a state folder that does not exist yet, without touching the lock", async () => {
    const layer = nativeHostLayerFromLoader(() => ({
      acquireLock: () => { throw new Error("Cannot open private ownership lock") },
      releaseLock: () => {},
    }))
    const missing = join(tmpdir(), `update-lease-missing-${process.pid}-${Date.now()}`, "state")
    expect(await Effect.runPromise(isUpdateInstallationActive(missing).pipe(Effect.provide(layer)))).toBe(false)
  })
})
