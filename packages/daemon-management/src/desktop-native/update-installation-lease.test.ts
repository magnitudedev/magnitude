import { Effect, Option } from "effect"
import { describe, expect, it } from "vitest"
import { acquireUpdateInstallationLease, isUpdateInstallationActive } from "./update-installation-lease"
import { NativeHost, nativeHostLayerFromLoader } from "./index"

// Exercise the actual scoped native adapter, rather than giving the lease a second lock implementation.
describe("update installation exclusion", () => {
  it("holds the helper lease until its scope exits and releases a passive probe immediately", async () => {
    let held = false
    const layer = nativeHostLayerFromLoader(() => ({
      acquireLock: (path: string) => { expect(path).toContain("update-installation.lock"); if (held) return null; held = true; return {} },
      releaseLock: () => { held = false },
    }))
    await Effect.runPromise(Effect.gen(function* () {
      expect(yield* isUpdateInstallationActive("/fixture/state")).toBe(false)
      expect(held).toBe(false)
      yield* Effect.scoped(Effect.gen(function* () {
        yield* acquireUpdateInstallationLease("/fixture/state")
        expect(held).toBe(true)
        expect(yield* isUpdateInstallationActive("/fixture/state")).toBe(true)
        expect((yield* Effect.scoped(acquireUpdateInstallationLease("/fixture/state")).pipe(Effect.either))._tag).toBe("Left")
        expect(held).toBe(true)
      }))
      expect(held).toBe(false)
    }).pipe(Effect.provide(layer)))
  })
})
